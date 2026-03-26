import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from scipy.stats import pearsonr

from src.data.dataset import load_processed, temporal_split, make_loaders, TARGET_COLS
from src.model import GameEncoder, PlayerDecoder, reparameterize


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    encoder = GameEncoder(
        input_dim=cfg["team_dim"],
        h_dim=cfg["h_dim_enc"],
        latent_dim=cfg["latent_dim"],
        dropout=0.0,
    ).to(device)

    model_type = cfg.get("model_type", "standard")
    if model_type == "film":
        from src.model_film import PlayerDecoderFiLM
        decoder = PlayerDecoderFiLM(
            latent_dim=cfg["latent_dim"],
            player_dim=cfg["player_dim"],
            h_dim=cfg["h_dim_dec"],
            output_dim=cfg["n_target_cols"],
            dropout=0.0,
        ).to(device)
    else:
        decoder = PlayerDecoder(
            latent_dim=cfg["latent_dim"],
            player_dim=cfg["player_dim"],
            h_dim=cfg["h_dim_dec"],
            output_dim=cfg["n_target_cols"],
            dropout=0.0,
        ).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    return encoder, decoder, ckpt, cfg


def diagnose(ckpt_path, num_samples=500, device="cpu", season_suffix="2019-26"):
    print(f"\n=== Z-Sensitivity Diagnostic ===")
    print(f"Checkpoint: {ckpt_path}")

    encoder, decoder, ckpt, cfg = load_model(ckpt_path, device)
    latent_dim = cfg["latent_dim"]

    Y_mean = ckpt["Y_mean"].to(device)
    Y_std  = ckpt["Y_std"].to(device)
    Xt_mean = ckpt["Xt_mean"].to(device)
    Xt_std  = ckpt["Xt_std"].to(device)
    Xp_mean = ckpt["Xp_mean"].to(device)
    Xp_std  = ckpt["Xp_std"].to(device)

    print(f"Loading data (season {season_suffix})...")
    df = load_processed(season_suffix=season_suffix)
    _, val_df = temporal_split(df)
    print(f"  val rows: {len(val_df)}")

    _, val_loader, _, _, _, _, _, _ = make_loaders(
        df[df["gameDateTimeEst"] <= val_df["gameDateTimeEst"].min()],  # dummy train
        val_df,
        batch_size=64,
    )

    # reload val loader with correct normalization from checkpoint
    from src.data.dataset import build_tensors, NBADataset
    from torch.utils.data import DataLoader

    X_team_val, X_pl_val, weights_val, Y_val, lines_val = build_tensors(val_df)
    Y_val_norm = (Y_val - ckpt["Y_mean"]) / ckpt["Y_std"]
    X_team_val_norm = (X_team_val - ckpt["Xt_mean"]) / ckpt["Xt_std"]
    X_pl_val_norm = (X_pl_val - ckpt["Xp_mean"]) / ckpt["Xp_std"]

    val_ds = NBADataset(X_team_val_norm, X_pl_val_norm, Y_val_norm, weights_val, lines_val)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"  val games: {len(val_ds)}")

    # ---- collect P(over|z) across samples ----
    all_p_over_std = []       # std of P(over|z) across z samples, per player-stat
    all_same_team_corrs = []  # Pearson corr between same-team players' P(over|z) traces
    all_kl_per_dim = []
    all_mu_z_std = []
    all_sigma_z = []

    example_games = []   # store a few for display

    lines_norm = torch.zeros(1)  # lines are 0 in normalized space (threshold = 0)

    with torch.no_grad():
        for batch_idx, (X_t, X_p, Y, weights, lines) in enumerate(val_loader):
            X_t, X_p, Y, weights = X_t.to(device), X_p.to(device), Y.to(device), weights.to(device)
            mask = (weights > 0).float()  # (batch, 16)

            mu, logvar = encoder(X_t)

            # KL stats
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            all_kl_per_dim.append(kl_per_dim.mean(dim=0).cpu())

            # z spread stats
            all_mu_z_std.append(mu.std(dim=0).cpu())  # across games in batch
            sigma_z = (0.5 * logvar).exp()
            all_sigma_z.append(sigma_z.mean(dim=0).cpu())

            # sample z and compute P(over|z) for each sample
            p_over_samples = []  # (num_samples, batch, 16, 3)
            for _ in range(num_samples):
                z = reparameterize(mu, logvar)
                mu_pred, logvar_pred = decoder(z, X_p)
                sigma_pred = (0.5 * logvar_pred).exp()
                # lines = 0 in normalized space
                z_score = mu_pred / sigma_pred
                p_over = 0.5 * (1.0 + torch.erf(z_score / (2.0 ** 0.5)))
                p_over_samples.append(p_over.cpu())

            p_over_tensor = torch.stack(p_over_samples, dim=0)  # (S, B, 16, 3)
            S, B, P, T = p_over_tensor.shape

            # std of P(over|z) across z samples: (B, 16, 3)
            p_over_std = p_over_tensor.std(dim=0)  # (B, 16, 3)

            # only keep non-padded players
            for b in range(B):
                valid = mask[b].cpu().bool()  # (16,)
                # per-player-stat std
                p_std = p_over_std[b][valid]  # (n_valid, 3)
                all_p_over_std.append(p_std.numpy())

                # same-team correlation: home is first 8, away is next 8
                n_home = int(valid[:8].sum().item())
                n_away = int(valid[8:].sum().item())

                p_traces = p_over_tensor[:, b, :, :]  # (S, 16, 3) — P(over|z) traces

                # collect same-team pairs across all stats
                for team_start, team_n in [(0, n_home), (8, n_away)]:
                    for i in range(team_start, team_start + team_n):
                        for j in range(i + 1, team_start + team_n):
                            for s in range(T):
                                trace_i = p_traces[:, i, s].numpy()
                                trace_j = p_traces[:, j, s].numpy()
                                if trace_i.std() > 1e-6 and trace_j.std() > 1e-6:
                                    r, _ = pearsonr(trace_i, trace_j)
                                    all_same_team_corrs.append(r)

                # store example games
                if len(example_games) < 3:
                    example_games.append({
                        "p_traces": p_traces[:, :min(3, P), :1].squeeze(-1).numpy(),  # (S, 3 players)
                        "p_std": p_std[:3, 0].numpy() if len(p_std) >= 3 else p_std[:, 0].numpy(),
                    })

    # ---- summarize ----
    all_p_over_std_flat = np.concatenate([x.ravel() for x in all_p_over_std])
    kl_per_dim_mean = torch.stack(all_kl_per_dim).mean(dim=0)
    mu_z_std_mean = torch.stack(all_mu_z_std).mean(dim=0)
    sigma_z_mean = torch.stack(all_sigma_z).mean(dim=0)

    print(f"\n--- P(over|z) std across {num_samples} z samples ---")
    print(f"  Mean  : {all_p_over_std_flat.mean():.4f}")
    print(f"  Std   : {all_p_over_std_flat.std():.4f}")
    print(f"  Median: {np.median(all_p_over_std_flat):.4f}")
    print(f"  p95   : {np.percentile(all_p_over_std_flat, 95):.4f}")
    print(f"  Max   : {all_p_over_std_flat.max():.4f}")

    print(f"\n--- Same-team P(over|z) trace correlation ---")
    if all_same_team_corrs:
        corrs = np.array(all_same_team_corrs)
        print(f"  n pairs : {len(corrs)}")
        print(f"  Mean    : {corrs.mean():.4f}")
        print(f"  Std     : {corrs.std():.4f}")
        print(f"  Median  : {np.median(corrs):.4f}")
    else:
        print("  No valid pairs found (all traces constant)")

    print(f"\n--- KL / latent dim ---")
    print(f"  KL/dim mean  : {kl_per_dim_mean.mean().item():.4f}")
    print(f"  KL/dim std   : {kl_per_dim_mean.std().item():.4f}")
    print(f"  KL/dim max   : {kl_per_dim_mean.max().item():.4f}")
    print(f"  KL/dim min   : {kl_per_dim_mean.min().item():.4f}")

    print(f"\n--- z distribution stats ---")
    print(f"  mu_z std across games (mean over dims): {mu_z_std_mean.mean().item():.4f}")
    print(f"  sigma_z mean (posterior spread):        {sigma_z_mean.mean().item():.4f}")

    print(f"\n--- Example games: P(over|z) across {num_samples} z samples ---")
    stat_names = TARGET_COLS
    for i, eg in enumerate(example_games):
        print(f"\n  Game {i+1}:")
        for p in range(eg["p_traces"].shape[1]):
            trace = eg["p_traces"][:, p]
            print(f"    Player {p+1} ({stat_names[0]}): mean={trace.mean():.3f}  std={trace.std():.4f}  "
                  f"range=[{trace.min():.3f}, {trace.max():.3f}]")

    return {
        "p_over_std_mean": float(all_p_over_std_flat.mean()),
        "p_over_std_std": float(all_p_over_std_flat.std()),
        "kl_per_dim_mean": float(kl_per_dim_mean.mean().item()),
        "same_team_corr_mean": float(np.mean(all_same_team_corrs)) if all_same_team_corrs else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/model_latest.pt")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--season", type=str, default="2019-26")
    args = parser.parse_args()

    diagnose(
        ckpt_path=args.ckpt,
        num_samples=args.num_samples,
        season_suffix=args.season,
    )

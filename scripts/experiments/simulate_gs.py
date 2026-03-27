"""
Simulation for two-stage Game State VAE.

Replaces z sampling with G sampling:
  encoder(X_team) → (mu_G, logvar_G)
  G_sample ~ N(mu_G, sigma_G)   ← all players in game share this
  decoder(G_sample, X_players) → (mu_pred, logvar_pred)
  P(over|G_sample) = Phi(mu_pred / sigma_pred)

Reports phi distribution, bets placed, and ROI.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from src.data.dataset import load_processed, temporal_split, TARGET_COLS
from src.data.game_state_dataset import make_loaders_gs, G_DIM
from src.model_gs import GCondDecoder, GameEncoder, reparameterize
from src.evaluate import phi_coefficient, extract_pairs, backtest

PLAYER_DIM = 24
H_DIM_DEC = 64
H_DIM_ENC = 128
DROPOUT = 0.3
NUM_SAMPLES = 500
THRESHOLD = 0.15
TOP_K = 10
DEVICE = "cpu"


def simulate_gs(encoder, decoder, loader, num_samples=NUM_SAMPLES, device=DEVICE):
    encoder.eval()
    decoder.eval()
    results = []

    with torch.no_grad():
        for X_t, X_p, Y, weights, lines, G, G_mask in loader:
            X_t = X_t.to(device)
            X_p = X_p.to(device)
            weights = weights.to(device)

            mask = (weights > 0).float()
            mu_G, logvar_G = encoder(X_t)

            # Sample G num_samples times — each sample = a different "game scenario"
            over_samples = []
            for _ in range(num_samples):
                G_sample = reparameterize(mu_G, logvar_G)
                mu_pred, logvar_pred = decoder(G_sample, X_p)
                sigma_pred = (0.5 * logvar_pred).exp()
                # threshold = 0 since targets are residuals (actual - h_stat)
                p_over = torch.distributions.Normal(0, 1).cdf(mu_pred / (sigma_pred + 1e-8))
                over_samples.append(p_over)

            over = torch.stack(over_samples, dim=0)  # (num_samples, batch, 16, 3)
            batch_size, n_players, n_stats = over.shape[1], over.shape[2], over.shape[3]

            A = over.permute(1, 2, 3, 0).reshape(batch_size, n_players * n_stats, num_samples).float()

            # Joint outcomes
            S = A.sum(dim=2)
            OO = A @ A.transpose(1, 2)
            OU = S.unsqueeze(2) - OO
            UO = S.unsqueeze(1) - OO
            UU = num_samples - OO - OU - UO

            over_mean = A.mean(dim=2)

            # Actual outcomes (residuals vs 0 threshold)
            actual_over = (Y > 0).int()
            n_vars = n_players * n_stats
            mask_flat = mask.unsqueeze(-1).expand(-1, -1, n_stats).reshape(batch_size, n_vars)
            actual_over_flat = actual_over.reshape(batch_size, n_vars) * mask_flat.int()

            results.append({
                "OO": OO, "OU": OU, "UO": UO, "UU": UU,
                "over_mean": over_mean,
                "actual_over": actual_over_flat,
                "mask_flat": mask_flat,
            })

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints_gs2")
    p.add_argument("--season", default="2019-26")
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    print(f"Loading data ({args.season})...")
    df = load_processed(season_suffix=args.season)
    _, val_df = temporal_split(df)
    _, val_loader, norm_stats = make_loaders_gs(val_df, val_df, batch_size=64, include_pbp=False)

    first_batch = next(iter(val_loader))
    team_dim = first_batch[0].shape[1]

    decoder = GCondDecoder(g_dim=G_DIM, player_dim=PLAYER_DIM, h_dim=H_DIM_DEC,
                            output_dim=len(TARGET_COLS), dropout=DROPOUT)
    encoder = GameEncoder(input_dim=team_dim, h_dim=H_DIM_ENC, g_dim=G_DIM, dropout=DROPOUT)

    dec_ckpt = torch.load(ckpt_dir / "decoder_stage1.pt", map_location=DEVICE, weights_only=False)
    enc_ckpt = torch.load(ckpt_dir / "encoder_stage2.pt", map_location=DEVICE, weights_only=False)
    decoder.load_state_dict(dec_ckpt["decoder"])
    encoder.load_state_dict(enc_ckpt["encoder"])
    print(f"Loaded: decoder epoch {dec_ckpt['epoch']}, encoder epoch {enc_ckpt['epoch']}")

    print(f"Running simulation ({args.num_samples} G samples per game)...")
    results = simulate_gs(encoder, decoder, val_loader, num_samples=args.num_samples)

    # Aggregate phi across all val games
    all_phi = []
    all_bets_pairs = []
    all_actual = []
    n_stats = len(TARGET_COLS)

    for r in results:
        phi = phi_coefficient(r["OO"], r["OU"], r["UO"], r["UU"], r["mask_flat"], n_stats)
        pairs = extract_pairs(phi, r["OO"], r["OU"], r["UO"], r["UU"],
                               args.num_samples, r["mask_flat"], TARGET_COLS,
                               threshold=args.threshold, top_k=TOP_K)
        all_phi.append(phi)
        all_bets_pairs.extend(pairs)
        all_actual.append(r["actual_over"])

    # Phi statistics
    phi_cat = torch.cat([p.reshape(-1) for p in all_phi])
    valid_phi = phi_cat[~torch.isnan(phi_cat)]
    print(f"\n=== Phi Statistics ===")
    print(f"Total pairs evaluated: {len(valid_phi):,}")
    print(f"Phi mean: {valid_phi.mean():.4f}  std: {valid_phi.std():.4f}")
    print(f"Phi > 0.15: {(valid_phi > 0.15).sum().item():,}  ({(valid_phi > 0.15).float().mean()*100:.2f}%)")
    print(f"Phi < -0.15: {(valid_phi < -0.15).sum().item():,}  ({(valid_phi < -0.15).float().mean()*100:.2f}%)")
    print(f"Phi percentiles [5, 25, 50, 75, 95]: {np.percentile(valid_phi.numpy(), [5,25,50,75,95]).round(3).tolist()}")

    # Backtest
    actual_over_cat = torch.cat(all_actual, dim=0)
    backtest_results = backtest(all_bets_pairs, actual_over_cat)

    print(f"\n=== Backtest (threshold={args.threshold}) ===")
    print(f"Bets placed: {backtest_results['bets']}")
    if backtest_results['bets'] > 0:
        print(f"Win rate:    {backtest_results['win_rate']:.3f}  (breakeven: {backtest_results['breakeven_wr']:.3f})")
        print(f"Net PnL:     {backtest_results['net_pnl']:.2f} units")
        print(f"ROI:         {backtest_results['roi']:.3f}")

        # Same-team vs cross-team phi distribution
        print(f"\nTop 10 pairs by |phi|:")
        recs = sorted(backtest_results['records'], key=lambda x: abs(x['phi']), reverse=True)[:10]
        for rec in recs:
            print(f"  phi={rec['phi']:+.3f} {rec['predicted_dir']} | {rec['i_label']} × {rec['j_label']} | {'WIN' if rec['won'] else 'LOSS'}")
    else:
        print("No bets placed — phi threshold not crossed.")

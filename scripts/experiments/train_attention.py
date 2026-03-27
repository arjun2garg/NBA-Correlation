"""
Track A: Train the attention decoder with PBP assist features.

Self-attention over 16 player slots creates explicit player-player interaction.
PBP assist network features add prior knowledge about passing relationships.
"""

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
import torch.nn as nn
import numpy as np
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from src.data.dataset import (
    load_processed, temporal_split, make_loaders,
    build_tensors, NBADataset, PLAYER_EXTRA_COLS, TARGET_COLS, STAT_COLS,
)
from torch.utils.data import DataLoader
from src.model import GameEncoder, reparameterize
from src.model_attention import PlayerDecoderAttention
from src.train import kl_divergence, masked_nll

# --- config ---
LATENT_DIM = 32
H_DIM_ENC = 128
H_DIM_ATT = 64
N_HEADS = 2
DROPOUT = 0.3
LR = 1e-3
BETA = 0.001
FREE_BITS = 0.5
WARMUP_EPOCHS = 15
NUM_EPOCHS = 150
BATCH_SIZE = 64
DEVICE = "cpu"
PBP_EXTRA_COLS = ["rolling_ast_given_rate", "rolling_ast_received_rate"]


def p_over_std_diagnostic(encoder, decoder, val_loader, num_samples=100, device="cpu"):
    """Compute std of P(over|z) across z samples — key z-sensitivity metric."""
    encoder.eval()
    decoder.eval()
    stds = []
    with torch.no_grad():
        for X_t, X_p, Y, weights, _ in val_loader:
            X_t, X_p = X_t.to(device), X_p.to(device)
            weights = weights.to(device)
            mask = (weights > 0).float()
            mu, logvar = encoder(X_t)
            p_samples = []
            for _ in range(num_samples):
                z = reparameterize(mu, logvar)
                mu_pred, logvar_pred = decoder(z, X_p, mask)
                sigma_pred = (0.5 * logvar_pred).exp()
                p_over = 0.5 * (1.0 + torch.erf(mu_pred / (sigma_pred * (2.0 ** 0.5))))
                p_samples.append(p_over.cpu())
            p_tensor = torch.stack(p_samples, dim=0)  # (S, B, 16, 3)
            for b in range(p_tensor.shape[1]):
                valid = mask[b].cpu().bool()
                p_std = p_tensor[:, b, valid, :].std(dim=0)  # (n_valid, 3)
                stds.append(p_std.numpy())
    all_stds = np.concatenate([s.ravel() for s in stds])
    return float(all_stds.mean()), float(all_stds.std())


def train_epoch_attn(encoder, decoder, optimizer, loader, beta, free_bits, device, grad_clip=1.0):
    encoder.train()
    decoder.train()
    totals = {"recon": 0.0, "kl": 0.0}
    for X_t, X_p, Y, weights, _ in loader:
        X_t, X_p, Y, weights = X_t.to(device), X_p.to(device), Y.to(device), weights.to(device)
        mask = (weights > 0).float()
        optimizer.zero_grad()
        mu, logvar = encoder(X_t)
        z = reparameterize(mu, logvar)
        mu_pred, logvar_pred = decoder(z, X_p, mask)
        recon = masked_nll(mu_pred, logvar_pred, Y, weights)
        kl = kl_divergence(mu, logvar, free_bits=free_bits)
        loss = recon + beta * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), max_norm=grad_clip
        )
        optimizer.step()
        totals["recon"] += recon.item()
        totals["kl"] += kl.item()
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def evaluate_attn(encoder, decoder, loader, beta, num_samples=10, device="cpu"):
    encoder.eval()
    decoder.eval()
    totals = {"recon": 0.0, "kl": 0.0}
    with torch.no_grad():
        for X_t, X_p, Y, weights, _ in loader:
            X_t, X_p, Y, weights = X_t.to(device), X_p.to(device), Y.to(device), weights.to(device)
            mask = (weights > 0).float()
            mu, logvar = encoder(X_t)
            kl = kl_divergence(mu, logvar)
            recon_vals = []
            for _ in range(num_samples):
                z = reparameterize(mu, logvar)
                mu_pred, logvar_pred = decoder(z, X_p, mask)
                recon_vals.append(masked_nll(mu_pred, logvar_pred, Y, weights).item())
            totals["recon"] += np.mean(recon_vals)
            totals["kl"] += kl.item()
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/exp_attention")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--season", type=str, default="2019-26")
    parser.add_argument("--beta", type=float, default=BETA)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CKPT_DIR = Path(args.ckpt_dir)
    CKPT_LATEST = CKPT_DIR / "model_latest.pt"
    LOG_FILE = CKPT_DIR / "train_log.txt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")

    log(f"Attention Config: BETA={args.beta}, epochs={args.epochs}, ckpt_dir={CKPT_DIR}")

    # --- Extract PBP features if needed ---
    pbp_feat_path = Path("data/processed/pbp_features.csv")
    if not pbp_feat_path.exists():
        log("Extracting PBP assist features (one-time, may take ~10 min)...")
        from src.data.pbp_features import extract_pbp_features
        extract_pbp_features()
        log("PBP features extracted.")
    else:
        log(f"PBP features already exist at {pbp_feat_path}")

    import pandas as pd
    pbp_feats = pd.read_csv(pbp_feat_path)
    # Align types: PBP has int64 IDs, input_data has int64 gameId and float64 personId
    pbp_feats["personId"] = pd.to_numeric(pbp_feats["personId"], errors="coerce").astype("int64")
    pbp_feats["gameId"] = pd.to_numeric(pbp_feats["gameId"], errors="coerce").astype("int64")
    log(f"  PBP features: {len(pbp_feats):,} rows")

    # --- Load & merge data ---
    log(f"Loading data (season {args.season})...")
    df = load_processed(season_suffix=args.season)
    df["personId"] = pd.to_numeric(df["personId"], errors="coerce").astype("int64")
    df["gameId"] = pd.to_numeric(df["gameId"], errors="coerce").astype("int64")
    # Merge PBP features; fill NaN with 0 for players absent from PBP
    df = df.merge(pbp_feats, on=["personId", "gameId"], how="left")
    df[PBP_EXTRA_COLS] = df[PBP_EXTRA_COLS].fillna(0.0)
    log(f"  {len(df)} rows after merge")

    train_df, val_df = temporal_split(df)
    log(f"  train={len(train_df)}  val={len(val_df)}")

    extra_cols = PLAYER_EXTRA_COLS + PBP_EXTRA_COLS
    log(f"  player extra cols: {extra_cols}")

    train_loader, val_loader, Y_mean, Y_std, Xt_mean, Xt_std, Xp_mean, Xp_std = make_loaders(
        train_df, val_df, batch_size=BATCH_SIZE, extra_player_cols=extra_cols
    )
    log("  make_loaders done")

    fb = next(iter(train_loader))
    team_dim = fb[0].shape[1]
    player_dim = fb[1].shape[2]
    log(f"  team_dim={team_dim}  player_dim={player_dim}")

    encoder = GameEncoder(team_dim, H_DIM_ENC, LATENT_DIM, DROPOUT).to(DEVICE)
    decoder = PlayerDecoderAttention(
        LATENT_DIM, player_dim, H_DIM_ATT, N_HEADS, len(TARGET_COLS), DROPOUT
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=LR
    )

    start_epoch = 0
    if not args.no_resume and CKPT_LATEST.exists():
        ckpt = torch.load(CKPT_LATEST, map_location=DEVICE, weights_only=False)
        cfg = ckpt.get("config", {})
        if cfg.get("team_dim") == team_dim and cfg.get("player_dim") == player_dim:
            encoder.load_state_dict(ckpt["encoder"])
            decoder.load_state_dict(ckpt["decoder"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            log(f"Resumed from epoch {start_epoch - 1}")
        else:
            log("Checkpoint config mismatch — starting fresh.")
    elif args.no_resume:
        log("--no-resume — starting from epoch 0.")
    else:
        log("No checkpoint — starting from epoch 0.")

    log(f"Training attention model epochs {start_epoch}–{args.epochs - 1}")

    for epoch in range(start_epoch, args.epochs):
        beta_t = args.beta * min(1.0, epoch / max(1, WARMUP_EPOCHS))
        train_m = train_epoch_attn(encoder, decoder, optimizer, train_loader, beta_t, FREE_BITS, DEVICE)
        val_m = evaluate_attn(encoder, decoder, val_loader, beta_t, num_samples=10, device=DEVICE)

        log(
            f"Epoch {epoch:03d}/{args.epochs - 1} | beta {beta_t:.5f} | "
            f"train recon {train_m['recon']:.4f}  kl/dim {train_m['kl'] / LATENT_DIM:.3f} | "
            f"val recon {val_m['recon']:.4f}  kl/dim {val_m['kl'] / LATENT_DIM:.3f}"
        )

        # Every 10 epochs: log P(over|z) std diagnostic
        if epoch % 10 == 0 and epoch > 0:
            p_std_mean, p_std_std = p_over_std_diagnostic(encoder, decoder, val_loader, num_samples=100)
            log(f"  [diag] P(over|z) std: mean={p_std_mean:.4f}  std={p_std_std:.4f}  "
                f"(target >0.05)")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "Y_mean": Y_mean, "Y_std": Y_std,
            "Xt_mean": Xt_mean, "Xt_std": Xt_std,
            "Xp_mean": Xp_mean, "Xp_std": Xp_std,
            "config": {
                "model_type": "attention",
                "latent_dim": LATENT_DIM,
                "h_dim_enc": H_DIM_ENC,
                "h_dim_dec": H_DIM_ATT,
                "n_heads": N_HEADS,
                "dropout": DROPOUT,
                "team_dim": team_dim,
                "player_dim": player_dim,
                "n_target_cols": len(TARGET_COLS),
                "extra_player_cols": extra_cols,
            },
        }, CKPT_LATEST)

    # Final diagnostic
    p_std_mean, p_std_std = p_over_std_diagnostic(encoder, decoder, val_loader, num_samples=200)
    log(f"\nFinal P(over|z) std: mean={p_std_mean:.4f}  std={p_std_std:.4f}")
    log("Attention training complete.")

"""
Track B: Train two-stage VAE with raw targets and game outcome anchoring.

Stage 1: z → GameOutcomeDecoder → predicted game outcome
Stage 2: (z, game_outcome, player_feats) → TwoStagePlayerDecoder → stats

Hypotheses:
  1. Anchoring z to real game outcomes forces z to carry game-state signal
  2. Raw targets (not residuals) preserve game-context correlation
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
import pandas as pd
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import (
    load_processed, temporal_split, build_tensors, NBADataset, TARGET_COLS,
)
from src.data.game_outcomes import load_game_outcomes, normalize_outcomes, OUTCOME_COLS
from src.model import GameEncoder, reparameterize
from src.model_twostage import GameOutcomeDecoder, TwoStagePlayerDecoder
from src.train import kl_divergence, masked_nll

# --- config ---
LATENT_DIM = 32
H_DIM_ENC = 128
H_DIM_DEC = 64
OUTCOME_DIM = 6
DROPOUT = 0.3
LR = 1e-3
BETA = 0.001
ALPHA = 0.5          # weight on game_mse loss
FREE_BITS = 0.5
WARMUP_EPOCHS = 15
NUM_EPOCHS = 150
BATCH_SIZE = 64
DEVICE = "cpu"


class TwoStageDataset(Dataset):
    def __init__(self, X_team, X_players, Y, weights, lines, G):
        self.X_team = X_team
        self.X_players = X_players
        self.Y = Y
        self.weights = weights
        self.lines = lines
        self.G = G  # game outcome vector (batch, outcome_dim)

    def __len__(self):
        return len(self.X_team)

    def __getitem__(self, idx):
        return (self.X_team[idx], self.X_players[idx], self.Y[idx],
                self.weights[idx], self.lines[idx], self.G[idx])


def build_twostage_loaders(train_df, val_df, outcomes_df, batch_size=64):
    """Build DataLoaders that include game outcome vectors."""
    # Build player tensors (raw targets)
    X_team_tr, X_pl_tr, w_tr, Y_tr, lines_tr = build_tensors(train_df, raw_targets=True)
    X_team_val, X_pl_val, w_val, Y_val, lines_val = build_tensors(val_df, raw_targets=True)

    # Get ordered gameIds from the tensor builds (need to align G)
    train_game_ids = _get_ordered_game_ids(train_df)
    val_game_ids = _get_ordered_game_ids(val_df)

    # Normalize raw targets using train stats
    Y_mean = Y_tr.mean(dim=(0, 1))
    Y_std = Y_tr.std(dim=(0, 1)) + 1e-6
    Y_tr_norm = (Y_tr - Y_mean) / Y_std
    Y_val_norm = (Y_val - Y_mean) / Y_std

    # Normalize team & player features
    Xt_mean = X_team_tr.mean(dim=0)
    Xt_std = X_team_tr.std(dim=0) + 1e-6
    X_team_tr_n = (X_team_tr - Xt_mean) / Xt_std
    X_team_val_n = (X_team_val - Xt_mean) / Xt_std

    Xp_mean = X_pl_tr.mean(dim=(0, 1))
    Xp_std = X_pl_tr.std(dim=(0, 1)) + 1e-6
    X_pl_tr_n = (X_pl_tr - Xp_mean) / Xp_std
    X_pl_val_n = (X_pl_val - Xp_mean) / Xp_std

    # Normalize game outcomes using train games
    train_game_ids_set = set(train_game_ids)
    outcomes_normed, outcome_stats = normalize_outcomes(outcomes_df, train_game_ids_set)

    # Build G tensors aligned with game order
    def _build_G(game_ids, outcomes_normed, outcome_dim=OUTCOME_DIM):
        g_rows = []
        oid_map = outcomes_normed.set_index("gameId")
        for gid in game_ids:
            if gid in oid_map.index:
                row = oid_map.loc[gid, OUTCOME_COLS].values.astype(np.float32)
            else:
                row = np.zeros(outcome_dim, dtype=np.float32)
            g_rows.append(row)
        return torch.tensor(np.stack(g_rows), dtype=torch.float32)

    G_tr = _build_G(train_game_ids, outcomes_normed)
    G_val = _build_G(val_game_ids, outcomes_normed)

    # Normalize lines using train stats (lines = h_stat, same as Y features)
    lines_tr_norm = (lines_tr - Y_mean) / Y_std
    lines_val_norm = (lines_val - Y_mean) / Y_std

    train_ds = TwoStageDataset(X_team_tr_n, X_pl_tr_n, Y_tr_norm, w_tr, lines_tr_norm, G_tr)
    val_ds = TwoStageDataset(X_team_val_n, X_pl_val_n, Y_val_norm, w_val, lines_val_norm, G_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    norm_stats = dict(
        Y_mean=Y_mean, Y_std=Y_std,
        Xt_mean=Xt_mean, Xt_std=Xt_std,
        Xp_mean=Xp_mean, Xp_std=Xp_std,
        outcome_stats=outcome_stats,
    )
    return train_loader, val_loader, norm_stats


def _get_ordered_game_ids(df):
    """Return gameIds in the order they appear after build_tensors processing."""
    game_ids = []
    for gid, game in df.groupby("gameId"):
        # Check if game has enough players (same check as build_game)
        home = game[game["home"] == 1]
        away = game[game["home"] == 0]
        if len(home) >= 6 and len(away) >= 6:  # n_starters + 1 bench
            game_ids.append(gid)
    return game_ids


def train_epoch_ts(encoder, outcome_dec, player_dec, optimizer, loader, beta, alpha, free_bits, device):
    encoder.train()
    outcome_dec.train()
    player_dec.train()
    totals = {"recon": 0.0, "kl": 0.0, "game_mse": 0.0}
    mse_loss = nn.MSELoss()

    for X_t, X_p, Y, weights, _, G in loader:
        X_t, X_p, Y, weights = X_t.to(device), X_p.to(device), Y.to(device), weights.to(device)
        G = G.to(device)

        optimizer.zero_grad()
        mu, logvar = encoder(X_t)
        z = reparameterize(mu, logvar)

        G_pred = outcome_dec(z)
        game_mse = mse_loss(G_pred, G)

        mu_pred, logvar_pred = player_dec(z, G_pred, X_p)
        recon = masked_nll(mu_pred, logvar_pred, Y, weights)
        kl = kl_divergence(mu, logvar, free_bits=free_bits)

        loss = recon + beta * kl + alpha * game_mse
        loss.backward()
        optimizer.step()

        totals["recon"] += recon.item()
        totals["kl"] += kl.item()
        totals["game_mse"] += game_mse.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def evaluate_ts(encoder, outcome_dec, player_dec, loader, beta, num_samples=10, device="cpu"):
    encoder.eval()
    outcome_dec.eval()
    player_dec.eval()
    totals = {"recon": 0.0, "kl": 0.0, "game_mse": 0.0}
    mse_loss = nn.MSELoss()

    with torch.no_grad():
        for X_t, X_p, Y, weights, _, G in loader:
            X_t, X_p, Y, weights = X_t.to(device), X_p.to(device), Y.to(device), weights.to(device)
            G = G.to(device)
            mu, logvar = encoder(X_t)
            kl = kl_divergence(mu, logvar)
            recon_vals = []
            game_mse_vals = []
            for _ in range(num_samples):
                z = reparameterize(mu, logvar)
                G_pred = outcome_dec(z)
                game_mse_vals.append(mse_loss(G_pred, G).item())
                mu_pred, logvar_pred = player_dec(z, G_pred, X_p)
                recon_vals.append(masked_nll(mu_pred, logvar_pred, Y, weights).item())
            totals["recon"] += np.mean(recon_vals)
            totals["kl"] += kl.item()
            totals["game_mse"] += np.mean(game_mse_vals)

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def p_over_std_diagnostic(encoder, outcome_dec, player_dec, val_loader, num_samples=100, device="cpu"):
    encoder.eval()
    outcome_dec.eval()
    player_dec.eval()
    stds = []
    with torch.no_grad():
        for X_t, X_p, Y, weights, _, G in val_loader:
            X_t, X_p = X_t.to(device), X_p.to(device)
            weights = weights.to(device)
            mask = (weights > 0).float()
            mu, logvar = encoder(X_t)
            p_samples = []
            for _ in range(num_samples):
                z = reparameterize(mu, logvar)
                G_pred = outcome_dec(z)
                mu_pred, logvar_pred = player_dec(z, G_pred, X_p)
                sigma_pred = (0.5 * logvar_pred).exp()
                p_over = 0.5 * (1.0 + torch.erf(mu_pred / (sigma_pred * (2.0 ** 0.5))))
                p_samples.append(p_over.cpu())
            p_tensor = torch.stack(p_samples, dim=0)
            for b in range(p_tensor.shape[1]):
                valid = mask[b].cpu().bool()
                p_std = p_tensor[:, b, valid, :].std(dim=0)
                stds.append(p_std.numpy())
    all_stds = np.concatenate([s.ravel() for s in stds])
    return float(all_stds.mean()), float(all_stds.std())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/exp_twostage")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--season", type=str, default="2019-26")
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
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

    log(f"TwoStage Config: BETA={args.beta}, ALPHA={args.alpha}, epochs={args.epochs}")

    # --- Load data ---
    log(f"Loading player data (season {args.season})...")
    df = load_processed(season_suffix=args.season)
    log(f"  {len(df)} rows loaded")
    train_df, val_df = temporal_split(df)
    log(f"  train={len(train_df)}  val={len(val_df)}")

    # --- Load game outcomes ---
    log("Loading game outcomes...")
    outcomes = load_game_outcomes()
    log(f"  {len(outcomes)} games with outcomes")

    # --- Build loaders ---
    log("Building loaders (raw targets)...")
    train_loader, val_loader, norm_stats = build_twostage_loaders(
        train_df, val_df, outcomes, batch_size=BATCH_SIZE
    )
    log("  done")

    fb = next(iter(train_loader))
    team_dim = fb[0].shape[1]
    player_dim = fb[1].shape[2]
    log(f"  team_dim={team_dim}  player_dim={player_dim}")

    encoder = GameEncoder(team_dim, H_DIM_ENC, LATENT_DIM, DROPOUT).to(DEVICE)
    outcome_dec = GameOutcomeDecoder(LATENT_DIM, OUTCOME_DIM, h_dim=64).to(DEVICE)
    player_dec = TwoStagePlayerDecoder(
        LATENT_DIM, player_dim, OUTCOME_DIM, H_DIM_DEC, len(TARGET_COLS), DROPOUT
    ).to(DEVICE)

    all_params = (list(encoder.parameters()) +
                  list(outcome_dec.parameters()) +
                  list(player_dec.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=LR)

    start_epoch = 0
    if not args.no_resume and CKPT_LATEST.exists():
        ckpt = torch.load(CKPT_LATEST, map_location=DEVICE, weights_only=False)
        cfg = ckpt.get("config", {})
        if cfg.get("team_dim") == team_dim and cfg.get("player_dim") == player_dim:
            encoder.load_state_dict(ckpt["encoder"])
            outcome_dec.load_state_dict(ckpt["outcome_dec"])
            player_dec.load_state_dict(ckpt["player_dec"])
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

    log(f"Training two-stage model epochs {start_epoch}–{args.epochs - 1}")

    for epoch in range(start_epoch, args.epochs):
        beta_t = args.beta * min(1.0, epoch / max(1, WARMUP_EPOCHS))
        train_m = train_epoch_ts(encoder, outcome_dec, player_dec, optimizer,
                                  train_loader, beta_t, args.alpha, FREE_BITS, DEVICE)
        val_m = evaluate_ts(encoder, outcome_dec, player_dec, val_loader,
                             beta_t, num_samples=10, device=DEVICE)

        log(
            f"Epoch {epoch:03d}/{args.epochs - 1} | beta {beta_t:.5f} | "
            f"train recon {train_m['recon']:.4f}  game_mse {train_m['game_mse']:.4f}  "
            f"kl/dim {train_m['kl'] / LATENT_DIM:.3f} | "
            f"val recon {val_m['recon']:.4f}  game_mse {val_m['game_mse']:.4f}  "
            f"kl/dim {val_m['kl'] / LATENT_DIM:.3f}"
        )

        if epoch % 10 == 0 and epoch > 0:
            p_std_mean, p_std_std = p_over_std_diagnostic(
                encoder, outcome_dec, player_dec, val_loader, num_samples=100
            )
            log(f"  [diag] P(over|z) std: mean={p_std_mean:.4f}  std={p_std_std:.4f}  "
                f"(target >0.05)")

        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "outcome_dec": outcome_dec.state_dict(),
            "player_dec": player_dec.state_dict(),
            "optimizer": optimizer.state_dict(),
            **{k: v for k, v in norm_stats.items() if k != "outcome_stats"},
            "outcome_stats": norm_stats["outcome_stats"],
            "config": {
                "model_type": "twostage",
                "latent_dim": LATENT_DIM,
                "h_dim_enc": H_DIM_ENC,
                "h_dim_dec": H_DIM_DEC,
                "outcome_dim": OUTCOME_DIM,
                "dropout": DROPOUT,
                "team_dim": team_dim,
                "player_dim": player_dim,
                "n_target_cols": len(TARGET_COLS),
            },
        }, CKPT_LATEST)

    # Final diagnostic
    p_std_mean, p_std_std = p_over_std_diagnostic(
        encoder, outcome_dec, player_dec, val_loader, num_samples=200
    )
    log(f"\nFinal P(over|z) std: mean={p_std_mean:.4f}  std={p_std_std:.4f}")
    log("Two-stage training complete.")

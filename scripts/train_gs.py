"""
Train game state anchored VAE (Phase 2).

Key addition: auxiliary loss forces z to encode actual game state G,
preventing the decoder from ignoring z (root cause of phi=0 in all prior experiments).

New hyperparameter: --lambda-gs controls the weight of the GS auxiliary loss.
Recommended sweep: 0.1, 1.0, 5.0, 10.0
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
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from src.data.dataset import load_processed, temporal_split, STAT_COLS, TARGET_COLS
from src.data.game_state_dataset import make_loaders_gs, G_DIM, GS_TARGET_COLS
from src.model_gs import GameEncoder, PlayerDecoder, GameStateHead
from src.train_gs import train_epoch_gs, evaluate_gs, compute_phi_sensitivity

# Default config (same as train.py for comparability)
LATENT_DIM = 32
H_DIM_ENC = 128
H_DIM_DEC = 64
DROPOUT = 0.3
LR = 1e-3
BETA_DEFAULT = 0.001
FREE_BITS = 0.5
WARMUP_EPOCHS = 15
NUM_EPOCHS_DEFAULT = 100
BATCH_SIZE = 64
LAMBDA_GS_DEFAULT = 1.0   # weight for game state auxiliary loss
DEVICE = "cpu"

# How often to run phi sensitivity diagnostic (expensive: 100 z samples per game)
DIAG_EVERY = 10


def parse_args():
    p = argparse.ArgumentParser(description="Train game state anchored NBA VAE")
    p.add_argument("--beta", type=float, default=BETA_DEFAULT)
    p.add_argument("--lambda-gs", type=float, default=LAMBDA_GS_DEFAULT,
                   help="Weight of game state auxiliary loss (higher = z must encode G more tightly)")
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS_DEFAULT)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints_gs")
    p.add_argument("--season", type=str, default="2019-26")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--diag-every", type=int, default=DIAG_EVERY,
                   help="Run P(over|z) std diagnostic every N epochs (0 to disable)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    BETA = args.beta
    LAMBDA_GS = args.lambda_gs
    NUM_EPOCHS = args.epochs
    CKPT_DIR = Path(args.ckpt_dir)
    CKPT_LATEST = CKPT_DIR / "model_latest.pt"
    LOG_FILE = CKPT_DIR / "train_log.txt"

    def log(msg):
        print(msg, flush=True)
        CKPT_DIR.mkdir(exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")

    log(f"Config: BETA={BETA}, lambda_gs={LAMBDA_GS}, epochs={NUM_EPOCHS}, season={args.season}")
    log(f"  LATENT_DIM={LATENT_DIM}, H_DIM_ENC={H_DIM_ENC}, H_DIM_DEC={H_DIM_DEC}")
    log(f"  G_DIM={G_DIM}, GS_TARGET_COLS={GS_TARGET_COLS}")

    log("Loading data...")
    df = load_processed(season_suffix=args.season)
    log(f"  {len(df)} rows loaded")
    train_df, val_df = temporal_split(df)
    log(f"  train={len(train_df)}  val={len(val_df)}")

    log("Building tensors and loaders (with game state G)...")
    train_loader, val_loader, norm_stats = make_loaders_gs(
        train_df, val_df, batch_size=BATCH_SIZE, include_pbp=False,
    )
    log("  make_loaders_gs done")

    first_batch = next(iter(train_loader))
    team_dim = first_batch[0].shape[1]
    player_dim = first_batch[1].shape[2]
    log(f"  team_dim={team_dim}  player_dim={player_dim}")

    encoder = GameEncoder(input_dim=team_dim, h_dim=H_DIM_ENC, latent_dim=LATENT_DIM, dropout=DROPOUT).to(DEVICE)
    gs_head = GameStateHead(latent_dim=LATENT_DIM, g_dim=G_DIM).to(DEVICE)
    decoder = PlayerDecoder(latent_dim=LATENT_DIM, player_dim=player_dim, h_dim=H_DIM_DEC,
                            output_dim=len(TARGET_COLS), dropout=DROPOUT).to(DEVICE)

    all_params = list(encoder.parameters()) + list(gs_head.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=LR)

    start_epoch = 0
    if not args.no_resume and CKPT_LATEST.exists():
        ckpt = torch.load(CKPT_LATEST, map_location=DEVICE, weights_only=False)
        cfg = ckpt.get("config", {})
        if (cfg.get("team_dim") == team_dim and cfg.get("latent_dim") == LATENT_DIM
                and cfg.get("g_dim") == G_DIM):
            encoder.load_state_dict(ckpt["encoder"])
            gs_head.load_state_dict(ckpt["gs_head"])
            decoder.load_state_dict(ckpt["decoder"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            log(f"Resumed from epoch {start_epoch - 1}")
        else:
            log("Config mismatch — starting fresh.")
    elif not args.no_resume:
        log("No checkpoint found — starting from epoch 0.")
    else:
        log("--no-resume: starting fresh.")

    log(f"Training epochs {start_epoch}–{NUM_EPOCHS - 1}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        beta_t = BETA * min(1.0, epoch / max(1, WARMUP_EPOCHS))

        train_m = train_epoch_gs(
            encoder, gs_head, decoder, optimizer, train_loader,
            beta=beta_t, lambda_gs=LAMBDA_GS, free_bits=FREE_BITS, device=DEVICE,
        )
        val_m = evaluate_gs(
            encoder, gs_head, decoder, val_loader,
            beta=beta_t, lambda_gs=LAMBDA_GS, num_samples=5, device=DEVICE,
        )

        log(
            f"Epoch {epoch:03d}/{NUM_EPOCHS-1} | beta {beta_t:.5f} | "
            f"train: recon {train_m['recon']:.4f} kl/dim {train_m['kl']/LATENT_DIM:.3f} gs {train_m['gs']:.4f} | "
            f"val: recon {val_m['recon']:.4f} kl/dim {val_m['kl']/LATENT_DIM:.3f} gs {val_m['gs']:.4f}"
        )

        # Periodic phi sensitivity diagnostic
        if args.diag_every > 0 and (epoch + 1) % args.diag_every == 0:
            diag = compute_phi_sensitivity(encoder, gs_head, decoder, val_loader,
                                           n_z_samples=100, device=DEVICE)
            log(
                f"  [DIAG] P(over|z) std={diag['p_over_std_mean']:.4f}  "
                f"GS R²={diag['gs_r2']:.3f}"
            )

        CKPT_DIR.mkdir(exist_ok=True)
        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "gs_head": gs_head.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            **norm_stats,
            "config": {
                "latent_dim": LATENT_DIM,
                "h_dim_enc": H_DIM_ENC,
                "h_dim_dec": H_DIM_DEC,
                "dropout": DROPOUT,
                "team_dim": team_dim,
                "player_dim": player_dim,
                "n_target_cols": len(TARGET_COLS),
                "g_dim": G_DIM,
                "lambda_gs": LAMBDA_GS,
                "beta": BETA,
            },
        }, CKPT_LATEST)

    log("Training complete.")

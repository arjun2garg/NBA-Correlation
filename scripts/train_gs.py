"""
Two-stage training for Game State VAE.

Stage 1: Train decoder supervised on actual G (run first, independently).
Stage 2: Freeze decoder, train encoder to predict G from pre-game features.

Usage:
  # Run Stage 1 first:
  python scripts/train_gs.py --stage 1 --epochs 80 --ckpt-dir checkpoints_gs

  # Then Stage 2:
  python scripts/train_gs.py --stage 2 --epochs 100 --ckpt-dir checkpoints_gs

  # Or run both sequentially:
  python scripts/train_gs.py --stage both --epochs1 80 --epochs2 100 --ckpt-dir checkpoints_gs
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.set_num_threads(1)

from src.data.dataset import load_processed, temporal_split, TARGET_COLS
from src.data.game_state_dataset import make_loaders_gs, G_DIM, GS_TARGET_COLS
from src.model_gs import GCondDecoder, GameEncoder
from src.train_gs import (
    train_decoder_epoch, eval_decoder,
    train_encoder_epoch, eval_encoder,
    compute_p_over_std,
)

PLAYER_DIM = 24
H_DIM_DEC = 64
H_DIM_ENC = 128
DROPOUT = 0.3
LR = 1e-3
BATCH_SIZE = 64
DEVICE = "cpu"

# Stage 2 hyperparameters
BETA_DEFAULT = 0.01      # KL weight for encoder training (higher than old VAE)
FREE_BITS = 0.5          # min KL/dim — prevents sigma_G collapsing to 0
DIAG_EVERY = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["1", "2", "both"], default="both")
    p.add_argument("--epochs1", type=int, default=80, help="Stage 1 epochs")
    p.add_argument("--epochs2", type=int, default=100, help="Stage 2 epochs")
    p.add_argument("--epochs", type=int, default=None, help="Override both epoch counts")
    p.add_argument("--beta", type=float, default=BETA_DEFAULT, help="KL weight for Stage 2")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints_gs")
    p.add_argument("--season", type=str, default="2019-26")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--diag-every", type=int, default=DIAG_EVERY)
    return p.parse_args()


def run_stage1(decoder, train_loader, val_loader, epochs, ckpt_dir, log, resume=True):
    ckpt_path = ckpt_dir / "decoder_stage1.pt"
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
    start_epoch = 0

    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        decoder.load_state_dict(ckpt["decoder"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log(f"[Stage 1] Resumed from epoch {start_epoch - 1}")
    else:
        log(f"[Stage 1] Starting from scratch")

    log(f"[Stage 1] Training decoder: G + player_feats → residuals  ({epochs} epochs)")

    for epoch in range(start_epoch, epochs):
        tr = train_decoder_epoch(decoder, optimizer, train_loader, device=DEVICE)
        va = eval_decoder(decoder, val_loader, device=DEVICE)
        log(f"  Stage1 Epoch {epoch:03d}/{epochs-1} | train NLL {tr['loss']:.4f} | val NLL {va['loss']:.4f}")

        torch.save({
            "epoch": epoch,
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, ckpt_path)

    log(f"[Stage 1] Done. Final val NLL: {va['loss']:.4f}")
    return decoder


def run_stage2(encoder, decoder, train_loader, val_loader, epochs, beta,
               ckpt_dir, log, diag_every, resume=True):
    ckpt_path = ckpt_dir / "encoder_stage2.pt"
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
    start_epoch = 0

    # Freeze decoder completely
    for p in decoder.parameters():
        p.requires_grad_(False)
    decoder.eval()

    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log(f"[Stage 2] Resumed from epoch {start_epoch - 1}")
    else:
        log(f"[Stage 2] Starting from scratch")

    log(f"[Stage 2] Training encoder: X_team → N(mu_G, sigma_G)  ({epochs} epochs)")
    log(f"[Stage 2] beta={beta}, free_bits={FREE_BITS}")

    va = {}
    for epoch in range(start_epoch, epochs):
        tr = train_encoder_epoch(encoder, decoder, optimizer, train_loader,
                                  beta=beta, free_bits=FREE_BITS, device=DEVICE)
        va = eval_encoder(encoder, val_loader, beta=beta, device=DEVICE)

        log(
            f"  Stage2 Epoch {epoch:03d}/{epochs-1} | "
            f"train: g_nll {tr['g_nll']:.4f} kl/dim {tr['kl']/G_DIM:.3f} | "
            f"val: g_nll {va['g_nll']:.4f} G_R²={va['g_r2']:.3f}"
        )

        if diag_every > 0 and (epoch + 1) % diag_every == 0:
            diag = compute_p_over_std(encoder, decoder, val_loader,
                                       n_samples=200, device=DEVICE)
            log(
                f"    [DIAG] P(over|G) std={diag['p_over_std_mean']:.4f}  "
                f"sigma_pred={diag['sigma_pred_mean']:.4f}  "
                f"(target std > 0.05)"
            )

        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, ckpt_path)

    log(f"[Stage 2] Done. Final val G_R²: {va.get('g_r2', 0):.3f}")
    return encoder


if __name__ == "__main__":
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)
    log_file = ckpt_dir / "train_log_twostage.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    if args.epochs is not None:
        epochs1 = epochs2 = args.epochs
    else:
        epochs1 = args.epochs1
        epochs2 = args.epochs2

    log(f"Two-stage training | stage={args.stage} | season={args.season}")
    log(f"epochs1={epochs1}, epochs2={epochs2}, beta={args.beta}")

    log("Loading data...")
    df = load_processed(season_suffix=args.season)
    train_df, val_df = temporal_split(df)
    log(f"  train={len(train_df)}  val={len(val_df)}")

    log("Building loaders...")
    train_loader, val_loader, norm_stats = make_loaders_gs(
        train_df, val_df, batch_size=BATCH_SIZE, include_pbp=False,
    )
    log(f"  G_DIM={G_DIM}, GS cols: {GS_TARGET_COLS}")

    first_batch = next(iter(train_loader))
    team_dim = first_batch[0].shape[1]
    log(f"  team_dim={team_dim}  player_dim={PLAYER_DIM}")

    decoder = GCondDecoder(g_dim=G_DIM, player_dim=PLAYER_DIM, h_dim=H_DIM_DEC,
                            output_dim=len(TARGET_COLS), dropout=DROPOUT).to(DEVICE)
    encoder = GameEncoder(input_dim=team_dim, h_dim=H_DIM_ENC, g_dim=G_DIM,
                           dropout=DROPOUT).to(DEVICE)

    resume = not args.no_resume

    if args.stage in ("1", "both"):
        # Load existing decoder if resuming stage 2 only
        dec_ckpt = ckpt_dir / "decoder_stage1.pt"
        if args.stage == "both" or not dec_ckpt.exists():
            decoder = run_stage1(decoder, train_loader, val_loader, epochs1,
                                  ckpt_dir, log, resume=resume)
        else:
            ckpt = torch.load(dec_ckpt, map_location=DEVICE, weights_only=False)
            decoder.load_state_dict(ckpt["decoder"])
            log(f"[Stage 1] Loaded from {dec_ckpt}")

    if args.stage in ("2", "both"):
        # Ensure decoder checkpoint exists for stage 2
        dec_ckpt = ckpt_dir / "decoder_stage1.pt"
        if dec_ckpt.exists() and args.stage == "2":
            ckpt = torch.load(dec_ckpt, map_location=DEVICE, weights_only=False)
            decoder.load_state_dict(ckpt["decoder"])
            log(f"[Stage 2] Loaded stage-1 decoder from {dec_ckpt}")
        encoder = run_stage2(encoder, decoder, train_loader, val_loader, epochs2,
                              args.beta, ckpt_dir, log, args.diag_every, resume=resume)

    log("Training complete.")
    log(f"Checkpoints in: {ckpt_dir}/")
    log(f"  Stage 1 decoder: {ckpt_dir}/decoder_stage1.pt")
    log(f"  Stage 2 encoder: {ckpt_dir}/encoder_stage2.pt")

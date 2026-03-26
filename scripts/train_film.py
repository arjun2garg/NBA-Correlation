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

from src.data.dataset import load_processed, temporal_split, make_loaders, TARGET_COLS
from src.model import GameEncoder, reparameterize
from src.model_film import PlayerDecoderFiLM
from src.train import train_epoch, evaluate, masked_nll, kl_divergence

# --- default config ---
LATENT_DIM = 32
H_DIM_ENC = 128
H_DIM_FILM = 64
DROPOUT = 0.3
LR = 1e-3
BETA_DEFAULT = 0.1
FREE_BITS = 0.5
WARMUP_EPOCHS = 15
NUM_EPOCHS_DEFAULT = 60
BATCH_SIZE = 64
DEVICE = "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train NBA VAE with FiLM decoder")
    parser.add_argument("--beta", type=float, default=BETA_DEFAULT)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/exp_film")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS_DEFAULT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--season", type=str, default="2019-26")
    return parser.parse_args()


def train_epoch_film(encoder, decoder, optimizer, loader, beta=0.001, free_bits=0.0, device="cpu"):
    encoder.train()
    decoder.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

    for X_t, X_p, Y, weights, _ in loader:
        X_t, X_p, Y, weights = X_t.to(device), X_p.to(device), Y.to(device), weights.to(device)

        optimizer.zero_grad()
        mu, logvar = encoder(X_t)
        z = reparameterize(mu, logvar)
        mu_pred, logvar_pred = decoder(z, X_p)

        recon = masked_nll(mu_pred, logvar_pred, Y, weights)
        kl = kl_divergence(mu, logvar, free_bits=free_bits)
        loss = recon + beta * kl
        loss.backward()
        optimizer.step()

        totals["loss"] += loss.item()
        totals["recon"] += recon.item()
        totals["kl"] += kl.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def evaluate_film(encoder, decoder, loader, beta=0.001, num_samples=1, device="cpu"):
    encoder.eval()
    decoder.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

    with torch.no_grad():
        for X_t, X_p, Y, weights, _ in loader:
            X_t, X_p, Y, weights = X_t.to(device), X_p.to(device), Y.to(device), weights.to(device)
            mu, logvar = encoder(X_t)
            kl = kl_divergence(mu, logvar)
            recon = torch.stack([
                masked_nll(*decoder(reparameterize(mu, logvar), X_p), Y, weights)
                for _ in range(num_samples)
            ]).mean()
            loss = recon + beta * kl
            totals["loss"] += loss.item()
            totals["recon"] += recon.item()
            totals["kl"] += kl.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


if __name__ == "__main__":
    args = parse_args()

    BETA = args.beta
    NUM_EPOCHS = args.epochs
    CKPT_DIR = Path(args.ckpt_dir)
    CKPT_LATEST = CKPT_DIR / "model_latest.pt"
    LOG_FILE = CKPT_DIR / "train_log.txt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")

    log(f"FiLM Config: BETA={BETA}, epochs={NUM_EPOCHS}, ckpt_dir={CKPT_DIR}, season={args.season}")
    log("Loading data...")
    df = load_processed(season_suffix=args.season)
    log(f"  {len(df)} rows loaded")
    train_df, val_df = temporal_split(df)
    log(f"  train={len(train_df)}  val={len(val_df)}")
    log("Building tensors and loaders...")
    train_loader, val_loader, Y_mean, Y_std, Xt_mean, Xt_std, Xp_mean, Xp_std = make_loaders(
        train_df, val_df, batch_size=BATCH_SIZE
    )
    log("  make_loaders done")

    first_batch = next(iter(train_loader))
    team_dim = first_batch[0].shape[1]
    player_dim = first_batch[1].shape[2]
    log(f"  team_dim={team_dim}  player_dim={player_dim}")

    encoder = GameEncoder(
        input_dim=team_dim, h_dim=H_DIM_ENC, latent_dim=LATENT_DIM, dropout=DROPOUT
    ).to(DEVICE)
    decoder = PlayerDecoderFiLM(
        latent_dim=LATENT_DIM, player_dim=player_dim, h_dim=H_DIM_FILM,
        output_dim=len(TARGET_COLS), dropout=DROPOUT
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=LR
    )

    start_epoch = 0

    if not args.no_resume and CKPT_LATEST.exists():
        ckpt = torch.load(CKPT_LATEST, map_location=DEVICE, weights_only=False)
        if "encoder" in ckpt and ckpt.get("config", {}).get("team_dim") == team_dim:
            encoder.load_state_dict(ckpt["encoder"])
            decoder.load_state_dict(ckpt["decoder"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            log(f"Resumed from epoch {start_epoch - 1}")
        else:
            log("Checkpoint config mismatch — starting fresh.")
    elif args.no_resume:
        log("--no-resume flag set — starting from epoch 0.")
    else:
        log("No checkpoint found — starting from epoch 0.")

    log(f"Training FiLM model epochs {start_epoch}–{NUM_EPOCHS - 1}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        beta_t = BETA * min(1.0, epoch / max(1, WARMUP_EPOCHS))
        train_metrics = train_epoch_film(
            encoder, decoder, optimizer, train_loader,
            beta=beta_t, free_bits=FREE_BITS, device=DEVICE
        )
        val_metrics = evaluate_film(
            encoder, decoder, val_loader, beta=beta_t, num_samples=10, device=DEVICE
        )

        log(
            f"Epoch {epoch:03d}/{NUM_EPOCHS - 1} | "
            f"beta {beta_t:.5f} | "
            f"train recon {train_metrics['recon']:.4f}  kl/dim {train_metrics['kl'] / LATENT_DIM:.3f} | "
            f"val recon {val_metrics['recon']:.4f}  kl/dim {val_metrics['kl'] / LATENT_DIM:.3f}"
        )

        CKPT_DIR.mkdir(exist_ok=True)
        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "Y_mean": Y_mean, "Y_std": Y_std,
            "Xt_mean": Xt_mean, "Xt_std": Xt_std,
            "Xp_mean": Xp_mean, "Xp_std": Xp_std,
            "config": {
                "latent_dim": LATENT_DIM,
                "h_dim_enc": H_DIM_ENC,
                "h_dim_dec": H_DIM_FILM,
                "dropout": DROPOUT,
                "team_dim": team_dim,
                "player_dim": player_dim,
                "n_target_cols": len(TARGET_COLS),
                "model_type": "film",
            },
        }, CKPT_LATEST)

    log("FiLM training complete.")

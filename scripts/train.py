import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from src.data.dataset import load_processed, temporal_split, make_loaders, STAT_COLS, TARGET_COLS
from src.model import GameEncoder, PlayerDecoder
from src.train import train_epoch, evaluate

# --- config ---
LATENT_DIM = 16
H_DIM_ENC = 64
H_DIM_DEC = 32
LR = 1e-3
BETA = 0.001        # target beta after warmup
FREE_BITS = 0.5     # min KL per latent dim — prevents posterior collapse
WARMUP_EPOCHS = 15  # linearly ramp beta from 0 → BETA over this many epochs
NUM_EPOCHS = 100
BATCH_SIZE = 32
DEVICE = "cpu"

if __name__ == "__main__":
    df = load_processed()
    train_df, val_df = temporal_split(df)
    train_loader, val_loader, Y_mean, Y_std, Xt_mean, Xt_std, Xp_mean, Xp_std = make_loaders(train_df, val_df, batch_size=BATCH_SIZE)

    team_dim = next(iter(train_loader))[0].shape[1]
    player_dim = next(iter(train_loader))[1].shape[2]

    encoder = GameEncoder(input_dim=team_dim, h_dim=H_DIM_ENC, latent_dim=LATENT_DIM).to(DEVICE)
    decoder = PlayerDecoder(latent_dim=LATENT_DIM, player_dim=player_dim, h_dim=H_DIM_DEC, output_dim=len(TARGET_COLS)).to(DEVICE)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

    for epoch in range(NUM_EPOCHS):
        beta_t = BETA * min(1.0, epoch / max(1, WARMUP_EPOCHS))
        train_metrics = train_epoch(encoder, decoder, optimizer, train_loader, beta=beta_t, free_bits=FREE_BITS, device=DEVICE)
        print(
            f"Epoch {epoch:03d} | "
            f"beta {beta_t:.5f} | "
            f"Loss {train_metrics['loss']:.4f} | "
            f"Recon {train_metrics['recon']:.4f} | "
            f"KL {train_metrics['kl']:.4f} | "
            f"KL/dim {train_metrics['kl'] / LATENT_DIM:.3f}"
        )

    val_metrics = evaluate(encoder, decoder, val_loader, beta=BETA, num_samples=10, device=DEVICE)
    print(
        f"\nVal | "
        f"Loss {val_metrics['loss']:.4f} | "
        f"Recon {val_metrics['recon']:.4f} | "
        f"KL {val_metrics['kl']:.4f}"
    )

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "Xt_mean": Xt_mean,
        "Xt_std": Xt_std,
        "Xp_mean": Xp_mean,
        "Xp_std": Xp_std,
        "config": {
            "latent_dim": LATENT_DIM,
            "h_dim_enc": H_DIM_ENC,
            "h_dim_dec": H_DIM_DEC,
            "team_dim": team_dim,
            "player_dim": player_dim,
            "n_target_cols": len(TARGET_COLS),
        },
    }, "checkpoints/model_latest.pt")
    print("Checkpoint saved → checkpoints/model_latest.pt")

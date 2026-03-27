"""
Phi correlation simulation for v7 (PlayerGDecoder with actual mins+FGA).

Previous inline phi computation was WRONG: it computed mean P(over) across G samples
then thresholded at 0.5, collapsing all within-game G variation.

Correct approach:
  For each val game:
    1. Use actual G (G_game + G_player), then perturb N_SIM times to simulate G sampling
    2. For each G sample, compute P(over_i | G) per player analytically
    3. Draw binary outcomes: X_i ~ Bernoulli(P(over_i | G))
    4. Stack binary (N_SIM, n_active) → compute phi from OO/OU/UO/UU counts

Also reports theoretical phi_max = sigma_P^2 / Var(X) = sigma_P^2 / 0.25

Usage:
  python scripts/simulate_player_g.py --ckpt-dir checkpoints_v7 --n-sim 500 --n-games 300
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from src.data.dataset import load_processed, temporal_split, TARGET_COLS
from scripts.train_player_g import (
    PlayerGDecoder, make_loaders_player_g,
    G_GAME_DIM, G_PLAYER_DIM, PLAYER_DIM, OUTPUT_DIM, RAW_DIR,
)

CKPT_DIR = "checkpoints_v7"
N_SIM = 500
N_GAMES = 300          # val games to evaluate (None = all)
SIGMA_G_GAME = 0.6     # noise on game-level G (1 normalized std unit ≈ empirical encoder spread)
SIGMA_G_PLAYER = 0.5   # noise on player-level G (mins/FGA)
DEVICE = "cpu"
PHI_THRESHOLD = 0.15


def phi_from_binary(binary):
    """
    Compute phi for all player pairs from binary (N_SIM, n_active) matrix.

    Returns (n_active, n_active) phi matrix.
    """
    N = binary.shape[0]
    n = binary.shape[1]
    OO = binary.T @ binary                          # (n, n) — count both over
    S = binary.sum(dim=0)                            # (n,) — count each over
    OU = S.unsqueeze(1) - OO                         # i over, j under
    UO = S.unsqueeze(0) - OO                         # i under, j over
    UU = N - OO - OU - UO                           # both under
    denom = ((OO + OU) * (UO + UU) * (OO + UO) * (OU + UU)).clamp(min=1e-8).sqrt()
    phi = (OO * UU - OU * UO).float() / denom
    return phi


def simulate(decoder, val_loader, n_sim=N_SIM, n_games=N_GAMES,
             sigma_g_game=SIGMA_G_GAME, sigma_g_player=SIGMA_G_PLAYER, device=DEVICE):
    decoder.eval()
    norm = torch.distributions.Normal(0, 1)

    phi_all = []
    phi_same_team = []
    phi_cross_team = []
    p_over_stds = []
    n_processed = 0

    with torch.no_grad():
        for batch in val_loader:
            X_t, X_p, Y, weights, _, G_game, G_player, G_mask = batch
            X_p = X_p.to(device)
            G_game = G_game.to(device)
            G_player = G_player.to(device)
            G_mask = G_mask.to(device)
            weights = weights.to(device)

            # Only process games with valid G
            valid_idx = torch.where(G_mask)[0]
            if len(valid_idx) == 0:
                continue

            for idx in valid_idx:
                if n_games is not None and n_processed >= n_games:
                    break

                x_p = X_p[idx].unsqueeze(0)       # (1, 16, 24)
                g_g = G_game[idx].unsqueeze(0)     # (1, 6)
                g_p = G_player[idx].unsqueeze(0)   # (1, 16, 2)
                w = weights[idx]                    # (16,)

                active_mask = w > 0                # (16,) bool
                active_idx_local = torch.where(active_mask)[0]
                n_active = len(active_idx_local)
                if n_active < 2:
                    continue

                # Sample G n_sim times and compute P(over) per sample
                p_samples = []  # list of (n_active,) tensors
                for _ in range(n_sim):
                    g_g_n = g_g + torch.randn_like(g_g) * sigma_g_game
                    g_p_n = g_p + torch.randn_like(g_p) * sigma_g_player
                    mu_p, lv_p = decoder(g_g_n, g_p_n, x_p)     # (1, 16, 3)
                    sigma_p = (0.5 * lv_p).exp()
                    p_over = norm.cdf(mu_p / (sigma_p + 1e-8))   # (1, 16, 3)
                    # Flatten: (n_active,) using active player slots
                    p_flat = p_over[0, active_idx_local, :].reshape(-1)  # (n_active*3,)
                    p_samples.append(p_flat)

                p_mat = torch.stack(p_samples)   # (n_sim, n_active*3)

                # P(over|G) std — key diagnostic
                p_over_stds.append(p_mat.std(dim=0).mean().item())

                # Correct Bernoulli sampling
                binary = torch.bernoulli(p_mat)  # (n_sim, n_active*3)

                phi = phi_from_binary(binary)    # (n_active*3, n_active*3)

                # Mask diagonal (self-pairs)
                n_vars = p_mat.shape[1]
                mask_diag = ~torch.eye(n_vars, dtype=torch.bool)

                # Same-team vs cross-team classification
                # Player slots: 0..7 = home, 8..15 = away; 3 stats per player
                # active_idx_local indexes into the 16-slot array
                slots = active_idx_local.tolist()
                is_home = torch.tensor([s < 8 for s in slots], dtype=torch.bool)  # (n_active,)
                # Expand per stat
                is_home_var = is_home.repeat_interleave(3)  # (n_active*3,)
                same_team_mask = (is_home_var.unsqueeze(0) == is_home_var.unsqueeze(1)) & mask_diag
                cross_team_mask = (is_home_var.unsqueeze(0) != is_home_var.unsqueeze(1)) & mask_diag

                phi_vals = phi[mask_diag]
                phi_all.extend(phi_vals.tolist())
                phi_same_team.extend(phi[same_team_mask].tolist())
                phi_cross_team.extend(phi[cross_team_mask].tolist())
                n_processed += 1

            if n_games is not None and n_processed >= n_games:
                break

    return {
        "phi_all": phi_all,
        "phi_same_team": phi_same_team,
        "phi_cross_team": phi_cross_team,
        "p_over_std_mean": float(np.mean(p_over_stds)),
        "n_games": n_processed,
    }


def phi_max_theoretical(sigma_p, mean_p=0.5):
    """
    phi_max = sigma_p^2 / Var(X)
    Var(X) = E[p(1-p)] + Var(p) = mean_p*(1-mean_p)  [when mean_p = 0.5]
    """
    var_x = mean_p * (1 - mean_p)
    return sigma_p ** 2 / var_x


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=CKPT_DIR)
    parser.add_argument("--season", default="2019-26")
    parser.add_argument("--n-sim", type=int, default=N_SIM)
    parser.add_argument("--n-games", type=int, default=N_GAMES)
    parser.add_argument("--sigma-g-game", type=float, default=SIGMA_G_GAME)
    parser.add_argument("--sigma-g-player", type=float, default=SIGMA_G_PLAYER)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = ckpt_dir / "decoder_player_g.pt"

    print(f"Loading data ({args.season})...")
    from src.data.dataset import load_processed, temporal_split
    df = load_processed(season_suffix=args.season)
    _, val_df = temporal_split(df)
    print(f"  val={len(val_df)}")

    print("Building val loaders...")
    _, val_loader, norm_stats = make_loaders_player_g(val_df, val_df, batch_size=64, raw_dir=RAW_DIR)

    decoder = PlayerGDecoder(
        g_game_dim=G_GAME_DIM, g_player_dim=G_PLAYER_DIM,
        player_dim=PLAYER_DIM, h_dim=64, output_dim=OUTPUT_DIM, dropout=0.3,
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    decoder.load_state_dict(ckpt["decoder"])
    print(f"Loaded decoder from epoch {ckpt['epoch']}")

    print(f"\nRunning phi simulation: {args.n_sim} G samples × {args.n_games} games")
    print(f"  sigma_G_game={args.sigma_g_game}, sigma_G_player={args.sigma_g_player}")
    print(f"  (Bernoulli sampling — correct method)")

    results = simulate(
        decoder, val_loader,
        n_sim=args.n_sim, n_games=args.n_games,
        sigma_g_game=args.sigma_g_game, sigma_g_player=args.sigma_g_player,
    )

    sigma_p = results["p_over_std_mean"]
    phi_max = phi_max_theoretical(sigma_p)

    print(f"\n=== P(over|G) Statistics ===")
    print(f"mean P(over|G) std across players: {sigma_p:.4f}")
    print(f"Theoretical phi_max (ρ=1):          {phi_max:.4f}")
    print(f"Required sigma_P for phi>0.15:       {(0.15 * 0.25)**0.5:.4f}")

    print(f"\n=== Phi Statistics (n_games={results['n_games']}) ===")
    for label, phi_list in [
        ("All pairs", results["phi_all"]),
        ("Same-team", results["phi_same_team"]),
        ("Cross-team", results["phi_cross_team"]),
    ]:
        arr = np.array(phi_list)
        if len(arr) == 0:
            print(f"{label}: no pairs")
            continue
        print(f"\n{label} ({len(arr):,} pairs):")
        print(f"  mean={arr.mean():.4f}  std={arr.std():.4f}")
        print(f"  median={np.median(arr):.4f}  max={arr.max():.4f}  min={arr.min():.4f}")
        print(f"  > {PHI_THRESHOLD}: {(arr > PHI_THRESHOLD).sum():,} ({(arr > PHI_THRESHOLD).mean()*100:.2f}%)")
        print(f"  < -{PHI_THRESHOLD}: {(arr < -PHI_THRESHOLD).sum():,} ({(arr < -PHI_THRESHOLD).mean()*100:.2f}%)")
        print(f"  percentiles [5,25,50,75,95]: {np.percentile(arr, [5,25,50,75,95]).round(3).tolist()}")

"""
Proportional betting backtest for v7 PlayerGDecoder.

For each player pair in each val game:
  - Simulate 500 G samples → estimate P(OO), P(OU), P(UO), P(UU)
  - Bet $1 total per pair, split proportionally across all 4 outcomes
  - Each outcome is a 2-leg parlay; payout = 2.645x net (at -110/-110)
  - Evaluate against actual outcomes

This is an oracle ceiling test: uses actual G (actual mins+FGA), not encoder predictions.

Key math:
  PnL per pair = P_sim(actual_outcome) * 2.645 - 1.0
  Expected PnL = 2.645 * sum_k(P_sim(k) * P_true(k)) - 1
  Under no signal (P_sim = P_true = 0.25): EV = 2.645 * 0.25 - 1 = -0.089 per dollar

Usage:
  python scripts/backtest_player_g.py --n-games 500
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from src.data.dataset import load_processed, temporal_split
from scripts.train_player_g import (
    PlayerGDecoder, make_loaders_player_g,
    G_GAME_DIM, G_PLAYER_DIM, PLAYER_DIM, OUTPUT_DIM, RAW_DIR,
)

CKPT_DIR = "checkpoints_v7"
N_SIM = 500
N_GAMES = 500
SIGMA_G_GAME = 0.6
SIGMA_G_PLAYER = 0.5
PARLAY_PAYOUT = 3.645   # total return at -110/-110 per leg (decimal odds: 1.909 * 1.909)
DEVICE = "cpu"


def run_backtest(decoder, val_loader, n_sim=N_SIM, n_games=N_GAMES,
                 sigma_g_game=SIGMA_G_GAME, sigma_g_player=SIGMA_G_PLAYER):
    decoder.eval()
    norm = torch.distributions.Normal(0, 1)

    results = {k: {"pnl": 0.0, "staked": 0.0, "n": 0}
               for k in ("same_player", "same_team", "cross_team")}
    n_processed = 0

    with torch.no_grad():
        for batch in val_loader:
            X_t, X_p, Y, weights, _, G_game, G_player, G_mask = batch
            X_p = X_p.to(DEVICE)
            G_game = G_game.to(DEVICE)
            G_player = G_player.to(DEVICE)
            G_mask = G_mask.to(DEVICE)
            weights = weights.to(DEVICE)
            Y = Y.to(DEVICE)

            for idx in torch.where(G_mask)[0]:
                if n_processed >= n_games:
                    break

                x_p = X_p[idx].unsqueeze(0)
                g_g = G_game[idx].unsqueeze(0)
                g_p = G_player[idx].unsqueeze(0)
                w = weights[idx]
                y = Y[idx]   # (16, 3)

                active_slots = torch.where(w > 0)[0]
                n_active = len(active_slots)
                if n_active < 2:
                    continue

                # Simulate n_sim G perturbations → binary outcomes (n_sim, n_active*3)
                p_samples = []
                for _ in range(n_sim):
                    g_g_n = g_g + torch.randn_like(g_g) * sigma_g_game
                    g_p_n = g_p + torch.randn_like(g_p) * sigma_g_player
                    mu_p, lv_p = decoder(g_g_n, g_p_n, x_p)
                    sigma_p = (0.5 * lv_p).exp()
                    p_over = norm.cdf(mu_p / (sigma_p + 1e-8))
                    p_samples.append(p_over[0, active_slots, :].reshape(-1))

                p_mat = torch.stack(p_samples)          # (n_sim, n_vars)
                binary = torch.bernoulli(p_mat)         # (n_sim, n_vars)
                n_vars = p_mat.shape[1]

                # Vectorized joint outcome probabilities for all pairs
                # OO[i,j] = fraction of sims where both i and j were over
                OO = (binary.T @ binary) / n_sim        # (n_vars, n_vars)
                S = binary.mean(dim=0)                  # (n_vars,)
                OU = S.unsqueeze(1) - OO                # i over, j under
                UO = S.unsqueeze(0) - OO                # i under, j over
                UU = 1 - OO - OU - UO

                # Actual outcomes
                y_active = y[active_slots, :]           # (n_active, 3)
                actual = (y_active > 0).float().reshape(-1)  # (n_vars,)
                ai = actual.unsqueeze(1)                # (n_vars, 1)
                aj = actual.unsqueeze(0)                # (1, n_vars)

                # P_sim of the actual outcome for each pair (i, j)
                prob_correct = (ai * aj * OO
                                + ai * (1-aj) * OU
                                + (1-ai) * aj * UO
                                + (1-ai) * (1-aj) * UU)

                # PnL = stake_on_correct * payout - 1  (stake_correct = P_sim(correct))
                pnl_mat = prob_correct * PARLAY_PAYOUT - 1.0

                # Pair type mask
                slot_idx = torch.arange(n_vars) // 3      # index into active_slots
                slot_val = active_slots[slot_idx]          # actual 0-15 slot number

                same_player = slot_val.unsqueeze(1) == slot_val.unsqueeze(0)
                home_i = slot_val.unsqueeze(1) < 8
                home_j = slot_val.unsqueeze(0) < 8
                same_team = (home_i == home_j) & ~same_player
                cross_team = (home_i != home_j)

                # Upper triangle only (each pair once)
                upper = torch.triu(torch.ones(n_vars, n_vars, dtype=torch.bool), diagonal=1)

                for label, mask in [("same_player", same_player & upper),
                                     ("same_team", same_team & upper),
                                     ("cross_team", cross_team & upper)]:
                    if mask.any():
                        pnl_vals = pnl_mat[mask]
                        results[label]["pnl"] += pnl_vals.sum().item()
                        results[label]["staked"] += mask.sum().item()
                        results[label]["n"] += mask.sum().item()

                n_processed += 1

            if n_processed >= n_games:
                break

    return results, n_processed


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

    print(f"Loading data ({args.season})...")
    df = load_processed(season_suffix=args.season)
    _, val_df = temporal_split(df)

    print("Building val loaders...")
    _, val_loader, _ = make_loaders_player_g(val_df, val_df, batch_size=32, raw_dir=RAW_DIR)

    decoder = PlayerGDecoder(
        g_game_dim=G_GAME_DIM, g_player_dim=G_PLAYER_DIM,
        player_dim=PLAYER_DIM, h_dim=64, output_dim=OUTPUT_DIM, dropout=0.3,
    )
    ckpt = torch.load(Path(args.ckpt_dir) / "decoder_player_g.pt",
                      map_location=DEVICE, weights_only=False)
    decoder.load_state_dict(ckpt["decoder"])
    print(f"Loaded decoder (epoch {ckpt['epoch']})\n")

    print(f"Running backtest: {args.n_games} games × {args.n_sim} G samples")
    print(f"Payout: {PARLAY_PAYOUT}x total return (−110/−110)  |  Expected ROI with no signal: −8.9%\n")

    results, n_games = run_backtest(
        decoder, val_loader,
        n_sim=args.n_sim, n_games=args.n_games,
        sigma_g_game=args.sigma_g_game,
        sigma_g_player=args.sigma_g_player,
    )

    print(f"Games evaluated: {n_games}\n")

    total_pnl = sum(v["pnl"] for v in results.values())
    total_staked = sum(v["staked"] for v in results.values())

    print(f"{'Type':<15} {'Pairs':>8} {'Staked':>10} {'PnL':>10} {'ROI':>8}")
    print("-" * 55)
    for label, r in results.items():
        if r["n"] == 0:
            continue
        roi = r["pnl"] / r["staked"] * 100 if r["staked"] > 0 else 0
        print(f"{label:<15} {r['n']:>8,} {r['staked']:>10,.0f}  {r['pnl']:>+9.1f}  {roi:>+7.2f}%")
    print("-" * 55)
    overall_roi = total_pnl / total_staked * 100 if total_staked > 0 else 0
    print(f"{'TOTAL':<15} {sum(v['n'] for v in results.values()):>8,} {total_staked:>10,.0f}  {total_pnl:>+9.1f}  {overall_roi:>+7.2f}%")
    print(f"\nNote: −8.9% ROI expected with zero signal at −110/−110 parlays")

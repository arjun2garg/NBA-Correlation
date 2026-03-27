"""
Threshold betting backtest: only bet on outcomes with P_sim > threshold.

Compares:
  1. Proportional (bet on all 4 outcomes proportionally) — previous script
  2. Top-1 only (bet $1 on the single most likely outcome)
  3. Threshold (bet $1 on each outcome where P_sim > threshold)
     threshold=0.378 = breakeven at -110/-110 payout (1/2.645)

Usage:
  python scripts/backtest_threshold.py --n-games 500
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
PARLAY_PAYOUT = 3.645   # total return at -110/-110 (decimal odds: 1.909 * 1.909)
# EV = P * 3.645 - 1 > 0  →  P > 1/3.645 = 0.274
BREAKEVEN = 1 / PARLAY_PAYOUT              # = 0.274
DEVICE = "cpu"


def run(decoder, val_loader, n_sim, n_games, sigma_g_game, sigma_g_player):
    decoder.eval()
    norm = torch.distributions.Normal(0, 1)

    # Track 3 strategies separately
    strategies = ["proportional", "top1", "threshold"]
    pair_types = ["same_player", "same_team", "cross_team"]

    stats = {
        s: {pt: {"pnl": 0.0, "staked": 0.0, "n_bets": 0, "n_pairs": 0}
            for pt in pair_types}
        for s in strategies
    }

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
                y = Y[idx]

                active_slots = torch.where(w > 0)[0]
                n_active = len(active_slots)
                if n_active < 2:
                    continue

                p_samples = []
                for _ in range(n_sim):
                    g_g_n = g_g + torch.randn_like(g_g) * sigma_g_game
                    g_p_n = g_p + torch.randn_like(g_p) * sigma_g_player
                    mu_p, lv_p = decoder(g_g_n, g_p_n, x_p)
                    sigma_p = (0.5 * lv_p).exp()
                    p_over = norm.cdf(mu_p / (sigma_p + 1e-8))
                    p_samples.append(p_over[0, active_slots, :].reshape(-1))

                p_mat = torch.stack(p_samples)
                binary = torch.bernoulli(p_mat)
                n_vars = p_mat.shape[1]

                OO = (binary.T @ binary) / n_sim
                S = binary.mean(dim=0)
                OU = S.unsqueeze(1) - OO
                UO = S.unsqueeze(0) - OO
                UU = 1 - OO - OU - UO

                probs = torch.stack([OO, OU, UO, UU], dim=0)  # (4, n_vars, n_vars)

                y_active = y[active_slots, :]
                actual = (y_active > 0).int().reshape(-1)
                ai = actual.unsqueeze(1)
                aj = actual.unsqueeze(0)

                # Actual outcome index (0=OO, 1=OU, 2=UO, 3=UU)
                outcome_idx = (1 - ai) * 2 + (1 - aj)   # (n_vars, n_vars) int
                # OO: ai=1,aj=1 → 0; OU: ai=1,aj=0 → 1; UO: ai=0,aj=1 → 2; UU: ai=0,aj=0 → 3

                # Pair type masks
                slot_idx = torch.arange(n_vars) // 3
                slot_val = active_slots[slot_idx]
                same_player = (slot_val.unsqueeze(1) == slot_val.unsqueeze(0))
                home_i = slot_val.unsqueeze(1) < 8
                home_j = slot_val.unsqueeze(0) < 8
                same_team = (home_i == home_j) & ~same_player
                cross_team = home_i != home_j
                upper = torch.triu(torch.ones(n_vars, n_vars, dtype=torch.bool), diagonal=1)

                for pt, type_mask in [("same_player", same_player),
                                       ("same_team", same_team),
                                       ("cross_team", cross_team)]:
                    mask = type_mask & upper
                    if not mask.any():
                        continue

                    # For each pair in mask, get the 4 outcome probs and actual outcome
                    pair_i, pair_j = torch.where(mask)

                    # (n_pairs, 4) outcome probabilities
                    p4 = probs[:, pair_i, pair_j].T   # (n_pairs, 4)
                    actual_k = outcome_idx[pair_i, pair_j]  # (n_pairs,)
                    p_actual = p4[torch.arange(len(pair_i)), actual_k]  # (n_pairs,)

                    n_p = len(pair_i)
                    stats["proportional"][pt]["n_pairs"] += n_p

                    # 1. Proportional: bet on all 4, stakes = probs
                    pnl_prop = p_actual * PARLAY_PAYOUT - 1.0
                    stats["proportional"][pt]["pnl"] += pnl_prop.sum().item()
                    stats["proportional"][pt]["staked"] += n_p
                    stats["proportional"][pt]["n_bets"] += n_p

                    # 2. Top-1: bet $1 on single most likely outcome
                    top_prob, top_k = p4.max(dim=1)
                    won_top1 = (top_k == actual_k).float()
                    pnl_top1 = won_top1 * PARLAY_PAYOUT - 1.0
                    stats["top1"][pt]["pnl"] += pnl_top1.sum().item()
                    stats["top1"][pt]["staked"] += n_p
                    stats["top1"][pt]["n_bets"] += n_p
                    stats["top1"][pt]["n_pairs"] += n_p

                    # 3. Threshold: bet $1 on each outcome where P_sim > BREAKEVEN
                    # Only count pairs where at least one outcome clears the threshold
                    for k in range(4):
                        bet_mask = p4[:, k] > BREAKEVEN
                        if bet_mask.any():
                            won = (actual_k[bet_mask] == k).float()
                            pnl_k = won * PARLAY_PAYOUT - 1.0
                            stats["threshold"][pt]["pnl"] += pnl_k.sum().item()
                            stats["threshold"][pt]["staked"] += bet_mask.sum().item()
                            stats["threshold"][pt]["n_bets"] += bet_mask.sum().item()
                    stats["threshold"][pt]["n_pairs"] += n_p

                n_processed += 1
            if n_processed >= n_games:
                break

    return stats, n_processed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=CKPT_DIR)
    parser.add_argument("--season", default="2019-26")
    parser.add_argument("--n-sim", type=int, default=N_SIM)
    parser.add_argument("--n-games", type=int, default=N_GAMES)
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

    print(f"Breakeven threshold: P > {BREAKEVEN:.3f}  (payout {PARLAY_PAYOUT}x at -110/-110)\n")

    results, n_games = run(
        decoder, val_loader,
        n_sim=args.n_sim, n_games=args.n_games,
        sigma_g_game=SIGMA_G_GAME, sigma_g_player=SIGMA_G_PLAYER,
    )

    print(f"Games: {n_games}\n")

    for strategy in ["proportional", "top1", "threshold"]:
        label = {
            "proportional": "Proportional (all 4 outcomes)",
            "top1": "Top-1 (most likely outcome only)",
            "threshold": f"Threshold (P > {BREAKEVEN:.3f} only)",
        }[strategy]
        print(f"=== {label} ===")

        total_pnl = sum(v["pnl"] for v in results[strategy].values())
        total_staked = sum(v["staked"] for v in results[strategy].values())
        total_bets = sum(v["n_bets"] for v in results[strategy].values())
        total_pairs = sum(v["n_pairs"] for v in results[strategy].values())

        print(f"  {'Type':<15} {'Bets':>8} {'Staked':>10} {'PnL':>10} {'ROI':>8}")
        print(f"  {'-'*52}")
        for pt in ["same_player", "same_team", "cross_team"]:
            r = results[strategy][pt]
            if r["staked"] == 0:
                continue
            roi = r["pnl"] / r["staked"] * 100
            print(f"  {pt:<15} {r['n_bets']:>8,} {r['staked']:>10,.0f}  {r['pnl']:>+9.1f}  {roi:>+7.2f}%")
        print(f"  {'-'*52}")
        roi_total = total_pnl / total_staked * 100 if total_staked > 0 else 0
        print(f"  {'TOTAL':<15} {total_bets:>8,} {total_staked:>10,.0f}  {total_pnl:>+9.1f}  {roi_total:>+7.2f}%")
        if strategy == "threshold":
            print(f"  (bets placed on {total_bets/total_pairs*100:.1f}% of pair-outcomes)")
        print()

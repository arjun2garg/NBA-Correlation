"""
Pure correlation backtest — three strategies.

Instead of betting on a specific outcome (OO, OU, UU, UO), we bet on the
DIRECTION of correlation:
  - "Same direction" bet: covers both OO and UU (players go same way)
  - "Opposite direction" bet: covers both OU and UO (players go opposite ways)

Three strategies compared:

  1. raw:      confidence = P_same - P_opp
               Leaks individual signal: high P_same when both players have high P(over)
               even if they aren't correlated.

  2. phi:      confidence = P_same - P_same_ind  = 2 * Cov(Xi, Xj)
               Independence-adjusted. Removes marginal effects. Equivalent to phi numerator.
               Expect ~-8.9% ROI if phi ≈ 0.

  3. balanced: Set each player's line to the model's median mu prediction for this game
               (median of mu_pred across 500 G samples). This forces P(over)=50% for each
               player in simulation, so OO=UU and OU=UO under independence.
               confidence = P_same_balanced - 0.5
               Actual outcome: y_residual > median_mu  (continuous Y vs model median)
               Under no correlation: balanced win rate ≈ 50% regardless of individual signal.
               Under genuine correlation: balanced win rate > 50%.

Diagnostic: 2×4 confusion table per predicted group.
  When predicting SAME (or OPP), track actual breakdown: OO / UU / OU / UO.
  Genuine correlation: OO≈UU within same group, OU≈UO within opposite group.
  Individual signal: OO >> UU when predicting SAME (both just have high P(over)).

Bet size: $1 per pair, equal $0.5 split within group.

Usage:
  python scripts/backtest_correlation.py --n-games 500
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
DEVICE = "cpu"

CONFUSION_KEYS = [
    "pred_same_act_oo", "pred_same_act_uu", "pred_same_act_ou", "pred_same_act_uo",
    "pred_opp_act_oo",  "pred_opp_act_uu",  "pred_opp_act_ou",  "pred_opp_act_uo",
]


def run(decoder, val_loader, n_sim, n_games, sigma_g_game, sigma_g_player):
    decoder.eval()
    norm = torch.distributions.Normal(0, 1)

    pair_types = ["same_player", "same_team", "cross_team"]
    strategies = ["raw", "phi", "balanced"]

    def empty_stat():
        d = {"pnl": 0.0, "staked": 0.0, "n": 0, "wins": 0}
        for k in CONFUSION_KEYS:
            d[k] = 0
        return d

    stats = {s: {pt: empty_stat() for pt in pair_types} for s in strategies}
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

                # Simulate G samples — store both mu and p(over)
                p_samples = []
                mu_samples = []
                for _ in range(n_sim):
                    g_g_n = g_g + torch.randn_like(g_g) * sigma_g_game
                    g_p_n = g_p + torch.randn_like(g_p) * sigma_g_player
                    mu_p, lv_p = decoder(g_g_n, g_p_n, x_p)
                    sigma_p = (0.5 * lv_p).exp()
                    p_over = norm.cdf(mu_p / (sigma_p + 1e-8))
                    p_samples.append(p_over[0, active_slots, :].reshape(-1))
                    mu_samples.append(mu_p[0, active_slots, :].reshape(-1))

                p_mat  = torch.stack(p_samples)   # (n_sim, n_vars)
                mu_mat = torch.stack(mu_samples)  # (n_sim, n_vars) — in normalized Y space
                binary = torch.bernoulli(p_mat)
                n_vars = p_mat.shape[1]

                # ── Joint probs from original binary ──────────────────────────
                OO = (binary.T @ binary) / n_sim
                S  = binary.mean(dim=0)
                OU = S.unsqueeze(1) - OO
                UO = S.unsqueeze(0) - OO
                UU = 1 - OO - OU - UO

                # Strategy 1: raw P_same - P_opp (leaks individual signal)
                conf_raw = (OO + UU) - (OU + UO)

                # Strategy 2: phi-based / independence-adjusted
                P_same_ind = (S.unsqueeze(1) * S.unsqueeze(0)
                              + (1 - S.unsqueeze(1)) * (1 - S.unsqueeze(0)))
                conf_phi = (OO + UU) - P_same_ind   # = 2 * Cov(Xi, Xj)

                # ── Balanced: set each player's line to median mu ─────────────
                # median_mu[k] = model's median prediction for player-stat k this game
                # In normalized Y space, same as the continuous residual targets
                median_mu = mu_mat.median(dim=0).values   # (n_vars,)

                # In simulation: binary where over = mu_sample > median_mu (→ 50% each)
                binary_bal = (mu_mat > median_mu.unsqueeze(0)).float()

                OO_b = (binary_bal.T @ binary_bal) / n_sim
                S_b  = binary_bal.mean(dim=0)
                OU_b = S_b.unsqueeze(1) - OO_b
                UO_b = S_b.unsqueeze(0) - OO_b
                UU_b = 1 - OO_b - OU_b - UO_b
                conf_bal = (OO_b + UU_b) - 0.5  # centered at 0; > 0 = predict same

                # Actual outcomes at balanced line:
                # y_active (n_active, 3) is the continuous normalized residual
                # median_mu (n_vars,) is in the same normalized space
                y_active = y[active_slots, :]                      # (n_active, 3)
                y_flat   = y_active.reshape(-1)                     # (n_vars,)
                actual_bal = (y_flat > median_mu).int()             # (n_vars,)

                # ── Standard actual outcomes (threshold = 0) ──────────────────
                actual_std = (y_flat > 0).int()   # (n_vars,)

                # ── Pair type masks ───────────────────────────────────────────
                slot_idx   = torch.arange(n_vars) // 3
                slot_val   = active_slots[slot_idx]
                same_player = (slot_val.unsqueeze(1) == slot_val.unsqueeze(0))
                home_i      = slot_val.unsqueeze(1) < 8
                home_j      = slot_val.unsqueeze(0) < 8
                same_team   = (home_i == home_j) & ~same_player
                cross_team  = home_i != home_j
                upper = torch.triu(torch.ones(n_vars, n_vars, dtype=torch.bool), diagonal=1)

                for pt, type_mask in [("same_player", same_player),
                                       ("same_team",   same_team),
                                       ("cross_team",  cross_team)]:
                    mask = type_mask & upper
                    if not mask.any():
                        continue

                    pair_i, pair_j = torch.where(mask)

                    # Build actual-outcome flags for each strategy's line
                    def outcome_flags(actual):
                        ai = actual.float().unsqueeze(1)  # (n_vars, 1)
                        aj = actual.float().unsqueeze(0)  # (1, n_vars)
                        oo = (ai == 1) & (aj == 1)
                        uu = (ai == 0) & (aj == 0)
                        ou = (ai == 1) & (aj == 0)
                        uo = (ai == 0) & (aj == 1)
                        same = oo | uu
                        return (oo[pair_i, pair_j], uu[pair_i, pair_j],
                                ou[pair_i, pair_j], uo[pair_i, pair_j],
                                same[pair_i, pair_j])

                    a_oo_s, a_uu_s, a_ou_s, a_uo_s, a_same_s = outcome_flags(actual_std)
                    a_oo_b, a_uu_b, a_ou_b, a_uo_b, a_same_b = outcome_flags(actual_bal)

                    for strategy, conf_mat, a_oo, a_uu, a_ou, a_uo, a_same in [
                        ("raw",      conf_raw, a_oo_s, a_uu_s, a_ou_s, a_uo_s, a_same_s),
                        ("phi",      conf_phi, a_oo_s, a_uu_s, a_ou_s, a_uo_s, a_same_s),
                        ("balanced", conf_bal, a_oo_b, a_uu_b, a_ou_b, a_uo_b, a_same_b),
                    ]:
                        conf = conf_mat[pair_i, pair_j]
                        predict_same = conf > 0

                        bet_size = torch.ones(len(pair_i))

                        won_same  = predict_same  & a_same
                        lost_same = predict_same  & ~a_same
                        won_opp   = ~predict_same & ~a_same
                        lost_opp  = ~predict_same & a_same

                        pnl = torch.zeros(len(pair_i))
                        pnl[won_same]  = bet_size[won_same]  * 0.5 * PARLAY_PAYOUT - bet_size[won_same]
                        pnl[lost_same] = -bet_size[lost_same]
                        pnl[won_opp]   = bet_size[won_opp]   * 0.5 * PARLAY_PAYOUT - bet_size[won_opp]
                        pnl[lost_opp]  = -bet_size[lost_opp]

                        st = stats[strategy][pt]
                        st["pnl"]    += pnl.sum().item()
                        st["staked"] += bet_size.sum().item()
                        st["n"]      += len(pair_i)
                        st["wins"]   += (won_same | won_opp).sum().item()

                        for flag, suffix in [(a_oo, "oo"), (a_uu, "uu"),
                                              (a_ou, "ou"), (a_uo, "uo")]:
                            st[f"pred_same_act_{suffix}"] += (predict_same  & flag).sum().item()
                            st[f"pred_opp_act_{suffix}"]  += (~predict_same & flag).sum().item()

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

    print(f"Running correlation backtest: {args.n_games} games × {args.n_sim} G samples\n")
    print(f"  raw:      confidence = P_same - P_opp (leaks individual signal)")
    print(f"  phi:      confidence = P_same - P_same_ind = 2*Cov (pure correlation)")
    print(f"  balanced: lines set to model median mu; confidence = P_same_balanced - 0.5")
    print(f"            actual outcome = y_residual > median_mu (continuous comparison)\n")

    results, n_games = run(
        decoder, val_loader,
        n_sim=args.n_sim, n_games=args.n_games,
        sigma_g_game=SIGMA_G_GAME, sigma_g_player=SIGMA_G_PLAYER,
    )

    print(f"Games: {n_games}\n")

    labels = {
        "raw":      "Raw P_same (leaks individual signal)",
        "phi":      "Phi-based / Cov(Xi,Xj) — pure correlation",
        "balanced": "Balanced lines (median mu) — correlation after centering",
    }

    for strategy in ["raw", "phi", "balanced"]:
        r_all = results[strategy]
        total_pnl    = sum(v["pnl"]    for v in r_all.values())
        total_staked = sum(v["staked"] for v in r_all.values())
        total_n      = sum(v["n"]      for v in r_all.values())
        total_wins   = sum(v["wins"]   for v in r_all.values())

        print(f"{'='*62}")
        print(f"=== {labels[strategy]} ===")
        print(f"{'='*62}")
        print(f"\n  {'Type':<15} {'Pairs':>8} {'Staked':>10} {'PnL':>10} {'ROI':>8} {'Win%':>7}")
        print("  " + "-" * 62)
        for pt, r in r_all.items():
            if r["n"] == 0:
                continue
            roi = r["pnl"] / r["staked"] * 100 if r["staked"] > 0 else 0
            wr  = r["wins"] / r["n"] * 100
            print(f"  {pt:<15} {r['n']:>8,} {r['staked']:>10,.1f}  {r['pnl']:>+9.1f}  {roi:>+7.2f}%  {wr:>6.1f}%")
        print("  " + "-" * 62)
        overall_roi = total_pnl / total_staked * 100 if total_staked > 0 else 0
        overall_wr  = total_wins / total_n * 100 if total_n > 0 else 0
        print(f"  {'TOTAL':<15} {total_n:>8,} {total_staked:>10,.1f}  {total_pnl:>+9.1f}  "
              f"{overall_roi:>+7.2f}%  {overall_wr:>6.1f}%")

        # 2×4 confusion table aggregated across all pair types
        agg = {k: sum(r_all[pt][k] for pt in r_all) for k in CONFUSION_KEYS}
        print(f"\n  --- Confusion: predicted group × actual outcome ---")
        print(f"  Genuine correlation → OO≈UU in pred-SAME, OU≈UO in pred-OPP")
        print(f"  Individual signal   → OO >> UU in pred-SAME\n")

        for pred_group in ["same", "opp"]:
            keys = [("oo", "OO (win)"), ("uu", "UU (win)"), ("ou", "OU (loss)"), ("uo", "UO (loss)")]
            total_pred = sum(agg[f"pred_{pred_group}_act_{s}"] for s, _ in keys)
            if total_pred == 0:
                continue
            print(f"  When predicting {pred_group.upper()} ({total_pred:,} pairs):")
            for suffix, lbl in keys:
                cnt = agg[f"pred_{pred_group}_act_{suffix}"]
                pct = cnt / total_pred * 100
                bar = "#" * int(pct / 2)
                print(f"    actual {lbl:<12}: {cnt:>8,}  ({pct:5.1f}%)  {bar}")
            print()
        print()

    print(f"Breakeven win rate: {1/(1 + 0.5*PARLAY_PAYOUT)*100:.1f}%  "
          f"(net per win: +{0.5*PARLAY_PAYOUT-1:.4f}, net per loss: -1.0000)")
    print(f"Baseline ROI with random direction: ~-8.9%")
    print(f"\nKey interpretation for 'balanced' strategy:")
    print(f"  Win% ≈ 50% → no genuine correlation (individual signal was all we had)")
    print(f"  Win% > 50% → genuine positive correlation exists")
    print(f"  Win% < 50% → genuine negative correlation exists")

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

from torch.utils.data import DataLoader

from src.data.dataset import load_processed, temporal_split, build_tensors, NBADataset, TARGET_COLS
from src.model import GameEncoder, PlayerDecoder
from src.simulate import simulate
from src.evaluate import phi_coefficient, extract_pairs, backtest

# --- config ---
CHECKPOINT    = "checkpoints/model_latest.pt"
NUM_SAMPLES   = 500
PHI_THRESHOLD = 0.15
TOP_K         = 10
PARLAY_ODDS   = -110
BATCH_SIZE    = 32
DEVICE        = "cpu"
SEED          = 42

if __name__ == "__main__":
    torch.manual_seed(SEED)

    # load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    cfg    = ckpt["config"]
    Y_mean = ckpt["Y_mean"]
    Y_std  = ckpt["Y_std"]
    Xt_mean = ckpt["Xt_mean"]
    Xt_std  = ckpt["Xt_std"]
    Xp_mean = ckpt["Xp_mean"]
    Xp_std  = ckpt["Xp_std"]

    encoder = GameEncoder(
        input_dim=cfg["team_dim"], h_dim=cfg["h_dim_enc"], latent_dim=cfg["latent_dim"],
        dropout=cfg.get("dropout", 0.0),
    ).to(DEVICE)
    decoder = PlayerDecoder(
        latent_dim=cfg["latent_dim"], player_dim=cfg["player_dim"],
        h_dim=cfg["h_dim_dec"], output_dim=cfg["n_target_cols"],
        dropout=cfg.get("dropout", 0.0),
    ).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    # build val loader using checkpoint normalization stats
    _, val_df = temporal_split(load_processed(season_suffix="2022-25"))
    X_team_v, X_pl_v, masks_v, Y_v, lines_v = build_tensors(val_df)
    Y_v      = (Y_v      - Y_mean)  / Y_std
    X_team_v = (X_team_v - Xt_mean) / Xt_std
    X_pl_v   = (X_pl_v   - Xp_mean) / Xp_std
    val_loader = DataLoader(
        NBADataset(X_team_v, X_pl_v, Y_v, masks_v, lines_v),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    print(f"Simulating {len(val_loader)} batches × {NUM_SAMPLES} samples...")
    results = simulate(encoder, decoder, val_loader, Y_mean, Y_std,
                       num_samples=NUM_SAMPLES, device=DEVICE)

    # aggregate backtest across all batches
    all_records = []
    for batch in results:
        phi = phi_coefficient(
            batch["OO"], batch["OU"], batch["UO"], batch["UU"], batch["mask_flat"],
            n_stats=len(TARGET_COLS),
        )
        pairs = extract_pairs(
            phi, batch["OO"], batch["OU"], batch["UO"], batch["UU"],
            NUM_SAMPLES, batch["mask_flat"],
            TARGET_COLS, threshold=PHI_THRESHOLD, top_k=TOP_K,
        )
        bt = backtest(pairs, batch["actual_over"], parlay_odds=PARLAY_ODDS)
        all_records.extend(bt["records"])

    # summary
    total_bets = len(all_records)
    wins       = sum(r["won"] for r in all_records)
    losses     = total_bets - wins
    net_pnl    = sum(r["pnl"] for r in all_records)
    win_rate   = wins / total_bets if total_bets > 0 else 0.0
    roi        = net_pnl / total_bets if total_bets > 0 else 0.0

    decimal    = 1 + 100 / abs(PARLAY_ODDS)
    multiplier = decimal ** 2
    breakeven  = 1 / multiplier

    print(f"\n=== Backtest Summary ===")
    print(f"Val games simulated : {len(results)} batches")
    print(f"Num samples per game: {NUM_SAMPLES}")
    print(f"Phi threshold       : {PHI_THRESHOLD}")
    print(f"Total bets          : {total_bets}")
    print(f"Wins / Losses       : {wins} / {losses}")
    print(f"Win rate            : {win_rate:.3f}  (breakeven: {breakeven:.3f})")
    print(f"Net P&L (stakes)    : {net_pnl:+.2f}")
    print(f"ROI                 : {roi:+.3f}")

    # per-direction breakdown
    print(f"\n=== Breakdown by Direction ===")
    for direction in ["OO", "UU", "OU", "UO"]:
        recs = [r for r in all_records if r["predicted_dir"] == direction]
        if not recs:
            continue
        n = len(recs)
        w = sum(r["won"] for r in recs)
        pnl = sum(r["pnl"] for r in recs)
        print(f"  {direction}: {n:4d} bets | {w:4d} wins | wr={w/n:.3f} | roi={pnl/n:+.3f}")

    # phi distribution across all valid cross-player pairs
    print(f"\n=== Phi Distribution (all cross-player pairs) ===")
    n_stats = len(TARGET_COLS)
    phi_vals = []
    for batch in results:
        phi_b = phi_coefficient(
            batch["OO"], batch["OU"], batch["UO"], batch["UU"], batch["mask_flat"],
            n_stats=n_stats,
        )
        batch_size, n_vars, _ = phi_b.shape
        upper = torch.triu(torch.ones(n_vars, n_vars, dtype=torch.bool), diagonal=1)
        for b in range(batch_size):
            vals = phi_b[b][upper]
            phi_vals.append(vals[~torch.isnan(vals)])
    phi_all = torch.cat(phi_vals)
    pos = (phi_all > 0).sum().item()
    neg = (phi_all < 0).sum().item()
    total = len(phi_all)
    print(f"  Positive phi: {pos:6d} ({pos/total:.3f})")
    print(f"  Negative phi: {neg:6d} ({neg/total:.3f})")
    print(f"  Mean phi    : {phi_all.mean().item():.4f}")
    print(f"  Std phi     : {phi_all.std().item():.4f}")
    bins = [(-1.0, -0.5), (-0.5, -0.15), (-0.15, 0.15), (0.15, 0.5), (0.5, 1.01)]
    for lo, hi in bins:
        count = ((phi_all >= lo) & (phi_all < hi)).sum().item()
        print(f"  [{lo:+.2f}, {hi:+.2f}): {count:6d} ({count/total:.3f})")

    # same-team vs cross-team phi breakdown
    print(f"\n=== Phi by Team Relationship ===")
    n_players_per_team = 8
    same_phi, cross_phi = [], []
    for batch in results:
        phi_b = phi_coefficient(
            batch["OO"], batch["OU"], batch["UO"], batch["UU"], batch["mask_flat"],
            n_stats=n_stats,
        )
        batch_size, n_vars, _ = phi_b.shape
        upper = torch.triu(torch.ones(n_vars, n_vars, dtype=torch.bool), diagonal=1)
        vars_per_team = n_players_per_team * n_stats
        for b in range(batch_size):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    v = phi_b[b, i, j].item()
                    if torch.isnan(phi_b[b, i, j]):
                        continue
                    same_team = (i // vars_per_team) == (j // vars_per_team)
                    if same_team:
                        same_phi.append(v)
                    else:
                        cross_phi.append(v)
    same_phi  = torch.tensor(same_phi)
    cross_phi = torch.tensor(cross_phi)
    print(f"  Same-team  pairs: {len(same_phi):6d} | mean phi={same_phi.mean():.4f} | "
          f"pct > 0.15: {(same_phi > 0.15).float().mean():.3f} | "
          f"pct < -0.15: {(same_phi < -0.15).float().mean():.3f}")
    print(f"  Cross-team pairs: {len(cross_phi):6d} | mean phi={cross_phi.mean():.4f} | "
          f"pct > 0.15: {(cross_phi > 0.15).float().mean():.3f} | "
          f"pct < -0.15: {(cross_phi < -0.15).float().mean():.3f}")

    # actual outcome distribution across all valid cross-player pairs
    print(f"\n=== Actual Joint Outcome Distribution ===")
    n_stats = len(TARGET_COLS)
    oo_count = ou_count = uo_count = uu_count = 0
    for batch in results:
        actual = batch["actual_over"]          # (batch, n_vars)
        mask   = batch["mask_flat"]            # (batch, n_vars)
        batch_size, n_vars = actual.shape
        for b in range(batch_size):
            for i in range(n_vars):
                if mask[b, i] == 0:
                    continue
                for j in range(i + 1, n_vars):
                    if mask[b, j] == 0:
                        continue
                    if i // n_stats == j // n_stats:   # same player
                        continue
                    ai, aj = actual[b, i].item(), actual[b, j].item()
                    if   ai == 1 and aj == 1: oo_count += 1
                    elif ai == 1 and aj == 0: ou_count += 1
                    elif ai == 0 and aj == 1: uo_count += 1
                    else:                     uu_count += 1
    total_pairs = oo_count + ou_count + uo_count + uu_count
    for label, count in [("OO", oo_count), ("OU", ou_count), ("UO", uo_count), ("UU", uu_count)]:
        print(f"  {label}: {count:6d}  ({count/total_pairs:.3f})")
    print(f"  Total cross-player pairs: {total_pairs}")
    over_rate = (oo_count + ou_count) / total_pairs  # marginal over rate for i
    print(f"  Marginal over rate (any single stat): {over_rate:.3f}")

    # per-stat predicted vs actual over rates
    print(f"\n=== Per-Stat Over Rates (Predicted vs Actual) ===")
    n_stats = len(TARGET_COLS)
    pred_over_sum  = torch.zeros(n_stats)
    pred_over_cnt  = torch.zeros(n_stats)
    actual_over_sum = torch.zeros(n_stats)
    actual_over_cnt = torch.zeros(n_stats)
    for batch in results:
        # over_mean: marginal P(over) per variable, shape (batch, n_vars)
        over_mean = batch["over_mean"]
        mask      = batch["mask_flat"]                   # (batch, n_vars)
        actual    = batch["actual_over"]                 # (batch, n_vars)
        for s in range(n_stats):
            stat_mask = mask[:, s::n_stats]             # (batch, n_players)
            pred_over_sum[s]   += (over_mean[:, s::n_stats] * stat_mask).sum()
            pred_over_cnt[s]   += stat_mask.sum()
            actual_over_sum[s] += (actual[:, s::n_stats] * stat_mask).sum()
            actual_over_cnt[s] += stat_mask.sum()
    print(f"  {'Stat':<12}  {'Pred Over%':>10}  {'Actual Over%':>12}")
    for s, name in enumerate(TARGET_COLS):
        pred_rate   = (pred_over_sum[s] / pred_over_cnt[s]).item() if pred_over_cnt[s] > 0 else float("nan")
        actual_rate = (actual_over_sum[s] / actual_over_cnt[s]).item() if actual_over_cnt[s] > 0 else float("nan")
        print(f"  {name:<12}  {pred_rate:>10.3f}  {actual_rate:>12.3f}")

    if all_records:
        print(f"\n=== Top 10 Bets by Phi ===")
        top = sorted(all_records, key=lambda r: abs(r["phi"]), reverse=True)[:10]
        for r in top:
            outcome = "WIN " if r["won"] else "LOSS"
            print(
                f"  [{outcome}] phi={r['phi']:.3f}  "
                f"{r['i_label'][1]}(slot {r['i_label'][0]}) & "
                f"{r['j_label'][1]}(slot {r['j_label'][0]})  "
                f"dir={r['predicted_dir']}  "
                f"actual=({r['i_actual']},{r['j_actual']})"
            )

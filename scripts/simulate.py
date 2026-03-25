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
NUM_SAMPLES   = 100
PHI_THRESHOLD = 0.15
TOP_K         = 10
PARLAY_ODDS   = -110
BATCH_SIZE    = 32
DEVICE        = "cpu"

if __name__ == "__main__":
    # load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]
    Y_mean = ckpt["Y_mean"]
    Y_std  = ckpt["Y_std"]

    encoder = GameEncoder(
        input_dim=cfg["team_dim"], h_dim=cfg["h_dim_enc"], latent_dim=cfg["latent_dim"]
    ).to(DEVICE)
    decoder = PlayerDecoder(
        latent_dim=cfg["latent_dim"], player_dim=cfg["player_dim"],
        h_dim=cfg["h_dim_dec"], output_dim=cfg["n_target_cols"]
    ).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    # build val loader using checkpoint Y_mean/Y_std (avoids double-normalize)
    _, val_df = temporal_split(load_processed())
    X_team_v, X_pl_v, masks_v, Y_v, lines_v = build_tensors(val_df)
    Y_v = (Y_v - Y_mean) / Y_std
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
            batch["OO"], batch["OU"], batch["UO"], batch["UU"], batch["mask_flat"]
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

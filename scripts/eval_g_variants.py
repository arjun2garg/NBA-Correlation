"""
Evaluate G encoding variants for AUC signal.

For each G variant, measures:
  1. AUC of GBT predicting (player_stat > h_stat) using only G features
  2. AUC of GBT using G + player features (combined)
  3. Key feature importances
  4. Predictability of each G dim from pre-game features (R²)

Usage:
  python scripts/eval_g_variants.py [--variant v1_baseline,v3_team_totals,...]
  python scripts/eval_g_variants.py  # runs all variants

Results tell us which G encoding has the most signal for player outcome prediction.
The best G encoding is one where:
  - High AUC (G alone): G contains useful information about player outcomes
  - AUC gain (combined - player-only): G adds incremental signal
  - Low pre-game R²: encoder needs to be uncertain → valid posterior variance
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.dataset import load_processed, temporal_split
from src.data.game_state import build_game_state_df, G_VARIANTS, get_variant_cols

ROOT = Path(__file__).resolve().parents[1]  # noqa


STAT_NAMES = ["points", "assists", "reboundsTotal"]
STAT_DISPLAY = ["points", "assists", "rebounds"]
H_STAT_COLS = ["h_points", "h_assists", "h_reboundsTotal"]


def get_player_feature_cols(df: pd.DataFrame) -> list:
    """Get historical feature columns (h_* prefix)."""
    return [c for c in df.columns if c.startswith("h_")]


def get_g_cols_for_player(df: pd.DataFrame, variant: str) -> list:
    """Get all G feature columns (home + away) for use in prediction."""
    team_cols = get_variant_cols(variant)
    return [f"home_{c}" for c in team_cols] + [f"away_{c}" for c in team_cols]


def evaluate_variant(
    train_merged: pd.DataFrame,
    val_merged: pd.DataFrame,
    variant: str,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a single G variant for AUC signal.

    Returns dict with per-stat results.
    """
    g_cols = get_g_cols_for_player(train_merged, variant)
    g_cols = [c for c in g_cols if c in train_merged.columns]

    player_feat_cols = get_player_feature_cols(train_merged)

    results = {}

    for stat_name, h_col, display in zip(STAT_NAMES, H_STAT_COLS, STAT_DISPLAY):
        if stat_name not in train_merged.columns or h_col not in train_merged.columns:
            continue

        # Binary: did player beat their historical average (residual > 0)?
        y_train = (train_merged[stat_name] > train_merged[h_col]).astype(int).values
        y_val = (val_merged[stat_name] > val_merged[h_col]).astype(int).values

        # -- G only --
        X_g_train = train_merged[g_cols].fillna(0).values
        X_g_val = val_merged[g_cols].fillna(0).values

        scaler_g = StandardScaler()
        X_g_train_s = scaler_g.fit_transform(X_g_train)
        X_g_val_s = scaler_g.transform(X_g_val)

        gbt_g = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        gbt_g.fit(X_g_train_s, y_train)
        auc_g = roc_auc_score(y_val, gbt_g.predict_proba(X_g_val_s)[:, 1])

        # -- Player only --
        X_p_train = train_merged[player_feat_cols].fillna(0).values
        X_p_val = val_merged[player_feat_cols].fillna(0).values

        scaler_p = StandardScaler()
        X_p_train_s = scaler_p.fit_transform(X_p_train)
        X_p_val_s = scaler_p.transform(X_p_val)

        gbt_p = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        gbt_p.fit(X_p_train_s, y_train)
        auc_p = roc_auc_score(y_val, gbt_p.predict_proba(X_p_val_s)[:, 1])

        # -- Combined --
        X_c_train = np.hstack([X_g_train_s, X_p_train_s])
        X_c_val = np.hstack([X_g_val_s, X_p_val_s])

        gbt_c = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        gbt_c.fit(X_c_train, y_train)
        auc_c = roc_auc_score(y_val, gbt_c.predict_proba(X_c_val)[:, 1])

        # Feature importance for G features
        g_importances = dict(zip(g_cols, gbt_g.feature_importances_))
        top5_g = sorted(g_importances.items(), key=lambda x: -x[1])[:5]

        results[display] = {
            "auc_g_only": auc_g,
            "auc_player_only": auc_p,
            "auc_combined": auc_c,
            "auc_g_adds": auc_c - auc_p,
            "top_g_features": top5_g,
        }

        if verbose:
            print(f"  {display:10s}: G={auc_g:.4f}  Player={auc_p:.4f}  "
                  f"Combined={auc_c:.4f}  (+{auc_c - auc_p:+.4f})")

    return results


def evaluate_g_predictability(
    train_merged: pd.DataFrame,
    val_merged: pd.DataFrame,
    variant: str,
) -> dict:
    """
    For each G dimension, measure how well pre-game player features predict it.

    R² close to 0 → encoder must be uncertain → valid correlation mechanism.
    R² close to 1 → G is fully predictable from pre-game → no uncertainty for encoder.
    """
    team_cols = get_variant_cols(variant)
    player_feat_cols = get_player_feature_cols(train_merged)

    X_train = train_merged[player_feat_cols].fillna(0).values
    X_val = val_merged[player_feat_cols].fillna(0).values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    r2_results = {}
    for prefix in ["home_", "away_"]:
        for col in team_cols:
            full_col = f"{prefix}{col}"
            if full_col not in train_merged.columns:
                continue

            y_tr = train_merged[full_col].fillna(train_merged[full_col].median()).values
            y_va = val_merged[full_col].fillna(val_merged[full_col].median()).values

            # Use GBT for R²
            gbt = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
            gbt.fit(X_train_s, y_tr)
            y_pred = gbt.predict(X_val_s)

            ss_res = ((y_pred - y_va) ** 2).sum()
            ss_tot = ((y_va - y_va.mean()) ** 2).sum() + 1e-8
            r2 = float(1 - ss_res / ss_tot)
            r2_results[full_col] = r2

    return r2_results


def run_all_variants(variants: list, season: str = "2019-26") -> dict:
    print(f"Loading processed data (season={season})...")
    df = load_processed(season_suffix=season)
    train_df, val_df = temporal_split(df)
    print(f"  Train: {len(train_df)} rows  Val: {len(val_df)} rows")

    # Build target columns from residuals
    # TARGET_COLS from dataset.py correspond to points, assists, rebounds
    all_results = {}

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Variant: {variant}  ({len(get_variant_cols(variant))*2} features)")
        print(f"{'='*60}")

        # Load game state for this variant
        needs_entropy = any("entropy" in c or "conc" in c for c in get_variant_cols(variant))
        try:
            game_gs = build_game_state_df(
                variant=variant,
                include_entropy=needs_entropy,
                use_cache=True,
            )
        except Exception as e:
            print(f"  ERROR loading variant: {e}")
            continue

        # Merge with player data
        # Player data has: gameId, personId, h_* features, and target residuals
        # We need to reconstruct from df which has raw player rows

        # Merge player data with game state
        train_merged = train_df.merge(game_gs, on="gameId", how="inner")
        val_merged = val_df.merge(game_gs, on="gameId", how="inner")

        if len(train_merged) == 0:
            print(f"  No data after merge! Check gameId matching.")
            continue

        print(f"  Merged: train={len(train_merged):,}  val={len(val_merged):,}")

        # AUC evaluation
        print("\n  AUC results:")
        auc_results = evaluate_variant(train_merged, val_merged, variant, verbose=True)

        # G predictability
        print("\n  G predictability (pre-game → actual G), R²:")
        r2_results = evaluate_g_predictability(train_merged, val_merged, variant)
        r2_vals = list(r2_results.values())
        print(f"    Mean R²: {np.mean(r2_vals):.4f}  |  Top 3:")
        for col, r2 in sorted(r2_results.items(), key=lambda x: -x[1])[:3]:
            print(f"      {col}: R²={r2:.4f}")

        # Summary
        print("\n  SUMMARY:")
        for display, res in auc_results.items():
            print(f"    {display}: G={res['auc_g_only']:.4f}, "
                  f"+{res['auc_g_adds']:+.4f} over player-only")

        all_results[variant] = {
            "auc": auc_results,
            "r2": r2_results,
            "n_train": len(train_merged),
            "n_val": len(val_merged),
        }

    # Final comparison table
    print(f"\n\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Variant':<25} {'G_dim':>6} {'pts_G':>8} {'ast_G':>8} {'reb_G':>8} "
          f"{'pts_+':>8} {'ast_+':>8} {'reb_+':>8} {'mean_R²':>8}")
    print("-" * 80)

    for variant, res in all_results.items():
        g_dim = len(get_variant_cols(variant)) * 2
        auc = res["auc"]
        r2_mean = np.mean(list(res["r2"].values())) if res["r2"] else 0

        pts = auc.get("points", {})
        ast = auc.get("assists", {})
        reb = auc.get("reboundsTotal", {})

        print(
            f"{variant:<25} {g_dim:>6} "
            f"{pts.get('auc_g_only', 0):>8.4f} "
            f"{ast.get('auc_g_only', 0):>8.4f} "
            f"{reb.get('auc_g_only', 0):>8.4f} "
            f"{pts.get('auc_g_adds', 0):>+8.4f} "
            f"{ast.get('auc_g_adds', 0):>+8.4f} "
            f"{reb.get('auc_g_adds', 0):>+8.4f} "
            f"{r2_mean:>8.4f}"
        )

    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variants", type=str, default=",".join(G_VARIANTS.keys()),
                   help="Comma-separated list of variants to evaluate")
    p.add_argument("--season", type=str, default="2019-26")
    args = p.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    print(f"Evaluating variants: {variants}")

    results = run_all_variants(variants, season=args.season)

"""
Phase 1 Validation: Does actual game state predict player over/under outcomes?

Merges actual in-game game state features (G) with player over/under labels.
Trains logistic regression and gradient boosted trees.

Reports:
  1. AUC with game state features only (no player identity)
  2. AUC with player features only (h_stat history, current baseline)
  3. AUC with game state + player features combined
  4. Feature importances from GBT

The incremental AUC from (1) → (3) measures what z must learn to capture.
If game state alone achieves AUC > 0.55, the signal is real.

Usage:
  python scripts/validate_game_state.py [--season 2019-26] [--stat points]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.game_state import build_game_state_df, GAME_STATE_COLS, TEAM_FEATURE_COLS


PLAYER_FEATURE_COLS = [
    "h_points", "h_assists", "h_reboundsTotal",
    "h_numMinutes", "h_usage_rate", "h_usage_share",
    "h_pace", "h_off_rating", "h_def_rating", "h_implied_total",
    "days_rest", "is_b2b", "home",
]

STAT_COLS = ["points", "assists", "reboundsTotal"]
H_STAT_COLS = ["h_points", "h_assists", "h_reboundsTotal"]


def load_player_data(season: str) -> pd.DataFrame:
    inp = pd.read_csv(ROOT / "data" / "processed" / f"input_data_{season}.csv")
    tgt = pd.read_csv(ROOT / "data" / "processed" / f"target_data_{season}.csv")
    df = inp.merge(tgt, on=["personId", "gameId", "home"], how="inner")
    df = df.dropna(subset=PLAYER_FEATURE_COLS + STAT_COLS)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    return df


def temporal_split(df: pd.DataFrame, train_frac: float = 0.8):
    df = df.sort_values("gameDateTimeEst")
    games = df["gameId"].unique()
    # Sort games by first occurrence date
    game_dates = df.groupby("gameId")["gameDateTimeEst"].min().sort_values()
    cutoff_game = game_dates.index[int(len(game_dates) * train_frac)]
    cutoff_date = game_dates[cutoff_game]
    train = df[df["gameDateTimeEst"] <= cutoff_date]
    val = df[df["gameDateTimeEst"] > cutoff_date]
    return train, val


def build_features(df: pd.DataFrame, game_state: pd.DataFrame, stat: str):
    """
    Build feature matrices and labels for a given stat.

    Returns: (X_game_state, X_player, X_combined, y, player_only_baseline_auc)
    """
    h_stat = f"h_{stat}"
    merged = df.merge(game_state, on="gameId", how="inner")

    # Label: actual > h_stat (over = 1)
    y = (merged[stat] > merged[h_stat]).astype(int)

    # Feature sets
    gs_cols = [c for c in GAME_STATE_COLS if c in merged.columns and merged[c].notna().mean() > 0.5]
    pl_cols = [c for c in PLAYER_FEATURE_COLS if c in merged.columns]

    X_gs = merged[gs_cols].fillna(merged[gs_cols].median())
    X_pl = merged[pl_cols].fillna(merged[pl_cols].median())
    X_combined = pd.concat([X_gs, X_pl], axis=1)

    return X_gs, X_pl, X_combined, y, merged


def eval_model(name: str, model, X_train, y_train, X_val, y_val) -> dict:
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, prob)
    ll = log_loss(y_val, prob)
    over_rate = y_val.mean()
    pred_over_rate = (prob > 0.5).mean()
    return {
        "name": name,
        "auc": auc,
        "log_loss": ll,
        "actual_over_rate": over_rate,
        "pred_over_rate": pred_over_rate,
        "n_val": len(y_val),
    }


def run_validation(season: str = "2019-26", stats: list = None, include_pbp: bool = True):
    if stats is None:
        stats = STAT_COLS

    print(f"Loading player data ({season})...")
    df = load_player_data(season)
    print(f"  {len(df):,} player-game rows, {df['gameId'].nunique():,} games")

    print("Building game state features...")
    game_state = build_game_state_df(include_pbp=include_pbp)
    game_state_overlap = game_state[game_state["gameId"].isin(df["gameId"])]
    print(f"  Game state available for {len(game_state_overlap):,} / {df['gameId'].nunique():,} games")

    train_df, val_df = temporal_split(df)
    train_games = set(train_df["gameId"].unique())
    val_games = set(val_df["gameId"].unique())
    print(f"  Train games: {len(train_games):,}, Val games: {len(val_games):,}")

    all_results = []
    feature_importance_results = {}

    for stat in stats:
        print(f"\n{'='*60}")
        print(f"Stat: {stat}")
        print('='*60)

        # Build features for train and val sets separately
        X_gs_tr, X_pl_tr, X_comb_tr, y_tr, merged_tr = build_features(train_df, game_state, stat)
        X_gs_val, X_pl_val, X_comb_val, y_val, merged_val = build_features(val_df, game_state, stat)

        # Filter to rows with game state available
        gs_tr_mask = merged_tr["gameId"].isin(game_state_overlap["gameId"])
        gs_val_mask = merged_val["gameId"].isin(game_state_overlap["gameId"])

        print(f"  Rows in train (with GS): {gs_tr_mask.sum():,}, val: {gs_val_mask.sum():,}")
        print(f"  Over rate — train: {y_tr[gs_tr_mask].mean():.3f}, val: {y_val[gs_val_mask].mean():.3f}")

        results = []

        # --- Baseline: always predict over_rate ---
        baseline_prob = np.full(gs_val_mask.sum(), y_tr[gs_tr_mask].mean())
        baseline_auc = roc_auc_score(y_val[gs_val_mask], baseline_prob) if y_val[gs_val_mask].nunique() > 1 else 0.5
        print(f"\n  Baseline (constant): AUC={baseline_auc:.4f}")

        # --- Model 1: Player features only (LR) ---
        lr_player = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ])
        r = eval_model(
            f"{stat}_player_only_LR",
            lr_player,
            X_pl_tr[gs_tr_mask], y_tr[gs_tr_mask],
            X_pl_val[gs_val_mask], y_val[gs_val_mask],
        )
        results.append(r)
        print(f"  Player-only LR:          AUC={r['auc']:.4f}, LL={r['log_loss']:.4f}")

        # --- Model 2: Game state only (LR) ---
        lr_gs = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ])
        r = eval_model(
            f"{stat}_game_state_only_LR",
            lr_gs,
            X_gs_tr[gs_tr_mask], y_tr[gs_tr_mask],
            X_gs_val[gs_val_mask], y_val[gs_val_mask],
        )
        results.append(r)
        print(f"  Game-state-only LR:      AUC={r['auc']:.4f}, LL={r['log_loss']:.4f}")

        # --- Model 3: Combined LR ---
        lr_combined = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ])
        r = eval_model(
            f"{stat}_combined_LR",
            lr_combined,
            X_comb_tr[gs_tr_mask], y_tr[gs_tr_mask],
            X_comb_val[gs_val_mask], y_val[gs_val_mask],
        )
        results.append(r)
        print(f"  Combined LR:             AUC={r['auc']:.4f}, LL={r['log_loss']:.4f}")

        # --- Model 4: Player features only (GBT) ---
        gbt_player = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        r = eval_model(
            f"{stat}_player_only_GBT",
            gbt_player,
            X_pl_tr[gs_tr_mask], y_tr[gs_tr_mask],
            X_pl_val[gs_val_mask], y_val[gs_val_mask],
        )
        results.append(r)
        print(f"  Player-only GBT:         AUC={r['auc']:.4f}, LL={r['log_loss']:.4f}")

        # --- Model 5: Game state only (GBT) ---
        gbt_gs = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        r = eval_model(
            f"{stat}_game_state_only_GBT",
            gbt_gs,
            X_gs_tr[gs_tr_mask], y_tr[gs_tr_mask],
            X_gs_val[gs_val_mask], y_val[gs_val_mask],
        )
        results.append(r)
        print(f"  Game-state-only GBT:     AUC={r['auc']:.4f}, LL={r['log_loss']:.4f}")

        # --- Model 6: Combined GBT (with feature importances) ---
        gbt_combined = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        r = eval_model(
            f"{stat}_combined_GBT",
            gbt_combined,
            X_comb_tr[gs_tr_mask], y_tr[gs_tr_mask],
            X_comb_val[gs_val_mask], y_val[gs_val_mask],
        )
        results.append(r)
        print(f"  Combined GBT:            AUC={r['auc']:.4f}, LL={r['log_loss']:.4f}")

        # Feature importances from combined GBT
        feat_names = list(X_comb_tr.columns)
        importances = gbt_combined.named_steps["GBT"] if hasattr(gbt_combined, 'named_steps') else gbt_combined
        imps = pd.Series(
            importances.feature_importances_,
            index=feat_names,
        ).sort_values(ascending=False)
        feature_importance_results[stat] = imps

        print(f"\n  Top 15 feature importances ({stat}):")
        for fname, fimp in imps.head(15).items():
            marker = " *** GAME STATE" if any(fname.startswith(p) for p in ["home_actual", "away_actual"]) else ""
            print(f"    {fname:40s}  {fimp:.4f}{marker}")

        # Incremental AUC analysis
        player_auc = next(r["auc"] for r in results if "player_only_GBT" in r["name"])
        combined_auc = next(r["auc"] for r in results if "combined_GBT" in r["name"])
        gs_only_auc = next(r["auc"] for r in results if "game_state_only_GBT" in r["name"])
        print(f"\n  === SUMMARY for {stat} ===")
        print(f"  Game state only AUC:  {gs_only_auc:.4f}  (target > 0.55)")
        print(f"  Player only AUC:      {player_auc:.4f}")
        print(f"  Combined AUC:         {combined_auc:.4f}")
        print(f"  Incremental delta:    {combined_auc - player_auc:+.4f}  (game state contribution)")

        all_results.extend(results)

    # Summary table
    print(f"\n{'='*60}")
    print("FULL RESULTS TABLE")
    print('='*60)
    results_df = pd.DataFrame(all_results)
    print(results_df[["name", "auc", "log_loss", "actual_over_rate", "n_val"]].to_string(index=False))

    # Save feature importances
    for stat, imps in feature_importance_results.items():
        out_path = ROOT / "experiments" / f"gs_feature_importance_{stat}.csv"
        imps.to_csv(out_path, header=["importance"])
        print(f"\nSaved feature importances to {out_path}")

    return results_df, feature_importance_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="2019-26")
    parser.add_argument("--stats", nargs="+", default=None)
    parser.add_argument("--no-pbp", action="store_true")
    args = parser.parse_args()

    stats = args.stats or STAT_COLS
    run_validation(season=args.season, stats=stats, include_pbp=not args.no_pbp)

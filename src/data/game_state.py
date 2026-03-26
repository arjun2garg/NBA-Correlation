"""
Compute actual in-game game state features from TeamStatistics and PlayByPlay.

These are POST-GAME features (no leakage concern for Phase 1 validation).
They serve as the "teacher" signal G that the VAE encoder must learn to predict.

Per-team features (both home and away):
  TeamStatistics-based (always available):
    actual_pace        — possessions per 48 min
    actual_3pa_rate    — 3PA / FGA
    actual_ast_rate    — assists / FGM
    actual_to_rate     — turnovers / possessions
    actual_margin      — teamScore - opponentScore
    actual_oreb_rate   — ORB / (FGA - FGM)

  PBP-based (requires PlayByPlay.parquet, ~2019+ coverage):
    actual_rim_rate        — FGA at rim (area='Restricted Area' or dist <= 4) / FGA
    actual_midrange_rate   — midrange FGA / FGA
    actual_assisted_rate   — assisted FGM / FGM

Output: one row per team per game, with gameId, teamId, home, and all feature columns.
Call build_game_state_df() to get the concatenated home+away game-level vector.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CACHE_PATH = ROOT / "data" / "processed" / "game_state_cache.csv"

# Features always available from TeamStatistics
TS_FEATURE_COLS = [
    "actual_pace",
    "actual_3pa_rate",
    "actual_ast_rate",
    "actual_to_rate",
    "actual_margin",
    "actual_oreb_rate",
]

# Features from PBP (may be NaN if PBP unavailable for a game)
PBP_FEATURE_COLS = [
    "actual_rim_rate",
    "actual_midrange_rate",
    "actual_assisted_rate",
]

# All per-team features
TEAM_FEATURE_COLS = TS_FEATURE_COLS + PBP_FEATURE_COLS

# Game-level vector: home_* + away_* concatenated
GAME_STATE_COLS = (
    [f"home_{c}" for c in TEAM_FEATURE_COLS]
    + [f"away_{c}" for c in TEAM_FEATURE_COLS]
)


def _compute_ts_features(ts: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team-game features from TeamStatistics rows."""
    ts = ts.copy()
    ts = ts.dropna(subset=["teamScore", "numMinutes", "fieldGoalsMade", "fieldGoalsAttempted"])

    # Possession estimate (Dean Oliver formula)
    ts["poss"] = (
        ts["fieldGoalsAttempted"]
        + 0.44 * ts["freeThrowsAttempted"].fillna(0)
        + ts["turnovers"].fillna(0)
        - ts["reboundsOffensive"].fillna(0)
    ).clip(lower=1)

    # numMinutes = 5 * game_minutes (e.g. 240 for a 48-min regulation game)
    ts["game_min"] = (ts["numMinutes"] / 5.0).clip(lower=40)

    ts["actual_pace"] = (ts["poss"] * 48.0 / ts["game_min"]).clip(60, 160)
    ts["actual_3pa_rate"] = ts["threePointersAttempted"] / (ts["fieldGoalsAttempted"] + 1e-6)
    ts["actual_ast_rate"] = ts["assists"] / (ts["fieldGoalsMade"] + 1e-6)
    ts["actual_to_rate"] = ts["turnovers"] / ts["poss"]
    ts["actual_margin"] = ts["teamScore"] - ts["opponentScore"]
    ts["actual_oreb_rate"] = ts["reboundsOffensive"] / (
        (ts["fieldGoalsAttempted"] - ts["fieldGoalsMade"]).clip(lower=1)
    )

    keep = ["gameId", "teamId", "home"] + TS_FEATURE_COLS
    return ts[keep].copy()


def _compute_pbp_features(pbp_path: Path) -> pd.DataFrame:
    """
    Compute per-team-game shot composition features from PlayByPlay.

    Returns DataFrame with: gameId, teamId, actual_rim_rate, actual_midrange_rate,
    actual_assisted_rate.
    """
    print("Loading PlayByPlay for shot features...")
    pbp = pd.read_parquet(pbp_path, columns=[
        "gameId", "teamId", "isFieldGoal", "shotResult",
        "shotDistance", "area", "assistPersonId",
    ])
    print(f"  Loaded {len(pbp):,} rows")

    # Field goal attempts only
    shots = pbp[pbp["isFieldGoal"] == True].copy()
    shots["gameId"] = pd.to_numeric(shots["gameId"], errors="coerce")
    shots["teamId"] = pd.to_numeric(shots["teamId"], errors="coerce")
    shots["assistPersonId"] = pd.to_numeric(shots["assistPersonId"], errors="coerce")
    shots = shots.dropna(subset=["gameId", "teamId"])

    shots["is_made"] = shots["shotResult"] == "Made"
    shots["is_assisted"] = shots["assistPersonId"].notnull()

    # Rim: Restricted Area or distance ≤ 4 ft
    shots["is_rim"] = (shots["area"] == "Restricted Area") | (shots["shotDistance"] <= 4)
    # Midrange: explicit area label or 10-22 ft non-rim
    shots["is_midrange"] = shots["area"] == "Mid-Range"

    agg = shots.groupby(["gameId", "teamId"]).agg(
        total_fga=("isFieldGoal", "count"),
        total_fgm=("is_made", "sum"),
        rim_fga=("is_rim", "sum"),
        midrange_fga=("is_midrange", "sum"),
    ).reset_index()

    # Assisted FGM: join on made shots where assistPersonId is not null
    made = shots[shots["is_made"]].copy()
    ast_counts = (
        made.groupby(["gameId", "teamId"])["is_assisted"]
        .sum()
        .reset_index(name="assisted_fgm")
    )
    agg = agg.merge(ast_counts, on=["gameId", "teamId"], how="left")
    agg["assisted_fgm"] = agg["assisted_fgm"].fillna(0)

    agg["actual_rim_rate"] = agg["rim_fga"] / (agg["total_fga"] + 1e-6)
    agg["actual_midrange_rate"] = agg["midrange_fga"] / (agg["total_fga"] + 1e-6)
    agg["actual_assisted_rate"] = agg["assisted_fgm"] / (agg["total_fgm"] + 1e-6)

    return agg[["gameId", "teamId"] + PBP_FEATURE_COLS].copy()


def load_team_game_state(
    raw_dir: Path = RAW_DIR,
    include_pbp: bool = True,
) -> pd.DataFrame:
    """
    Load per-team-game state features.

    Returns one row per (gameId, teamId) with all TEAM_FEATURE_COLS.
    PBP features are filled with NaN if include_pbp=False or PBP unavailable.
    """
    ts = pd.read_csv(raw_dir / "TeamStatistics.csv")
    features = _compute_ts_features(ts)

    if include_pbp:
        pbp_path = raw_dir / "PlayByPlay.parquet"
        if pbp_path.exists():
            pbp_feats = _compute_pbp_features(pbp_path)
            features = features.merge(pbp_feats, on=["gameId", "teamId"], how="left")
        else:
            print("Warning: PlayByPlay.parquet not found, skipping PBP features")
            for col in PBP_FEATURE_COLS:
                features[col] = np.nan
    else:
        for col in PBP_FEATURE_COLS:
            features[col] = np.nan

    return features


def build_game_state_df(
    raw_dir: Path = RAW_DIR,
    include_pbp: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build game-level state vectors with home_ and away_ prefixed features.

    Returns one row per game with columns: gameId + GAME_STATE_COLS.
    Caches the TS-only result to data/processed/game_state_cache.csv for fast reloads.
    """
    if use_cache and not include_pbp and CACHE_PATH.exists():
        gs = pd.read_csv(CACHE_PATH)
        print(f"Loaded game state from cache: {len(gs):,} games")
        return gs

    team_gs = load_team_game_state(raw_dir=raw_dir, include_pbp=include_pbp)

    home = team_gs[team_gs["home"] == 1][["gameId"] + TEAM_FEATURE_COLS].copy()
    away = team_gs[team_gs["home"] == 0][["gameId"] + TEAM_FEATURE_COLS].copy()

    home = home.rename(columns={c: f"home_{c}" for c in TEAM_FEATURE_COLS})
    away = away.rename(columns={c: f"away_{c}" for c in TEAM_FEATURE_COLS})

    # Some games have duplicate rows — keep first
    home = home.drop_duplicates(subset=["gameId"])
    away = away.drop_duplicates(subset=["gameId"])

    game_gs = home.merge(away, on="gameId", how="inner")
    print(f"Game state vectors: {len(game_gs):,} games, {len(GAME_STATE_COLS)} features each")

    if use_cache and not include_pbp:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        game_gs.to_csv(CACHE_PATH, index=False)
        print(f"Cached game state to {CACHE_PATH}")

    return game_gs


if __name__ == "__main__":
    gs = build_game_state_df(include_pbp=True)
    print(gs[GAME_STATE_COLS].describe().round(3))

"""
Track B: Build per-game outcome vectors for two-stage VAE.

Outcome vector (6-dim): [home_score, away_score, home_pace, away_pace, home_poss, away_poss]

Sources:
  - data/raw/TeamStatistics.csv  → home_score, away_score
  - data/raw/PlayerStatisticsAdvanced.csv → pace, poss per team per game
    (take first player row per team-game as the team estimate)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def load_game_outcomes(raw_dir=RAW_DIR):
    """
    Returns a DataFrame with columns:
      gameId, home_score, away_score, home_pace, away_pace, home_poss, away_poss
    One row per game.
    """
    # --- Scores from TeamStatistics ---
    ts = pd.read_csv(raw_dir / "TeamStatistics.csv",
                     usecols=["gameId", "home", "teamScore", "opponentScore"])
    home_ts = ts[ts["home"] == 1][["gameId", "teamScore", "opponentScore"]].rename(
        columns={"teamScore": "home_score", "opponentScore": "away_score"}
    )
    # drop duplicates (some games appear twice if home/away both listed)
    home_ts = home_ts.drop_duplicates(subset=["gameId"])

    # --- Pace & possessions from PlayerStatisticsAdvanced ---
    adv_cols = ["gameId", "teamId", "home", "pace", "poss"]
    adv = pd.read_csv(raw_dir / "PlayerStatisticsAdvanced.csv", usecols=adv_cols)
    adv = adv.dropna(subset=["pace", "poss"])

    # Take first row per (gameId, teamId) as team estimate
    adv_team = adv.groupby(["gameId", "teamId", "home"], as_index=False).first()

    home_adv = adv_team[adv_team["home"] == 1][["gameId", "pace", "poss"]].rename(
        columns={"pace": "home_pace", "poss": "home_poss"}
    ).drop_duplicates(subset=["gameId"])

    away_adv = adv_team[adv_team["home"] == 0][["gameId", "pace", "poss"]].rename(
        columns={"pace": "away_pace", "poss": "away_poss"}
    ).drop_duplicates(subset=["gameId"])

    # Merge everything
    outcomes = home_ts.merge(home_adv, on="gameId", how="inner")
    outcomes = outcomes.merge(away_adv, on="gameId", how="inner")
    outcomes = outcomes.dropna()

    print(f"Game outcomes loaded: {len(outcomes):,} games, cols={list(outcomes.columns)}")
    return outcomes


OUTCOME_COLS = ["home_score", "away_score", "home_pace", "away_pace", "home_poss", "away_poss"]


def normalize_outcomes(outcomes_df, train_game_ids=None):
    """
    Normalize outcome columns using train-set stats.

    Returns (normalized_df, stats_dict) where stats_dict has mean/std for each col.
    If train_game_ids is None, uses all rows.
    """
    if train_game_ids is not None:
        train = outcomes_df[outcomes_df["gameId"].isin(train_game_ids)]
    else:
        train = outcomes_df

    stats = {}
    for col in OUTCOME_COLS:
        mu = float(train[col].mean())
        sd = float(train[col].std()) + 1e-6
        stats[col] = (mu, sd)

    normed = outcomes_df.copy()
    for col in OUTCOME_COLS:
        mu, sd = stats[col]
        normed[col] = (normed[col] - mu) / sd

    return normed, stats


if __name__ == "__main__":
    outcomes = load_game_outcomes()
    print(outcomes.describe())

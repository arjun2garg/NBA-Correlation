"""
Track B: Build per-game outcome vectors for two-stage VAE.

Outcome vector (6-dim): [home_score, away_score, home_pace, away_pace, home_poss, away_poss]

Source: data/raw/TeamStatistics.csv only — same gameId format as input_data.
Pace and possessions are estimated from box score stats in TeamStatistics.

Possession estimate: poss ≈ FGA + 0.44*FTA + TOV  (Dean Oliver formula, per team)
Pace = mean_possessions_per_team * 48 / numMinutes
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
    cols = [
        "gameId", "home", "teamScore", "opponentScore",
        "fieldGoalsAttempted", "freeThrowsAttempted", "turnovers", "numMinutes",
    ]
    ts = pd.read_csv(raw_dir / "TeamStatistics.csv", usecols=cols)
    ts = ts.dropna(subset=["teamScore", "numMinutes"])

    # Estimate possessions: FGA + 0.44*FTA + TOV (simple estimate without ORB)
    ts["poss_est"] = (
        ts["fieldGoalsAttempted"]
        + 0.44 * ts["freeThrowsAttempted"]
        + ts["turnovers"]
    )
    # numMinutes is total team minutes (5 players × minutes); use it for pace
    # pace = poss_per_48_per_team = poss / (numMinutes / (5 * 48)) * 48
    # Simplify: pace = poss * 240 / numMinutes (if numMinutes is team total = 5×game_minutes)
    # But TeamStatistics numMinutes = actual game minutes played (usually 240 for a 48-min game)
    ts["pace_est"] = ts["poss_est"] * 48.0 / (ts["numMinutes"] / 5.0 + 1e-6)

    home = ts[ts["home"] == 1][["gameId", "teamScore", "opponentScore", "pace_est", "poss_est"]].rename(
        columns={
            "teamScore": "home_score",
            "opponentScore": "away_score",
            "pace_est": "home_pace",
            "poss_est": "home_poss",
        }
    ).drop_duplicates(subset=["gameId"])

    away = ts[ts["home"] == 0][["gameId", "pace_est", "poss_est"]].rename(
        columns={"pace_est": "away_pace", "poss_est": "away_poss"}
    ).drop_duplicates(subset=["gameId"])

    outcomes = home.merge(away, on="gameId", how="inner")
    outcomes = outcomes.dropna()
    outcomes = outcomes[["gameId"] + OUTCOME_COLS]

    print(f"Game outcomes loaded: {len(outcomes):,} games, cols={list(outcomes.columns)}")
    return outcomes


OUTCOME_COLS = ["home_score", "away_score", "home_pace", "away_pace", "home_poss", "away_poss"]


def normalize_outcomes(outcomes_df, train_game_ids=None):
    """
    Normalize outcome columns using train-set stats.

    Returns (normalized_df, stats_dict) where stats_dict has mean/std for each col.
    If train_game_ids is None or no overlap found, falls back to using all rows.
    """
    if train_game_ids is not None:
        train = outcomes_df[outcomes_df["gameId"].isin(train_game_ids)]
        if len(train) == 0:
            # Fallback: no overlap between train_game_ids and outcomes — use all
            train = outcomes_df
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

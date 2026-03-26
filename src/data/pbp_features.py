"""
Track A: Extract rolling assist features from PlayByPlay.parquet.

Outputs data/processed/pbp_features.csv with per-player-game:
  - rolling_ast_given_rate: rolling 20-game (assists_given / team_fgm)
  - rolling_ast_received_rate: rolling 20-game (times_assisted / own_fgm)

Uses prior-game rolling (shift(1)) to avoid leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PBP_PATH = ROOT / "data" / "raw" / "PlayByPlay.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "pbp_features.csv"


def extract_pbp_features(pbp_path=PBP_PATH, output_path=OUTPUT_PATH, window=20):
    print("Loading PlayByPlay.parquet (this may take a few minutes)...")
    df = pd.read_parquet(pbp_path, columns=[
        "gameId", "gameDateTimeEst", "personId", "teamId",
        "isFieldGoal", "shotResult", "assistPersonId",
    ])
    print(f"  Loaded {len(df):,} rows")

    # Made field goals only
    made = df[(df["isFieldGoal"] == True) & (df["shotResult"] == "Made")].copy()
    print(f"  Made FGs: {len(made):,}")

    # Convert IDs to int64 (they come as strings from PBP parquet)
    made["personId"] = pd.to_numeric(made["personId"], errors="coerce").astype("Int64")
    made["teamId"] = pd.to_numeric(made["teamId"], errors="coerce").astype("Int64")
    made["assistPersonId"] = pd.to_numeric(made["assistPersonId"], errors="coerce").astype("Int64")
    made["gameId"] = pd.to_numeric(made["gameId"], errors="coerce").astype("Int64")
    made = made.dropna(subset=["personId", "teamId", "gameId"])
    made["gameDateTimeEst"] = pd.to_datetime(made["gameDateTimeEst"], utc=True).dt.tz_convert(None)

    # Per-game team FGM
    team_fgm = (
        made.groupby(["gameId", "teamId"])
        .size()
        .reset_index(name="team_fgm")
    )

    # Per-player-game FGM (scorer)
    player_fgm = (
        made.groupby(["gameId", "personId", "teamId", "gameDateTimeEst"])
        .size()
        .reset_index(name="fgm_cnt")
    )
    # Take first teamId per (gameId, personId) in case of rare multi-team edge cases
    player_fgm = player_fgm.sort_values("fgm_cnt", ascending=False)
    player_fgm = player_fgm.groupby(["gameId", "personId"], as_index=False).first()

    # Per-player-game assists given (player appears as assistPersonId)
    has_assist = made.dropna(subset=["assistPersonId"])
    ast_given = (
        has_assist.groupby(["gameId", "assistPersonId"])
        .size()
        .reset_index(name="ast_given_cnt")
        .rename(columns={"assistPersonId": "personId"})
    )

    # Per-player-game assists received (scorer was assisted)
    ast_received = (
        has_assist.groupby(["gameId", "personId"])
        .size()
        .reset_index(name="ast_received_cnt")
    )

    # Also need player-game base for players who assisted but didn't score
    # Use all unique (personId, gameId) from made FGs (either as scorer or assister)
    all_player_games = pd.concat([
        player_fgm[["gameId", "personId", "teamId", "gameDateTimeEst"]],
        has_assist[["gameId", "assistPersonId", "teamId", "gameDateTimeEst"]]
        .rename(columns={"assistPersonId": "personId"}),
    ]).drop_duplicates(subset=["gameId", "personId"])

    # Build features table
    features = all_player_games.copy()
    features = features.merge(player_fgm[["gameId", "personId", "fgm_cnt"]], on=["gameId", "personId"], how="left")
    features = features.merge(team_fgm, on=["gameId", "teamId"], how="left")
    features = features.merge(ast_given, on=["gameId", "personId"], how="left")
    features = features.merge(ast_received, on=["gameId", "personId"], how="left")

    features["fgm_cnt"] = features["fgm_cnt"].fillna(0).astype(float)
    features["team_fgm"] = features["team_fgm"].fillna(1).astype(float)
    features["ast_given_cnt"] = features["ast_given_cnt"].fillna(0).astype(float)
    features["ast_received_cnt"] = features["ast_received_cnt"].fillna(0).astype(float)

    # Raw per-game rates
    features["ast_given_raw"] = features["ast_given_cnt"] / features["team_fgm"]
    features["ast_received_raw"] = np.where(
        features["fgm_cnt"] > 0,
        features["ast_received_cnt"] / features["fgm_cnt"],
        0.0,
    )

    # Sort per player by date for rolling
    features = features.sort_values(["personId", "gameDateTimeEst"]).reset_index(drop=True)

    def _rolling_prior(group, col, w):
        """Rolling mean over prior w games (shift(1) to exclude current game)."""
        return group[col].shift(1).rolling(w, min_periods=1).mean()

    print("Computing rolling assist rates per player...")
    features["rolling_ast_given_rate"] = (
        features.groupby("personId", group_keys=False)
        .apply(lambda g: _rolling_prior(g, "ast_given_raw", window))
    )
    features["rolling_ast_received_rate"] = (
        features.groupby("personId", group_keys=False)
        .apply(lambda g: _rolling_prior(g, "ast_received_raw", window))
    )
    features[["rolling_ast_given_rate", "rolling_ast_received_rate"]] = (
        features[["rolling_ast_given_rate", "rolling_ast_received_rate"]].fillna(0)
    )

    # Output: (personId as int64, gameId as int64, rolling features)
    out = features[["personId", "gameId", "rolling_ast_given_rate", "rolling_ast_received_rate"]].copy()
    out["personId"] = pd.to_numeric(out["personId"], errors="coerce").astype("Int64")
    out["gameId"] = pd.to_numeric(out["gameId"], errors="coerce").astype("Int64")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out):,} rows to {output_path}")
    return out


if __name__ == "__main__":
    extract_pbp_features()

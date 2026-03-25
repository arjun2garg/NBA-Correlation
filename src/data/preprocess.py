import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Three regular seasons: 2022-23, 2023-24, 2024-25
SEASON_START = "2022-10-01"
SEASON_END = "2025-04-14"

STATS_TO_DECAY = [
    "points", "assists", "blocks", "steals",
    "fieldGoalsAttempted", "fieldGoalsMade",
    "threePointersAttempted", "threePointersMade",
    "freeThrowsAttempted", "freeThrowsMade",
    "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
    "foulsPersonal", "turnovers", "numMinutes",
]

TARGET_COLS = ["personId", "gameId", "home", "points", "assists", "reboundsTotal"]
INPUT_COLS = [
    "personId", "gameId", "home", "gameDateTimeEst",
    "h_points", "h_assists", "h_blocks", "h_steals",
    "h_fieldGoalsAttempted", "h_fieldGoalsMade",
    "h_threePointersAttempted", "h_threePointersMade",
    "h_freeThrowsAttempted", "h_freeThrowsMade",
    "h_reboundsDefensive", "h_reboundsOffensive", "h_reboundsTotal",
    "h_foulsPersonal", "h_turnovers", "h_numMinutes",
]


def exp_time_decay_feature(df, stat_col, time_col="gameDateTimeEst", player_col="personId", beta=0.99):
    """Compute exponentially time-decayed historical average of stat_col per player.

    For each game, the feature is a weighted average of all prior games,
    with weights = beta ^ days_ago. First game per player is NaN.
    """
    out = []
    for _, g in df.groupby(player_col, sort=False):
        times = g[time_col].values
        values = g[stat_col].values
        hist_vals = []
        for i in range(len(g)):
            if i == 0:
                hist_vals.append(np.nan)
                continue
            days_ago = (times[i] - times[:i]) / np.timedelta64(1, "D")
            weights = beta ** days_ago
            hist_vals.append(np.sum(values[:i] * weights) / np.sum(weights))
        out.extend(hist_vals)
    return out


def load_player_stats(path=None):
    path = path or RAW_DIR / "PlayerStatistics.csv"
    stats = pd.read_csv(path, low_memory=False)
    mask = (
        (stats["gameDateTimeEst"] >= SEASON_START) &
        (stats["gameDateTimeEst"] <= SEASON_END) &
        (stats["gameType"] == "Regular Season")
    )
    stats = stats[mask]
    stats = stats[stats["numMinutes"].notna()]
    stats = stats.drop(columns=["gameLabel", "gameSubLabel", "seriesGameNumber"], errors="ignore")
    stats["gameDateTimeEst"] = pd.to_datetime(stats["gameDateTimeEst"], utc=True, errors="coerce")
    return stats


def add_decay_features(df, beta=0.99):
    df = df.sort_values(["personId", "gameDateTimeEst"])
    for stat in STATS_TO_DECAY:
        df["h_" + stat] = exp_time_decay_feature(df, stat_col=stat, beta=beta)
    return df


def build_input_target(df):
    target = df[TARGET_COLS]
    inputs = df[INPUT_COLS]
    return inputs, target


def run(raw_path=None, out_dir=None, season_suffix="2022-25"):
    out_dir = Path(out_dir) if out_dir else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading player stats...")
    df = load_player_stats(raw_path)

    print(f"Building decay features for {len(df)} rows...")
    df = add_decay_features(df)

    inputs, target = build_input_target(df)

    input_path = out_dir / f"input_data_{season_suffix}.csv"
    target_path = out_dir / f"target_data_{season_suffix}.csv"
    inputs.to_csv(input_path, index=False)
    target.to_csv(target_path, index=False)
    print(f"Saved input  → {input_path}")
    print(f"Saved target → {target_path}")

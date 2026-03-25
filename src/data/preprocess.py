import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Three regular seasons: 2022-23, 2023-24, 2024-25
SEASON_START = "2022-10-01"
SEASON_END = "2025-04-14"

# (raw_column, h_column_name, beta)
# h_blocks removed — adds no incremental signal once opponent data is present
STATS_TO_DECAY = [
    ("points",                  "h_points",                 0.99),
    ("assists",                 "h_assists",                0.99),
    ("steals",                  "h_steals",                 0.995),
    ("fieldGoalsAttempted",     "h_fieldGoalsAttempted",    0.99),
    ("fieldGoalsMade",          "h_fieldGoalsMade",         0.995),
    ("threePointersAttempted",  "h_threePointersAttempted", 0.98),
    ("threePointersMade",       "h_threePointersMade",      0.995),
    ("freeThrowsAttempted",     "h_freeThrowsAttempted",    0.99),
    ("freeThrowsMade",          "h_freeThrowsMade",         0.99),
    ("reboundsDefensive",       "h_reboundsDefensive",      0.99),
    ("reboundsOffensive",       "h_reboundsOffensive",      0.99),
    ("reboundsTotal",           "h_reboundsTotal",          0.99),
    ("foulsPersonal",           "h_foulsPersonal",          0.99),
    ("turnovers",               "h_turnovers",              0.99),
    ("numMinutes",              "h_numMinutes",             0.97),
    # Advanced stats from PlayerStatisticsAdvanced.csv / TeamStatisticsAdvanced.csv
    ("usage_rate",              "h_usage_rate",             0.97),
    ("usage_share",             "h_usage_share",            0.97),
    ("team_pace",               "h_pace",                   0.98),
    ("team_off_rating",         "h_off_rating",             0.99),
    ("team_def_rating",         "h_def_rating",             0.99),
    ("implied_total",           "h_implied_total",          0.99),
]

TARGET_COLS = ["personId", "gameId", "home", "points", "assists", "reboundsTotal"]
INPUT_COLS = [
    "personId", "gameId", "home", "gameDateTimeEst",
    # Decay features
    "h_points", "h_assists", "h_steals",
    "h_fieldGoalsAttempted", "h_fieldGoalsMade",
    "h_threePointersAttempted", "h_threePointersMade",
    "h_freeThrowsAttempted", "h_freeThrowsMade",
    "h_reboundsDefensive", "h_reboundsOffensive", "h_reboundsTotal",
    "h_foulsPersonal", "h_turnovers", "h_numMinutes",
    # Advanced decay features
    "h_usage_rate", "h_usage_share",
    "h_pace", "h_off_rating", "h_def_rating", "h_implied_total",
    # Point-in-time game features (not decayed)
    "days_rest", "is_b2b",
    # Rolling covariance features
    "cov_pts_ast", "cov_pts_reb",
]


def exp_time_decay_feature(df, stat_col, time_col="gameDateTimeEst", player_col="personId", beta=0.99):
    """Compute exponentially time-decayed historical average of stat_col per player.

    For each game, the feature is a weighted average of all prior games,
    with weights = beta ^ days_ago. First game per player is NaN.
    NaN values in stat_col are excluded from the weighted average.
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
            prior_vals = values[:i]
            valid = ~np.isnan(prior_vals.astype(float))
            if not valid.any():
                hist_vals.append(np.nan)
            else:
                hist_vals.append(
                    np.sum(prior_vals[valid].astype(float) * weights[valid])
                    / np.sum(weights[valid])
                )
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
    # Cast all numeric stat columns to float (CSV may load them as object due to blank rows)
    numeric_cols = [
        "numMinutes", "points", "assists", "blocks", "steals",
        "fieldGoalsAttempted", "fieldGoalsMade",
        "threePointersAttempted", "threePointersMade",
        "freeThrowsAttempted", "freeThrowsMade",
        "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
        "foulsPersonal", "turnovers",
    ]
    for col in numeric_cols:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce")
    return stats


def enrich_with_advanced(df):
    """Compute advanced features from box-score data and join rest-day info.

    Advanced stats (usage rate, pace, off/def rating, implied total) are derived
    directly from PlayerStatistics.csv box-score columns — no separate advanced
    CSV is needed, as those files only cover the 2025-26 season.

    Rest days and back-to-back flags are computed from TeamStatistics.csv schedule.
    """
    df = df.copy()

    # Team identifier: (gameId, home) uniquely identifies each team per game.
    # home=1 → home team, home=0 → away team.
    df["_gid"] = df["gameId"].astype(str)
    df["_home"] = df["home"].astype(int)

    # --- Compute team-level box-score totals per game ---
    poss_col = df["fieldGoalsAttempted"] + 0.44 * df["freeThrowsAttempted"] + df["turnovers"] - df["reboundsOffensive"]
    df["_poss_contrib"] = poss_col

    team_agg = df.groupby(["_gid", "_home"]).agg(
        _team_fga=("fieldGoalsAttempted", "sum"),
        _team_fta=("freeThrowsAttempted", "sum"),
        _team_tov=("turnovers", "sum"),
        _team_oreb=("reboundsOffensive", "sum"),
        _team_pts=("points", "sum"),
        _team_min=("numMinutes", "sum"),
    ).reset_index()

    # Possessions ≈ FGA + 0.44*FTA + TOV - OREB
    team_agg["_team_poss"] = (
        team_agg["_team_fga"] + 0.44 * team_agg["_team_fta"]
        + team_agg["_team_tov"] - team_agg["_team_oreb"]
    ).clip(lower=1.0)

    # Pace = possessions per 48 minutes (team_min / 5 ≈ game minutes)
    team_agg["team_pace"] = (team_agg["_team_poss"] * 48 * 5 / team_agg["_team_min"].replace(0, np.nan)).round(1)

    # Off rating = 100 * points / possessions
    team_agg["team_off_rating"] = (100 * team_agg["_team_pts"] / team_agg["_team_poss"]).round(1)

    # Merge team totals back onto players (for usage_rate and pace)
    df = df.merge(team_agg[["_gid", "_home", "_team_fga", "_team_fta", "_team_tov",
                              "_team_poss", "_team_min", "team_pace", "team_off_rating"]],
                  on=["_gid", "_home"], how="left")

    # Usage rate: 100 * (FGA + 0.44*FTA + TOV) * (team_min/5) / (min * team_total)
    player_poss = df["fieldGoalsAttempted"] + 0.44 * df["freeThrowsAttempted"] + df["turnovers"]
    team_total_poss_rate = df["_team_fga"] + 0.44 * df["_team_fta"] + df["_team_tov"]
    df["usage_rate"] = (
        100 * player_poss * (df["_team_min"] / 5)
        / (df["numMinutes"].replace(0, np.nan) * team_total_poss_rate.replace(0, np.nan))
    )

    # Usage share = player's possession rate / team total (simpler normalized version)
    df["usage_share"] = player_poss / team_total_poss_rate.replace(0, np.nan)

    # Def rating: opponent's off rating. Join opponent team agg.
    opp_agg = team_agg[["_gid", "_home", "team_off_rating"]].copy()
    opp_agg["_opp_home"] = 1 - opp_agg["_home"]  # flip home flag to get opponent
    opp_agg = opp_agg.rename(columns={"team_off_rating": "team_def_rating"})
    df = df.merge(opp_agg[["_gid", "_opp_home", "team_def_rating"]].rename(columns={"_opp_home": "_home"}),
                  on=["_gid", "_home"], how="left")

    # Implied total = own off_rating + opponent def_rating (opponent's off_rating)
    df["implied_total"] = df["team_off_rating"] + df["team_def_rating"]

    # --- Rest days from TeamStatistics.csv ---
    team_games = pd.read_csv(RAW_DIR / "TeamStatistics.csv", low_memory=False,
                              usecols=["gameId", "teamId", "gameDateTimeEst"])
    team_games = team_games[
        (team_games["gameDateTimeEst"] >= SEASON_START) &
        (team_games["gameDateTimeEst"] <= SEASON_END)
    ].copy()
    team_games["gameId"] = team_games["gameId"].astype(str)
    team_games["gameDateTimeEst"] = pd.to_datetime(team_games["gameDateTimeEst"], utc=True, errors="coerce")
    team_games = team_games.sort_values(["teamId", "gameDateTimeEst"])

    team_games["days_rest"] = team_games.groupby("teamId")["gameDateTimeEst"].diff().dt.days
    next_date = team_games.groupby("teamId")["gameDateTimeEst"].shift(-1)
    team_games["is_b2b_first"] = (next_date - team_games["gameDateTimeEst"]).dt.days == 1
    team_games["is_b2b_second"] = team_games["days_rest"] == 1
    team_games["is_b2b"] = (team_games["is_b2b_first"] | team_games["is_b2b_second"]).astype(int)
    team_games["days_rest"] = team_games["days_rest"].clip(upper=7).fillna(7)

    # Join rest days via (gameId, teamId). TeamStatistics has teamId; need to match to our df.
    # Use a separate TeamStatistics join: match gameId + home flag via PlayerteamName.
    # Simpler: join TeamStatistics onto df by gameId + playerteamName == teamName.
    ts_lookup = team_games[["gameId", "teamId", "days_rest", "is_b2b"]].copy()
    ts_lookup.columns = ["_gid", "_ts_teamId", "days_rest", "is_b2b"]

    # Build a (gameId, teamName) → teamId lookup, then join via playerteamName
    ts_full = pd.read_csv(RAW_DIR / "TeamStatistics.csv", low_memory=False,
                           usecols=["gameId", "teamId", "teamName"])
    ts_full["gameId"] = ts_full["gameId"].astype(str)
    ts_full = ts_full.rename(columns={"gameId": "_gid", "teamName": "playerteamName"})

    ts_with_rest = ts_full.merge(ts_lookup[["_gid", "_ts_teamId", "days_rest", "is_b2b"]],
                                  left_on=["_gid", "teamId"], right_on=["_gid", "_ts_teamId"], how="left")
    ts_with_rest = ts_with_rest[["_gid", "playerteamName", "days_rest", "is_b2b"]].drop_duplicates()

    df = df.merge(ts_with_rest, on=["_gid", "playerteamName"], how="left")
    df["days_rest"] = df["days_rest"].fillna(7)
    df["is_b2b"] = df["is_b2b"].fillna(0).astype(int)

    drop_cols = ["_gid", "_home", "_poss_contrib", "_team_fga", "_team_fta", "_team_tov",
                 "_team_poss", "_team_min"]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def add_decay_features(df):
    df = df.sort_values(["personId", "gameDateTimeEst"])
    for raw_col, h_col, beta in STATS_TO_DECAY:
        if raw_col not in df.columns:
            df[h_col] = np.nan
            continue
        df[h_col] = exp_time_decay_feature(df, stat_col=raw_col, beta=beta)
    return df


def compute_rolling_covariances(df, window=20, min_periods=5):
    """Compute rolling Pearson correlations per player over their last `window` games.

    Uses raw per-game stats as inputs. Returns 0 when fewer than min_periods
    games are available.
    """
    df = df.sort_values(["personId", "gameDateTimeEst"])

    def _roll_corr(g):
        pts = g["points"]
        ast = g["assists"]
        reb = g["reboundsTotal"]
        r_pa = pts.rolling(window, min_periods=min_periods).corr(ast).fillna(0.0).clip(-1.0, 1.0)
        r_pr = pts.rolling(window, min_periods=min_periods).corr(reb).fillna(0.0).clip(-1.0, 1.0)
        return pd.DataFrame({"cov_pts_ast": r_pa, "cov_pts_reb": r_pr}, index=g.index)

    corr = df.groupby("personId", group_keys=False).apply(_roll_corr)
    df["cov_pts_ast"] = corr["cov_pts_ast"]
    df["cov_pts_reb"] = corr["cov_pts_reb"]
    return df


def validate(df):
    """Assert key data quality invariants after enrichment."""
    assert df["h_usage_rate"].notna().mean() > 0.95, "h_usage_rate has too many NaN"
    assert df["h_usage_share"].notna().mean() > 0.95, "h_usage_share has too many NaN"
    assert df["h_pace"].notna().mean() > 0.95, "h_pace has too many NaN"
    assert df["h_usage_share"].between(0, 1).all(), "usage_share out of bounds [0, 1]"
    assert df["h_pace"].between(85, 130).all(), f"pace values implausible: {df['h_pace'].describe()}"
    assert df["days_rest"].between(0, 7).all(), "days_rest out of bounds [0, 7]"
    print("Validation passed.")


def build_input_target(df):
    available_cols = [c for c in INPUT_COLS if c in df.columns]
    target = df[TARGET_COLS]
    inputs = df[available_cols]
    return inputs, target


def run(raw_path=None, out_dir=None, season_suffix="2022-25", skip_validation=False):
    out_dir = Path(out_dir) if out_dir else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading player stats...")
    df = load_player_stats(raw_path)
    print(f"  {len(df)} rows loaded")

    print("Enriching with advanced stats...")
    df = enrich_with_advanced(df)
    print(f"  Enrichment complete — {df.shape[1]} columns")

    print(f"Building decay features for {len(df)} rows...")
    df = add_decay_features(df)

    print("Computing rolling covariances...")
    df = compute_rolling_covariances(df)

    df_valid = df.dropna(subset=["h_numMinutes"])

    if not skip_validation:
        validate(df_valid)

    inputs, target = build_input_target(df_valid)

    input_path = out_dir / f"input_data_{season_suffix}.csv"
    target_path = out_dir / f"target_data_{season_suffix}.csv"
    inputs.to_csv(input_path, index=False)
    target.to_csv(target_path, index=False)
    print(f"Saved input  → {input_path}  ({len(inputs)} rows, {len(inputs.columns)} cols)")
    print(f"Saved target → {target_path}")

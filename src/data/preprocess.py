import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ADV_DIR = ROOT / "data" / "advanced"

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
    # Advanced stats (requires data/advanced/ parquets from ingest/fetch_advanced_stats.py)
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
    # New advanced decay features
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
    return stats


def load_advanced_stats(adv_dir=None):
    """Load all available advanced stats parquets, concatenated across seasons."""
    d = Path(adv_dir) if adv_dir else ADV_DIR
    if not d.exists():
        return None, None, None, None

    def _load_all(pattern):
        frames = [pd.read_parquet(f) for f in sorted(d.glob(pattern))]
        return pd.concat(frames, ignore_index=True) if frames else None

    player_adv  = _load_all("boxscore_advanced_*.parquet")
    team_adv    = _load_all("team_advanced_*.parquet")
    gamelog     = _load_all("team_gamelog_*.parquet")
    home_away   = _load_all("game_home_away_*.parquet")
    return player_adv, team_adv, gamelog, home_away


def enrich_with_advanced(df, player_adv, team_adv, gamelog, home_away):
    """Join advanced stats onto player dataframe and compute derived features.

    Adds: usage_rate, usage_share, team_pace, team_off_rating, team_def_rating,
          implied_total, days_rest, is_b2b.
    """
    if home_away is None or player_adv is None:
        raise RuntimeError(
            "Advanced stats not found in data/advanced/. "
            "Run `python ingest/fetch_advanced_stats.py` first."
        )

    df = df.copy()
    df["gameId_str"] = df["gameId"].astype(str)
    df["personId_str"] = df["personId"].astype(str)

    # Derive teamId from home/away mapping
    home_away = home_away.copy()
    home_away["GAME_ID"] = home_away["GAME_ID"].astype(str)
    home_away["HOME_TEAM_ID"] = home_away["HOME_TEAM_ID"].astype(str)
    home_away["AWAY_TEAM_ID"] = home_away["AWAY_TEAM_ID"].astype(str)

    df = df.merge(home_away[["GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"]],
                  left_on="gameId_str", right_on="GAME_ID", how="left")
    df["teamId_str"] = np.where(df["home"] == 1, df["HOME_TEAM_ID"], df["AWAY_TEAM_ID"])
    df = df.drop(columns=["GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"])

    # --- Join player-level advanced stats ---
    player_adv = player_adv.copy()
    player_adv["GAME_ID"] = player_adv["GAME_ID"].astype(str)
    player_adv["PLAYER_ID"] = player_adv["PLAYER_ID"].astype(str)

    df = df.merge(
        player_adv[["GAME_ID", "PLAYER_ID", "USG_PCT"]].rename(columns={
            "GAME_ID": "gameId_str", "PLAYER_ID": "personId_str",
        }),
        on=["gameId_str", "personId_str"],
        how="left",
    )
    df["usage_rate"] = df["USG_PCT"]

    # Within-team usage share per game
    team_usg_total = df.groupby(["gameId_str", "teamId_str"])["USG_PCT"].transform("sum")
    df["usage_share"] = df["USG_PCT"] / team_usg_total.replace(0, np.nan)

    # --- Join team-level advanced stats ---
    team_adv = team_adv.copy()
    team_adv["GAME_ID"] = team_adv["GAME_ID"].astype(str)
    team_adv["TEAM_ID"] = team_adv["TEAM_ID"].astype(str)

    df = df.merge(
        team_adv[["GAME_ID", "TEAM_ID", "PACE", "OFF_RATING", "DEF_RATING"]].rename(columns={
            "GAME_ID": "gameId_str", "TEAM_ID": "teamId_str",
            "PACE": "team_pace", "OFF_RATING": "team_off_rating", "DEF_RATING": "team_def_rating",
        }),
        on=["gameId_str", "teamId_str"],
        how="left",
    )

    # Implied total: own OFF_RATING + opponent DEF_RATING
    game_team_ratings = (
        df.groupby(["gameId_str", "teamId_str"])[["team_off_rating", "team_def_rating"]]
        .first()
        .reset_index()
    )
    opp_def = game_team_ratings[["gameId_str", "teamId_str", "team_def_rating"]].copy()
    opp_def.columns = ["gameId_str", "opp_teamId_str", "opp_def_rating"]

    game_pairs = game_team_ratings.merge(opp_def, on="gameId_str")
    game_pairs = game_pairs[game_pairs["teamId_str"] != game_pairs["opp_teamId_str"]].copy()
    game_pairs["implied_total"] = game_pairs["team_off_rating"] + game_pairs["opp_def_rating"]

    df = df.merge(
        game_pairs[["gameId_str", "teamId_str", "implied_total"]],
        on=["gameId_str", "teamId_str"],
        how="left",
    )

    # --- Join rest days from team game log ---
    if gamelog is not None and not gamelog.empty:
        gamelog = gamelog.copy()
        gamelog["GAME_ID"] = gamelog["GAME_ID"].astype(str)
        gamelog["TEAM_ID"] = gamelog["TEAM_ID"].astype(str)

        df = df.merge(
            gamelog[["GAME_ID", "TEAM_ID", "days_rest", "is_b2b"]].rename(columns={
                "GAME_ID": "gameId_str", "TEAM_ID": "teamId_str",
            }),
            on=["gameId_str", "teamId_str"],
            how="left",
        )
    else:
        df["days_rest"] = np.nan
        df["is_b2b"] = 0

    # Clip and fill rest days: NaN (first game of season) → 7
    df["days_rest"] = df["days_rest"].clip(upper=7).fillna(7).astype(float)
    df["is_b2b"] = df["is_b2b"].fillna(0).astype(int)

    df = df.drop(columns=["USG_PCT", "gameId_str", "personId_str", "teamId_str"], errors="ignore")
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

    Uses raw per-game stats (not decay averages) as inputs.
    Returns 0 when fewer than min_periods games are available.
    """
    df = df.sort_values(["personId", "gameDateTimeEst"])

    def _roll_corr(g):
        pts = g["points"]
        ast = g["assists"]
        reb = g["reboundsTotal"]
        r_pa = pts.rolling(window, min_periods=min_periods).corr(ast).fillna(0.0)
        r_pr = pts.rolling(window, min_periods=min_periods).corr(reb).fillna(0.0)
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
    assert df["h_pace"].between(85, 115).all(), f"pace values implausible: {df['h_pace'].describe()}"
    assert df["days_rest"].between(0, 7).all(), "days_rest out of bounds [0, 7]"
    print("Validation passed.")


def build_input_target(df):
    # Only keep rows where all INPUT_COLS are present
    available_cols = [c for c in INPUT_COLS if c in df.columns]
    target = df[TARGET_COLS]
    inputs = df[available_cols]
    return inputs, target


def run(raw_path=None, out_dir=None, season_suffix="2022-25", adv_dir=None, skip_validation=False):
    out_dir = Path(out_dir) if out_dir else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading player stats...")
    df = load_player_stats(raw_path)
    print(f"  {len(df)} rows loaded")

    print("Loading advanced stats...")
    player_adv, team_adv, gamelog, home_away = load_advanced_stats(adv_dir)

    if home_away is not None:
        print("Enriching with advanced stats...")
        df = enrich_with_advanced(df, player_adv, team_adv, gamelog, home_away)
        print(f"  Enrichment complete — {df.shape[1]} columns")
    else:
        print("WARNING: No advanced stats found in data/advanced/. Run ingest/fetch_advanced_stats.py.")
        print("         Proceeding with basic features only.")
        # Fill new columns with NaN so downstream code doesn't break
        for raw_col, h_col, _ in STATS_TO_DECAY:
            if raw_col not in df.columns:
                df[raw_col] = np.nan
        df["days_rest"] = 7.0
        df["is_b2b"] = 0
        df["cov_pts_ast"] = 0.0
        df["cov_pts_reb"] = 0.0

    print(f"Building decay features for {len(df)} rows...")
    df = add_decay_features(df)

    print("Computing rolling covariances...")
    df = compute_rolling_covariances(df)

    df_valid = df.dropna(subset=["h_numMinutes"])

    if home_away is not None and not skip_validation:
        validate(df_valid)

    inputs, target = build_input_target(df_valid)

    input_path = out_dir / f"input_data_{season_suffix}.csv"
    target_path = out_dir / f"target_data_{season_suffix}.csv"
    inputs.to_csv(input_path, index=False)
    target.to_csv(target_path, index=False)
    print(f"Saved input  → {input_path}  ({len(inputs)} rows, {len(inputs.columns)} cols)")
    print(f"Saved target → {target_path}")

"""Fetch advanced NBA stats from nba_api and write raw parquets to data/advanced/.

Usage:
    python ingest/fetch_advanced_stats.py [--seasons 2022-23 2023-24 2024-25] [--force]

Re-runnable (idempotent): skips seasons whose output files already exist unless
--force is passed. Prints progress per season and per game batch.
"""
import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADV_DIR = ROOT / "data" / "advanced"

SLEEP = 0.6  # seconds between API calls — do not reduce
DEFAULT_SEASONS = ["2022-23", "2023-24", "2024-25"]


def fetch_with_retry(endpoint_cls, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            time.sleep(SLEEP)
            return endpoint_cls(**kwargs).get_data_frames()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = SLEEP * (attempt + 2)
            print(f"    Retry {attempt + 1}/{retries - 1} after {wait:.1f}s: {e}")
            time.sleep(wait)


def get_game_ids(season):
    """Return unique regular-season game IDs and the home/away team mapping."""
    from nba_api.stats.endpoints import LeagueGameFinder

    print(f"  Fetching game IDs for {season}...")
    dfs = fetch_with_retry(
        LeagueGameFinder,
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    games_df = dfs[0]

    # Build home/away team mapping using MATCHUP column ("X vs. Y" = home, "X @ Y" = away)
    home_df = games_df[games_df["MATCHUP"].str.contains(r"\bvs\b", na=False)][["GAME_ID", "TEAM_ID"]].copy()
    away_df = games_df[games_df["MATCHUP"].str.contains("@", na=False)][["GAME_ID", "TEAM_ID"]].copy()
    home_df = home_df.rename(columns={"TEAM_ID": "HOME_TEAM_ID"})
    away_df = away_df.rename(columns={"TEAM_ID": "AWAY_TEAM_ID"})
    game_home_away = home_df.merge(away_df, on="GAME_ID", how="inner")
    game_home_away["GAME_ID"] = game_home_away["GAME_ID"].astype(str)

    game_ids = games_df["GAME_ID"].astype(str).unique().tolist()
    print(f"  Found {len(game_ids)} unique game IDs")
    return game_ids, game_home_away


def fetch_boxscore_advanced(game_ids, season):
    """Fetch per-player and per-team advanced box scores for all games."""
    from nba_api.stats.endpoints import BoxScoreAdvancedV2

    player_rows, team_rows = [], []
    total = len(game_ids)

    for i, game_id in enumerate(game_ids):
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == total:
            print(f"  BoxScore {i + 1}/{total}")
        try:
            dfs = fetch_with_retry(BoxScoreAdvancedV2, game_id=game_id)
            player_df = dfs[0]
            team_df = dfs[1]

            if player_df is None or player_df.empty:
                warnings.warn(f"Empty player data for game_id={game_id}")
                continue

            p_cols = ["GAME_ID", "TEAM_ID", "PLAYER_ID", "USG_PCT", "PACE", "OFF_RATING", "DEF_RATING", "MIN"]
            t_cols = ["GAME_ID", "TEAM_ID", "PACE", "OFF_RATING", "DEF_RATING"]

            player_rows.append(player_df[[c for c in p_cols if c in player_df.columns]])
            if team_df is not None and not team_df.empty:
                team_rows.append(team_df[[c for c in t_cols if c in team_df.columns]])

        except Exception as e:
            warnings.warn(f"Failed game_id={game_id}: {e}")
            continue

    player_adv = pd.concat(player_rows, ignore_index=True) if player_rows else pd.DataFrame()
    team_adv = pd.concat(team_rows, ignore_index=True) if team_rows else pd.DataFrame()

    # Normalize types
    for df in [player_adv, team_adv]:
        if "GAME_ID" in df.columns:
            df["GAME_ID"] = df["GAME_ID"].astype(str)
        if "TEAM_ID" in df.columns:
            df["TEAM_ID"] = df["TEAM_ID"].astype(str)
    if "PLAYER_ID" in player_adv.columns:
        player_adv["PLAYER_ID"] = player_adv["PLAYER_ID"].astype(str)

    return player_adv, team_adv


def fetch_team_gamelogs(season):
    """Fetch game logs for all 30 teams and compute rest/b2b flags."""
    from nba_api.stats.endpoints import TeamGameLog
    from nba_api.stats.static import teams as nba_teams

    all_teams = nba_teams.get_teams()
    all_rows = []

    for i, team in enumerate(all_teams):
        team_id = team["id"]
        if (i + 1) % 10 == 0 or i + 1 == len(all_teams):
            print(f"  TeamGameLog {i + 1}/{len(all_teams)}")
        try:
            dfs = fetch_with_retry(
                TeamGameLog,
                team_id=team_id,
                season=season,
                season_type_all_star="Regular Season",
            )
            df = dfs[0]
            if df is None or df.empty:
                continue
            df = df[["GAME_ID", "GAME_DATE"]].copy()
            df["TEAM_ID"] = str(team_id)
            all_rows.append(df)
        except Exception as e:
            warnings.warn(f"Failed TeamGameLog team_id={team_id}: {e}")

    if not all_rows:
        return pd.DataFrame()

    gamelog = pd.concat(all_rows, ignore_index=True)
    gamelog["GAME_ID"] = gamelog["GAME_ID"].astype(str)
    gamelog["GAME_DATE"] = pd.to_datetime(gamelog["GAME_DATE"], format="mixed", dayfirst=False)
    gamelog = gamelog.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    # days since previous game (NaN for first game of season)
    gamelog["days_rest"] = gamelog.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days

    # is_b2b_second: days_rest == 1 (played yesterday)
    gamelog["is_b2b_second"] = gamelog["days_rest"] == 1

    # is_b2b_first: next game is tomorrow
    next_date = gamelog.groupby("TEAM_ID")["GAME_DATE"].shift(-1)
    gamelog["is_b2b_first"] = (next_date - gamelog["GAME_DATE"]).dt.days == 1

    gamelog["is_b2b"] = (gamelog["is_b2b_first"] | gamelog["is_b2b_second"]).astype(int)

    return gamelog[["GAME_ID", "TEAM_ID", "GAME_DATE", "days_rest", "is_b2b"]]


def fetch_league_team_stats(season):
    """Fetch season-level team defensive stats."""
    from nba_api.stats.endpoints import LeagueDashTeamStats

    print(f"  Fetching LeagueDashTeamStats for {season}...")
    dfs = fetch_with_retry(
        LeagueDashTeamStats,
        measure_type_detailed_defense="Advanced",
        per_mode_simple="PerGame",
        season=season,
    )
    df = dfs[0]
    keep = ["TEAM_ID", "DEF_RATING", "PACE"]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()
    df["TEAM_ID"] = df["TEAM_ID"].astype(str)
    return df


def run_season(season, force=False):
    """Fetch and save all advanced data for a single season."""
    ADV_DIR.mkdir(parents=True, exist_ok=True)

    player_path   = ADV_DIR / f"boxscore_advanced_{season}.parquet"
    team_path     = ADV_DIR / f"team_advanced_{season}.parquet"
    gamelog_path  = ADV_DIR / f"team_gamelog_{season}.parquet"
    league_path   = ADV_DIR / f"league_team_stats_{season}.parquet"
    home_away_path = ADV_DIR / f"game_home_away_{season}.parquet"

    all_exist = all(p.exists() for p in [player_path, team_path, gamelog_path, league_path, home_away_path])
    if all_exist and not force:
        print(f"[{season}] All files exist — skipping (pass --force to re-fetch)")
        return

    print(f"\n=== Season {season} ===")

    # --- Game IDs and home/away mapping ---
    game_ids, game_home_away = get_game_ids(season)
    game_home_away.to_parquet(home_away_path, index=False)
    print(f"  Saved {len(game_home_away)} rows → {home_away_path.name}")

    # --- Per-game advanced box scores ---
    if not player_path.exists() or not team_path.exists() or force:
        player_adv, team_adv = fetch_boxscore_advanced(game_ids, season)
        player_adv.to_parquet(player_path, index=False)
        team_adv.to_parquet(team_path, index=False)
        print(f"  Saved {len(player_adv)} player rows → {player_path.name}")
        print(f"  Saved {len(team_adv)} team rows → {team_path.name}")

    # --- Team game logs ---
    if not gamelog_path.exists() or force:
        print(f"  Fetching team game logs for {season}...")
        gamelog = fetch_team_gamelogs(season)
        gamelog.to_parquet(gamelog_path, index=False)
        print(f"  Saved {len(gamelog)} rows → {gamelog_path.name}")

    # --- League-level team stats ---
    if not league_path.exists() or force:
        league_stats = fetch_league_team_stats(season)
        league_stats.to_parquet(league_path, index=False)
        print(f"  Saved {len(league_stats)} rows → {league_path.name}")

    print(f"[{season}] Done.")


def detect_seasons_from_processed(processed_dir=None):
    """Infer NBA seasons from processed input files by reading game dates."""
    d = Path(processed_dir) if processed_dir else ROOT / "data" / "processed"
    files = sorted(d.glob("input_data_*.csv"))
    if not files:
        return DEFAULT_SEASONS

    all_dates = []
    for f in files:
        try:
            dates = pd.read_csv(f, usecols=["gameDateTimeEst"])["gameDateTimeEst"]
            all_dates.extend(pd.to_datetime(dates, utc=True, errors="coerce").dropna().tolist())
        except Exception:
            continue

    if not all_dates:
        return DEFAULT_SEASONS

    min_year = min(d.year for d in all_dates)
    max_year = max(d.year for d in all_dates)

    seasons = []
    for year in range(min_year - 1, max_year + 1):
        # NBA regular season runs ~Oct year to ~Apr year+1
        if any(pd.Timestamp(year, 10, 1, tzinfo=pd.Timestamp.now().tzinfo) <= d <= pd.Timestamp(year + 1, 7, 1, tzinfo=pd.Timestamp.now().tzinfo) for d in all_dates):
            seasons.append(f"{year}-{str(year + 1)[2:]}")

    return sorted(set(seasons)) or DEFAULT_SEASONS


def main():
    parser = argparse.ArgumentParser(description="Fetch advanced NBA stats from nba_api")
    parser.add_argument(
        "--seasons", nargs="+", default=None,
        help="NBA seasons to fetch (e.g. 2022-23 2023-24 2024-25). "
             "Defaults to seasons detected from existing processed files.",
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch even if files exist")
    args = parser.parse_args()

    seasons = args.seasons or detect_seasons_from_processed()
    print(f"Fetching seasons: {seasons}")

    for season in seasons:
        run_season(season, force=args.force)

    print("\nAll seasons complete.")


if __name__ == "__main__":
    main()

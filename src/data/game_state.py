"""
Compute actual in-game game state features from TeamStatistics and PlayByPlay.

These are POST-GAME features (no leakage concern for Phase 1 validation).
They serve as the "teacher" signal G that the VAE encoder must learn to predict.

G VARIANTS (each is a named configuration of features):

  v1_baseline (12-dim) — original 6 TS features × 2 teams:
    actual_pace, actual_3pa_rate, actual_ast_rate, actual_to_rate,
    actual_margin, actual_oreb_rate

  v2_four_factors (22-dim) — v1 + four factors + game script (q-score data required):
    + actual_eFG_pct, actual_ft_rate
    + actual_margin_q2, actual_margin_q3, actual_biggest_lead

  v3_team_totals (18-dim) — team counting totals + key rates:
    actual_pts, actual_ast, actual_reb
    + actual_pace, actual_margin, actual_eFG_pct, actual_ft_rate,
      actual_ast_rate, actual_to_rate

  v4_rich (36-dim) — all TS-derived features (quarter data partial coverage):
    all v2 + team counting totals + bench/paint/fastbreak rates

  v5_totals_only (6-dim) — minimal, just counting totals per team:
    actual_pts, actual_ast, actual_reb

  v6_entropy (16-dim) — team totals + distribution/concentration stats:
    actual_pts, actual_ast, actual_reb, actual_margin, actual_pace
    + actual_scoring_entropy, actual_minutes_concentration, actual_ast_rate

Output: one row per team per game, with gameId, teamId, home, and all feature columns.
Call build_game_state_df(variant=...) to get the concatenated home+away game-level vector.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Named G variants: which per-team columns each uses
# ---------------------------------------------------------------------------

G_VARIANTS = {
    "v1_baseline": [
        "actual_pace", "actual_3pa_rate", "actual_ast_rate",
        "actual_to_rate", "actual_margin", "actual_oreb_rate",
    ],
    "v2_four_factors": [
        # v1
        "actual_pace", "actual_3pa_rate", "actual_ast_rate",
        "actual_to_rate", "actual_margin", "actual_oreb_rate",
        # four factors additions
        "actual_eFG_pct", "actual_ft_rate",
        # game script (q-score coverage: ~25% of training data)
        "actual_margin_q2", "actual_margin_q3", "actual_biggest_lead",
    ],
    "v3_team_totals": [
        # team counting totals (tell the decoder the game's "pie")
        "actual_pts", "actual_ast", "actual_reb",
        # context
        "actual_pace", "actual_margin",
        "actual_eFG_pct", "actual_ft_rate", "actual_ast_rate", "actual_to_rate",
    ],
    "v4_rich": [
        # all non-PBP features
        "actual_pace", "actual_poss",
        "actual_eFG_pct", "actual_ft_rate", "actual_to_rate", "actual_oreb_rate",
        "actual_3pa_rate", "actual_ast_rate",
        "actual_pts", "actual_ast", "actual_reb",
        "actual_margin", "actual_margin_q2", "actual_margin_q3", "actual_biggest_lead",
        "actual_bench_rate", "actual_paint_rate", "actual_fb_rate",
    ],
    "v5_totals_only": [
        # minimal: just counting totals per team
        "actual_pts", "actual_ast", "actual_reb",
    ],
    "v6_entropy": [
        # team totals + distribution stats
        "actual_pts", "actual_ast", "actual_reb",
        "actual_margin", "actual_pace",
        "actual_scoring_entropy", "actual_minutes_conc",
        "actual_ast_rate",
    ],
}

DEFAULT_VARIANT = "v1_baseline"

# Legacy aliases for backward compatibility with game_state_dataset.py
TS_FEATURE_COLS = G_VARIANTS["v1_baseline"]
TEAM_FEATURE_COLS = G_VARIANTS["v1_baseline"]
GAME_STATE_COLS = (
    [f"home_{c}" for c in G_VARIANTS["v1_baseline"]]
    + [f"away_{c}" for c in G_VARIANTS["v1_baseline"]]
)

# PBP-derived (optional)
PBP_TEAM_COLS = [
    "actual_rim_rate",
    "actual_midrange_rate",
    "actual_assisted_rate",
]


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_ts_features(ts: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-team-game features from TeamStatistics rows."""
    ts = ts.copy()
    ts = ts.dropna(subset=["teamScore", "numMinutes", "fieldGoalsMade", "fieldGoalsAttempted"])
    # Drop rows with nonsensical FGA (data artifacts)
    ts = ts[ts["fieldGoalsAttempted"] >= 5].copy()

    fga = ts["fieldGoalsAttempted"]
    fgm = ts["fieldGoalsMade"]
    ts3pa = ts["threePointersAttempted"].fillna(0)
    ts3pm = ts["threePointersMade"].fillna(0)
    fta = ts["freeThrowsAttempted"].fillna(0)
    tov = ts["turnovers"].fillna(0)
    orb = ts["reboundsOffensive"].fillna(0)

    # Possession estimate (Dean Oliver formula)
    ts["poss"] = (fga + 0.44 * fta + tov - orb).clip(lower=1)

    # game_min: numMinutes = 5 * game_minutes (e.g. 240 for 48-min regulation)
    ts["game_min"] = (ts["numMinutes"] / 5.0).clip(lower=40)

    ts["actual_poss"] = ts["poss"]
    ts["actual_pace"] = (ts["poss"] * 48.0 / ts["game_min"]).clip(60, 160)
    ts["actual_3pa_rate"] = (ts3pa / fga).clip(0, 1)
    ts["actual_ast_rate"] = (ts["assists"].fillna(0) / (fgm + 1e-6)).clip(0, 2)
    ts["actual_to_rate"] = (tov / ts["poss"]).clip(0, 0.5)
    ts["actual_margin"] = ts["teamScore"] - ts["opponentScore"]
    ts["actual_oreb_rate"] = (orb / (fga - fgm).clip(lower=1)).clip(0, 1)

    # Four factors
    ts["actual_eFG_pct"] = ((fgm + 0.5 * ts3pm) / fga).clip(0, 1)
    ts["actual_ft_rate"] = (fta / fga).clip(0, 3)

    # Team counting totals
    ts["actual_pts"] = ts["teamScore"]
    ts["actual_ast"] = ts["assists"].fillna(0)
    ts["actual_reb"] = ts["reboundsTotal"].fillna(0)

    # Quarter score margins (raw quarter points stored for later margin computation)
    ts["_q1"] = ts["q1Points"].fillna(np.nan)
    ts["_q2"] = ts["q2Points"].fillna(np.nan)
    ts["_q3"] = ts["q3Points"].fillna(np.nan)

    ts["actual_biggest_lead"] = ts["biggestLead"].fillna(np.nan)

    # Distribution features (only meaningful where benchPoints etc. are available)
    ts["actual_bench_rate"] = (ts["benchPoints"].fillna(np.nan) / (ts["teamScore"] + 1e-6)).clip(0, 1)
    ts["actual_paint_rate"] = (ts["pointsInThePaint"].fillna(np.nan) / (ts["teamScore"] + 1e-6)).clip(0, 1)
    ts["actual_fb_rate"] = (ts["pointsFastBreak"].fillna(np.nan) / (ts["teamScore"] + 1e-6)).clip(0, 1)

    keep_cols = ["gameId", "teamId", "home"] + [
        "actual_poss", "actual_pace", "actual_3pa_rate", "actual_ast_rate",
        "actual_to_rate", "actual_margin", "actual_oreb_rate",
        "actual_eFG_pct", "actual_ft_rate",
        "actual_pts", "actual_ast", "actual_reb",
        "actual_biggest_lead",
        "actual_bench_rate", "actual_paint_rate", "actual_fb_rate",
        "_q1", "_q2", "_q3",
    ]
    return ts[[c for c in keep_cols if c in ts.columns]].copy()


def _add_quarter_margins(team_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative score margin at Q2 (halftime) and Q3 from each team's perspective."""
    team_df = team_df.copy()

    home_rows = team_df[team_df["home"] == 1][["gameId", "_q1", "_q2", "_q3"]].copy()
    away_rows = team_df[team_df["home"] == 0][["gameId", "_q1", "_q2", "_q3"]].copy()

    merged = home_rows.merge(away_rows, on="gameId", suffixes=("_h", "_a"), how="inner")
    merged["game_margin_q2"] = (
        merged["_q1_h"].fillna(0) + merged["_q2_h"].fillna(0)
        - merged["_q1_a"].fillna(0) - merged["_q2_a"].fillna(0)
    )
    merged["game_margin_q3"] = (
        merged["_q1_h"].fillna(0) + merged["_q2_h"].fillna(0) + merged["_q3_h"].fillna(0)
        - merged["_q1_a"].fillna(0) - merged["_q2_a"].fillna(0) - merged["_q3_a"].fillna(0)
    )
    # Mark as NaN if either team's quarter data was missing
    q2_valid = (
        merged["_q1_h"].notna() & merged["_q2_h"].notna() &
        merged["_q1_a"].notna() & merged["_q2_a"].notna()
    )
    q3_valid = q2_valid & merged["_q3_h"].notna() & merged["_q3_a"].notna()
    merged.loc[~q2_valid, "game_margin_q2"] = np.nan
    merged.loc[~q3_valid, "game_margin_q3"] = np.nan

    game_margins = merged[["gameId", "game_margin_q2", "game_margin_q3"]]

    team_df = team_df.merge(game_margins, on="gameId", how="left")
    team_df["actual_margin_q2"] = np.where(
        team_df["home"] == 1,
        team_df["game_margin_q2"],
        -team_df["game_margin_q2"],
    )
    team_df["actual_margin_q3"] = np.where(
        team_df["home"] == 1,
        team_df["game_margin_q3"],
        -team_df["game_margin_q3"],
    )

    team_df = team_df.drop(
        columns=["game_margin_q2", "game_margin_q3", "_q1", "_q2", "_q3"], errors="ignore"
    )
    return team_df


def _compute_entropy_features(raw_dir: Path) -> pd.DataFrame:
    """
    Compute per-team-game scoring entropy and minutes concentration from PlayerStatistics.

    scoring_entropy: Shannon entropy of each player's share of team points.
      High = balanced scoring, low = one star dominated.

    minutes_conc: Herfindahl-Hirschman Index (HHI) of minutes distribution.
      High (near 1) = short rotation, low = many players got minutes.

    Returns DataFrame with: gameId, home (0/1), actual_scoring_entropy, actual_minutes_conc.
    """
    print("Computing scoring entropy from PlayerStatistics...")
    ps = pd.read_csv(raw_dir / "PlayerStatistics.csv", low_memory=False)
    ps["gameId"] = pd.to_numeric(ps["gameId"], errors="coerce")
    ps["home"] = pd.to_numeric(ps["home"], errors="coerce")
    ps["points"] = pd.to_numeric(ps["points"], errors="coerce").fillna(0)
    ps["numMinutes"] = pd.to_numeric(ps["numMinutes"], errors="coerce")
    ps = ps.dropna(subset=["gameId", "home"])

    records = []
    for (game_id, home_flag), grp in ps.groupby(["gameId", "home"]):
        # Scoring entropy
        pts = grp["points"].values.clip(0)
        team_pts = pts.sum()
        if team_pts > 0:
            p = pts / team_pts
            p = p[p > 0]
            entropy = float(-np.sum(p * np.log(p + 1e-10)))
        else:
            entropy = np.nan

        # Minutes concentration (HHI)
        mins = grp["numMinutes"].dropna().values.clip(0)
        team_mins = mins.sum()
        if team_mins > 0:
            m = mins / team_mins
            hhi = float(np.sum(m ** 2))
        else:
            hhi = np.nan

        records.append({
            "gameId": game_id,
            "home": int(home_flag),
            "actual_scoring_entropy": entropy,
            "actual_minutes_conc": hhi,
        })

    print(f"  Computed entropy for {len(records):,} team-games")
    return pd.DataFrame(records)


def _compute_pbp_features(pbp_path: Path) -> pd.DataFrame:
    """Compute per-team-game shot composition features from PlayByPlay."""
    print("Loading PlayByPlay for shot features...")
    pbp = pd.read_parquet(pbp_path, columns=[
        "gameId", "teamId", "isFieldGoal", "shotResult",
        "shotDistance", "area", "assistPersonId",
    ])
    print(f"  Loaded {len(pbp):,} rows")

    shots = pbp[pbp["isFieldGoal"] == True].copy()
    shots["gameId"] = pd.to_numeric(shots["gameId"], errors="coerce")
    shots["teamId"] = pd.to_numeric(shots["teamId"], errors="coerce")
    shots["assistPersonId"] = pd.to_numeric(shots["assistPersonId"], errors="coerce")
    shots = shots.dropna(subset=["gameId", "teamId"])

    shots["is_made"] = shots["shotResult"] == "Made"
    shots["is_assisted"] = shots["assistPersonId"].notnull()
    shots["is_rim"] = (shots["area"] == "Restricted Area") | (shots["shotDistance"] <= 4)
    shots["is_midrange"] = shots["area"] == "Mid-Range"

    agg = shots.groupby(["gameId", "teamId"]).agg(
        total_fga=("isFieldGoal", "count"),
        total_fgm=("is_made", "sum"),
        rim_fga=("is_rim", "sum"),
        midrange_fga=("is_midrange", "sum"),
    ).reset_index()

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

    return agg[["gameId", "teamId"] + PBP_TEAM_COLS].copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_variant_cols(variant: str = DEFAULT_VARIANT) -> list:
    """Return the per-team feature columns for a given G variant."""
    if variant not in G_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(G_VARIANTS.keys())}")
    return G_VARIANTS[variant]


def load_team_game_state(
    raw_dir: Path = RAW_DIR,
    include_pbp: bool = False,
    include_entropy: bool = False,
) -> pd.DataFrame:
    """
    Load per-team-game state features (all variants).

    Returns one row per (gameId, teamId) with all computed features.
    """
    ts = pd.read_csv(raw_dir / "TeamStatistics.csv")
    features = _compute_ts_features(ts)
    features = _add_quarter_margins(features)

    if include_entropy:
        entropy_df = _compute_entropy_features(raw_dir)
        # Merge on gameId + home (not teamId, since entropy uses home flag)
        features = features.merge(entropy_df, on=["gameId", "home"], how="left")

    if include_pbp:
        pbp_path = raw_dir / "PlayByPlay.parquet"
        if pbp_path.exists():
            pbp_feats = _compute_pbp_features(pbp_path)
            features = features.merge(pbp_feats, on=["gameId", "teamId"], how="left")
        else:
            print("Warning: PlayByPlay.parquet not found, skipping PBP features")
            for col in PBP_TEAM_COLS:
                features[col] = np.nan
    else:
        for col in PBP_TEAM_COLS:
            features[col] = np.nan

    return features


def build_game_state_df(
    raw_dir: Path = RAW_DIR,
    include_pbp: bool = False,
    include_entropy: bool = False,
    use_cache: bool = True,
    variant: str = DEFAULT_VARIANT,
) -> pd.DataFrame:
    """
    Build game-level state vectors with home_ and away_ prefixed features.

    Returns one row per game with columns: gameId + [home_{col}, away_{col} for col in variant].
    """
    needs_entropy = include_entropy or "entropy" in variant or "conc" in "".join(get_variant_cols(variant))
    cache_suffix = "_entropy" if needs_entropy else ""
    cache_path = ROOT / "data" / "processed" / f"game_state_full_cache{cache_suffix}.csv"

    if use_cache and not include_pbp and cache_path.exists():
        team_gs = pd.read_csv(cache_path)
        print(f"Loaded game state from cache: {len(team_gs):,} rows")
    else:
        team_gs = load_team_game_state(
            raw_dir=raw_dir, include_pbp=include_pbp, include_entropy=needs_entropy
        )
        if use_cache and not include_pbp:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            team_gs.to_csv(cache_path, index=False)
            print(f"Cached game state to {cache_path}")

    variant_cols = get_variant_cols(variant)
    available = [c for c in variant_cols if c in team_gs.columns]
    missing = [c for c in variant_cols if c not in team_gs.columns]
    if missing:
        print(f"Warning: missing columns for variant '{variant}': {missing}")

    home = team_gs[team_gs["home"] == 1][["gameId"] + available].copy()
    away = team_gs[team_gs["home"] == 0][["gameId"] + available].copy()

    home = home.rename(columns={c: f"home_{c}" for c in available})
    away = away.rename(columns={c: f"away_{c}" for c in available})

    home = home.drop_duplicates(subset=["gameId"])
    away = away.drop_duplicates(subset=["gameId"])

    game_gs = home.merge(away, on="gameId", how="inner")
    n_feats = len(available) * 2
    print(f"Game state ({variant}): {len(game_gs):,} games, {n_feats} features")

    return game_gs


if __name__ == "__main__":
    print("=== All G Variants ===")
    for v, cols in G_VARIANTS.items():
        print(f"\n{v} ({len(cols)*2} total features = {len(cols)} per team × 2):")
        for c in cols:
            print(f"  {c}")

    print("\n\nBuilding full feature table (with entropy)...")
    team_gs = load_team_game_state(include_entropy=True)
    print(f"Loaded {len(team_gs):,} team-game rows")

    all_feat_cols = [c for c in [
        "actual_pace", "actual_poss", "actual_eFG_pct", "actual_ft_rate",
        "actual_to_rate", "actual_oreb_rate", "actual_3pa_rate", "actual_ast_rate",
        "actual_pts", "actual_ast", "actual_reb",
        "actual_margin", "actual_margin_q2", "actual_margin_q3",
        "actual_biggest_lead", "actual_bench_rate", "actual_paint_rate", "actual_fb_rate",
        "actual_scoring_entropy", "actual_minutes_conc",
    ] if c in team_gs.columns]

    print("\nFeature stats:")
    print(team_gs[all_feat_cols].describe().round(3).to_string())

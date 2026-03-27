"""
Quick data-level phi analysis.

For each candidate stat, computes actual pairwise phi coefficients between players
in the same game using h_stat as the line. No model needed — answers whether
real-world inter-player correlations exist for these stats.

Stats tested: points, assists, reboundsTotal, threePointersMade, blocks, steals, turnovers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parents[1]

STATS = ["points", "assists", "reboundsTotal", "threePointersMade", "blocks", "steals", "turnovers"]

N_PLAYERS_PER_TEAM = 8  # top 8 by minutes


def compute_game_phi(over_matrix):
    """
    over_matrix: (n_players, n_stats) binary 0/1
    Returns all cross-player phi values (excluding same-player pairs).
    """
    n_players, n_stats = over_matrix.shape
    phis = []
    # all pairs of (player_i, stat_a) vs (player_j, stat_b) where i != j
    for i in range(n_players):
        for j in range(i + 1, n_players):
            for a in range(n_stats):
                for b in range(n_stats):
                    x = over_matrix[i, a]
                    y = over_matrix[j, b]
                    # single-game: treat across games
                    phis.append((i, a, j, b, x, y))
    return phis


def analyze(seasons=None, min_minutes=10, n_players_per_team=N_PLAYERS_PER_TEAM):
    raw = pd.read_csv(ROOT / "data/raw/PlayerStatistics.csv")
    inp = pd.read_csv(ROOT / "data/processed/input_data_2024-25.csv")

    # merge h_stat columns in
    h_cols = ["h_" + s for s in STATS if "h_" + s in inp.columns]
    available_h = {s: "h_" + s for s in STATS if "h_" + s in inp.columns}

    merge_cols = ["personId", "gameId", "home"] + list(available_h.values())
    inp_sub = inp[merge_cols]
    df = raw.merge(inp_sub, on=["personId", "gameId", "home"], how="inner")

    # filter out very low minute players
    df = df[df["numMinutes"] >= min_minutes]
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])

    print(f"Loaded {len(df):,} player-games across {df['gameId'].nunique():,} games\n")

    # For each stat, compute binary over/under per player-game
    stat_available = [s for s in STATS if "h_" + s in inp.columns and s in df.columns]
    print(f"Stats available: {stat_available}\n")

    # For phi computation we need to aggregate across many games.
    # Strategy: for each game, get top-N players per team by numMinutes,
    # compute their over/under, then aggregate cross-game phi.

    # Collect binary vectors per player slot per game
    # We'll assign player slots 0..N-1 home, N..2N-1 away
    records = []
    for game_id, gdf in df.groupby("gameId"):
        home = gdf[gdf["home"] == 1].sort_values("numMinutes", ascending=False).head(n_players_per_team)
        away = gdf[gdf["home"] == 0].sort_values("numMinutes", ascending=False).head(n_players_per_team)
        if len(home) < 3 or len(away) < 3:
            continue

        for team_label, tdf in [("home", home), ("away", away)]:
            for slot, (_, row) in enumerate(tdf.iterrows()):
                rec = {"gameId": game_id, "team": team_label, "slot": slot}
                for s in stat_available:
                    actual = row[s]
                    h_stat = row["h_" + s]
                    if pd.isna(actual) or pd.isna(h_stat):
                        rec[s] = np.nan
                    else:
                        rec[s] = int(actual > h_stat)
                records.append(rec)

    game_df = pd.DataFrame(records)
    print(f"Built {len(game_df):,} player-slot records from {game_df['gameId'].nunique():,} games\n")

    # --- Phi analysis per stat ---
    print("=" * 65)
    print(f"{'Stat':<22} {'Mean phi':>10} {'Std phi':>10} {'|phi|>0.10':>12} {'|phi|>0.15':>12}")
    print("=" * 65)

    stat_results = {}
    for stat in stat_available:
        stat_df = game_df[["gameId", "team", "slot", stat]].dropna()

        # for each game, for each pair of (team, slot) != same player, compute phi
        # aggregate across all games: treat pairs as Bernoulli samples
        # phi_pearson = corr(x, y) over all games (same slot pairing)

        # Pair players by team and slot rank within a game, compute correlation over games
        # Cross-team pairs: home slot i vs away slot j
        # Same-team pairs: home slot i vs home slot j (i != j)

        all_phi = []

        # pivot: gameId x (team, slot) -> binary
        pivoted = stat_df.pivot_table(index="gameId", columns=["team", "slot"], values=stat)
        pivoted = pivoted.dropna(how="all")

        home_cols = [("home", s) for s in range(n_players_per_team) if ("home", s) in pivoted.columns]
        away_cols = [("away", s) for s in range(n_players_per_team) if ("away", s) in pivoted.columns]
        all_cols = home_cols + away_cols

        for (t1, s1), (t2, s2) in combinations(all_cols, 2):
            if t1 == t2 and s1 == s2:
                continue  # same player
            col1, col2 = (t1, s1), (t2, s2)
            if col1 not in pivoted.columns or col2 not in pivoted.columns:
                continue
            pair = pivoted[[col1, col2]].dropna()
            if len(pair) < 50:
                continue
            x = pair[col1].values
            y = pair[col2].values
            # phi for binary = pearson correlation
            if x.std() < 1e-6 or y.std() < 1e-6:
                continue
            phi = np.corrcoef(x, y)[0, 1]
            all_phi.append(phi)

        if not all_phi:
            print(f"{stat:<22} {'N/A':>10}")
            continue

        arr = np.array(all_phi)
        stat_results[stat] = arr
        pct_10 = (np.abs(arr) > 0.10).mean() * 100
        pct_15 = (np.abs(arr) > 0.15).mean() * 100
        print(f"{stat:<22} {arr.mean():>+10.4f} {arr.std():>10.4f} {pct_10:>11.1f}% {pct_15:>11.1f}%")

    print("=" * 65)

    # --- Cross-stat analysis: same player slot pairs across teams ---
    print("\n\nCross-stat pair analysis (same-team player pairs)")
    print("Top 10 stat combinations by mean |phi|:\n")

    cross_results = []
    for stat_a, stat_b in combinations(stat_available, 2):
        same_team_phis = []
        for team in ["home", "away"]:
            cols_a = [(team, s) for s in range(n_players_per_team)]
            cols_b = [(team, s) for s in range(n_players_per_team)]
            try:
                piv_a = game_df.pivot_table(index="gameId", columns=["team", "slot"], values=stat_a)
                piv_b = game_df.pivot_table(index="gameId", columns=["team", "slot"], values=stat_b)
            except Exception:
                continue

            for s1 in range(n_players_per_team):
                for s2 in range(s1 + 1, n_players_per_team):
                    ca, cb = (team, s1), (team, s2)
                    if ca not in piv_a.columns or cb not in piv_b.columns:
                        continue
                    pair = pd.DataFrame({"a": piv_a[ca], "b": piv_b[cb]}).dropna()
                    if len(pair) < 50:
                        continue
                    x, y = pair["a"].values, pair["b"].values
                    if x.std() < 1e-6 or y.std() < 1e-6:
                        continue
                    same_team_phis.append(np.corrcoef(x, y)[0, 1])

        if same_team_phis:
            arr = np.array(same_team_phis)
            cross_results.append((stat_a, stat_b, arr.mean(), np.abs(arr).mean(), arr.std()))

    cross_results.sort(key=lambda x: -x[3])
    print(f"{'Stat A':<22} {'Stat B':<22} {'Mean phi':>10} {'Mean|phi|':>10}")
    print("-" * 70)
    for stat_a, stat_b, mean_phi, mean_abs, std_phi in cross_results[:15]:
        print(f"{stat_a:<22} {stat_b:<22} {mean_phi:>+10.4f} {mean_abs:>10.4f}")

    # --- Over rate check ---
    print("\n\nActual over rates (should be near 50% if h_stat is a good line):")
    for stat in stat_available:
        col = game_df[stat].dropna()
        print(f"  {stat:<25} {col.mean():.1%}  ({len(col):,} obs)")


if __name__ == "__main__":
    analyze()

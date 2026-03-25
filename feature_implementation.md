# Layer 2 Integration Spec: `nba_api` Advanced Features

## Purpose

This document is a complete implementation guide for integrating `swar/nba_api` advanced
game-level data into the existing multi-file feature pipeline. The existing pipeline
(Layer 1) ingests historical player box scores from disk, computes exponential decay
averages, and saves pre-computed feature parquets/CSVs. This spec adds a new ingest stage
and extends the feature computation stage with game-environment and usage-hierarchy signals
that the VAE correlation model currently lacks.

Do **not** modify the model architecture or training code. The goal is only to produce
richer pre-computed feature files that slot into the existing `X_team` and `X_players`
tensors.

---

## 1. Dependency

Add to `requirements.txt` (or `pyproject.toml`):

```
nba_api>=1.11.4
```

`nba_api` requires Python ≥ 3.10. It wraps `stats.nba.com` and returns pandas DataFrames.
It is subject to rate limiting — every fetch function must include a `time.sleep(0.6)`
after each API call. Do not remove or reduce this sleep.

---

## 2. New File to Create: `ingest/fetch_advanced_stats.py`

This script is the **only** place that calls `nba_api`. It writes raw data to disk so the
rest of the pipeline never calls the API directly. It should be re-runnable (idempotent):
if output files already exist for a season, skip that season unless a `--force` flag is
passed.

### 2a. What to fetch and from which endpoints

#### Per-game advanced box scores (player + team level)
**Endpoint:** `nba_api.stats.endpoints.BoxScoreAdvancedV2`  
**Call pattern:** One call per `GAME_ID`.  
**Key columns to retain:**

From the `PlayerStats` result set:
- `GAME_ID`, `TEAM_ID`, `PLAYER_ID`
- `USG_PCT` — usage rate (the proportion of team possessions used while on floor)
- `PACE` — game pace in possessions per 48 min (same value for all players in the game)
- `OFF_RATING` — individual offensive rating
- `DEF_RATING` — individual defensive rating
- `MIN` — minutes played (use to weight bench aggregation)

From the `TeamStats` result set (same endpoint, different result set):
- `GAME_ID`, `TEAM_ID`
- `PACE` — team-level pace for the game
- `OFF_RATING`, `DEF_RATING` — team offensive and defensive ratings for the game

> **Implementation note:** `BoxScoreAdvancedV2` requires a `game_id` parameter. To get
> all game IDs for a season, use `LeagueGameFinder` (see below). Process game IDs in
> batches with sleep between calls.

#### All game IDs for a season
**Endpoint:** `nba_api.stats.endpoints.LeagueGameFinder`  
**Parameters:** `season_nullable='YYYY-YY'`, `league_id_nullable='00'`  
**Key columns:** `GAME_ID`, `GAME_DATE`, `TEAM_ID`, `TEAM_ABBREVIATION`  
Use this to build the list of game IDs to iterate over for `BoxScoreAdvancedV2`.

#### Team game log (for rest days and back-to-back flags)
**Endpoint:** `nba_api.stats.endpoints.TeamGameLog`  
**Parameters:** `team_id`, `season`, `season_type_all_star='Regular Season'`  
**Call pattern:** One call per team per season.  
**Key columns:** `GAME_ID`, `GAME_DATE`, `TEAM_ID`  

Compute from this data (do not call a separate endpoint):
- `days_rest` — calendar days since previous game for this team (NaN for first game of season)
- `is_b2b` — boolean, True when `days_rest == 1`
- `is_b2b_first` — True for the first game of a back-to-back (rest == days until next game == 1)
- `is_b2b_second` — True for the second game of a back-to-back (days_rest == 1)

#### Opponent defensive stats (rolling team-level)
**Endpoint:** `nba_api.stats.endpoints.LeagueDashTeamStats`  
**Parameters:** `measure_type_detailed_defense='Advanced'`, `per_mode_simple='PerGame'`,
`season=season`  
**Call pattern:** One call per season (returns all teams).  
**Key columns:** `TEAM_ID`, `DEF_RATING`, `OPP_PTS`, `OPP_REB`, `OPP_AST`, `PACE`

> This gives season-long opponent allowed stats per team. The pipeline will apply
> exponential decay to these values using the same per-game enrichment approach described
> in Section 4.

### 2b. Output files

Write all outputs to a `data/advanced/` directory (create if missing) with the following
naming convention:

| File | Contents |
|---|---|
| `data/advanced/boxscore_advanced_{season}.parquet` | Per-player advanced stats from `BoxScoreAdvancedV2` (PlayerStats result set), one row per player per game |
| `data/advanced/team_advanced_{season}.parquet` | Per-team advanced stats from `BoxScoreAdvancedV2` (TeamStats result set), one row per team per game |
| `data/advanced/team_gamelog_{season}.parquet` | Team game log with computed `days_rest`, `is_b2b`, `is_b2b_second` columns |
| `data/advanced/league_team_stats_{season}.parquet` | Season-level opponent stats from `LeagueDashTeamStats` |

Seasons to cover: same range as the existing Layer 1 data. Determine this by inspecting
what seasons are already present in the Layer 1 feature files rather than hardcoding.

### 2c. Rate limiting pattern

```python
import time
from nba_api.stats.endpoints import BoxScoreAdvancedV2

SLEEP = 0.6  # seconds between API calls — do not reduce

def fetch_with_retry(endpoint_cls, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            time.sleep(SLEEP)
            return endpoint_cls(**kwargs).get_data_frames()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(SLEEP * (attempt + 2))
```

---

## 3. New Derived Columns to Compute in the Transform Stage

After the raw fetch files exist, compute the following columns in the existing transform
stage (wherever decay stats are currently calculated). Add these computations **before**
the decay loop so they are available as inputs.

### 3a. Usage rate (per player per game)

```
usage_rate = USG_PCT  # already a rate from BoxScoreAdvancedV2, no additional math needed
```

### 3b. Within-team usage share (per player per game)

This is the key within-team feature for negative correlation. For each game and each team:

```
team_total_usg = sum of USG_PCT for all players on that team in that game
usage_share = player_USG_PCT / team_total_usg
```

`usage_share` ranges from 0 to 1 and represents that player's fraction of the team's
total usage — the star player will be near 0.3–0.4, deep bench near 0.02–0.05.

### 3c. Implied game total

```
implied_total = team_OFF_RATING + opponent_DEF_RATING
```

Where `team_OFF_RATING` and `opponent_DEF_RATING` both come from `BoxScoreAdvancedV2`
TeamStats for the two teams in the game. This is a per-game scalar shared by all players
in the game and belongs in `X_team`.

### 3d. Rest days and back-to-back (per team per game)

Compute from `team_gamelog_{season}.parquet` as described in Section 2a. Join to
player-level rows on `(GAME_ID, TEAM_ID)`.

---

## 4. Decay Application for New Features

The existing pipeline computes exponential decay averages with a uniform `β=0.99`. The
new features should use **stat-specific β values** as follows. Update the decay
computation loop to accept a `beta` parameter per column rather than using a single global
value.

| Column | Recommended β | Rationale |
|---|---|---|
| `h_numMinutes` | 0.97 | Highly volatile; role changes should propagate fast |
| `h_points`, `h_assists` | 0.99 | Moderate stability |
| `h_threePointersMade`, `h_fieldGoalsMade` (percentages) | 0.995 | Shooting % stabilizes slowly |
| `h_threePointersAttempted` | 0.98 | Attempt rate stabilizes faster than % |
| `h_rebounds` (all) | 0.99 | Moderate |
| `h_steals` | 0.995 | Highly variable, slow to stabilize |
| `h_blocks` | 0.995 | Highly variable |
| `h_turnovers` | 0.99 | Moderate |
| `h_foulsPersonal` | 0.99 | Moderate |
| **New:** `h_usage_rate` | 0.97 | Tracks role; treat like minutes |
| **New:** `h_usage_share` | 0.97 | Tracks role; treat like minutes |
| **New:** `h_pace` (team) | 0.98 | Moderate; team system changes faster than player skills |
| **New:** `h_off_rating` (team) | 0.99 | Moderate |
| **New:** `h_def_rating` (opponent) | 0.99 | Moderate |

The decay formula is unchanged: `h_stat = β * h_stat_prev + (1 - β) * current_stat`.
Only the per-column β values change.

---

## 5. New Feature Columns and Their Routing

After computing decays, the following new columns must exist in the pre-computed feature
file(s) and be routed as described.

### Goes into `X_players` (per-player features, shape `[num_games, 16, *]`)

| Column | Type | Description |
|---|---|---|
| `h_usage_rate` | float | Decay-averaged usage rate |
| `h_usage_share` | float | Decay-averaged within-team usage fraction |
| `cov_pts_ast` | float | See Section 6 |
| `cov_pts_reb` | float | See Section 6 |

### Goes into `X_team` (game encoder input, shape `[num_games, 192+]`)

| Column | Type | Description |
|---|---|---|
| `h_pace_home` | float | Home team decay-averaged pace |
| `h_pace_away` | float | Away team decay-averaged pace |
| `h_implied_total` | float | `team_OFF_RATING + opp_DEF_RATING` decay-averaged |
| `h_off_rating_home` | float | Home team decay-averaged offensive rating |
| `h_def_rating_home` | float | Home team decay-averaged defensive rating |
| `h_off_rating_away` | float | Away team decay-averaged offensive rating |
| `h_def_rating_away` | float | Away team decay-averaged defensive rating |
| `days_rest_home` | int | Home team days since last game (clip at 7; fill first game with 7) |
| `days_rest_away` | int | Away team days since last game (clip at 7; fill first game with 7) |
| `is_b2b_home` | int (0/1) | Home team on back-to-back |
| `is_b2b_away` | int (0/1) | Away team on back-to-back |

**Important:** `days_rest_home` / `days_rest_away` are **not** decay-averaged — they are
point-in-time values for each specific game. Do not apply decay to them.

### Remove from feature set

| Column | Reason |
|---|---|
| `h_blocks` | Adds no predictive signal once opponent FG data is present (per RAPTOR). Removes double-counting of rim protection with `h_fieldGoalsMade`. |

---

## 6. Cross-Stat Covariance Features

These capture each player's *role archetype* and are computed over a rolling window of
the player's recent games (suggested window: 20 most recent games, or all games in the
current season if fewer than 20).

For each player, compute:

```
cov_pts_ast = Pearson correlation(points, assists) over rolling window
cov_pts_reb = Pearson correlation(points, reboundsTotal) over rolling window
```

Use the **raw per-game stats** (not the decay averages) as inputs to the correlation.
Set `cov_pts_ast = 0` and `cov_pts_reb = 0` when fewer than 5 games are available in
the window (insufficient data for a meaningful correlation).

These values are **not** themselves decay-averaged — they are computed fresh for each
game using the lookback window. Store them alongside the decay columns in the per-player
feature file.

---

## 7. Join Keys and Data Flow

The join key throughout is `(GAME_ID, PLAYER_ID)` for player-level data and
`(GAME_ID, TEAM_ID)` for team-level data. These keys are already present in Layer 1
box score data and in all `nba_api` outputs described above.

Recommended enrichment order in the transform stage:

1. Load Layer 1 box scores for a player.
2. Join `boxscore_advanced_{season}.parquet` on `(GAME_ID, PLAYER_ID)` → adds
   `USG_PCT`, `OFF_RATING`, `DEF_RATING`, `PACE` per game.
3. Join `team_gamelog_{season}.parquet` on `(GAME_ID, TEAM_ID)` → adds `days_rest`,
   `is_b2b_second`.
4. Compute `usage_share` using team-level sum of `USG_PCT` per game.
5. Compute `implied_total` using team and opponent `OFF_RATING` / `DEF_RATING`.
6. Apply stat-specific decay to all columns (Section 4).
7. Compute rolling cross-stat covariances (Section 6).
8. Write enriched feature file to disk in the same format and location as existing
   Layer 1 feature files, with new columns appended.

---

## 8. Validation Checks to Add

After running the enriched feature computation, assert the following before proceeding
to training:

```python
# No NaN in key new columns (except days_rest which is NaN for first game)
assert df['h_usage_rate'].notna().mean() > 0.95
assert df['h_pace_home'].notna().mean() > 0.95
assert df['h_usage_share'].notna().mean() > 0.95

# usage_share should be bounded (0, 1) — values outside this indicate join errors
assert df['h_usage_share'].between(0, 1).all(), "usage_share out of bounds"

# pace should be in a plausible NBA range
assert df['h_pace_home'].between(85, 115).all(), "pace values implausible"

# days_rest clipped at 7
assert df['days_rest_home'].dropna().between(0, 7).all()
```

---

## 9. Summary of Files to Create or Modify

| File | Action | Description |
|---|---|---|
| `ingest/fetch_advanced_stats.py` | **Create** | Fetches all `nba_api` data to disk |
| `transform/compute_features.py` (or equivalent) | **Modify** | Add new decay columns, usage_share, implied_total, cross-stat covariances; switch to stat-specific β; remove `h_blocks` |
| `data/advanced/` | **Create dir** | Storage for raw `nba_api` output parquets |
| `requirements.txt` | **Modify** | Add `nba_api>=1.11.4` |

The model code, dataloader, and training loop should require **no changes** as long as
the feature file schema is extended consistently and the tensor construction code reads
column names dynamically rather than by positional index.

---

## 10. Notes for Claude Code

- Look at the existing transform stage to find where the decay loop is implemented.
  The stat-specific β change (Section 4) must be made there.
- Look at how `X_team` is currently assembled from the pooled roster rows. The new
  game-environment scalars (`h_pace_home`, `h_implied_total`, `days_rest_home`, etc.)
  should be appended to that vector — check the current shape `[num_games, 192]` to
  understand how concatenation is done and extend it accordingly.
- Look at how `X_players` is assembled (currently `[num_games, 16, 17]`) and append
  the new per-player columns (`h_usage_rate`, `h_usage_share`, `cov_pts_ast`,
  `cov_pts_reb`) to the last dimension.
- Do **not** change target columns `Y` (residuals for points, assists, reboundsTotal).
- The fetch script (`ingest/fetch_advanced_stats.py`) should be runnable standalone
  (`python ingest/fetch_advanced_stats.py --seasons 2018-19 2019-20 ...`) and should
  print progress per season and per game batch so failures are easy to spot.
- Handle the case where `BoxScoreAdvancedV2` returns empty data for a game (rare but
  happens for postponed/replayed games) by logging a warning and continuing.
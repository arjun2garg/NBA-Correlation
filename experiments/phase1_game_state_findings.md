# Phase 1 Validation: Game State Signal Findings

Date: 2026-03-26

## Setup

- Data: `input_data_2019-26.csv` + `target_data_2019-26.csv` (170,772 player-games, 8,051 games)
- Temporal split: 80/20 (train: 6,444 games, val: 1,607 games)
- Models: Logistic Regression + Gradient Boosted Trees (GBT, 200 trees, depth 4)
- Task: predict `actual_stat > h_stat` (over/under own historical average)

## Game State Features Used (G vector, from actual post-game data)

From TeamStatistics.csv (both home and away teams):
- `actual_pace`: possessions per 48 min
- `actual_3pa_rate`: 3PA / FGA
- `actual_ast_rate`: assists / FGM
- `actual_to_rate`: turnovers / possessions
- `actual_margin`: teamScore - opponentScore
- `actual_oreb_rate`: ORB / (FGA - FGM)

From PlayByPlay.parquet (coverage: 8,041/8,051 games):
- `actual_rim_rate`: FGA at rim / total FGA
- `actual_midrange_rate`: midrange FGA / total FGA
- `actual_assisted_rate`: assisted FGM / total FGM

## AUC Results

| Stat | GS-only GBT | Player-only GBT | Combined GBT | Delta (GS adds) |
|---|---|---|---|---|
| Points | 0.5412 | 0.5613 | 0.6041 | **+0.0428** |
| Assists | **0.5568** | 0.5912 | 0.6267 | **+0.0355** |
| Rebounds | 0.5467 | 0.5640 | 0.6069 | **+0.0429** |

Target threshold (GS alone > 0.55): **Assists passes (0.5568), Points/Rebounds close (0.54)**

## Key Findings

### 1. Game State Signal Is Real
Adding game state to player features provides +0.035 to +0.043 AUC across all stats.
The combined model reaches 0.60-0.63 AUC — meaningful discriminative power.

### 2. Most Important Game State Features

**Points:**
- `actual_margin` (0.087) — blowout proxy, starters get pulled early
- `actual_pace` (0.030) — more possessions = more scoring opportunities
- `actual_ast_rate` (0.029) — ball movement style
- `actual_to_rate` (0.027) — chaos factor

**Assists:**
- `away_actual_ast_rate` (0.098), `home_actual_ast_rate` (0.082) — most important by far
- `actual_margin` (0.033–0.034) — blowout factor
- `actual_to_rate` and `actual_pace` also relevant

**Rebounds:**
- `actual_oreb_rate` (0.049–0.075) — most important
- `actual_margin` (0.060–0.065)
- `actual_pace`, `actual_to_rate`

### 3. PBP Features Add Marginal Incremental Signal

With PBP features (rim_rate, midrange_rate, assisted_rate):
- Game-state-only AUC: +0.0013–0.0021 improvement vs without PBP
- `actual_rim_rate` and `actual_midrange_rate` did NOT appear in any top-15 feature importance list
- `actual_assisted_rate` also absent — its signal is captured by `actual_ast_rate` in TS
- **Conclusion: TeamStatistics features are sufficient; PBP adds complexity without much gain**

### 4. Critical Insight: Margin Is a Game Outcome, Not an Input

`actual_margin` is the most important feature (game score difference), but it's a post-game outcome.
At inference time, we won't know the margin. This is fine for Phase 1 validation.
For Phase 2 (VAE), the encoder must **predict** the expected margin from pre-game features
(team win rates, strength of schedule, etc.). The uncertainty in this prediction drives z variance.

This is actually ideal: all players share the same game outcome (same margin, same pace, same
ast_rate), so z uncertainty = "what kind of game will this be?" → correlated variation across all players.

## G Vector for Phase 2 (Recommended)

Use only TeamStatistics features (always available, no PBP complexity):

**12-dimensional G per game** (6 features × 2 teams):
```
home_actual_pace, home_actual_3pa_rate, home_actual_ast_rate,
home_actual_to_rate, home_actual_margin, home_actual_oreb_rate,
away_actual_pace, away_actual_3pa_rate, away_actual_ast_rate,
away_actual_to_rate, away_actual_margin, away_actual_oreb_rate
```

Note: home_actual_margin = -away_actual_margin, so this is redundant by 1 dim.
Could reduce to 11 dims, but keeping both is fine for the VAE.

## Implications for Phase 2 VAE Design

The signal exists: game state features G explain ~4% AUC beyond player history.
The VAE must:
1. Learn to predict G from pre-game X_team features (encoder → z → G_pred head)
2. Use G-shaped z in the decoder to predict outcomes

If the encoder can achieve R² > 0.2 on predicting G from X_team, the z variance will be
meaningful enough to produce P(over|z) std well above the current ±2-4% failure mode.

Key question to investigate next: How well can pre-game X_team features predict G?
(i.e., what's the R² of X_team → actual_pace, actual_ast_rate, etc.?)

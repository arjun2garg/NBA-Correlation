# Game State Encoding Plan

## Problem

The VAE decoder ignores z because player features (h_points, h_assists, etc.) already
give a good enough prediction of the residual without needing game context. z becomes a
free parameter the decoder learns to ignore.

All overnight experiments confirmed this: P(over|z) std stuck at 0.015–0.024 regardless
of architecture (attention, two-stage, MI). The signal exists (same-team trace correlation
0.49–0.55) but the absolute scale is too small to produce phi > 0.15.

## Root Cause

The encoder and decoder share redundant features (both see historical player stats). The
decoder shortcuts through player features rather than routing through z. Even with better
encoder features, the residual target `actual - h_stat` has high sigma_pred (~0.94
normalized) because h_stat is an imprecise line.

## Proposed Approach: Two-Phase Game State Encoding

### Phase 1: Validate the Signal (Supervised, No VAE)

Compute a rich game state vector G from **actual in-game data** (post-game, no leakage
concern in this phase). Train a simple supervised model:

```
G (game state features) + player identity features → P(player goes over h_stat)
```

Measure AUC. If game state features contribute incremental AUC above player features
alone, the signal is real and worth building on.

Key validation metric: AUC of game state features alone (no player identity). Target > 0.55.
Incremental AUC when adding game state to player features. This delta is what z needs to capture.

### Phase 2: VAE Learns to Approximate Game State

Redesign the VAE so:

- **Encoder** → z ≈ G (trained to approximate the game state from pre-game features)
- **Decoder** → P(over) using z + player identity features

The decoder is essentially Phase 1's model with z substituted for actual G. Because Phase
1 validated that G predicts outcomes, the decoder has real signal and cannot collapse.

At inference, the encoder's uncertainty in predicting G from pre-game features becomes
posterior variance in z. Since all players in a game share the same z, sampling z multiple
times produces **correlated variation across players** — which is exactly what phi needs.

This is a teacher-student setup: Phase 1 is the teacher (has actual G), Phase 2 trains the
student (predicts G from pre-game context) and inherits the teacher's decoder.

---

## Game State Features to Compute

### From TeamStatistics.csv (available for all seasons)

| Feature | Formula | Why it matters |
|---|---|---|
| `actual_pace` | poss × 48 / game_min | Scales everyone's counting stats up/down |
| `actual_3pa_rate` | team_3PA / team_FGA | Guards/wings vs. bigs game — affects scoring distribution |
| `actual_ast_rate` | assists / FGM | Ball movement — directly predicts assist residuals |
| `actual_to_rate` | turnovers / poss | Chaos factor — reduces counting stats for everyone |
| `actual_fastbreak_rate` | fastBreakPts / teamScore | Transition game — affects role players and minutes |
| `actual_bench_share` | benchPts / teamScore | High bench share → starters played fewer minutes |
| `actual_margin` | teamScore - opponentScore | Blowout proxy — starters get pulled in big leads/deficits |
| `actual_paint_rate` | pointsInThePaint / teamScore | Big-man vs. perimeter game |
| `actual_oreb_rate` | reboundsOffensive / (FGA - FGM) | Second chance possessions |

Compute separately for home and away team. The game state vector G is the concatenation
of both teams' stats (so G captures the interaction, not just one team's view).

### From PlayByPlay.parquet (higher effort, likely higher signal)

| Feature | What it captures |
|---|---|
| `transition_rate` | Fraction of possessions in transition vs. halfcourt |
| `avg_seconds_per_possession` | How much of the shot clock teams used on average |
| `assisted_fg_rate` | Fraction of FGM that were assisted (team ball movement vs. ISO) |
| `rim_shot_rate` | Fraction of FGA at rim (~4 ft) — paint-dominant game |
| `midrange_rate` | Fraction of FGA from midrange — often signals ISO-heavy game |
| `putback_rate` | Offensive rebound → immediate attempt rate |

---

## Implementation Steps

1. **Compute game state features** — new file `src/data/game_state.py`
   - TeamStatistics features: straightforward aggregation, one row per team per game
   - PBP features: extend `src/data/pbp_features.py` with shot location, clock, transition logic

2. **Phase 1 validation** — new script `scripts/validate_game_state.py`
   - Merge game state features with player over/under labels
   - Train logistic regression and gradient boosted trees
   - Report: AUC with game state only, AUC with player features only, AUC combined
   - Report: feature importances (which game state features matter most)

3. **Redesign encoder features** — modify `src/data/dataset.py`
   - Remove per-player stat aggregates from X_team (they're already in X_players)
   - Replace with matchup interaction features: pace mismatch, net off/def advantage,
     win% differential, style clash (3PA rate diff), etc.
   - X_team becomes a compact game-context vector, not a team stat aggregate

4. **Phase 2 VAE** — modify training to anchor z to game state G
   - Auxiliary loss: `MSE(z_decoded_to_G, actual_G)` during training
   - Decoder conditioned on z that has been shaped to represent game state
   - Evaluate by checking if P(over|z) std rises above 0.05 threshold

---

## Why This Solves the Collapse Problem

Current failure mode: decoder sees player features → predicts player history → ignores z.

With this approach:
- Phase 1 proves G predicts outcomes that player features alone cannot
- Decoder is trained on G, which has demonstrated value → cannot ignore it
- z ≈ G during training → z cannot be ignored without hurting the auxiliary loss
- At inference, z uncertainty = "what game state will this be?" → samples correlated across all players

The shared z across all 16 players per game is the correlation mechanism. Game state
uncertainty (fast game? slow game? blowout?) shifts all players together, producing the
joint distribution structure needed for correlated parlays.

---

## Open Questions

- How much of G is predictable from pre-game features? (determines posterior variance in z)
- Which game state dimensions matter most for player over/under? (Phase 1 validation answers this)
- Does PBP signal justify the extraction complexity, or do TeamStatistics features suffice?

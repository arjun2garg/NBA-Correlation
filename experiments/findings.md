# NBA Correlated Parlays — Project Findings

**Goal:** Build a model to find positive-EV correlated parlays on NBA player props
(points, assists, rebounds, and other box score stats) by learning inter-player
dependencies through a shared latent variable.

**Conclusion:** No exploitable correlation signal exists in NBA box score stats after
conditioning on each player's own historical baseline. The signal is in the data —
no architecture change can conjure it.

---

## The Core Idea

A Variational Autoencoder (VAE) with a shared latent variable `z` encodes game-level
context (pace, matchups, rest, etc.) from pre-game team features. All player stat
predictions in a game are conditioned on the same sampled `z`, so correlations between
players emerge through shared game context.

**Simulation pipeline:** Sample `z` many times → decode to player stats → compute
`P(over | z)` per player per sample → derive pairwise phi (Matthews correlation)
across players → bet on pairs with high `|phi|`.

**Over/under threshold:** Each player's line is their own exponential-decay historical
average (`h_stat`). Target is the residual `actual - h_stat`. "Over" = player outperforms
their own history.

---

## Experiment Progression

### Phase 1: Baseline VAE

**Architecture:** Game encoder (`X_team → mu, logvar`), player decoder
(`z + X_players → mu_pred, logvar_pred`). Probabilistic decoder trained with Gaussian
NLL to avoid the over-prediction bias a deterministic decoder produces.

**Key fix found:** Normalizing all features (`X_team`, `X_players`, `Y`) was critical.
Without it, the `home` flag was drowned out by large-scale features, and the decoder
predicted "over" ~70% of the time vs. ~50% actual.

**Result:**
- Val recon: 0.465, KL/dim: 0.31
- `P(over|z)` std across z samples: **0.024** (target for phi > 0.15: > 0.05)
- Phi mean: 0.0007, phi std: 0.0011 — 0 pairs above threshold
- Decoder learned to mostly ignore `z`; `BETA=0.001` makes KL contribute only ~0.016 to
  total loss so the decoder has no incentive to attend to `z`

---

### Phase 2: Architecture & Regularization Sweep

Tested increasing `BETA`, FiLM conditioning, self-attention decoder, two-stage VAE,
and mutual information (MI) variance loss. All experiments on 7 seasons of data
(~170k player-game rows, 1,579 val games).

| Experiment | P(over\|z) std | Notes |
|---|---|---|
| Baseline (BETA=0.001) | 0.024 | Starting point |
| BETA=0.01 | 0.020 | Worse — higher KL pressure collapses mu_z |
| BETA=0.1 | 0.018 | KL/dim drops to 0.12; z pushed toward N(0,I) |
| BETA=1.0 | 0.010 | Severe posterior collapse |
| FiLM conditioning | 0.017 | Same-team trace correlation improves (0.92); absolute std unchanged |
| Self-attention decoder + PBP features | 0.019 | Players co-move but magnitude too small |
| Two-stage VAE (game outcome anchoring) | 0.015 | Worst result; game_mse competes with recon |
| MI variance loss (λ=0.3) | 0.020 | NaN beyond λ≈0.3; gradient explosion |

**Key finding from sweeps:**

Increasing `BETA` makes things worse, not better. Higher KL pressure forces the posterior
toward `N(0,I)`, reducing `mu_z` variability across games — z becomes less informative,
not more. The `free_bits=0.5` floor prevents collapse in the loss, but cannot override the
gradient that prefers a flat posterior when `BETA` is large.

**The bottleneck is not architecture:** Same-team `P(over|z)` trace correlations of 0.49–0.92
confirm `z` does co-move players within a game. The problem is magnitude: co-movement at ±2%
absolute scale produces phi ≈ 0.003 regardless of how coherent the co-movement is.

---

### Phase 3: Game State Encoding

Instead of learning `z` end-to-end, factored the problem: train a decoder conditioned on
actual game state `G` (pace, assist rate, margin, etc.), then train an encoder to predict `G`
from pre-game features. This guarantees decoder z-sensitivity by construction.

**Phase 3a — Supervised validation:**

Actual game state `G` adds real AUC lift to predicting whether a player beats their h_stat:

| Stat | Player history alone | + Game state | G adds |
|---|---|---|---|
| Points | 0.561 | 0.604 | +0.043 |
| Assists | 0.591 | 0.627 | +0.036 |
| Rebounds | 0.564 | 0.607 | +0.043 |

Most important G features: `actual_margin` (blowout proxy), `actual_pace`,
`actual_ast_rate`. PBP-derived features (rim rate, midrange rate) added < 0.002 AUC.

Pre-game predictability of G is low (mean R² ≈ 0.054 across dimensions), which is
expected: game outcomes are hard to forecast from history alone.

**Phase 3b — Two-stage model results:**

The game-state-conditioned decoder with encoder trained to predict `G`:
- `sigma_pred` = 0.940 (unchanged from baseline)
- `P(over|G)` std = **0.050** — best result to date, but still below the needed 0.097
- Theoretical phi ceiling: ~0.04; simulation phi mean: 0.0032, 0 bets placed

**Tested 6 G vector variants** (baseline, four-factors, team totals, rich combined,
totals-only, scoring entropy + minutes HHI). All converged to `sigma_pred` ≈ 0.93–0.94.
Adding more game-level aggregate features provides no additional reduction in
player-level outcome noise.

---

### Phase 4: Player-Specific Game State (Best Result)

**Hypothesis:** Game-level `G` is an aggregate. It tells you the game was high-pace but
can't say *which* players benefited. Give the decoder actual per-player minutes and FGA
alongside game totals — player-specific conditioning.

**Architecture:** `G_game` (6-dim team totals) shared across roster +
`G_player` (actual `mins`, `fga`) per slot → decoder → `(mu_pred, logvar_pred)`.

**Result (v7):**

| Metric | Baseline | v7 Player-Specific G | Change |
|---|---|---|---|
| sigma_pred | 0.940 | 0.770 | −18% |
| P(over\|G) std | 0.024 | **0.128** | +5.3× |
| Theoretical phi_max | 0.002 | **0.066** | +33× |
| Simulated phi mean | 0.0007 | 0.0048 | +6.9× |
| Bets above phi=0.15 | 0 | 0 | — |

Minutes + FGA explain ~18% of residual player outcome variance — the largest single
reduction of any experiment. `P(over|G)` std reached 0.128, the first time it
exceeded 0.05. But phi_max = 0.066 < 0.15 threshold.

**Mathematical ceiling:**

```
phi_max = sigma_P^2 / 0.25

sigma_P = 0.128  →  phi_max = 0.066
```

For phi > 0.15, need sigma_P > 0.194. v7 reaches 66% of this. The remaining
`sigma_pred = 0.770` is player-level noise unexplained by any available feature.

---

### Phase 5: Correlation vs. Individual Signal Isolation

With oracle `G` (actual mins + FGA known at prediction time), the v7 model shows
strong oracle ROI. Three backtest strategies were run to determine whether this
comes from inter-player correlation or individual player predictability.

| Strategy | ROI | Win% |
|---|---|---|
| Proportional (all 4 outcomes by sim prob) | +37.5% | — |
| Top-1 direction only | +86.3% | — |
| Phi-based / pure covariance | −8.8% | 50.0% |
| Balanced lines (forces P(over)=50% per player) | −9.2% | 49.8% |

**The balanced-line test is definitive:** Setting each player's line to the model's own
median prediction forces individual signal to zero. Win rate drops to 49.8% —
indistinguishable from random. The confusion table (OO/OU/UO/UU) is near-uniform
(21–28% per cell) regardless of predicted direction.

**Conclusion:** The +37–86% oracle ROI comes entirely from individual player
predictability (knowing mins + FGA shifts P(over) from ~50% to 0–99%). There is
zero inter-player correlation signal. Shared game state (pace, margin) explains only
~4% AUC lift over player history and does not create exploitable joint variation
between players.

---

### Phase 6: Alternative Stat Exploration

Tested whether 3PM, blocks, steals, or turnovers show more cross-player correlation
than points/assists/rebounds, using actual pairwise phi computed directly from the data
(no model — pure empirical check).

| Stat | Mean phi | Std | % pairs > 0.15 |
|---|---|---|---|
| points | +0.010 | 0.033 | 0.0% |
| assists | +0.009 | 0.031 | 0.0% |
| reboundsTotal | +0.017 | 0.025 | 0.0% |
| threePointersMade | −0.001 | 0.028 | 0.0% |
| blocks | +0.006 | 0.028 | 0.0% |
| steals | +0.009 | 0.027 | 0.0% |
| turnovers | +0.015 | 0.029 | 0.0% |

Best cross-stat same-team pair: **points↔assists** at mean |phi| = 0.048.

No stat or stat combination shows signal above noise. Blocks and 3PM also have
poorly-calibrated `h_stat` baselines (32.5% and 44% over rates respectively) —
exponential decay averages are biased upward for rare/discrete stats.

---

## Root Cause Summary

**The NBA has too much variance.** After conditioning on each player's own historical
average, residual game-to-game variability is large (`sigma_pred ≈ 0.77–0.94`
normalized), and the fraction of that variability explained by shared game context is
small (~4% AUC lift). This is a data fact, not a modeling failure.

The math makes the barrier precise:

```
phi_max = Var(P(over|G)) / 0.25

Current best: Var(P(over|G))^0.5 = 0.128  →  phi_max = 0.066
Required:     Var(P(over|G))^0.5 = 0.194  →  phi_max = 0.150
```

To close the gap, shared game context must explain a larger fraction of individual
player outcome variance. With box score features alone, this ceiling has been reached.

---

## What Would Actually Work

### 1. Real sportsbook lines (highest impact)

Sharp books embed game context, injury information, and public money into their lines.
Using actual over/under lines instead of `h_stat` proxies would:
- Reduce `sigma_pred` from ~0.94 to ~0.50–0.60 (lines already encode the predictable part)
- The same game-state signal that currently produces P(over|G) std = 0.128 would
  likely reach std ≈ 0.20+, putting phi within range
- Individual player edges against sharp lines are rare; this path requires an information
  advantage (e.g. injury news before markets adjust)

### 2. Lines as the actual edge, not correlation

The balanced-line backtest showed that correlation alone produces −9% ROI.
Correlation only amplifies existing individual edges. The correct framing is:
find legs where the line is mispriced, then check if correlated legs exist to
build a parlay. Correlation is a multiplier, not the primary source of edge.

### 3. Day-of lineup and injury shocks

A starter sitting out is a discrete shared event that re-allocates minutes across
teammates. This creates genuine cross-player correlation (everyone else's usage goes up)
but requires access to injury information before lines move.

---

## Technical Artifacts

**Architecture (final state):**
- Encoder: `X_team (256-dim) → 128 → 128 → (mu_z, logvar_z)` with dropout=0.3
- Decoder: `(z + X_players) → 64 → 64 → (mu_pred, logvar_pred)` per player per stat
- Targets: residuals `actual_stat − h_stat`; line = 0 for all players
- Loss: minutes-weighted Gaussian NLL + KL with free_bits=0.5, BETA=0.001 with warmup

**Data:**
- 7 seasons (2019-26), ~170k player-game rows, 8,051 games
- 21 exponentially-decayed player features + 6 advanced features + 3 extra player features
- Top 8 players per team by historical minutes, 16 per game total

**Phi computation:** `P(over|z)` computed analytically via `Φ(mu_pred / sigma_pred)`,
then correlated across 500 z samples. Threshold: `|phi| > 0.15`.

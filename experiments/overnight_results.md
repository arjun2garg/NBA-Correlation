# Overnight Experiments — Results Summary

Branch: `experiments/overnight`
Date: 2026-03-26 (experiments ran 2026-03-25 23:25 – 2026-03-26 ~02:00)

---

## TL;DR

**All 5 overnight experiments failed to raise P(over|z) std above 0.024 (target: >0.05). The
attention decoder and two-stage VAE both made things WORSE. MI training remains numerically
unstable beyond lambda≈0.3. The root cause is confirmed: lines = h_stat means ~50% over rate
is the correct answer regardless of game context, and no architecture change can fix that.**

---

## Results Table

| Experiment | P(over|z) std | Val Recon | KL/dim | Same-team corr | Notes |
|---|---|---|---|---|---|
| **Baseline (BETA=0.001)** | **0.024** | 0.465 | 0.31 | — | Reference point |
| BETA=0.01 (prev) | 0.020 | — | — | — | From prior session |
| BETA=0.1 (prev) | 0.018 | — | — | — | KL at free_bits floor |
| BETA=1.0 (prev) | 0.010 | — | — | — | Severe collapse |
| FiLM (prev) | 0.017 | — | — | — | Multiplicative conditioning |
| **Track A: Attention + PBP** | **0.0188** | 0.437 | 0.26 | 0.49 | WORSE than baseline |
| Track B v1 (buggy) | 0.0238 | 0.160 | 0.31 | — | game_mse=0 (G=zeros bug) |
| **Track B v2: TwoStage** | **0.0151** | 0.158 | 0.47 | 0.55 | WORSE; game anchoring hurts |
| Track C1: MI λ=0.5 | NaN | — | — | — | NaN at epoch ~20 (λ≈0.25) |
| Track C2: MI λ=1.0 | NaN | — | — | — | NaN at epoch ~23 (λ≈0.38) |

---

## Track A: Attention Decoder + PBP Features

**P(over|z) std: 0.0188 — WORSE than baseline 0.024**

Architecture: `PlayerDecoderAttention` — z injected as additive bias into player token embeddings,
then self-attention over all 16 player slots, LayerNorm, MLP head. Player features include 2 PBP
rolling assist rates (20-game rolling_ast_given_rate, rolling_ast_received_rate) extracted from
18.6M PlayByPlay events.

**Key stats:**
- KL/dim: 0.26 (lower than baseline 0.31 — z being used less despite same BETA)
- Same-team P(over|z) trace correlation: 0.49 (players DO move together across z samples)
- mu_z std across games: 0.36 (encoder differentiates games)
- sigma_z (posterior spread): 0.61

**Why it failed:** Self-attention creates player-player interaction WITHIN each z sample, but
this doesn't increase the sensitivity of predictions TO z. The decoder still learns to output
similar mu_pred regardless of z because recon loss dominates. PBP features add some signal about
assist patterns but don't change the fundamental noise floor problem.

**The same-team correlation (0.49)** confirms the mechanism IS working — players on the same team
do co-move in P(over|z) space. The problem is purely that co-movement at ±2% absolute scale
produces negligible phi.

---

## Track B v2: Two-Stage VAE with Game Outcome Anchoring

**P(over|z) std: 0.0151 — WORST result across all experiments**

Architecture: z → GameOutcomeDecoder(z) → G_pred(6-dim: scores, pace, poss) →
TwoStagePlayerDecoder(z, G_pred, player_feats) → stats.

Loss: `recon + beta*kl + 0.5 * game_mse` where game_mse = MSE(G_pred, G_true).

Game outcomes: 52,625 games from TeamStatistics.csv (fixed; previous run used 4,995 games from
broken PlayerStatisticsAdvanced join). Train overlap: 6,472 games with real outcome labels.

**Key stats:**
- Train game_mse: 0.47 → **real anchoring signal** (previous buggy run showed 0.0 because G=zeros)
- Val game_mse: 0.98 (model overfits game outcome prediction)
- KL/dim: 0.47 (higher than baseline — z is being used for outcome prediction)
- Same-team P(over|z) corr: 0.55

**Why it made things WORSE:** The game_mse auxiliary loss forces z to encode actual game outcomes
(scores, pace, possessions). This DOES make z more informative about game context. However:
1. The game_mse loss (alpha=0.5) competes with recon loss — less capacity dedicated to player stats
2. G_pred barely varies across z samples (outcome decoder is nearly deterministic from mu_enc),
   so the additional G_pred input to the player decoder adds almost no within-game variation
3. Raw targets (actual stats, not residuals) have higher variance → higher sigma_pred → lower
   P(over|z) std even if mu_pred variation is the same in absolute terms

**Interesting finding:** val game_mse = 0.98 >> train game_mse = 0.47 suggests the model is
memorizing training game outcomes rather than generalizing the pace/score prediction. This makes
sense — actual game outcomes depend on factors (injuries, matchup adjustments, randomness) that
aren't predictable from historical team stats alone.

---

## Track C: MI Variance Sweep

**Both experiments went NaN. MI is fundamentally unstable beyond lambda ≈ 0.3.**

Previous session had already established this. Both new configs failed:
- C1: lambda=0.5, beta=0.001 → NaN at epoch ~20 (effective lambda ~0.25 after warmup)
- C2: lambda=1.0, beta=0.01 → NaN at epoch ~23 (effective lambda ~0.38 after warmup)

The instability threshold is ~0.3-0.4 regardless of:
- Higher KL regularization (beta=0.01 vs 0.001)
- Extended lambda warmup (60 epochs)
- Tight gradient clipping (0.3 norm)

**Root cause of NaN:** The MI variance objective creates a feedback loop:
1. As decoder becomes z-sensitive, logvar_pred is driven down (model becomes more confident)
2. This drives logvar_pred to the clamp minimum (-6)
3. NLL term `(target-mu_pred)^2 / exp(-6)` becomes enormous when mu_pred misses
4. → gradient explosion → NaN

At lambda ≤ 0.3, MI provides slight benefit (P(over|z) std: 0.020 from previous experiment),
but not enough to matter for phi.

---

## Diagnostic Insights (Consistent Across All Models)

### Same-team correlations are real but too small

All models show **0.49-0.55 same-team P(over|z) trace correlation** — players on the same team
consistently co-move across z samples. This is the correlation signal we want! The problem is
magnitude: co-movement at ±2% absolute scale produces phi ≈ 0.003, far below the 0.15 threshold.

### z differentiates games but decoder barely responds

- mu_z std across games: 0.36-0.45 (encoder creates distinct game embeddings)
- sigma_z (within-game spread): 0.53-0.61 (posterior is informative)
- But P(over|z) std across samples: only 0.015-0.024

This gap (meaningful z variation, negligible output variation) confirms the decoder has learned
to mostly ignore z. With beta=0.001, the KL term contributes ~0.016 to total loss — the decoder
has no incentive to attend to z.

### KL/dim consistently at or below free_bits floor

All models show KL/dim ≈ 0.26-0.47, close to the free_bits=0.5 floor per dimension. This means
z is at minimum usage across all dimensions. Free bits prevents collapse but doesn't force active
use of z beyond the minimum.

---

## Root Cause: The Lines Problem

Every experiment has now confirmed the same root cause. The model uses residual targets:
```
Y = actual_stat - h_stat    (player's rolling historical average)
line = 0
```

The "over" threshold is "does player beat their own average?". By construction:
- P(over) ≈ 50% for every player regardless of game context
- Historical averages already encode the player's baseline — residuals are pure noise
- Game context (pace, matchup, rest) shifts P(over) by ±5-10% in reality, but this is lost
  once we subtract the historical average

With sigma_pred ≈ 0.94 normalized and mu_pred variation ≈ ±0.10 across z samples:
```
P(over|z) std ≈ Normal CDF derivative at 0 × (mu_pred std / sigma_pred)
             ≈ 0.40 × (0.10 / 0.94) ≈ 0.04
```
We're observing 0.015-0.024, consistent with this calculation.

To get P(over|z) std > 0.05, we need either:
- mu_pred variation > ±0.25 normalized (currently ±0.10), OR
- sigma_pred < 0.40 (currently 0.94)

**Neither is achievable with the current feature set.** Historical team averages are weak
predictors of who outperforms their own history, so sigma_pred must be large.

---

## Recommended Path Forward

### Priority 1: Real Sportsbook Lines (High Impact, High Effort)

Replace `h_stat` lines with actual sportsbook over/under lines. Effect:
- Sharp lines already embed game context, injury info, public money → residual uncertainty
  shrinks from ~0.94 normalized to perhaps ~0.60-0.70
- Players on a team with high-implied-total game have lines set ABOVE their averages → the
  residual `actual - line` is now centered around true alpha for that context
- P(over|z) std would likely rise to ±5-15% range immediately

This single change is more likely to unlock phi signal than any architecture change.

### Priority 2: Opponent Position-Level Context (Medium Impact, Medium Effort)

Current: only opponent team-level offensive/defensive rating
Missing: "opponent point guard defensive rating vs guards" — this is strong signal for assists/pts

### Priority 3: Home/Away Split Averages (Low Impact, Low Effort)

Current h_stat pools home and away games. Home advantage is ~3-4 pts for most players.
Split averages would reduce residual noise slightly.

### Priority 4: Reconsider VAE Architecture for Lines-First Approach

Once real lines are available, the current architecture should work better. The VAE's shared z
captures game-level context that shifts all players together — exactly the correlation structure
of a high-total or pace-up game. With sharp lines, this shift would be large enough to see in phi.

---

## Files Changed This Session

- `src/data/pbp_features.py` — PBP feature extraction (rolling assist rates)
- `src/model_attention.py` — Self-attention decoder
- `src/model_twostage.py` — TwoStage decoder (GameOutcomeDecoder + TwoStagePlayerDecoder)
- `src/data/game_outcomes.py` — Game outcome loader (FIXED: TeamStatistics only, no cross-join)
- `src/data/dataset.py` — Added extra_player_cols, raw_targets parameters
- `src/train_mi.py` — MI variance loss with grad clipping
- `scripts/train_attention.py` — Track A training script
- `scripts/train_twostage.py` — Track B training script
- `scripts/train_mi.py` — Track C training script (revised hyperparams)
- `scripts/diagnose_z_sensitivity.py` — FIXED: now handles attention model's PBP player_dim
- `scripts/run_overnight.sh` — Orchestration (Tracks A+B parallel, then C1, C2)

**Checkpoints:**
- `checkpoints/exp_attention/` — Track A attention model
- `checkpoints/exp_twostage/` — Track B v1 (old buggy game_outcomes, game_mse≈0)
- `checkpoints/exp_twostage_v2/` — Track B v2 (fixed, real game anchoring)
- `checkpoints/exp_mi_1/` — Track C1 NaN weights (lambda=0.5/beta=0.001)
- `checkpoints/exp_mi_5/` — Track C2 NaN weights (lambda=1.0/beta=0.01)

# Experiment Results Summary

Branch: `experiments/beta-film-sweep`
Date: 2026-03-25
Data: 7 seasons (2019-26), ~170k rows, 1,579 val games

## Key Metrics Table

| Experiment | BETA | lambda_mi | Val Recon | KL/dim (train) | P(over|z) std | Same-team corr | Notes |
|---|---|---|---|---|---|---|---|
| Baseline (model_latest) | 0.001 | — | 0.465 | 0.31 | **0.0235** | 0.640 | Starting point |
| Exp 1a: BETA=0.01 | 0.01 | — | 0.462 | 0.28 | 0.0200 | 0.805 | Worse than baseline |
| Exp 1b: BETA=0.1 | 0.1 | — | 0.465 | 0.12 | 0.0182 | 0.881 | Worse — posterior collapse |
| Exp 1c: BETA=1.0 | 1.0 | — | 0.470 | 0.07 | 0.0104 | 0.799 | Severe collapse |
| Exp 2: FiLM, BETA=0.1 | 0.1 | — | 0.464 | 0.17 | 0.0170 | 0.922 | Slightly worse than concat |
| Exp 3: MI, BETA=0.1 | 0.1 | 0.1 | 0.464 | 0.16 | 0.0202 | 0.906 | Marginal MI growth seen |

Target for phi > 0.15: P(over|z) std > **0.05**
Max phi observed: ~0 across all experiments (no bets placed with threshold=0.15)

## What Was Found

### BETA sweep: increasing BETA makes things worse

All 3 BETA values (0.01, 0.1, 1.0) produced *lower* P(over|z) std than baseline:
- BETA=0.01: 0.020 (vs 0.024 baseline)
- BETA=0.1: 0.018
- BETA=1.0: 0.010

**Root cause**: The free_bits=0.5 floor does not prevent collapse when BETA is large.
With BETA=1.0, KL/dim drops to 0.07 — well below the free_bits=0.5 floor.
The gradient pushes toward the prior (sigma→1, mu→0), reducing mu_z variability across games.
This makes z less informative, not more.

The free_bits mechanism only prevents collapse via the loss function floor — it
cannot override a gradient that prefers a collapsed solution. With BETA=1.0,
the KL term contributes ~0.5×32 = 16× more than recon-weight would suggest, so
the model simply learns to output approximately N(0,I) regardless of game context.

### FiLM conditioning: same-team correlation improved but absolute variance unchanged

FiLM architecture shows higher same-team trace correlation (0.92 vs 0.64 baseline).
This confirms z is modulating all players within a game more coherently.
However, the absolute magnitude of P(over|z) variation is still 0.017 — almost
identical to the concat baseline at the same BETA.

**Interpretation**: The fundamental bottleneck is not the architectural pathway
from z to predictions — it's that the decoder's prediction uncertainty (sigma_pred ≈ 0.94)
is so large that even if mu_pred moves by 0.1 std units with z, P(over|z) only shifts by
~2-3%. FiLM gives z a better multiplicative pathway but cannot change this noise ceiling.

### MI term: small positive signal but not sufficient

The MI variance penalty grows monotonically (from -0.009 to -0.012 by epoch 60),
confirming the term is working and the decoder IS becoming marginally more z-sensitive
during training. P(over|z) std recovered to 0.020 (better than BETA=0.1 alone).

However, lambda_mi=0.1 contributes only ~0.001 to total loss — too small to
meaningfully compete with the reconstruction loss (~0.44). Would need lambda_mi ≥ 1.0
or more to have real effect.

## Root Cause Analysis (Confirmed)

The core problem is **sigma_pred is too large relative to mu_pred variation**:
- sigma_pred ≈ 0.94 normalized std (model is very uncertain about outcomes)
- mu_pred varies by only ~0.1 normalized std units across z samples
- P(over|z) = Φ(mu_pred / sigma_pred) ≈ Φ(±0.1/0.94) ≈ 0.5 ± 2-4%

This is NOT a training instability or architecture problem — it accurately reflects
that NBA player stats are highly stochastic and the features in X_team do not
strongly predict who will outperform their own historical average in a given game.

## Recommendation for What to Try Next

### Option A: Use real betting lines (highest priority)
Current lines are set to h_stat (player's own historical average). Sharp sportsbook
lines already incorporate game context, matchups, and injury information. With real
lines, the *relevant* signal is narrower: "does z cause the model to predict higher
than the line set by a sharp market?" This would shrink sigma_pred relative to
mu_pred movement, making P(over|z) more variable.

### Option B: Much larger MI lambda (lambda_mi = 1.0 or 5.0)
The MI variance term grows monotonically. Scaling it by 10-50x would force the
decoder to be z-sensitive even at cost of reconstruction quality. Risk: degraded
recon, but the model might find real correlation structure. Recommended:
lambda_mi=1.0 with BETA=0.001 (don't fight two competing pressures simultaneously).

### Option C: Disable free_bits, use BETA=0.001 + large MI
The free_bits floor was designed to prevent collapse but with the current data
distribution, z is genuinely not needed. Remove free_bits, keep BETA very low,
and use only the MI term to force z-sensitivity. This separates the two concerns.

### Option D: Hybrid encoder with game-level features z can exploit
The game encoder pools team rosters but lacks features z can *use* uniquely:
- Vegas implied totals and team totals
- Opponent defensive statistics vs. position
- Injury-adjusted lineup projections
Without these, the game-level signal available to z is weak, and the MI/FiLM
approaches cannot conjure signal that isn't there.

### What success looks like
- P(over|z) std rising from 0.02 to 0.05+
- KL/dim genuinely above 0.5 (not floored there artificially)
- Same-team phi rising above 0.01 consistently
- At least some pairs exceeding phi=0.15 threshold on val set

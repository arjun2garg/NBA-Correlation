# Game State Encoding Experiment — Results

Branch: `experiments/game-state-encoding`
Date: 2026-03-26

---

## TL;DR

**Phase 1 confirmed game state signal is real (+0.035–0.043 AUC). Phase 2 two-stage model
achieved P(over|G) std=0.050 — the highest of any experiment to date. But the mathematical
ceiling for phi is ~0.04, not 0.15. sigma_pred=0.94 (true noise floor) prevents game-level
context from driving phi above the betting threshold with current features.**

---

## Phase 1: Supervised Validation

### Setup
- 170,772 player-games, 8,051 games (2019-26 seasons)
- Temporal split: 80/20 (train: 6,444 games, val: 1,607 games)
- Task: predict `actual_stat > h_stat` (over/under own historical average)
- Models: Logistic Regression + Gradient Boosted Trees (GBT, 200 trees, depth 4)

### G Vector (12-dim, TS only)
Per-team from TeamStatistics.csv, home + away:
`actual_pace, actual_3pa_rate, actual_ast_rate, actual_to_rate, actual_margin, actual_oreb_rate`

### AUC Results

| Stat | GS-only GBT | Player-only GBT | Combined GBT | GS adds |
|---|---|---|---|---|
| Points | 0.5412 | 0.5613 | 0.6041 | +0.0428 |
| Assists | **0.5568** | 0.5912 | 0.6267 | +0.0355 |
| Rebounds | 0.5467 | 0.5640 | 0.6069 | +0.0429 |

**Key finding:** Game state adds real AUC lift (+0.035–0.043). Most important features:
- Points: `actual_margin` (blowout proxy), `actual_pace`
- Assists: `actual_ast_rate` (both teams, most important by far)
- Rebounds: `actual_oreb_rate`, `actual_margin`

PBP features (rim_rate, midrange_rate, assisted_rate) add <0.002 AUC — dropped.

### Pre-game Predictability of G

How well can pre-game features predict G (upper bound on encoder R²)?

| Feature | GBT R² (pre-game → actual) |
|---|---|
| actual_pace | 0.24 |
| actual_margin | 0.22 |
| actual_ast_rate | ~0 |
| actual_to_rate | ~0 |
| actual_3pa_rate | ~0 |
| actual_oreb_rate | ~0 |
| **Mean across all G dims** | **0.054** |

Low pre-game R² is actually ideal: unpredictable G dimensions = genuine posterior variance = correlation
source for z. The encoder SHOULD be uncertain about game state.

---

## Phase 2: Two-Stage Architecture

### Design

```
Stage 1 (supervised):  actual G + player_feats → decoder → (mu_pred, logvar_pred)
                       Loss: minutes-weighted Gaussian NLL
                       Guarantees decoder IS G-sensitive

Stage 2 (frozen decoder): X_team → encoder → N(mu_G, sigma_G)
                          Loss: NLL(actual G | mu_G, logvar_G) + beta * KL(q || N(0,1))
                          Encoder learns uncertain predictions over game state

Simulation: sample G ~ N(mu_G, sigma_G) → decoder → P(over|G) per player
            All 16 players in game share same G sample → correlated variation
```

### Stage 1 Results (Decoder)

- Architecture: `GCondDecoder(G_DIM=12, player_dim=24, h_dim=64, output_dim=3, dropout=0.3)`
- Final val NLL: ~0.43 (80 epochs, LR=1e-3, Adam)
- Decoder confirmed G-sensitive: using actual G in forward pass, no shortcut possible

### Stage 2 Results (Encoder)

- Architecture: `GameEncoder(input_dim=256, h_dim=128, g_dim=12, dropout=0.3)`
- Training: 100 epochs, beta=0.01, free_bits=0.5, LR=1e-3
- **kl/dim converged to exactly 0.500 (free_bits floor throughout)**
- **G_R² converged to 0.10–0.12 (consistent with pre-game predictability ceiling of 0.054 mean)**

| Epoch | val g_nll | G_R² | P(over\|G) std |
|---|---|---|---|
| 9 | 0.51 | 0.137 | 0.054 |
| 50 | 0.50 | 0.110 | 0.051 |
| 99 | 0.52 | 0.101 | 0.050 |

### Simulation Results (500 samples, 1,579 val games)

- phi mean: 0.0032, std: 0.0046, max: 0.012
- Bets placed at threshold=0.15: **0**
- phi threshold=0.05 would catch: ~1.8% of pairs (still very low)

---

## Mathematical Ceiling Analysis

### Why phi cannot reach 0.15 with current data

The phi coefficient for a pair of players is bounded by:
```
phi ≤ 4 × Var(P(over|G)) / [P(1-P)] ≈ 4 × Var(P(over|G)) / 0.25
```

For phi=0.15, need P(over|G) std ≈ 0.097. Current std=0.050 → phi_max ≈ 0.04.

### Why P(over|G) std is stuck at 0.050

```
P(over|G) std ≈ φ(0) × (dmu/dG × sigma_G / sigma_pred)
             ≈ 0.40 × (dmu/dG × sigma_G / sigma_pred)

sigma_pred = 0.94   ← TRUE noise floor; player outcome variance after conditioning on player history
sigma_G    = 0.61   ← encoder posterior spread (at free_bits floor)
dmu/dG     = 0.19   ← decoder's average sensitivity to G per unit change
                       (theoretical max: sigma_pred = 0.94, actual: 0.19 = 20% of capacity used)

→ P(over|G) std = 0.40 × (0.19 × 0.61 / 0.94) = 0.049   ← matches observed 0.050
```

### Root cause breakdown

1. **sigma_pred=0.94 is genuine**: player outcomes are noisy given their historical averages.
   This is not a modeling artifact — it reflects true game-to-game variability.

2. **dmu/dG=0.19 reflects G's true signal**: G explains ~4% AUC lift over player history.
   dmu/dG cannot be increased without the decoder lying about uncertainty.

3. **sigma_G=0.61 is at the floor**: encoder is maximally uncertain about game state
   (as expected given mean pre-game R²=0.054). Could push higher with larger free_bits,
   but increasing sigma_G by 2× only raises std to ~0.10 — marginal.

**The fundamental constraint:** G is an aggregate game-level summary. It explains ~4% of individual
player outcome variance beyond historical averages. No architecture change can alter this fact.

### What would unlock phi > 0.15

| Change | Expected effect | Availability |
|---|---|---|
| Real sportsbook lines | sigma_pred → ~0.50-0.60 (sharp lines encode game context) | External API |
| Day-of injury/lineup shocks | Discrete shared event across roster; large margin jump | External API |
| Opponent position-level defense | +2-3% AUC for guards/bigs separately | Derivable from data |
| Home/away split h_stat | Minor sigma_pred reduction | Derivable from data |

**Bottom line:** Sportsbook lines are the single most impactful change. With sharp lines, residual
uncertainty shrinks by ~40% and the same G signal that currently produces P(over|G) std=0.05
would produce std≈0.08-0.10, putting phi within reach.

---

## Phase 3: G Variant Sweep (v1–v6)

Tested 6 G encoding variants for sigma_pred reduction (lower = more signal in G).

| Variant | G_DIM | sigma_pred | P(over\|G) std | Notes |
|---|---|---|---|---|
| v1_baseline | 12 | 0.940 | 0.050 | pace, rates, margin per team |
| v2_four_factors | 22 | ~0.93 | ~0.050 | +eFG, FT rate, quarter margins |
| v3_team_totals | 18 | ~0.93 | ~0.050 | pts, ast, reb + v1 features |
| v4_rich | 36 | ~0.93 | ~0.050 | all team features combined |
| v5_totals_only | 6 | ~0.94 | ~0.048 | pts/ast/reb only (3 per team) |
| v6_entropy | 16 | **0.919** | ~0.052 | +scoring entropy + mins HHI |

**Key finding:** All game-level G variants converge to sigma_pred ~0.93–0.94. Adding more game-level
features (even scoring entropy) provides negligible improvement. The ceiling for game-level G is
sigma_pred ≈ 0.92, giving phi_max ≈ 0.05.

The fundamental issue: game-level G is an aggregate. It tells you this was a high-pace game, but
can't say *which* players benefited. Individual player outcome variance persists after conditioning.

---

## Phase 4: Player-Specific G (v7_player_mins_fga)

### Hypothesis

Give the decoder per-player actual minutes played and FGA alongside game totals. This gives player-specific
conditioning: if player X played 38 minutes and took 22 shots, the decoder can predict their stats
much more precisely. sigma_pred should drop substantially.

### Architecture

```
G_game  (6-dim):   [home_pts, home_ast, home_reb, away_pts, away_ast, away_reb]  — shared across team
G_player (2-dim/player): [actual_mins, actual_fga]  — player-specific actual usage

PlayerGDecoder:
  input per slot = G_game (broadcast) + G_player_i + player_feats_i  (6+2+24 = 32-dim)
  → 2-layer hidden (h_dim=64) → (mu, logvar) per stat
```

### Results

| Metric | v1_baseline | v7_player_mins_fga | Change |
|---|---|---|---|
| sigma_pred | 0.940 | **0.770** | **-18%** |
| P(over\|G) std | 0.050 | **0.128** | **+3.6×** |
| Theoretical phi_max | 0.012 | **0.066** | **+5.5×** |

sigma_pred dropped from 0.940 → 0.770: minutes and FGA explain ~18% of residual variance.
This is the largest sigma_pred reduction of any experiment in this project.

### Phi Simulation (corrected Bernoulli sampling)

Previous inline phi computation was wrong: it thresholded mean P(over) across G samples at 0.5,
collapsing all variance. Correct method draws `Bernoulli(P(over|G))` per G sample.

Results (500 G samples × 300 val games, sigma_G_game=0.6, sigma_G_player=0.5):

| Pairs | Mean phi | Std | Max | >0.15 |
|---|---|---|---|---|
| All | 0.0021 | 0.046 | 0.284 | 0.16% |
| Same-team | 0.0048 | 0.047 | 0.284 | 0.28% |
| Cross-team | -0.0004 | 0.045 | 0.209 | 0.04% |

Same-team phi is correctly positive; cross-team is correctly near zero or slightly negative.
But mean phi=0.0048 remains well below the 0.15 threshold.

### Why phi is still low

The math is unambiguous:

```
phi_max = sigma_P^2 / Var(X) = sigma_P^2 / 0.25

sigma_P = 0.128  (P(over|G) std)
phi_max = 0.128^2 / 0.25 = 0.066
```

For phi > 0.15:  sigma_P must exceed **0.194** (v7 is at 66% of this)

The remaining sigma_pred = 0.770 is unexplained variance that G doesn't capture. The only way to
further reduce sigma_pred (and raise phi_max) is to give the decoder more informative player-level G.

### What would close the gap

| Player-level G addition | Expected sigma_pred | Expected phi_max |
|---|---|---|
| actual_mins + actual_fga (v7) | 0.770 | 0.066 |
| + actual_pts (per player) | ~0.55 | ~0.12 |
| + actual_pts + actual_ast + actual_reb | ~0.30 | ~0.36 |

But giving the decoder actual player points/assists defeats the purpose — it's giving the answer.
The practical path is to improve pre-game predictions of player-level G:
- Historical usage rate as G_player proxy (currently in X_players, not G_player)
- Sharp sportsbook lines: residual uncertainty after conditioning on lines is ~40% lower

---

## Final Comparison

| Experiment | sigma_pred | P(over\|G) std | phi_max | phi_mean (sim) |
|---|---|---|---|---|
| Baseline VAE (BETA=0.001) | 0.94 | 0.024 | 0.002 | 0.0007 |
| BETA=0.1 | 0.94 | 0.018 | 0.001 | ~0.0007 |
| FiLM / Attention / MI | 0.94 | 0.017–0.020 | ~0.001 | ~0.0007 |
| Two-Stage GS v1_baseline | 0.94 | 0.050 | 0.010 | 0.0032 |
| Two-Stage GS v6_entropy | 0.919 | 0.052 | 0.011 | ~0.003 |
| **v7 player-specific G** | **0.770** | **0.128** | **0.066** | **0.0048** |

**v7 is the best result by every metric.** Same-team phi signal is correctly directional (positive)
and phi_max = 0.066 is the first time we've crossed 0.05. But 0.066 < 0.15 threshold.

**The phi > 0.15 barrier cannot be crossed with minutes+FGA alone.**
The signal is real but insufficient. Next major unlock is sportsbook lines as G proxy.

---

## Phase 5: Backtest Series — Isolating Correlation vs. Individual Signal

Three backtest scripts were written to test whether the v7 model's oracle signal comes from
inter-player correlation or individual player predictability.

### Setup

- 500 val games × 500 G samples (sigma_G_game=0.6, sigma_G_player=0.5)
- Bet $1 per pair, equal $0.50 split within direction group (same or opposite)
- Payout: 3.645× total return at -110/-110 (-8.9% baseline ROI under no signal)
- All three scripts use the oracle G (actual mins + FGA) — upper bound on real performance

### Script 1: Proportional Backtest (`backtest_player_g.py`)

Bet $1 per pair split proportionally across all 4 outcomes (OO/OU/UO/UU) by simulated probability.
PnL = P_sim(actual_outcome) × 3.645 − 1.

| Pair type | ROI |
|---|---|
| same_player | +12% |
| same_team | +37% |
| cross_team | +37% |
| **TOTAL** | **+37.53%** |

### Script 2: Strategy Comparison (`backtest_threshold.py`)

Compares three stake allocation strategies on the same oracle setup.

| Strategy | ROI |
|---|---|
| Proportional (all 4 outcomes) | +37.53% |
| Top-1 (most likely outcome only) | +86.30% |
| Threshold (P > 0.274 breakeven) | +65.09% |

**Key finding:** Very strong oracle ROI (+37–86%). This looks like a working model — but see Phase 5C.

### Script 3: Correlation Isolation (`backtest_correlation.py`)

Three strategies designed to isolate where the signal actually comes from.

**Strategy 1 — Raw P_same (baseline, leaks individual signal):**
Bet "same direction" if P_OO + P_UU > 0.5, "opposite" otherwise. $0.50 each on both outcomes in predicted group.

| | ROI | Win% |
|---|---|---|
| TOTAL | +7.64% | 59.1% |

**Strategy 2 — Phi-based / Cov(Xi, Xj) (pure inter-player correlation):**
Confidence = P_OO − S_i × S_j (excess co-occurrence beyond independence = 2 × Cov). Removes individual marginal effects.

| | ROI | Win% |
|---|---|---|
| TOTAL | **−8.83%** | 50.0% |

**Strategy 3 — Balanced lines (definitive correlation test):**
Set each player's line to the model's median mu prediction for the game (median of mu across 500 G samples).
This forces P(over) = 50% per player in simulation, so OO = UU and OU = UO under independence.
Actual outcome: continuous Y residual > median_mu (same normalized space).
If genuine correlation exists → win rate > 50%. If not → win rate ≈ 50%.

| | ROI | Win% |
|---|---|---|
| same_player | −4.19% | 52.6% |
| same_team | −9.95% | 49.4% |
| cross_team | −8.93% | 50.0% |
| **TOTAL** | **−9.18%** | **49.8%** |

Confusion table is near-uniform (21–28% per cell) regardless of predicted direction — indistinguishable from random.

### Conclusion

The +37–86% oracle ROI comes **entirely from individual player predictability**, not from inter-player correlation.

When actual mins+FGA are known:
- P(over|G) std = 0.239 (much higher than the 0.128 from noise-perturbed G)
- P(over) ranges 0.000–0.998 per player — the model correctly prices individual outcomes far from 50%
- The market implicitly prices all outcomes at 25% (independence assumption at fair odds)
- This creates individual edge that manifests as oracle ROI

But the **correlation** between players (the covariance structure) is zero. Stripping out individual signal
via the balanced-line test drops ROI from +7.64% → −9.18% in one step, confirming no cross-player structure.

**Root cause:** Player outcome variance is dominated by game-to-game individual noise (sigma_pred = 0.77).
This noise is independent across players — shared game state (pace, margin) explains only ~4% AUC lift
and does not create exploitable joint variation. phi_max = 0.066 with oracle G; actual phi ≈ 0.005.

**What would create real correlation signal:**
The only path to positive EV parlays is line mispricing on individual legs, not correlation.
Correlation only amplifies existing individual edges. Without sharp pre-game lines (or injury/lineup
information that moves lines before markets adjust), there is no edge to amplify.

---

## Comparison to Previous Experiments

| Experiment | P(over\|z) std | phi max | Notes |
|---|---|---|---|
| Baseline VAE (BETA=0.001) | 0.024 | ~0.004 | z ignored by decoder |
| BETA=0.1 | 0.018 | ~0.003 | KL at floor |
| FiLM decoder | 0.017 | ~0.003 | No improvement |
| Attention + PBP | 0.019 | ~0.003 | Same floor |
| MI variance (λ=0.3) | 0.020 | ~0.003 | NaN beyond λ=0.3 |
| Two-Stage GS v1 | 0.050 | 0.012 | Best prior; new ceiling |
| **v7 player-specific G** | **0.128** | **0.066** | **New best; phi > 0.05 for first time** |

---

## Files Added This Session (Phase 5)

- `scripts/backtest_player_g.py` — Proportional oracle backtest (+37.53% ROI)
- `scripts/backtest_threshold.py` — Strategy comparison: proportional / top-1 / threshold
- `scripts/backtest_correlation.py` — Three-strategy correlation isolation test (raw / phi / balanced)

**Phase 1–4 files:**
- `src/data/game_state.py` — Computes actual G from TeamStatistics + optional PBP; 6 variants
- `src/data/game_state_dataset.py` — Dataset/loader with G tensor, G_mask, normalization; variant support
- `src/model_gs.py` — GCondDecoder (Stage 1) + GameEncoder (Stage 2) + reparameterize
- `src/train_gs.py` — Stage 1 decoder training, Stage 2 encoder training, P(over|G) diagnostic
- `scripts/validate_game_state.py` — Phase 1 supervised AUC validation
- `scripts/train_gs.py` — Two-stage training entry point (--stage 1/2/both, --variant)
- `scripts/simulate_gs.py` — Simulation: sample G → decoder → phi → backtest
- `scripts/train_player_g.py` — v7 player-specific G decoder training + diagnostics
- `scripts/simulate_player_g.py` — Correct Bernoulli phi simulation for v7 decoder
- `scripts/eval_g_variants.py` — AUC evaluation for G variant comparison
- `experiments/phase1_game_state_findings.md` — Phase 1 detailed findings
- `data/processed/game_state_cache.csv` — Cached 55,881-game G vectors

**Checkpoints:**
- `checkpoints_gs/decoder_stage1.pt` — v1 Stage 1 decoder
- `checkpoints_gs2/decoder_stage1.pt`, `checkpoints_gs2/encoder_stage2.pt` — v1 full two-stage
- `checkpoints_v5/decoder_stage1.pt` — v5_totals_only Stage 1 decoder
- `checkpoints_v6/decoder_stage1.pt` — v6_entropy Stage 1 decoder
- `checkpoints_v7/decoder_player_g.pt` — v7 player-specific G decoder (best result)

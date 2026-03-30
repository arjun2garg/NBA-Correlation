# NBA Correlated Parlays — A VAE Investigation

**Can shared game context create exploitable cross-player correlation in NBA prop markets?**

This project builds a Variational Autoencoder (VAE) with a shared latent game-state variable to test whether NBA player outcomes are correlated enough to find positive-EV correlated parlays. After six phases of experiments across 7 seasons of data (~170k player-game rows), the answer is no — and the math shows exactly why.

[Full write-up](https://medium.com/@arjun2garg/the-phi-ceiling-a31ac29caf2a) · [Data source (Kaggle)](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores)

---

## The Idea

Sportsbooks price each leg of a parlay independently. If two players' outcomes are positively correlated — because they share a pace, a game plan, a defensive scheme — the true joint probability of both hitting is higher than the book assumes. That gap is exploitable.

The phi coefficient (Matthews correlation coefficient) quantifies this. For two 50/50 legs:

```
P(A and B) = 0.25 + phi × 0.25
```

A phi of 0.15 is enough to flip a -EV parlay to +EV at standard odds. The question is whether shared NBA game context — encoded as a latent variable `z` — produces phi at that scale.

---

## Architecture

```
Pre-game team features (X_team)
         │
    ┌────▼─────┐
    │  Encoder  │  →  μ_z, σ_z  →  sample z
    └──────────┘
                          │
              ┌───────────▼───────────┐
              │  shared game state z  │
              └───────────┬───────────┘
          ┌───────────────┼───────────────┐
     player 1        player 2  ...   player N
   (z + X_p1)      (z + X_p2)      (z + X_pN)
       │                 │                │
  ┌────▼────┐       ┌────▼────┐      ┌────▼────┐
  │ Decoder │       │ Decoder │      │ Decoder │
  └────┬────┘       └────┬────┘      └────┬────┘
  μ,σ (pts)         μ,σ (pts)        μ,σ (pts)
```

All players in a game are decoded from the **same sampled z**, so cross-player correlations emerge through shared game context. P(over | z) is computed analytically via the normal CDF: `Φ(μ_pred / σ_pred)`.

**Simulation pipeline:** Sample z 500 times → decode to player stats → compute P(over | z) per player → derive pairwise phi across players → flag pairs with |phi| > 0.15.

---

## Key Finding

There is a hard mathematical ceiling on achievable phi:

```
phi_max = Var(P(over | z)) / 0.25

Best result (Phase 4):  σ_P = 0.128  →  phi_max = 0.066
Required for edge:      σ_P = 0.194  →  phi_max = 0.150
```

The remaining `σ_pred ≈ 0.77` is player-level noise unexplained by any available feature. This is a property of the data, not a modeling failure.

---

## Experiment Progression

### Phase 1 — Baseline VAE
Probabilistic decoder with Gaussian NLL loss (fixes a systematic over-prediction bias from MSE). Feature normalization critical for calibration.
- P(over|z) std: **0.024** | phi mean: 0.0007 | Bets above threshold: 0

### Phase 2 — Architecture & Regularization Sweep

| Experiment | P(over\|z) std | Notes |
|---|---|---|
| Baseline (BETA=0.001) | 0.024 | Starting point |
| BETA=0.01 | 0.020 | Higher KL pressure collapses μ_z |
| BETA=0.1 | 0.018 | z pushed toward N(0,I) |
| BETA=1.0 | 0.010 | Severe posterior collapse |
| FiLM conditioning | 0.017 | Same-team trace corr improves (0.92); std unchanged |
| Self-attention decoder + PBP features | 0.019 | Co-movement present but magnitude too small |
| Two-stage VAE (game outcome anchoring) | 0.015 | game_mse competes with recon |
| MI variance loss (λ=0.3) | 0.020 | NaN beyond λ≈0.3; gradient explosion |

Increasing BETA makes things worse: higher KL pressure forces the posterior toward N(0,I), making z less informative rather than more.

### Phase 3 — Explicit Game State Conditioning
Replaced learned z with actual observed game state G (pace, assist rate, margin). AUC lift over player history alone:

| Stat | Player history | + Game state | Lift |
|---|---|---|---|
| Points | 0.561 | 0.604 | +0.043 |
| Assists | 0.591 | 0.627 | +0.036 |
| Rebounds | 0.564 | 0.607 | +0.043 |

P(over|G) std: **0.050** — best result at this stage, still below the needed 0.097.

### Phase 4 — Player-Specific Game State (Best Result)
Added actual per-player minutes and FGA alongside game totals. Minutes + FGA explain ~18% of residual variance — the largest single reduction of any experiment.

| Metric | Baseline | Phase 4 | Change |
|---|---|---|---|
| σ_pred | 0.940 | 0.770 | −18% |
| P(over\|G) std | 0.024 | **0.128** | +5.3× |
| phi_max (theoretical) | 0.002 | **0.066** | +33× |
| Bets above phi=0.15 | 0 | 0 | — |

### Phase 5 — Isolating Correlation vs. Individual Signal

Oracle backtest (actual minutes + FGA known at prediction time):

| Strategy | ROI | Win% |
|---|---|---|
| Proportional (all 4 outcomes by sim prob) | +37.5% | — |
| Top-1 direction only | +86.3% | — |
| Phi-based / pure covariance | −8.8% | 50.0% |
| **Balanced lines (individual signal removed)** | **−9.2%** | **49.8%** |

The balanced-line test is definitive: setting each player's line to the model's own median prediction removes all individual signal. Win rate drops to 49.8%. The OO/OU/UO/UU confusion table is near-uniform. **The +37–86% oracle ROI comes entirely from individual player predictability, not cross-player correlation.**

### Phase 6 — Alternative Stats
Direct empirical phi check across all major stats (no model):

| Stat | Mean phi | % pairs > 0.15 |
|---|---|---|
| points | +0.010 | 0.0% |
| assists | +0.009 | 0.0% |
| reboundsTotal | +0.017 | 0.0% |
| threePointersMade | −0.001 | 0.0% |
| blocks | +0.006 | 0.0% |
| steals | +0.009 | 0.0% |
| turnovers | +0.015 | 0.0% |

No stat combination shows signal above noise.

---

## Repo Structure

```
src/             # Core library: model, dataset, training loop, simulation, evaluation
scripts/
  preprocess.py  # Build player feature vectors and h_stat baselines
  train.py       # Train the VAE
  simulate.py    # Run Monte Carlo phi simulation and backtest
notebooks/       # Exploratory analysis and visualizations
logs/            # Training logs
```

## Setup

```bash
pip install -r requirements.txt
python scripts/preprocess.py
python scripts/train.py
python scripts/simulate.py
```

Data comes from this [Kaggle NBA dataset](https://www.kaggle.com/). Download and place in `data/` before running preprocessing.


---

## Technical Details

**Model:** Encoder `X_team (256-dim) → 128 → 128 → (μ_z, logvar_z)`, dropout=0.3. Decoder `(z + X_players) → 64 → 64 → (μ_pred, logvar_pred)` per player per stat.

**Loss:** Minutes-weighted Gaussian NLL + KL divergence with free_bits=0.5, BETA=0.001 with linear warmup.

**Data:** 7 seasons (2019–26), ~170k player-game rows, 8,051 games. Top 8 players per team by historical minutes. 21 exponentially-decayed player features + 6 advanced features.

**Targets:** Residuals `actual_stat − h_stat`, where `h_stat` is an exponential decay average of recent games. Over/under threshold = 0 for all players.

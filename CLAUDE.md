# CLAUDE.md

## Purpose

This repository is in an **early, evolving state**.
Do not assume the project structure, file organization, or pipeline are finalized.

Your role is to:

* Help design and refine the **project structure**
* Suggest clean, minimal implementations
* Avoid over-engineering
* Prefer **iterative improvements over full rewrites**

---

## Project Goal

Build a model to learn **correlations between NBA player stats** for prop-style evaluation. Given a game context and over/under lines for different stats for each player, we want to be able to built positive EV correlated parlays.

Specifically:

* Capture **inter-player dependencies**
  * teammates (e.g., assists ↔ points)
  * opponents (e.g., rebounds competition)

* Enable **stochastic simulation** to estimate:
  * distributions of outcomes
  * correlations between players
  * probabilities for over/under and parlays

---

## Modeling Approach

The core model is a **Variational Autoencoder (VAE)** with a shared latent variable.

### Key Idea

A latent variable `z` represents **game-level context**.

* All player predictions in a game depend on the same `z`
* Correlations between players emerge through the sampling of this shared latent

---

## Model Architecture

### 1. Game Encoder

* Input: team-level features (`X_team`)
* Output: `mu`, `logvar` for latent variable `z`

`z` captures:

* pace
* matchup dynamics
* game environment
* latent stochastic factors

---

### 2. Player Decoder

* Input:

  * sampled `z`
  * player-level features (`X_players`)
* Output:

  * `[points, assists, rebounds]` per player

Handles:

* variable number of players via masking

---

### 3. Reparameterization

```
z = mu + sigma * eps
```

Used for:

* training (backprop)
* simulation (sampling multiple outcomes)

---

## Data Representation (Target Shape)

Do not assume exact file formats yet. Work with these tensor abstractions:

* `X_team`: `[num_games, num_team_features]`
* `X_players`: `[num_games, max_players, num_player_features]`
* `player_mask`: `[num_games, max_players]`
* `Y`: `[num_games, max_players, 3]`

Masking is required for variable roster sizes.

---

## Loss Design

Reconstruction loss (minutes-weighted MSE — starters weighted higher than bench):

```
((pred - target)^2 * h_numMinutes_weight).sum() / (weights.sum() * n_stats)
```

KL divergence with free bits (prevents posterior collapse):

```
kl_per_dim = -0.5 * (1 + logvar - mu^2 - exp(logvar))
kl = sum(clamp(mean_per_dim(kl_per_dim), min=free_bits))
```

Total loss:

```
loss = recon + beta * kl
```

* `beta` annealed linearly from 0 → target over `WARMUP_EPOCHS`
* `free_bits = 0.5` nats/dim enforces minimum latent usage

---

## Simulation Goal (Critical)

The model is not just predictive—it is **generative**.

You should help design:

* Monte Carlo sampling pipelines
* Efficient repeated decoding
* Correlation estimation between players

---

## Sampling methodology

* Sample latent variables `z ~ N(mu, logvar)` multiple times per batch.
* Decode each sample to get predicted stats, then threshold vs `lines` → binary `over` (1) / `under` (0).
* Stack results → tensor of shape `(batch, n_vars, num_samples)` where `n_vars = n_players * n_stats`.
* Treat samples as draws from a joint distribution over players.
* Use matrix multiplication (`A @ Aᵀ`) to compute pairwise co-occurrence counts.
* Derive all joint outcomes (OO, OU, UO, UU) and normalize to get probabilities/correlations.


## Current State of the Repo

The pipeline is fully implemented and working end-to-end. Key files:

```
src/
  data/
    preprocess.py   — exponential decay features, season filtering, saves input/target CSVs
    dataset.py      — temporal split, team pooling, top-8 players per team, residual targets,
                      minutes-based weights, feature normalization
  model.py          — GameEncoder, PlayerDecoder, reparameterize
  train.py          — weighted MSE loss, KL divergence with free-bits, train/eval loops
  simulate.py       — Monte Carlo sampling, joint outcome counts (OO/OU/UO/UU)
  evaluate.py       — phi coefficient, extract_pairs (all 4 directions), parlay backtest
scripts/
  preprocess.py     — entry point for preprocessing
  train.py          — training entry point with KL annealing and checkpoint saving
  simulate.py       — simulation entry point with backtest reporting
```

### Data Design

* **Targets**: residuals — `actual_stat - h_stat` (player's own exponential decay average)
* **Lines**: set to 0 (threshold is "does player outperform their own history?")
* **Players**: top 8 per team by `h_numMinutes`, padded to 16 total
* **Weights**: `h_numMinutes` values — starters contribute more to loss than bench players
* **Features normalized**: `X_team`, `X_players`, and `Y` all normalized using training-set
  mean/std, saved in checkpoint for consistent use at inference

### Training Configuration

* `LATENT_DIM = 16`, `H_DIM_ENC = 64`, `H_DIM_DEC = 32`
* `BETA = 0.001` with 15-epoch linear warmup from 0
* `FREE_BITS = 0.5` nats/dim — prevents posterior collapse by enforcing minimum KL per dimension
* `NUM_EPOCHS = 100`
* Checkpoint saved to `checkpoints/model_latest.pt` (gitignored)

### Simulation Configuration

* `NUM_SAMPLES = 500`, `SEED = 42` (fixed for reproducibility)
* `PHI_THRESHOLD = 0.15`, `TOP_K = 10` per game
* Bets on all 4 directions: OO, UU (positive phi), OU, UO (negative phi)
* Same-player pairs explicitly excluded from phi computation

### Known Issues and Findings

**Feature normalization is critical.** Without normalizing `X_team` and `X_players`:
* The `home` flag (0/1) is drowned out by large-scale features (h_numMinutes ~25, h_points ~12)
* The decoder learns biased mean outputs per stat (rebounds +0.05 vs points -0.08 in norm space)
* This causes rebounds to be predicted "over" ~70% of the time vs actual ~49%
* Pairwise OO was inflated to 34.6% vs actual 26.2%, OU/UO severely underrepresented

**After normalization:**
* KL/dim improved from 0.65 → 2.2 — z is used much more effectively
* Per-stat predicted over rates are much better calibrated
* OU/UO bets now appear at ~6-12% of total bets (were ~0% before)
* Mean phi dropped from 0.061 → 0.007 (correctly centered, matching actual near-uniform data)

**Free bits prevent posterior collapse.** Without free_bits, increasing BETA causes KL → 0
(complete collapse). With `free_bits=0.5`, the model maintains ≥0.5 nats/dim regardless of BETA.

**Current signal is low.** Win rate ~26% vs 27.4% breakeven across all phi thresholds.
Root cause: only one season (~880 training games) — insufficient data to learn generalizable
inter-player correlation patterns. Train recon ~0.68 vs val recon ~1.35 indicates overfitting.

### Next Steps

1. **More data**: Add 2022-23, 2023-24 seasons (~3x training data) — primary bottleneck
2. **Real betting lines**: Replace h_stat proxies with actual sportsbook lines
3. **Reduce overfitting**: Smaller model or dropout given limited training data

---

## Guidelines for Contributions

When suggesting code or structure:

### 1. Prefer Minimal Structure First

Start with:

* a small number of files
* clear separation only where necessary

Avoid:

* large boilerplate setups
* unnecessary abstractions

---

### 2. Respect Core Architectural Split

Keep this separation clear:

* Encoder (game-level)
* Decoder (player-level)
* Training logic
* Simulation logic

---

### 3. Design for Iteration

This project will iterate heavily on:

* features
* latent dimension
* loss weighting
* simulation strategy

Favor designs that are:

* easy to modify
* easy to debug
* not tightly coupled

---

### 4. Avoid Hidden Assumptions

Do not:

* assume exact feature engineering steps
* assume specific datasets
* assume fixed dimensions

Always:

* ask or infer from context
* keep implementations flexible

---

### 5. Simulation Is First-Class

Treat simulation as equally important as training.

Design code so that:

* encoder and decoder can be used independently
* `z` can be sampled repeatedly without friction

---

## What to Help With

You should actively help with:

* Designing folder structure (incrementally)
* Writing PyTorch model components
* Building training loops
* Implementing masking correctly
* Creating simulation pipelines
* Debugging shape and tensor issues
* Improving correlation learning

---

## What to Avoid

* Do not assume this is a standard supervised learning project
* Do not collapse everything into a single deterministic model
* Do not remove stochastic components
* Do not ignore masking or variable player counts

---

## Notes

* Correlation learning via shared latent `z` is the core idea
* Model usefulness depends heavily on **simulation quality**, not just MSE
* Simplicity is preferred early; complexity should be introduced only when needed

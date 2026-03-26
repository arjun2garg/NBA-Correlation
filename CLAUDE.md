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

* `LATENT_DIM = 32`, `H_DIM_ENC = 128`, `H_DIM_DEC = 64`, `DROPOUT = 0.3`
* Decoder has two hidden layers: `(latent+player) → 64 → 64 → output`
* `BETA = 0.001` with 15-epoch linear warmup from 0
* `FREE_BITS = 0.5` nats/dim — prevents posterior collapse by enforcing minimum KL per dimension
* `NUM_EPOCHS = 150`, `BATCH_SIZE = 64`
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

**More data (3 seasons) reduced but did not eliminate overfitting.**
* 3x data (~3,600 games across 2022-23, 2023-24, 2024-25) vs ~880 games before
* Without dropout: train recon ~0.755 vs val recon ~1.344 — still significant overfitting
* Win rate 24.4% (below 27.4% breakeven), ROI -11.2%
* Per-stat over-prediction bias: ~56% predicted over vs ~49% actual across all stats

**Dropout (0.3) + smaller hidden dims resolved the train/val gap.**
* Architecture: encoder `input → 48 → 48 → latent`, decoder `(latent+player) → 24 → output`
* Train recon ~0.997 vs val recon ~1.083 — gap reduced from 0.59 → 0.09
* Win rate 27.9% (above 27.4% breakeven), ROI +1.7%
* KL/dim dropped to ~0.58 (was 1.89) — dropout compresses encoder, z carries less information
* Same-team phi mean jumped to 0.167 (was 0.013) — later found to be an artifact of over-prediction bias, not real signal
* Per-stat bias persists: points ~52% (good), rebounds ~55%, assists ~59% vs ~48-50% actual

**Advanced features added (usage rate, pace, ratings, rest days, back-to-backs, rolling correlations) partially address signal gaps.**
* Pace, offensive/defensive ratings, usage share, and rest context are now included
* Rolling per-player correlations (pts↔ast, pts↔reb) added as player-level features
* Assists over-prediction bias (~59%) persists — no position-level opponent defensive context yet
* No game-level metadata from external sources (Vegas lines, injury reports)

**Root cause of over-prediction bias identified and fixed (probabilistic decoder).**
* The deterministic decoder output point estimates with std ~0.19 normalized vs actual std ~1.0
* Since Y_mean > 0 (players slightly outperform h_stat on average), predictions clustered above 0 → 67–70% predicted over vs ~50% actual
* Fix: `PlayerDecoder` now outputs `(mu_pred, logvar_pred)` per player per stat; trained with Gaussian NLL
* During simulation, `P(over | z) = Φ(mu_pred / sigma_pred)` is computed analytically via normal CDF instead of binary sampling
* Result: predicted over rates now 50–51% across all stats, matching actual ~46–50%
* Actual joint distribution confirmed near-uniform: OO=24.8%, OU=25.1%, UO=24.1%, UU=26.0%

**Phi signal collapsed to ~0 after fixing the bias.**
* With the probabilistic decoder, `sigma_pred ≈ 0.94` (model is uncertain about all players)
* `mu_pred` std ~0.16 normalized — z barely moves predictions relative to noise floor
* `P(over | z)` std ≈ 0.05–0.07 across all z samples — effectively constant at 0.5
* Previous same-team phi = 0.167 was entirely an artifact of the over-prediction bias, not real signal
* The model correctly fits the marginal distribution but finds no pairwise correlation signal

**Expanded to 7 seasons (2019-26) and larger architecture — phi still zero.**
* Data expanded from 3 seasons (~57k rows) to 7 seasons (170,823 rows: 2019-20 through 2025-26)
* Architecture scaled up: `LATENT_DIM 16→32`, `H_DIM_ENC 48→128`, `H_DIM_DEC 24→64`, decoder now has 2 hidden layers
* Train/val gap is negligible (0.424 vs 0.465) — no overfitting
* KL/dim at end of training: ~0.31 (actual posterior), below the `free_bits=0.5` floor — z is barely used
* Simulation: phi mean = 0.0007, std = 0.0011 — all 1.7M pairs in [-0.15, +0.15), 0 bets placed
* Predicted over rates: points 52.1%, assists 52.2%, rebounds 52.2% vs actual 49.7%, 47.5%, 48.2% — slight persistent bias

**Root cause of phi=0 diagnosed: decoder is insensitive to z.**
* z does vary meaningfully between games: mu_z std = 0.46, range [-1.65, +1.80]
* Within a game, sigma_z ≈ 0.60 (posterior spread across z samples)
* BUT P(over|z) std across 500 z samples is only ±2–4% per player
* This means the decoder's mu_pred barely responds to z movement — same prediction regardless of z
* Cross-player correlations in P(over|z) across z samples are real and strong (0.7–0.98 for same-team players), but the absolute variance is so small (±2–4%) that the covariance is ~0.001, producing phi ≈ 0.004
* The phi metric captures the *correlation contribution* to joint probability. With ±2–4% P(over) variation, even perfect correlation between players produces negligible phi
* This is NOT because sigma_pred squashes P(over) toward 0.5 — it is because mu_pred does not respond to z. These are distinct failure modes.
* With `BETA=0.001`, the KL term contributes ~0.016 to total loss. Reconstruction dominates, and the decoder learns to ignore z entirely.

**Key distinction: the model is not guessing randomly.**
* Mean P(over) per game ranges from 47% to 72% across val set — the encoder is differentiating games
* Predicted vs actual game-level over rate has 0.57 Pearson correlation across 1,579 val games
* Top quintile of predicted P(over): model predicts 58.6%, actual is 57.7% — well-calibrated
* The signal exists at the game level but cannot be exploited for parlays with h_stat as lines, because the individual mispricing relative to true probability is what creates parlay edge, not correlation alone

---

## Feature Engineering (Current)

The `h_` prefix denotes exponentially time-decayed historical averages (per-day decay, computed as a weighted average of all prior games). Each stat has its own beta tuned for signal stability.

### Decay Stats (15 per player)

`h_blocks` removed — adds no incremental signal once opponent data is present.

| Feature | Beta | Description |
|---|---|---|
| `h_points` | 0.99 | Scoring average |
| `h_assists` | 0.99 | Assist average |
| `h_reboundsTotal` | 0.99 | Total rebound average |
| `h_reboundsDefensive` | 0.99 | Defensive rebound average |
| `h_reboundsOffensive` | 0.99 | Offensive rebound average |
| `h_steals` | 0.995 | Steal average |
| `h_turnovers` | 0.99 | Turnover average |
| `h_foulsPersonal` | 0.99 | Personal foul average |
| `h_fieldGoalsMade` | 0.995 | FGM average |
| `h_fieldGoalsAttempted` | 0.99 | FGA average |
| `h_threePointersMade` | 0.995 | 3PM average |
| `h_threePointersAttempted` | 0.98 | 3PA average |
| `h_freeThrowsMade` | 0.99 | FTM average |
| `h_freeThrowsAttempted` | 0.99 | FTA average |
| `h_numMinutes` | 0.97 | Minutes average (also used as loss weight) |

### Advanced Decay Features (6 per player)

Box-score derived. No external API — computed from `PlayerStatistics.csv` and `TeamStatistics.csv`.

| Feature | Beta | Description |
|---|---|---|
| `h_usage_rate` | 0.97 | 100 × (FGA + 0.44×FTA + TOV) × (team_min/5) / (min × team_total) |
| `h_usage_share` | 0.97 | Simplified possession share per player |
| `h_pace` | 0.98 | Team possessions per 48 minutes |
| `h_off_rating` | 0.99 | 100 × team_points / team_possessions (own team) |
| `h_def_rating` | 0.99 | Opponent's offensive rating (defensive challenge faced) |
| `h_implied_total` | 0.99 | h_off_rating + h_def_rating |

All 21 decay + advanced features form `STAT_COLS` in `dataset.py`.

### Game-Level Features (point-in-time, per team)

Not decayed — computed fresh for each game from the schedule.

| Feature | Description |
|---|---|
| `days_rest` | Days since last game, clipped to [0, 7] |
| `is_b2b` | Back-to-back flag (1 if yes, 0 if no) |

Stored as `GAME_TEAM_COLS` in `dataset.py`, appended as team-level scalars to `X_team`.

### Per-Player Extra Features

| Feature | Description |
|---|---|
| `home` | Binary flag — 1 if player's team is home, 0 if away |
| `cov_pts_ast` | 20-game rolling Pearson correlation between points and assists |
| `cov_pts_reb` | 20-game rolling Pearson correlation between points and rebounds |

Stored as `PLAYER_EXTRA_COLS` in `dataset.py`. These are appended per player in `X_players` but **not** pooled into `X_team`.

### How Features Are Used

**`X_team` (game encoder input):**
Team-level context built by pooling each team's roster into 6 rows (5 starters + 1 minutes-weighted bench aggregate), each with the 21 `STAT_COLS`. Home and away team blocks are concatenated, then per-team scalars (`days_rest`, `is_b2b`) for both teams are appended → shape `[num_games, 21×6×2 + 2×2]` = `[num_games, 256]`.

**`X_players` (player decoder input):**
Top 8 players per team (by `h_numMinutes`), 16 per game total. Each player has all 21 `STAT_COLS` + 3 `PLAYER_EXTRA_COLS` → shape `[num_games, 16, 24]`.

**Targets (`Y`):**
Residuals — `actual_stat - h_stat` for `[points, assists, reboundsTotal]`. The over/under threshold is therefore **0** for every player.

### What Is Missing (Key Gaps)

* No opponent context at position level (e.g., opponent defensive rating vs. guards vs. bigs)
* No game-level metadata from external sources (Vegas lines, injury reports)
* No home/away split in historical averages — only overall averages used

### Distributional Assumption Gap

The current decoder models player stats as Gaussian (via NLL loss). This is wrong in a fundamental way: points, assists, and rebounds are **non-negative integers**. A Gaussian allows negative values and treats the distribution as symmetric, neither of which holds for NBA stats.

Better candidate distributions:
* **Poisson** — natural for count data, variance = mean. Simple but may be underdispersed (real NBA stats tend to have variance > mean).
* **Negative Binomial** — count data with overdispersion (variance > mean). Likely a better fit than Poisson for bursty stats like assists and points.
* **Log-normal** — non-negative, right-skewed continuous approximation. Easy to implement (just model log(stat+1) as Gaussian). Would naturally prevent negative predictions.

This matters for simulation: if the decoder can output negative mu_pred, the P(over) computation via normal CDF is distorted for players near zero. A log-normal or negative binomial decoder would give better-calibrated P(over) estimates, especially for low-usage bench players.

---

### Next Steps

**Most critical: force the decoder to be z-sensitive.**

The decoder currently ignores z because `BETA=0.001` makes the KL term negligible (~0.016 contribution to loss). Until z meaningfully modulates mu_pred, phi will remain zero regardless of architecture size or data volume. The path forward:

1. **Increase BETA** — try `BETA=0.01` or `BETA=0.1`. This forces the encoder to use z and the decoder to respond to it. Monitor KL/dim (want >1.0) and whether P(over|z) std across z samples rises from ±4% to ±10–20%.
2. **Opponent position-level defensive context** — most impactful remaining feature gap
3. **Real betting lines** — replace h_stat proxies with actual sportsbook lines; with sharp lines, residual uncertainty shrinks and z-induced P(over) variation becomes a larger fraction of total uncertainty
4. **Home/away split historical averages** — current averages pool both contexts

**What success looks like:** P(over|z) std across z samples rising to ±10–20%, phi std rising above 0.01, and eventually pairs exceeding the 0.15 threshold. If BETA=0.1 causes val recon to degrade significantly with no phi improvement, the signal may genuinely not be capturable by this architecture.

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

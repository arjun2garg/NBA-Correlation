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

Reconstruction loss (masked MSE):

```
((pred - target)^2 * mask).mean()
```

KL divergence:

```
-0.5 * sum(1 + logvar - mu^2 - exp(logvar))
```

Total loss:

```
loss = recon + beta * kl
```

* `beta` may be annealed during training

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

* Structure is **not finalized**
* Some work exists in notebooks
* Scripts and modules may be missing or incomplete
* `requirements.txt` may be incomplete or empty

Do not assume:

* stable file layout
* working training pipeline
* clean separation of concerns

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

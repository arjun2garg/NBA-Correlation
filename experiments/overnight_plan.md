# Overnight Experiments Plan

Branch: `experiments/overnight`
Date: 2026-03-25

## Context

The VAE's decoder ignores z: sigma_pred (~0.94 normalized) swamps mu_pred variation (~±0.10), giving
P(over|z) std ≈ ±2%. Phi ≈ 0, no bets placed. Three root causes:
1. Residual targets (actual - h_stat) partially remove game-context correlation
2. Decoder predicts each player independently — no player-player interaction mechanism
3. z has no anchor to real game outcomes

Key data available:
- `data/raw/PlayByPlay.parquet` — 18.6M events, 83 cols (assist attribution, substitutions, court coords)
- `data/raw/PlayerStatisticsAdvanced.csv` — actual game-level pace, offRating, defRating, poss
- `data/raw/TeamStatistics.csv` — team totals per game

---

## Track A: Attention Decoder + PBP Assist Features

**Hypothesis**: Self-attention over all 16 players in the decoder will create explicit player-player
interaction, so player A's prediction directly influences player B's. Assist network features from
PlayByPlay add prior knowledge about passing relationships.

### A1: PBP Feature Extraction — `src/data/pbp_features.py`

Load PlayByPlay.parquet with polars lazy scan (memory-efficient). Filter to made field goals
(actionType in ['2pt', '3pt'] where shotResult == 'Made' or similar). Compute per-player-game:
- `assist_given_cnt`: rows where assistPersonId == personId
- `fgm_cnt`: made FGs by this player

Rolling 20-game (sorted by gameDateTimeEst per player):
- `rolling_ast_given_rate`: rolling(assist_given_cnt / team_fgm_cnt)
- `rolling_ast_received_rate`: rolling(assist_received_cnt / fgm_cnt)

Output: CSV `data/processed/pbp_features.csv` with (personId, gameId, rolling_ast_given_rate,
rolling_ast_received_rate). Merge into input_data in dataset.py.

### A2: Attention Decoder — `src/model_attention.py`

```python
class PlayerDecoderAttention(nn.Module):
    def __init__(self, latent_dim, player_dim, h_dim=64, n_heads=2, output_dim=3, dropout=0.3):
        self.z_proj = nn.Linear(latent_dim, h_dim)
        self.p_proj = nn.Linear(player_dim, h_dim)
        self.attn = nn.MultiheadAttention(h_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(h_dim)
        self.mlp = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))
        self.mu_head = nn.Linear(h_dim, output_dim)
        self.logvar_head = nn.Linear(h_dim, output_dim)

    def forward(self, z, player_feats, mask):
        z_exp = self.z_proj(z).unsqueeze(1).expand(-1, player_feats.size(1), -1)
        h = self.p_proj(player_feats) + z_exp
        key_pad = ~mask.bool()  # padded slots → True = ignore
        h_attn, _ = self.attn(h, h, h, key_padding_mask=key_pad)
        h = self.norm(h + h_attn)
        h = self.mlp(h)
        return self.mu_head(h), self.logvar_head(h).clamp(-6, 2)
```

### A3: Training — `scripts/train_attention.py`

- BETA=0.001, FREE_BITS=0.5, 150 epochs
- Pass (weights > 0) as attention mask
- Save to `checkpoints/exp_attention/`
- Log P(over|z) std every 10 epochs inline

**Success**: P(over|z) std > 0.05, same-team phi mean > 0.01

---

## Track B: Two-Stage VAE with Raw Targets

**Hypothesis**: (1) Anchoring z to real game outcomes forces it to carry game-state signal.
(2) Raw targets preserve game-context correlation that residuals destroy.

### B1: Game Outcome Targets — `src/data/game_outcomes.py`

From TeamStatistics.csv: home_score, away_score per game.
From PlayerStatisticsAdvanced.csv: actual pace, poss per team per game (take first player row per team).

Build per-game 6-dim vector: [home_score, away_score, home_pace, away_pace, home_poss, away_poss].
Normalize using train-set mean/std. Save normalization stats to checkpoint.

### B2: Raw Target Variant — `build_tensors_raw` in dataset.py

Instead of Y = actual - h_stat, use Y = actual (raw points/assists/rebounds).
Lines = h_stat (stored separately, not subtracted). Game-context correlation preserved:
fast game → all players score more → shared positive residuals relative to any fixed baseline.

### B3: Two-Stage Model — `src/model_twostage.py`

```python
class GameOutcomeDecoder(nn.Module):
    # z (batch, latent) → game outcome (batch, 6), small 2-layer MLP

class TwoStagePlayerDecoder(nn.Module):
    # input: concat(z_exp, game_outcome_exp, player_feats)
    # same structure as existing PlayerDecoder but wider input
    # game_outcome_exp: (batch, 16, 6) — same game outcome broadcast to all players
```

Loss: `player_nll + alpha * game_mse + beta * kl`
Start with alpha=0.5, beta=0.001.

### B4: Training — `scripts/train_twostage.py`

- 150 epochs, save to `checkpoints/exp_twostage/`
- Log: game_recon, player_recon, KL/dim, P(over|z) std every 10 epochs

**Success**: game_recon decreasing (z anchored to real outcomes), P(over|z) std > 0.05

---

## Track C: Large MI Sweep

Extend `scripts/train_mi.py` with `--lambda-mi` CLI arg if not already present.
Run sequentially:
- lambda_mi=1.0, 150 epochs → `checkpoints/exp_mi_1/`
- lambda_mi=5.0, 150 epochs → `checkpoints/exp_mi_5/`

**Success**: P(over|z) std > 0.05 at lambda=5.0

---

## Orchestration — `scripts/run_overnight.sh`

```bash
#!/bin/bash
set -e
cd "/Users/ag7/Documents/ML Projects/NBA Correlation"
PYTHON=".venv/bin/python"
mkdir -p logs checkpoints/exp_attention checkpoints/exp_twostage checkpoints/exp_mi_1 checkpoints/exp_mi_5

echo "=== Track C: MI sweep ===" | tee logs/progress.log
$PYTHON scripts/train_mi.py --beta 0.001 --lambda-mi 1.0 --ckpt-dir checkpoints/exp_mi_1 --epochs 150 --no-resume > logs/mi_1.log 2>&1 && echo "MI 1.0 done" >> logs/progress.log
$PYTHON scripts/train_mi.py --beta 0.001 --lambda-mi 5.0 --ckpt-dir checkpoints/exp_mi_5 --epochs 150 --no-resume > logs/mi_5.log 2>&1 && echo "MI 5.0 done" >> logs/progress.log

echo "=== Tracks A and B (parallel) ===" | tee -a logs/progress.log
$PYTHON scripts/train_attention.py > logs/attention.log 2>&1 &
PID_A=$!
$PYTHON scripts/train_twostage.py > logs/twostage.log 2>&1 &
PID_B=$!
wait $PID_A && echo "Attention done" >> logs/progress.log
wait $PID_B && echo "Twostage done" >> logs/progress.log

echo "=== Diagnostics ===" | tee -a logs/progress.log
for ckpt in exp_attention exp_twostage exp_mi_1 exp_mi_5; do
    $PYTHON scripts/diagnose_z_sensitivity.py --checkpoint checkpoints/$ckpt/model_latest.pt >> logs/diagnostics.log 2>&1
    echo "$ckpt diagnostics done" >> logs/progress.log
done

echo "ALL DONE" | tee -a logs/progress.log
```

---

## Additional Experiments (if time allows / for future runs)

### D: Lineup Stint Modeling
Extract actual 5-man lineup stints from PlayByPlay.parquet using substitution events
(actionType == 'substitution', subsInPersonId tracks who enters). Per stint, compute:
- Possessions played, points scored, assists, rebounds
- Duration in seconds / game clock units

Model stints instead of individual game totals. Lineup performance is much more tightly
coupled to game context — z should explain a larger fraction of variance.
High lift: 2-3hr data pipeline before any training.

### E: Opponent Position-Level Defensive Context
From TeamStatistics and PlayByPlay, compute how each team defends by position:
- Opponent PPG allowed to guards vs. bigs (from PlayerStatisticsAdvanced)
- This adds position-level matchup context currently missing from features

### F: Conditional VAE with Lineup as Explicit Input
Rather than pooling bench players, condition the encoder on the actual starting lineup:
each game's actual starting 5 (from first substitution events in PBP).
Smaller encoder input but higher quality signal.

### G: Flow-Based Decoder
Replace Gaussian decoder with a normalizing flow. Flows can model non-Gaussian distributions
(NBA stats are count-like, right-skewed). A flow conditioned on z could be more expressive
about how z changes the full outcome distribution, not just the mean.

### H: Contrastive z Training
Explicitly train z to be different for games with different outcomes:
- Take pairs of games with similar pre-game features but different actual outcomes
- Contrastive loss: push their z representations apart
- This forces z to carry the "what actually happened" information even without explicit labels

### I: Reduced Target Space — High-Variance Players Only
Filter to players with high stat variance (e.g., stars with large h_std).
Their residuals vs h_stat may show more game-context correlation than average players.
Smaller dataset but potentially stronger signal.

---

## Key Metrics to Watch (all experiments)

| Metric | Current | Target | Meaning |
|--------|---------|--------|---------|
| P(over\|z) std | 0.023 | > 0.05 | z meaningfully shifts predictions |
| KL/dim (genuine) | 0.31 | > 0.5 | z actually used, not just floored |
| Same-team phi mean | ~0 | > 0.01 | Real correlation structure |
| Same-team phi pct > 0.15 | 0% | > 1% | Bettable pairs exist |
| Val recon | 0.465 | < 0.55 | Didn't sacrifice too much accuracy |

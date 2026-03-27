#!/bin/bash
# Overnight experiment orchestration
# Revised order: A+B first (novel, stable), then MI sweeps at achievable lambdas
#
# Tracks:
#   A: Attention decoder + PBP assist features (~2h)
#   B: Two-stage VAE with raw targets (~2h, parallel with A)
#   C1: MI sweep lambda=0.5, beta=0.001  (~2h, after A+B)
#   C2: MI sweep lambda=1.0, beta=0.01   (~2h, after C1)
#
# MI instability note: lambda=1.0 with beta=0.001 and no KL regularization
# creates a gradient explosion from the variance term. Using lambda=0.5 (moderate)
# and lambda=1.0 with beta=0.01 (higher KL) for stable training.

set -e
cd "/Users/ag7/Documents/ML Projects/NBA Correlation"
PYTHON=".venv/bin/python"

mkdir -p logs checkpoints/exp_attention checkpoints/exp_twostage \
         checkpoints/exp_mi_1 checkpoints/exp_mi_5

ts() { date "+%H:%M:%S"; }

echo "[$(ts)] === Overnight experiments starting ===" | tee logs/progress.log

# ---- Tracks A and B: run in parallel (novel experiments, ~2h each) ----
echo "[$(ts)] === Tracks A (attention) and B (twostage) starting in parallel ===" | tee -a logs/progress.log

$PYTHON scripts/train_attention.py \
    --ckpt-dir checkpoints/exp_attention \
    --epochs 150 \
    --no-resume \
    > logs/attention.log 2>&1 &
PID_A=$!

$PYTHON scripts/train_twostage.py \
    --ckpt-dir checkpoints/exp_twostage \
    --epochs 150 \
    --no-resume \
    > logs/twostage.log 2>&1 &
PID_B=$!

wait $PID_A && echo "[$(ts)] Track A (attention) done" | tee -a logs/progress.log || echo "[$(ts)] Track A FAILED" | tee -a logs/progress.log
wait $PID_B && echo "[$(ts)] Track B (twostage) done" | tee -a logs/progress.log || echo "[$(ts)] Track B FAILED" | tee -a logs/progress.log

# ---- Track C: MI sweep (sequential, stable hyperparameters) ----
# lambda=0.5, beta=0.001: 5x previous MI experiment, moderate increase
echo "[$(ts)] === Track C1: MI sweep lambda=0.5 beta=0.001 ===" | tee -a logs/progress.log
$PYTHON scripts/train_mi.py \
    --beta 0.001 \
    --lambda-mi 0.5 \
    --ckpt-dir checkpoints/exp_mi_1 \
    --epochs 150 \
    --no-resume \
    > logs/mi_1.log 2>&1 && echo "[$(ts)] MI lambda=0.5 done" | tee -a logs/progress.log || echo "[$(ts)] MI lambda=0.5 FAILED" | tee -a logs/progress.log

# lambda=1.0, beta=0.01: 10x previous MI experiment with stronger KL
echo "[$(ts)] === Track C2: MI sweep lambda=1.0 beta=0.01 ===" | tee -a logs/progress.log
$PYTHON scripts/train_mi.py \
    --beta 0.01 \
    --lambda-mi 1.0 \
    --ckpt-dir checkpoints/exp_mi_5 \
    --epochs 150 \
    --no-resume \
    > logs/mi_5.log 2>&1 && echo "[$(ts)] MI lambda=1.0 done" | tee -a logs/progress.log || echo "[$(ts)] MI lambda=1.0 FAILED" | tee -a logs/progress.log

# ---- Diagnostics ----
echo "[$(ts)] === Running z-sensitivity diagnostics ===" | tee -a logs/progress.log

for ckpt_dir in exp_attention exp_mi_1 exp_mi_5; do
    if [ -f "checkpoints/$ckpt_dir/model_latest.pt" ]; then
        echo "[$(ts)] Diagnosing $ckpt_dir..." | tee -a logs/progress.log
        $PYTHON scripts/diagnose_z_sensitivity.py \
            --ckpt "checkpoints/$ckpt_dir/model_latest.pt" \
            --num-samples 500 \
            >> logs/diagnostics.log 2>&1
        echo "[$(ts)] $ckpt_dir done" | tee -a logs/progress.log
    else
        echo "[$(ts)] Skipping $ckpt_dir (no checkpoint)" | tee -a logs/progress.log
    fi
done

# twostage has different checkpoint format
if [ -f "checkpoints/exp_twostage/model_latest.pt" ]; then
    echo "[$(ts)] Diagnosing exp_twostage..." | tee -a logs/progress.log
    $PYTHON scripts/diagnose_z_sensitivity.py \
        --ckpt "checkpoints/exp_twostage/model_latest.pt" \
        --num-samples 500 \
        >> logs/diagnostics.log 2>&1
    echo "[$(ts)] exp_twostage done" | tee -a logs/progress.log
fi

echo "[$(ts)] === ALL EXPERIMENTS DONE ===" | tee -a logs/progress.log
echo "Key results in logs/diagnostics.log"
echo "Training logs in logs/{attention,twostage,mi_1,mi_5}.log"

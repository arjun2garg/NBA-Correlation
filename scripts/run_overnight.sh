#!/bin/bash
# Overnight experiment orchestration
# Tracks A (attention), B (twostage), C (MI sweep)
# Goal: find correlation signal; force decoder to be z-sensitive

set -e
cd "/Users/ag7/Documents/ML Projects/NBA Correlation"
PYTHON=".venv/bin/python"

mkdir -p logs checkpoints/exp_attention checkpoints/exp_twostage \
         checkpoints/exp_mi_1 checkpoints/exp_mi_5

# Timestamp helper
ts() { date "+%H:%M:%S"; }

echo "[$(ts)] === Overnight experiments starting ===" | tee logs/progress.log

# ---- Track C: MI sweep (sequential, must finish before A/B to not compete for CPU) ----
echo "[$(ts)] === Track C: MI sweep lambda=1.0 ===" | tee -a logs/progress.log
$PYTHON scripts/train_mi.py \
    --beta 0.001 \
    --lambda-mi 1.0 \
    --ckpt-dir checkpoints/exp_mi_1 \
    --epochs 150 \
    --no-resume \
    > logs/mi_1.log 2>&1
echo "[$(ts)] MI lambda=1.0 done" | tee -a logs/progress.log

echo "[$(ts)] === Track C: MI sweep lambda=5.0 ===" | tee -a logs/progress.log
$PYTHON scripts/train_mi.py \
    --beta 0.001 \
    --lambda-mi 5.0 \
    --ckpt-dir checkpoints/exp_mi_5 \
    --epochs 150 \
    --no-resume \
    > logs/mi_5.log 2>&1
echo "[$(ts)] MI lambda=5.0 done" | tee -a logs/progress.log

# ---- Tracks A and B: run in parallel ----
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

wait $PID_A
echo "[$(ts)] Track A (attention) done" | tee -a logs/progress.log

wait $PID_B
echo "[$(ts)] Track B (twostage) done" | tee -a logs/progress.log

# ---- Diagnostics ----
echo "[$(ts)] === Running diagnostics on all checkpoints ===" | tee -a logs/progress.log

for ckpt_dir in exp_mi_1 exp_mi_5 exp_attention; do
    echo "[$(ts)] Diagnosing $ckpt_dir..." | tee -a logs/progress.log
    $PYTHON scripts/diagnose_z_sensitivity.py \
        --ckpt "checkpoints/$ckpt_dir/model_latest.pt" \
        --num-samples 500 \
        >> logs/diagnostics.log 2>&1
    echo "[$(ts)] $ckpt_dir diagnostics done" | tee -a logs/progress.log
done

# twostage has a custom checkpoint format; run its inline diagnostic via a quick script
echo "[$(ts)] Diagnosing exp_twostage (custom)..." | tee -a logs/progress.log
$PYTHON scripts/diagnose_z_sensitivity.py \
    --ckpt "checkpoints/exp_twostage/model_latest.pt" \
    --num-samples 500 \
    >> logs/diagnostics.log 2>&1
echo "[$(ts)] exp_twostage diagnostics done" | tee -a logs/progress.log

echo "[$(ts)] === ALL EXPERIMENTS DONE ===" | tee -a logs/progress.log
echo "Results summary in logs/diagnostics.log"
echo "Training logs in logs/{attention,twostage,mi_1,mi_5}.log"

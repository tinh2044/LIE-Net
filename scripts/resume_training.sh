#!/bin/bash

# Resume training script for Low Light Image Enhancement
# Usage: ./scripts/resume_training.sh [checkpoint_path] [start_epoch]

# Default parameters
CHECKPOINT_PATH=${1:-"./checkpoints/Low_light/latest.pth"}
START_EPOCH=${2:-0}
BATCH_SIZE=4
EPOCHS=200
DEVICE="cuda"
SEED=42
NUM_WORKERS=4
CFG_PATH="./configs/low_light.yaml"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Resuming training from checkpoint: $CHECKPOINT_PATH"
echo "Starting from epoch: $START_EPOCH"

# Run training with resume
python main_new.py \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --resume "$CHECKPOINT_PATH" \
    --start_epoch "$START_EPOCH" \
    --num_workers "$NUM_WORKERS" \
    --cfg_path "$CFG_PATH" \
    2>&1 | tee logs/resume_training_$(date +%Y%m%d_%H%M%S).log 
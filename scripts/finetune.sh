#!/bin/bash

# Fine-tuning script for Low Light Image Enhancement
# Usage: ./scripts/finetune.sh [pretrained_model_path] [dataset_config]

# Default parameters
PRETRAINED_MODEL=${1:-"./checkpoints/Low_light/best.pth"}
DATASET_CONFIG=${2:-"./configs/uieb.yaml"}
BATCH_SIZE=4
EPOCHS=100
DEVICE="cuda"
SEED=42
NUM_WORKERS=4

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Fine-tuning from pre-trained model: $PRETRAINED_MODEL"
echo "Using dataset config: $DATASET_CONFIG"

# Run fine-tuning
python main_new.py \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --finetune "$PRETRAINED_MODEL" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    --cfg_path "$DATASET_CONFIG" \
    2>&1 | tee logs/finetune_$(date +%Y%m%d_%H%M%S).log 
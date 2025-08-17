#!/bin/bash
BATCH_SIZE=4
EPOCHS=200
FINETUNE=""
DEVICE="cpu"
SEED=42
RESUME=""
START_EPOCH=0
EVAL_FLAG=""
TEST_ON_LAST_EPOCH="False"
NUM_WORKERS=0
CFG_PATH="./configs/lolv2-real.yaml"

mkdir -p logs
python main.py \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --finetune "$FINETUNE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --resume "$RESUME" \
    --start_epoch "$START_EPOCH" \
    $EVAL_FLAG \
    --num_workers "$NUM_WORKERS" \
    --cfg_path "$CFG_PATH" \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log 
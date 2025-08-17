#!/bin/bash

# Inference script for Low Light Image Enhancement
# Usage: ./scripts/inference.sh [input_dir] [output_dir] [checkpoint_path]

# Default parameters
INPUT_DIR=${1:-"./test_images"}
OUTPUT_DIR=${2:-"./enhanced_images"}
CHECKPOINT_PATH=${3:-"./checkpoints/Low_light/best.pth"}
DEVICE="cuda"
SHOW_FLOPS="True"
SHOW_PARAMS="True"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting inference..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Device: $DEVICE"

# Run inference
python interface_new.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --device "$DEVICE" \
    --show_flops "$SHOW_FLOPS" \
    --show_params "$SHOW_PARAMS" \
    2>&1 | tee logs/inference_$(date +%Y%m%d_%H%M%S).log

echo "Inference completed! Enhanced images saved to: $OUTPUT_DIR" 
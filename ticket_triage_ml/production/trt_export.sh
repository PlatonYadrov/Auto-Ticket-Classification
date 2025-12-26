#!/bin/bash
# TensorRT Export Script
# Converts ONNX model to TensorRT engine for optimized inference
#
# Prerequisites:
# - NVIDIA GPU with TensorRT installed
# - trtexec command available in PATH
#
# Usage:
#   ./trt_export.sh [--fp16] [--int8]
#
# This script is optional and will not block CPU-only runs.
# For CPU inference, use the ONNX Runtime implementation instead.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
ONNX_MODEL="${PROJECT_ROOT}/artifacts/model.onnx"
TRT_OUTPUT="${PROJECT_ROOT}/artifacts/model.plan"

# Default settings
FP16_MODE=""
INT8_MODE=""
MAX_BATCH_SIZE=32
MAX_WORKSPACE_SIZE=4294967296  # 4GB

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fp16)
            FP16_MODE="--fp16"
            shift
            ;;
        --int8)
            INT8_MODE="--int8"
            shift
            ;;
        --max-batch-size)
            MAX_BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if trtexec is available
if ! command -v trtexec &> /dev/null; then
    echo "Error: trtexec not found in PATH"
    echo "Please install TensorRT and ensure trtexec is available"
    echo "For CPU-only inference, use the ONNX Runtime implementation"
    exit 1
fi

# Check if ONNX model exists
if [ ! -f "$ONNX_MODEL" ]; then
    echo "Error: ONNX model not found at $ONNX_MODEL"
    echo "Please run 'poetry run python -m ticket_triage_ml.commands export_onnx' first"
    exit 1
fi

echo "Converting ONNX model to TensorRT engine..."
echo "Input: $ONNX_MODEL"
echo "Output: $TRT_OUTPUT"
echo "Max batch size: $MAX_BATCH_SIZE"
echo "Precision modes: ${FP16_MODE:-none} ${INT8_MODE:-none}"

# Create output directory if needed
mkdir -p "$(dirname "$TRT_OUTPUT")"

# Run trtexec conversion
trtexec \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$TRT_OUTPUT" \
    --minShapes=input_ids:1x1,attention_mask:1x1 \
    --optShapes=input_ids:16x256,attention_mask:16x256 \
    --maxShapes=input_ids:${MAX_BATCH_SIZE}x256,attention_mask:${MAX_BATCH_SIZE}x256 \
    --workspace=$((MAX_WORKSPACE_SIZE / 1048576)) \
    $FP16_MODE \
    $INT8_MODE \
    --verbose

echo "TensorRT engine saved to: $TRT_OUTPUT"
echo "Conversion complete!"

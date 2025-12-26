#!/bin/bash
# Self-check script that replicates grader steps
# This script simulates what the course grader will do

set -e

echo "======================================"
echo "Auto Ticket Classification - Self Check Script"
echo "======================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# Step 1: Check if in a clean state
echo ""
echo "Step 1: Checking repository state..."
if [ -d ".git" ]; then
    echo "  Git repository detected"
else
    echo "  Warning: Not a git repository"
fi

# Step 2: Create virtual environment and install dependencies
echo ""
echo "Step 2: Installing dependencies with Poetry..."
if command -v poetry &> /dev/null; then
    poetry install --no-interaction
    echo "  Dependencies installed successfully"
else
    echo "  Error: Poetry not found. Please install Poetry first."
    exit 1
fi

# Step 3: Install pre-commit hooks
echo ""
echo "Step 3: Installing pre-commit hooks..."
poetry run pre-commit install
echo "  Pre-commit hooks installed"

# Step 4: Run pre-commit on all files
echo ""
echo "Step 4: Running pre-commit on all files..."
poetry run pre-commit run -a || {
    echo "  Warning: pre-commit found issues (may auto-fix some)"
    echo "  Running pre-commit again after auto-fixes..."
    poetry run pre-commit run -a
}
echo "  Pre-commit checks passed"

# Step 5: Download/prepare data
echo ""
echo "Step 5: Preparing data..."
poetry run python -m ticket_triage_ml.commands download_data
echo "  Data prepared"

# Step 6: Preprocess data
echo ""
echo "Step 6: Preprocessing data..."
poetry run python -m ticket_triage_ml.commands preprocess
echo "  Preprocessing complete"

# Step 7: Run training (minimal epochs for quick check)
echo ""
echo "Step 7: Running training (minimal config)..."
poetry run python -m ticket_triage_ml.commands train \
    --overrides='["train.max_epochs=1", "train.batch_size=4"]'
echo "  Training complete"

# Step 8: Export to ONNX
echo ""
echo "Step 8: Exporting model to ONNX..."
poetry run python -m ticket_triage_ml.commands export_onnx
echo "  ONNX export complete"

# Step 9: Run inference (single text)
echo ""
echo "Step 9: Running single-text inference..."
poetry run python -m ticket_triage_ml.commands infer \
    --text="I cannot access my account and need help resetting my password"
echo "  Single-text inference complete"

# Step 10: Verify outputs exist
echo ""
echo "Step 10: Verifying outputs..."

REQUIRED_FILES=(
    "data/processed/train.parquet"
    "data/processed/val.parquet"
    "data/processed/test.parquet"
    "artifacts/label_maps.json"
    "artifacts/tokenizer/tokenizer_config.json"
    "artifacts/model.onnx"
    "checkpoints/best.ckpt"
    "plots/loss_curve.png"
    "plots/f1_curve.png"
    "plots/confusion_matrix_topic.png"
)

ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (missing)"
        ALL_EXIST=false
    fi
done

echo ""
echo "======================================"
if [ "$ALL_EXIST" = true ]; then
    echo "✓ ALL CHECKS PASSED"
    echo "======================================"
    exit 0
else
    echo "✗ SOME CHECKS FAILED"
    echo "======================================"
    exit 1
fi

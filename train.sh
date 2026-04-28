#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  train.sh  –  One-shot script to preprocess data and train
# ─────────────────────────────────────────────────────────────
set -e

CONFIG="configs/train.yaml"

echo "========================================"
echo " Step 1: Preprocess & sample BANKING77"
echo "========================================"
python scripts/preprocess_data.py --config "$CONFIG"

echo ""
echo "========================================"
echo " Step 2: Fine-tune with Unsloth"
echo "========================================"
python scripts/train.py --config "$CONFIG"

echo ""
echo "Training complete. Checkpoint saved under outputs/banking-intent/best_model"

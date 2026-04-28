#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  inference.sh  –  Run a single prediction or interactive mode
#
#  Single message:
#    bash inference.sh "I want to check my account balance"
#
#  Interactive mode:
#    bash inference.sh
# ─────────────────────────────────────────────────────────────
set -e

CONFIG="configs/inference.yaml"
MESSAGE="$1"   # optional: first argument is the message

if [ -n "$MESSAGE" ]; then
    python scripts/inference.py --config "$CONFIG" --message "$MESSAGE"
else
    python scripts/inference.py --config "$CONFIG"
fi

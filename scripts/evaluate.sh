#!/usr/bin/env bash
# Run evaluation on the held-out test set (zero-shot and MCTS modes).
# Usage: bash scripts/evaluate.sh ROUND [MODEL_PATH]
#
# Example:
#   bash scripts/evaluate.sh 1
#   bash scripts/evaluate.sh 1 models/agentq-round1

set -euo pipefail

ROUND=${1:-1}
MODEL=${2:-models/agentq-round${ROUND}}

echo "Evaluating round ${ROUND} model: ${MODEL}"
nvidia-smi || { echo "ERROR: nvidia-smi failed — check GPU"; exit 1; }

echo "--- Zero-shot evaluation ---"
CUDA_VISIBLE_DEVICES=0 python -m src.evaluate \
    --model "${MODEL}" \
    --test-set data/test_set.json \
    --mode zero_shot \
    --output "results/eval_round${ROUND}_zero_shot.json"

echo "--- MCTS evaluation ---"
CUDA_VISIBLE_DEVICES=0 python -m src.evaluate \
    --model "${MODEL}" \
    --test-set data/test_set.json \
    --mode mcts \
    --output "results/eval_round${ROUND}_mcts.json"

echo "Evaluation complete. Results in results/"

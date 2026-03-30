#!/usr/bin/env bash
# Launch MCTS data collection in a detached tmux session.
# Usage: bash scripts/collect.sh [ROUND] [MODEL_PATH]
#
# Examples:
#   bash scripts/collect.sh                     # round 1, base model
#   bash scripts/collect.sh 2 models/agentq-round1-merged

set -euo pipefail

ROUND=${1:-1}
MODEL=${2:-models/qwen2.5-coder-7b}
SESSION="mcts-round${ROUND}"
OUTPUT="trajectories/round${ROUND}.jsonl"

echo "Starting MCTS collection: round=${ROUND}, model=${MODEL}"
echo "Output: ${OUTPUT}"
echo "tmux session: ${SESSION}"

# Check GPU availability before launching
nvidia-smi || { echo "ERROR: nvidia-smi failed — check GPU"; exit 1; }

tmux new-session -d -s "${SESSION}" \
    "CUDA_VISIBLE_DEVICES=1 python3.11 -m src.mcts \
        --config configs/mcts_config.yaml \
        --model ${MODEL} \
        --dataset data/debugbench.json \
        --output ${OUTPUT}; \
    echo 'MCTS collection complete for round ${ROUND}'"

echo "Session '${SESSION}' started."
echo "  Attach:  tmux attach -t ${SESSION}"
echo "  Detach:  Ctrl-B then D"

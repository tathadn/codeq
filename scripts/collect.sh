#!/usr/bin/env bash
# Launch MCTS data collection in a detached screen session.
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
LOG="/tmp/mcts-round${ROUND}.log"

echo "Starting MCTS collection: round=${ROUND}, model=${MODEL}"
echo "Output: ${OUTPUT}"
echo "screen session: ${SESSION}"
echo "log file: ${LOG}"

# Check GPU availability before launching
nvidia-smi || { echo "ERROR: nvidia-smi failed — check GPU"; exit 1; }

VENV_PY="/home/lamassunobackup/tdebnath/codeqA/.venv/bin/python"

screen -dmS "${SESSION}" bash -c "CUDA_VISIBLE_DEVICES=0 ${VENV_PY} -m src.mcts \
        --config configs/mcts_config.yaml \
        --model ${MODEL} \
        --dataset data/debugbench.json \
        --output ${OUTPUT} 2>&1 | tee ${LOG}; \
    echo 'MCTS collection complete for round ${ROUND}'"

echo "Session '${SESSION}' started."
echo "  Attach:  screen -r ${SESSION}"
echo "  Detach:  Ctrl-A then D"
echo "  Log:     tail -f ${LOG}"

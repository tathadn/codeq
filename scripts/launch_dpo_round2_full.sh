#!/bin/bash
# Full DPO Round 2 training run on GPU 0 (fp32, filtered pairs).
# Designed to be launched detached: `screen -dmS dpo_round2_full scripts/launch_dpo_round2_full.sh`
# All output goes to logs/dpo_round2_full.log so the screen session can die freely.

set -u

cd /home/lamassunobackup/tdebnath/codeqB

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=logs/dpo_round2_full.log
mkdir -p logs

{
    echo "=== DPO Round 2 full run ==="
    echo "host:    $(hostname)"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "gpu:     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "config:  configs/train_config_round2_fp32.yaml"
    echo "data:    data/preferences/round2_filtered.jsonl"
    echo "output:  models/agentq-round2"
    echo "==="
} > "$LOG"

.venv/bin/python -u -m src.train_dpo \
    --config configs/train_config_round2_fp32.yaml \
    --model models/qwen2.5-coder-7b \
    --preferences data/preferences/round2_filtered.jsonl \
    --output models/agentq-round2 \
    --round 2 \
    >> "$LOG" 2>&1

EXIT_CODE=$?
{
    echo "==="
    echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "exit:     ${EXIT_CODE}"
    echo "==="
} >> "$LOG"
exit "$EXIT_CODE"

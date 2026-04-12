#!/bin/bash
# Full DPO Round 1 training run on GPU 1 (fp32, filtered pairs).
# Designed to be launched detached: `screen -dmS dpo_round1_full scripts/launch_dpo_round1_full.sh`
# All output goes to logs/dpo_round1_full.log so the screen session can die freely.

set -u

cd /home/lamassunobackup/tdebnath/codeqB

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=1
export TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=logs/dpo_round1_full.log
mkdir -p logs

{
    echo "=== DPO Round 1 full run ==="
    echo "host:    $(hostname)"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "gpu:     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "config:  configs/train_config_round1.yaml"
    echo "data:    data/preferences/round1_filtered.jsonl"
    echo "output:  models/agentq-round1"
    echo "==="
} > "$LOG"

.venv/bin/python -u -m src.train_dpo \
    --config configs/train_config_round1.yaml \
    --model models/qwen2.5-coder-7b \
    --preferences data/preferences/round1_filtered.jsonl \
    --output models/agentq-round1 \
    --round 1 \
    --resume_from_checkpoint checkpoints/round1/checkpoint-30 \
    >> "$LOG" 2>&1

EXIT_CODE=$?
{
    echo "==="
    echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "exit:     ${EXIT_CODE}"
    echo "==="
} >> "$LOG"
exit "$EXIT_CODE"

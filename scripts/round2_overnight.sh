#!/bin/bash
set -e
echo "=== Round 2 Overnight Run ===" | tee /tmp/round2_overnight.log
date | tee -a /tmp/round2_overnight.log

echo "=== Step 1: Full rewrite eval - base model ===" | tee -a /tmp/round2_overnight.log
CUDA_VISIBLE_DEVICES=0 python3.11 -m src.evaluate \
  --model /home/lamassunobackup/tdebnath/codeqA/models/qwen2.5-coder-7b \
  --test-set data/test_set.json --mode full_rewrite 2>&1 | tee -a /tmp/round2_overnight.log

echo "=== Step 2: Full rewrite eval - round 2 adapter ===" | tee -a /tmp/round2_overnight.log
CUDA_VISIBLE_DEVICES=0 python3.11 -m src.evaluate \
  --model /home/lamassunobackup/tdebnath/codeqA/models/qwen2.5-coder-7b \
  --adapter /home/lamassunobackup/tdebnath/codeqA/models/agentq-round2 \
  --test-set data/test_set.json --mode full_rewrite 2>&1 | tee -a /tmp/round2_overnight.log

echo "=== Step 3: MCTS eval - round 2 adapter, 50 tasks ===" | tee -a /tmp/round2_overnight.log
CUDA_VISIBLE_DEVICES=0 python3.11 -m src.evaluate \
  --model /home/lamassunobackup/tdebnath/codeqA/models/qwen2.5-coder-7b \
  --adapter /home/lamassunobackup/tdebnath/codeqA/models/agentq-round2 \
  --test-set data/test_set.json --mode mcts --limit 50 2>&1 | tee -a /tmp/round2_overnight.log

echo "=== Done ===" | tee -a /tmp/round2_overnight.log
date | tee -a /tmp/round2_overnight.log

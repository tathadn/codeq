# CodeQ: MCTS + DPO Self-Improving Code Debugger

An autonomous code debugging agent that improves itself through Monte Carlo Tree Search data collection and Direct Preference Optimization fine-tuning, inspired by [Agent Q (Putta et al., 2024)](https://arxiv.org/abs/2408.07199).

---

## Architecture: Two-Machine MCTS + DPO Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Machine A  (2× H100 NVL 95GB — inference + data collection) │
│                                                              │
│  1. MCTS search over buggy code repair space                 │
│     • Proposes K full-rewrite solutions per node (temp 0.8)  │
│     • Tests each in subprocess sandbox immediately           │
│     • AI self-critique ranks rewrites (temp 0.2)             │
│     • UCB1 selection balances exploit vs. explore            │
│     • Backpropagates pass/fail reward                        │
│                                                              │
│  2. Preference pair construction                             │
│     • Pairwise sibling comparison: blended Q = α·Q_mcts     │
│       + (1-α)·Q_ai                                           │
│     • Emit (prompt, chosen_rewrite, rejected_rewrite) if     │
│       |Q_chosen - Q_rejected| > θ_threshold                  │
│                                                              │
│  3. Evaluation (zero_shot / mcts / full_rewrite modes)       │
│  4. LoRA adapter merge for next iteration                    │
└──────────────────────┬──────────────────────────────────────┘
                       │  scp preference pairs  ↓
                       │  scp LoRA adapters     ↑
┌──────────────────────▼──────────────────────────────────────┐
│  Machine B  (1× H100 94GB — DPO training)                    │
│                                                              │
│  5. DPO fine-tuning with LoRA (r=32, α=64, 7 target modules) │
│     • Model in full bf16 (~14 GB), gradient checkpointing    │
│     • TRL DPOTrainer, β=0.1, effective batch size 16         │
│     • ~19 min per round on 1,352 preference pairs            │
│  6. Export adapter_model.safetensors → Machine A             │
└─────────────────────────────────────────────────────────────┘
```

**Iteration loop:** Machine A collects → Machine B trains → Machine A evaluates and merges → repeat.

---

## Tech Stack

| Component | Library / Tool |
|-----------|---------------|
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Inference (Machine A) | HuggingFace `transformers` + `bitsandbytes` 4-bit NF4 |
| Training (Machine B) | HuggingFace `trl` (`DPOTrainer`), `peft` (LoRA), full bf16 |
| MCTS | Custom UCB1 implementation (`src/mcts.py`) |
| Sandbox | `subprocess` + `RLIMIT_AS/CPU/FSIZE` + `pytest` |
| Config | `pydantic` + YAML |
| Experiment tracking | W&B |
| Tests | `pytest` (no GPU required — all model calls mocked) |

---

## Round 1 Results

| Run | Mode | Model | Tasks (deduped) | Solved | Pass Rate |
|-----|------|-------|-----------------|--------|-----------|
| Baseline | `zero_shot` structured | Base | 123 | ~1 | ~0.4% † |
| A | `full_rewrite` | Base | 50 | 18 | **36.0%** |
| B | `full_rewrite` | Round 1 DPO | 50 | 19 | **38.0%** |
| C | `mcts` rewrite (K=2, r=10, d=2) | Base | 5 ‡ | 5 | **100%** |

† The structured zero-shot baseline was an eval artifact: the model mostly failed to emit `THOUGHT/CODE_ACTION` format. `full_rewrite` is the correct zero-shot baseline.
‡ Dry run of the refactored MCTS rewrite mode; full eval pending.

**Round 1 DPO training summary:** Peak reward accuracy 71.9% at step 150 (overtraining after step 150 → settled at 59.7% eval). Best checkpoint is step 150. Training: ~19 min, 1,352 preference pairs, 81M trainable LoRA params (1.05% of 7B).

---

## Reproducing

### Setup (both machines)

```bash
pip install -r requirements.txt
# Download base model once:
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir models/qwen2.5-coder-7b
```

### Machine A — MCTS data collection

```bash
# Always use GPU 1 (GPU 0 is shared)
CUDA_VISIBLE_DEVICES=1 python -m src.mcts \
  --config configs/mcts_config.yaml \
  --model models/qwen2.5-coder-7b \
  --dataset data/debugbench.json \
  --output trajectories/round1.jsonl

# Build preference pairs (CPU only)
python -m src.preferences \
  --input trajectories/round1.jsonl \
  --output data/preferences/round1.jsonl

# Sync to Machine B
bash scripts/sync_to_b.sh
```

### Machine B — DPO training

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train_dpo \
  --config configs/train_config.yaml \
  --model models/qwen2.5-coder-7b \
  --preferences data/preferences/round1.jsonl \
  --output models/agentq-round1

bash scripts/sync_to_a.sh
```

### Machine A — Evaluation

```bash
# Merge adapter into base for next round's MCTS
python -m src.merge_lora \
  --base models/qwen2.5-coder-7b \
  --adapter models/agentq-round1 \
  --output models/agentq-round1-merged

# Evaluate
CUDA_VISIBLE_DEVICES=1 python -m src.evaluate \
  --model models/agentq-round1-merged \
  --test-set data/test_set.json \
  --mode full_rewrite
```

### Tests

```bash
pytest tests/ -v        # no GPU required — all model calls mocked
```

---

## Reference

Putta, S., Mills, E., Garg, N., Motwani, S., Finn, C., Garg, D., & Rafailov, R. (2024).
**Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents.**
https://arxiv.org/abs/2408.07199

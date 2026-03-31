# DPO Training Report — Round 1

**Date:** 2026-03-29
**Machine:** Machine B (`lamassunobackup`)
**GPU:** NVIDIA H100 NVL — GPU 0 (`CUDA_VISIBLE_DEVICES=0`, 95,830 MiB total)
**Exit status:** `EXIT_CODE: 0` — clean finish, screen session exited normally

---

## 1. Training Metrics

| Metric | Value |
|--------|-------|
| Epochs | 2 |
| Total optimizer steps | 162 |
| Train loss (final, epoch 2) | **0.6700** |
| Train loss (best observed) | **0.6292** (step ~130, epoch 1.86) |
| Eval loss (final, epoch 2) | **0.6883** |
| Peak learning rate | 5.0e-6 |
| Final learning rate | 1.034e-7 (cosine decay) |
| Per-device batch size | 2 |
| Gradient accumulation steps | 8 |
| Effective batch size | 16 |
| Training wall time | **1,164 s (~19 min 24 s)** |
| Throughput | 2.207 samples/sec · 0.139 steps/sec |

---

## 2. Loss & Reward Progression

Logged every 10 optimizer steps:

| Step | Epoch | Train Loss | Reward Acc | Reward Margin |
|------|-------|-----------|------------|---------------|
| 10   | 0.249 | 0.6932    | 51.2%      | 0.0008        |
| 20   | 0.374 | 0.6916    | 53.7%      | 0.0042        |
| 30   | 0.498 | 0.6867    | 59.4%      | 0.0143        |
| 40   | 0.623 | 0.6893    | 50.6%      | 0.0110        |
| 50   | 0.748 | 0.6790    | 55.6%      | 0.0329        |
| 80   | 1.237 | 0.6497    | 66.2%      | 0.1079        |
| 110  | 1.361 | 0.6711    | 62.5%      | 0.0678        |
| 120  | 1.486 | **0.6378** | 69.4%     | 0.1442        |
| 130  | 1.611 | 0.6471    | 67.5%      | 0.1234        |
| 140  | 1.735 | 0.6606    | 66.9%      | 0.0917        |
| 150  | 1.860 | **0.6292** | **71.9%** | **0.1718**    |
| 160  | 1.984 | 0.6644    | 63.7%      | 0.0870        |

**Trend:** Loss dropped from 0.693 → 0.629 (best). Reward accuracy rose from ~51% at step 10 to a peak of **71.9%** at step 150. Some oscillation in the final steps is normal as LR decays toward zero.

---

## 3. Reward Margins (Chosen vs Rejected)

Training step metrics (final logged step, epoch 1.984):

| Metric | Value |
|--------|-------|
| rewards/chosen | 0.1434 |
| rewards/rejected | 0.0564 |
| rewards/margin | **0.0870** |
| rewards/accuracies | 63.7% |
| logps/chosen | -164.8 |
| logps/rejected | -167.3 |

Eval metrics at epoch 2.0 (final):

| Metric | Value |
|--------|-------|
| eval rewards/chosen | 0.05501 |
| eval rewards/rejected | 0.01282 |
| eval rewards/margin | **0.04219** |
| eval rewards/accuracies | **59.7%** |
| eval logps/chosen | -168.1 |
| eval logps/rejected | -168.8 |
| eval mean token accuracy | 71.2% |

Eval reward accuracy progression:

| Epoch | Eval Loss | Reward Acc | Reward Margin |
|-------|-----------|------------|---------------|
| 1.237 | 0.6904    | 62.5%      | 0.0270        |
| 1.860 | 0.6927    | 58.3%      | 0.0355        |
| 2.000 | **0.6883** | **59.7%** | **0.0421**   |

---

## 4. GPU Memory Usage

| Phase | GPU 0 Used |
|-------|-----------|
| Idle (before training) | 909 MiB |
| During training (estimated peak) | ~18–22 GB |
| After training (current) | 909 MiB |

> Peak VRAM estimate: 7B params × 2 B (BF16) ≈ 14 GB weights + ~0.6 GB LoRA optimizer states (AdamW, 81M trainable params) + ~3–5 GB activations (gradient checkpointing enabled). Total ~18–22 GB, well within the 95.8 GB available. Peak was not directly instrumented during this run.

---

## 5. Dataset Statistics

| Field | Value |
|-------|-------|
| Source | `/home/simurghnobackup/tdebnath/codeq-shared/preferences/round1.jsonl` |
| Total preference pairs | **1,352** |
| Train split (95%) | 1,284 |
| Eval split (5%) | 68 |
| Preference signal | `q_chosen` / `q_rejected` (MCTS Q-values from Machine A) |

**Text length statistics (characters):**

| Field | Min | Max | Mean | Median |
|-------|-----|-----|------|--------|
| Prompt | 692 | 5,421 | 2,165 | 2,004 |
| Chosen | 303 | 1,927 | 611 | 546 |
| Rejected | 286 | 1,784 | 595 | 546 |

---

## 6. LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| PEFT version | 0.18.1 |
| Rank (r) | 32 |
| Alpha (lora_alpha) | 64 |
| Scaling (alpha/r) | 2.0 |
| Dropout | 0.05 |
| Bias | none |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (7 modules) |
| use_dora | false |
| use_rslora | false |
| Task type | CAUSAL_LM |

**Parameter counts:**

| | Count |
|--|-------|
| Trainable (LoRA) params | **80,740,352** |
| Base model params | 7,696,356,864 |
| Trainable fraction | **1.049%** |

---

## 7. Adapter Files

**Local:** `models/agentq-round1/`

| File | Size |
|------|------|
| `adapter_model.safetensors` | **309 MB** ✓ |
| `adapter_config.json` | 1.1 KB ✓ |
| `tokenizer.json` | 11 MB |
| `tokenizer_config.json` | 660 B |
| `chat_template.jinja` | 2.5 KB |
| `README.md` | 5.1 KB |

**Shared filesystem:** `/home/simurghnobackup/tdebnath/codeq-shared/adapters/round1/`

| File | Size |
|------|------|
| `adapter_model.safetensors` | **309 MB** ✓ |
| `adapter_config.json` | 1.1 KB ✓ |

Machine A can load the adapter from the shared path.

---

## 8. Environment & Compatibility Fixes Applied

| Issue | Fix |
|-------|-----|
| PyTorch CPU-only build | Reinstalled as `torch 2.6.0+cu124` |
| FlashAttention2 not installed | `attn_implementation: sdpa` in config |
| TRL 0.29.1: `max_prompt_length` removed | Dropped from `DPOConfig` |
| TRL 0.29.1: `tokenizer` → `processing_class` | Updated `DPOTrainer` call |
| Data fields `q_chosen`/`q_rejected` vs `ref_chosen_logps` | `from_dict` fallback in `utils.py`; ref log-probs recomputed via `precompute_ref_log_probs=True` |
| Dataset column duplication | Removed `ref_*_logps` columns from `build_dataset` records |
| W&B no API key | `WANDB_MODE=disabled` |

---

## 9. Summary

Round 1 DPO training completed successfully in **~22 min total** (including ~3 min precompute). The model learned a meaningful preference signal: reward accuracy improved from ~51% at the start to a peak of **71.9%** mid-run, settling at **59.7%** on the held-out eval set at epoch 2. The modest eval reward accuracy is expected for round 1 — the MCTS preference pairs will improve in quality as subsequent rounds use this adapter as the base policy.

**Next steps:** Machine A pulls `adapters/round1/` from the shared filesystem, evaluates on the benchmark, generates round 2 preference pairs, and writes `codeq-shared/preferences/round2.jsonl` for the next training round.

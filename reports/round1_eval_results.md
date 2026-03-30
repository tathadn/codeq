# Round 1 Evaluation Results

## Summary

| Run | Mode | Model | N | Solved | Pass Rate |
|-----|------|-------|---|--------|-----------|
| Baseline (old) | `zero_shot` (structured format) | Base | 263 | 1 | 0.38% |
| A | `full_rewrite` | Base | 50 | 18 | 36.0% |
| B | `full_rewrite` | Round 1 DPO adapter | 50 | 19 | 38.0% |
| C | `mcts` (K=2, rollouts=5, depth=4) | Base | 20 | — | pending |

## Key Finding: Evaluation Format Was Masking Model Capability

The structured `zero_shot` mode required the model to output a PLAN/THOUGHT/CODE_ACTION format before any code was executed. The base model largely failed to follow this format, resulting in near-zero pass rate (0.38%, 1/263).

Switching to `full_rewrite` — a simple prompt asking for the complete corrected code — revealed the model's actual capability: **36.0% pass rate on the same task distribution**. The 0.38% figure was an eval artifact, not a true measure of model quality.

**Implication:** All previous zero-shot baselines using the structured format should be treated as invalid. The `full_rewrite` mode is now the canonical zero-shot baseline.

## DPO Improvement (Round 1)

- Base `full_rewrite`: 36.0% (18/50)
- Round 1 DPO `full_rewrite`: 38.0% (19/50)
- **Delta: +2pp, +1 problem solved**

Modest but directionally correct. One round of DPO on MCTS-generated preference pairs shifted the model in the right direction. The signal is small on 50 problems; more problems or rounds needed to confirm.

## Round 1 Training Metrics

- Preference pairs generated: 1,352
- Training time: 19 min
- Peak reward accuracy: **71.9% at step 150**
- Final reward accuracy: 59.7% (settled after step 150)
- Diagnosis: **overtraining** — the model improved through step 150 then degraded. Best checkpoint is step 150, not the final checkpoint.

The round 1 DPO adapter (`models/agentq-round1`) is the final checkpoint, which explains why the +2pp gain is lower than what step 150 likely achieved.

## Pending: Eval C (MCTS)

Running in screen session `mcts-eval`. Check progress:
```
grep "Evaluating task" /tmp/mcts_eval_c.log
tail -20 /tmp/mcts_eval_c.log
```
Results written to `results/eval_C_base_mcts_k2_r5_d4.json` on completion.

Expected runtime: 3–5 hours (10–15 min/task × 20 tasks).

---

# Round 2 Plan

## Goals

1. Recover the step-150 gains lost to overtraining
2. Improve DPO signal quality
3. Establish a valid MCTS baseline (eval C result)

## Changes for Round 2

### 1. Use Best Checkpoint (Step 150)

The peak reward accuracy of 71.9% at step 150 vs. 59.7% at end-of-training indicates the model overfit the preference pairs. For round 2:

- Save checkpoints every 25–50 steps during DPO training
- Evaluate `full_rewrite` pass rate on a small validation split at each checkpoint
- Use the checkpoint with highest pass rate, not the final one
- Add `--load_best_model_at_end` or manual checkpoint selection logic to `src/train_dpo.py`

### 2. Early Stopping

- Monitor reward accuracy on a held-out validation split
- Stop training when validation reward accuracy stops improving (patience = 2–3 evals)
- Prevents the step-150 → degradation pattern from repeating

### 3. Add Few-Shot Examples to `full_rewrite` Prompt

The `full_rewrite` prompt is currently zero-shot. Adding 1–2 examples of (buggy code → fixed code) in the system prompt may improve format compliance and fix quality without any additional training.

### 4. WandB Logging

- Enable WandB in `train_dpo.py` to track reward accuracy, loss, and per-step metrics
- Log eval pass rate at checkpoints so training and eval curves are visible together
- Use `wandb_project: codeq-dpo-round2` to separate from round 1 runs

### 5. More Preference Pairs (if MCTS baseline warrants it)

If eval C shows MCTS solves significantly more than 36% (e.g., >50%), collect more MCTS trajectories before round 2 DPO. Higher-quality chosen/rejected pairs from harder tasks will produce stronger training signal.

## Decision Gate

Wait for eval C result before starting round 2 data collection:
- If MCTS >> 36%: collect more trajectories, proceed with round 2 DPO
- If MCTS ≈ 36%: MCTS search is not helping much; reconsider search strategy or K/rollout budget

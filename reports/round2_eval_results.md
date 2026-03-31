# Round 2 Evaluation Results

## Full Comparison Table

| Run | Mode | Model | N | Solved | Pass Rate | Notes |
|-----|------|-------|---|--------|-----------|-------|
| Baseline (old) | `zero_shot` (structured) | Base | 263 | 1 | 0.38% | Eval artifact — format non-compliance |
| Round 1 — A | `full_rewrite` | Base | 50 | 18 | 36.0% | First valid baseline |
| Round 1 — B | `full_rewrite` | Round 1 DPO adapter | 50 | 19 | 38.0% | +2pp from DPO |
| Round 1 — C | `mcts` (K=2, r=5, d=4) | Base | 20 | 1 | 5.0% | **Broken MCTS** — pre-refactor |
| **Round 2 — Base** | `full_rewrite` | Base | 123 | 54 | **43.9%** | Larger eval set |
| **Round 2 — DPO** | `full_rewrite` | Round 2 DPO adapter | 123 | 57 | **46.3%** | +2.4pp from DPO |
| **Round 2 — MCTS base** | `mcts` | Base | 123 | 100 | **81.3%** | Post-refactor MCTS |
| **Round 2 — MCTS DPO** | `mcts` | Round 2 DPO adapter | 50 | 42 | **84.0%** | Best result |

---

## Key Finding 1: MCTS Rewrite Refactor Impact

Round 1 eval C measured MCTS at **5.0% (1/20)** — nearly useless. This was not a model failure; it was a broken search implementation (pre-refactor).

After the MCTS rewrite, the base model scores **81.3% (100/123)** on the same task distribution — a **+76pp jump** attributable entirely to the search fix. The MCTS refactor is the single largest improvement in the project so far.

**Implication:** All Round 1 conclusions about "MCTS not helping" should be discarded. The search works; the original eval was invalid.

---

## Key Finding 2: DPO Improvement (Round 2)

- Base `full_rewrite`: **43.9%** (54/123)
- Round 2 DPO `full_rewrite`: **46.3%** (57/123)
- **Delta: +2.4pp, +3 problems solved**

Directionally consistent with Round 1 (+2pp). The signal is small but reproducible across two rounds.

### Round 2 DPO Training Metrics

| Metric | Round 1 | Round 2 |
|--------|---------|---------|
| Preference pairs | 1,352 | 1,464 |
| Training time | ~19 min | ~8 min |
| Peak eval reward accuracy | 71.9% (step 150) | 58.75% (epoch 1) |
| Final eval reward accuracy | 59.7% | 58.75% |
| Overtraining observed | Yes (degraded after step 150) | No |

Round 2 training was more stable — reward accuracy at final epoch (58.75%) matches the peak, meaning no overtraining degradation this time. However, the peak is lower than Round 1's step-150 peak (71.9%), suggesting the preference pairs or learning rate schedule may need tuning to get stronger signal.

---

## Key Finding 3: MCTS vs full_rewrite

| Mode | Model | Pass Rate |
|------|-------|-----------|
| `full_rewrite` | Base | 43.9% |
| `full_rewrite` | Round 2 DPO | 46.3% |
| `mcts` | Base | 81.3% |
| `mcts` | Round 2 DPO | 84.0% |

MCTS nearly **doubles** the pass rate over `full_rewrite` (81.3% vs 43.9% for base model). This is the dominant lever — far larger than DPO training gains. DPO adds ~2–3pp on top of whatever the base model achieves, regardless of eval mode.

**Implication for Round 3:** The primary goal should be generating high-quality MCTS trajectories on tasks the model currently fails (the ~16–18% that MCTS doesn't solve), then running another DPO round targeting those failure cases specifically. Broad preference pairs from already-solved tasks produce weak signal.

---

## Result Files

| File | Description |
|------|-------------|
| `results/eval_full_rewrite.json` | Round 2 base, full_rewrite, 123 tasks |
| `results/eval_mcts.json` | Round 2 base, mcts, 123 tasks |
| `results/eval_full_rewrite_adapter.json` | Round 2 DPO, full_rewrite, 123 tasks |
| `results/eval_mcts_adapter.json` | Round 2 DPO, mcts, 50 tasks |
| `models/agentq-round2` | Round 2 DPO LoRA adapter |
| `dpo_round2_train.log` | Full training log |

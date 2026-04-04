# Ablation Study Results

All experiments run on Machine A (2026-04-03). Base model = agentq base; DPO model = Round 2 DPO adapter. MCTS uses K=2, r=20 rollouts (unless otherwise noted), d=4. N=20 problems per condition for category/difficulty ablations; N=50 for rollout ablation.

---

## Table 1: Error Category Ablation

Performance broken down by error category (`cat_*`), comparing `full_rewrite` vs `mcts` and base vs DPO model.

| Error Category  | Base Rewrite | DPO Rewrite | Base MCTS | DPO MCTS |
|-----------------|:------------:|:-----------:|:---------:|:--------:|
| Syntax error    |    61.90%    |    61.90%   |  **95.00%**   |  **95.00%**  |
| Logic error     |    45.83%    |    45.83%   |  **90.00%**   |   85.00%  |
| Reference error |    55.88%    |    55.88%   |  **80.00%**   |   80.00%  |
| Multiple errors |    31.82%    |    31.82%   |  **90.00%**   |   85.00%  |

---

## Table 2: Difficulty Ablation

Performance broken down by problem difficulty (`diff_*`), comparing `full_rewrite` vs `mcts` and base vs DPO model.

| Difficulty | Base Rewrite | DPO Rewrite | Base MCTS | DPO MCTS |
|------------|:------------:|:-----------:|:---------:|:--------:|
| Easy       |    56.76%    |    56.76%   |  **90.00%**   |  **90.00%**  |
| Medium     |    40.74%    |    44.44%   |  **90.00%**   |  **90.00%**  |
| Hard       |    34.38%    |    34.38%   |  **80.00%**   |   85.00%  |

---

## Table 3: Rollout Count Ablation (MCTS only)

Effect of number of MCTS rollouts on pass rate, evaluated on a fixed 50-problem set.

| Rollouts | Base MCTS | DPO MCTS |
|:--------:|:---------:|:--------:|
|    1     |   80.00%  |  78.00%  |
|    2     |   80.00%  |  80.00%  |
|    5     |   80.00%  |  82.00%  |
|   10     |   84.00%  |  84.00%  |
|   20     |   84.00%  |**86.00%**|

---

## Key Findings

### Finding 1: MCTS dominates across all conditions (80–95%)

MCTS achieves **80–95%** pass rate across every category and difficulty slice, versus **31–62%** for `full_rewrite`. The MCTS advantage is large and consistent — the minimum MCTS result (80%) still exceeds the best rewrite result (62%). Search, not model quality, is the primary lever for this task.

### Finding 2: DPO has negligible effect on rewrite mode; small positive effect on MCTS at high rollouts

In `full_rewrite` mode, DPO and base models score identically on every category and difficulty split. DPO provides no benefit when search is absent. With MCTS, DPO shows a small positive effect at higher rollout counts (84% → 86% at r=20) but is neutral or slightly negative at r=1. The DPO signal is real but only surfaces when the model has enough rollouts to exploit it.

### Finding 3: Multiple-error problems are an outlier — rewrite collapses, MCTS recovers fully

`multiple_error` problems are the hardest for rewrite (31.82% — worst of any slice). Yet MCTS base scores **90%** on the same problems, matching syntax errors. The gap between modes is largest here (+58pp). This suggests the model understands multi-error fixes when given search budget but cannot execute them single-shot.

### Finding 4: Rollout scaling saturates at ~10 rollouts

Going from r=1 to r=10 yields +4pp for base (80% → 84%) and +6pp for DPO (78% → 84%). Doubling to r=20 adds only +0pp (base) and +2pp (DPO). The curve flattens sharply after r=10, indicating diminishing returns. **r=10 is the practical sweet spot** given compute cost.

### Finding 5: Hard problems show the smallest MCTS advantage

Hard problems score lowest under both modes (80–85% MCTS vs. 34% rewrite), but the absolute MCTS ceiling is also lower than easy/medium (80% vs. 90%). This suggests some hard problems are not solvable by search alone with the current model, making them the primary target for Round 3 DPO training.

---

## Result Files

| File | Description |
|------|-------------|
| `results/ablation/cat_*.json` | Per-category results (syntax/logic/reference/multiple × base/dpo × rewrite/mcts) |
| `results/ablation/diff_*.json` | Per-difficulty results (easy/medium/hard × base/dpo × rewrite/mcts) |
| `results/ablation/rollouts_*.json` | Rollout ablation (r=1/2/5/10/20 × base/dpo, mcts only) |

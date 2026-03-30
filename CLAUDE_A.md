# CodeQ: MCTS + DPO Code Debugging Agent
# Machine A — Data Collection & Inference Node

An autonomous code debugging agent that learns to fix bugs through Monte Carlo Tree Search, AI self-critique, and Direct Preference Optimization. Inspired by Agent Q (Putta et al., 2024).

**This is Machine A.** It handles MCTS data collection, preference pair construction, evaluation, and subprocess-based sandboxed code execution. The model runs in 4-bit quantization for inference only (~4-6 GB VRAM). Machine B (separate server) handles DPO training.

## Tech Stack

- Python 3.10, PyTorch 2.x (CUDA 12.1)
- Hugging Face: `transformers`, `accelerate`, `peft`, `bitsandbytes`, `datasets`
- Base model: `Qwen/Qwen2.5-Coder-7B-Instruct` (loaded in 4-bit via `BitsAndBytesConfig`)
- Docker for sandboxed code execution
- W&B for experiment tracking

## This Machine's Role

- Run MCTS search to collect debugging trajectories (the most time-consuming step, 12-18 hours)
- Run the AI self-critique (critic) to rank proposed actions at each MCTS node
- Execute code fixes safely in subprocess sandbox (resource-limited, optionally network-isolated)
- Build preference pairs from MCTS tree data (CPU-only)
- Run evaluation (zero-shot and MCTS modes) on held-out test set
- Merge LoRA adapters received from Machine B into base model for next iteration
- Send trajectory/preference data to Machine B; receive LoRA adapters back

## Hardware

- **Two NVIDIA H100 NVL 95GB GPUs**
- GPU 0 is **shared with other users** and is often busy
- GPU 1 is **typically free** — prefer it for all jobs
- ALWAYS run `nvidia-smi` before launching GPU jobs
- ALWAYS prefix GPU commands with `CUDA_VISIBLE_DEVICES=1`
- Model loaded in 4-bit uses ~4-6 GB VRAM — leaves room for other users
- Use `tmux` for all long-running jobs (MCTS collection, evaluation)

## Project Structure

```
codeq/
├── CLAUDE.md                # This file
├── README.md
├── requirements.txt
├── Dockerfile.sandbox       # NOT USED — Docker unavailable on this machine
├── src/
│   ├── __init__.py
│   ├── agent.py             # Agent prompt templates, action parsing, observation formatting
│   ├── mcts.py              # MCTSDebugger: tree search engine with UCB1 selection
│   ├── critic.py            # AI self-critique: iterative action ranking
│   ├── sandbox.py           # CodeSandbox: Docker-based safe code execution
│   ├── preferences.py       # Preference pair construction from MCTS trees
│   ├── train_dpo.py         # [NOT USED ON THIS MACHINE — runs on Machine B]
│   ├── evaluate.py          # Evaluation harness (zero-shot and MCTS modes)
│   ├── merge_lora.py        # Merge LoRA adapters into base model between rounds
│   └── utils.py             # Shared helpers: logging, serialization, config
├── configs/
│   ├── mcts_config.yaml     # MCTS hyperparameters (K, rollouts, depth, c_exp, alpha)
│   └── eval_config.yaml     # Evaluation settings
├── data/
│   ├── debugbench.json      # Training bug dataset
│   ├── test_set.json        # Held-out evaluation set
│   └── preferences/         # Generated DPO preference pairs per round
├── trajectories/            # MCTS rollout data per round
├── models/
│   ├── qwen2.5-coder-7b/   # Base model (downloaded once)
│   ├── agentq-round1/       # LoRA adapters received from Machine B
│   └── agentq-round1-merged/ # Merged model for next round's MCTS
├── scripts/
│   ├── collect.sh           # Launch MCTS collection in tmux
│   ├── sync_to_b.sh         # Send preference data to Machine B
│   ├── sync_from_b.sh       # Receive LoRA adapters from Machine B
│   └── evaluate.sh          # Run evaluation suite
└── tests/
    ├── test_agent.py
    ├── test_mcts.py
    ├── test_sandbox.py
    └── test_preferences.py
```

## Commands for This Machine

```bash
# Setup
pip install -r requirements.txt
# huggingface-cli is installed as ~/.local/bin/hf (add ~/.local/bin to PATH or use full path)
~/.local/bin/hf download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir models/qwen2.5-coder-7b

# Run tests
pytest tests/ -v
pytest tests/test_mcts.py -v           # MCTS logic only
pytest tests/test_sandbox.py -v        # sandbox only

# === PRIMARY WORKFLOW: MCTS Data Collection ===
# ALWAYS run inside tmux in case SSH drops
tmux new -s mcts
CUDA_VISIBLE_DEVICES=1 python -m src.mcts \
  --config configs/mcts_config.yaml \
  --model models/qwen2.5-coder-7b \
  --dataset data/debugbench.json \
  --output trajectories/round1.jsonl
# Detach: Ctrl-B then D. Reattach: tmux attach -t mcts

# For round 2+ use the merged model from previous round:
CUDA_VISIBLE_DEVICES=1 python -m src.mcts \
  --config configs/mcts_config.yaml \
  --model models/agentq-round1-merged \
  --dataset data/debugbench.json \
  --output trajectories/round2.jsonl

# === Build preference pairs (CPU only, no GPU needed) ===
python -m src.preferences \
  --input trajectories/round1.jsonl \
  --output data/preferences/round1.jsonl \
  --alpha 0.5 --threshold 0.2

# === Transfer data to Machine B for training ===
scp data/preferences/round1.jsonl user@MACHINE_B_HOST:~/codeq/data/preferences/

# === Receive trained LoRA adapters from Machine B ===
scp -r user@MACHINE_B_HOST:~/codeq/models/agentq-round1/adapter_* models/agentq-round1/
scp user@MACHINE_B_HOST:~/codeq/models/agentq-round1/adapter_config.json models/agentq-round1/

# === Merge LoRA into base for next iteration ===
python -m src.merge_lora \
  --base models/qwen2.5-coder-7b \
  --adapter models/agentq-round1 \
  --output models/agentq-round1-merged

# === Evaluation ===
CUDA_VISIBLE_DEVICES=1 python -m src.evaluate \
  --model models/agentq-round1 \
  --test-set data/test_set.json \
  --mode zero_shot

CUDA_VISIBLE_DEVICES=1 python -m src.evaluate \
  --model models/agentq-round1 \
  --test-set data/test_set.json \
  --mode mcts
```

## Architecture: Modules on This Machine

### MCTS Engine (`src/mcts.py`)

The core search loop. Each task (buggy code + tests) becomes a search tree:

1. **Root** = initial buggy code state + failing test output
2. **Expansion** = LLM proposes K=4 actions (temperature 0.8 for diversity)
3. **Critic ranking** = same LLM ranks proposals (temperature 0.2)
4. **Selection** = UCB1 balances exploitation (high Q) vs exploration (low visit count)
5. **Rollout** = greedy policy execution until tests pass or max depth (8 steps)
6. **Backpropagation** = reward (0 or 1) updates Q-values bottom-up

Key types:
- `MCTSNode`: dataclass with `state`, `children`, `visit_count`, `total_value`, `ai_score`, `q_value` property, `ucb1()` method
- `MCTSDebugger`: main class with `search(task) -> dict`, `propose_actions()`, `rank_actions()`, `_rollout()`, `_apply_action()`

Download and load model with 4-bit quantization:
```python
# Download once: ~/.local/bin/hf download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir ./models/qwen2.5-coder-7b

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(path, quantization_config=quant_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(path)
```

Qwen2.5 uses ChatML-style templates (`<|im_start|>`, `<|im_end|>`). Always use `tokenizer.apply_chat_template()` for prompt formatting — never construct ChatML tags manually:
```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": observation},
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
```

### Agent Actions (`src/agent.py`)

Composite actions following Agent Q paper (Section 3.1):
- `PLAN` (first step only): high-level fix strategy
- `THOUGHT`: chain-of-thought reasoning about current state
- `CODE_ACTION`: one of `EDIT <start>-<end> <code>`, `INSERT <after> <code>`, `DELETE <start>-<end>`, `RUN_TESTS`, `SUBMIT`
- `EXPLANATION`: why this action helps
- `STATUS`: `CONTINUE` or `DONE`

Parse with regex, always handle parse failures gracefully (return None, count as metric).

### AI Self-Critique (`src/critic.py`)

Ranks K proposed actions by iteratively asking the LLM to pick the best remaining action. Returns normalized scores in [0, 1]. Uses low temperature (0.2) for consistent rankings.

### Code Sandbox (`src/sandbox.py`)

**Docker is not available on this machine.** Uses `subprocess.run` with resource limits and an optional network namespace:
- **Memory**: `RLIMIT_AS` = 512 MB virtual address space (applied via `preexec_fn`)
- **CPU time**: `RLIMIT_CPU` = 30 s hard stop + `subprocess.run(timeout=30)` wall-clock kill
- **File writes**: `RLIMIT_FSIZE` = 10 MB per file
- **Network**: disabled via `unshare -n` if unprivileged user namespaces are available, otherwise unrestricted
- Runs pytest in a `tempfile.TemporaryDirectory`; parses stdout+stderr for pass/fail
- Return `{"success": bool, "output": str, "all_passed": bool}`

### Preference Pairs (`src/preferences.py`)

Extract step-level pairs from MCTS tree (Algorithm 1 in paper):
- At each internal node, compare sibling children pairwise
- Blended Q-value: `Q = alpha * Q_mcts + (1-alpha) * Q_ai`
- Only create pair if `|Q_chosen - Q_rejected| > theta_threshold`
- Output format: `{"prompt": str, "chosen": str, "rejected": str}`

## Key Formulas

- **UCB1**: `a* = argmax[Q(h,a) + c_exp * sqrt(log(N(h)) / (1 + N(h')))]`
- **Backprop**: `Q(h,a) <- (Q(h,a)*N(h,a) + R) / (N(h,a) + 1)`
- **Blended Q**: `Q = alpha * Q_mcts + (1-alpha) * Q_ai`

## Code Style

- Python 3.10+, type hints on all function signatures
- Use `dataclasses` for data structures, `pydantic` for config validation
- Prefer composition over inheritance
- Docstrings: Google style, include Args/Returns/Raises
- Use `logging` module (not print), configure with `logging.basicConfig`
- Constants in UPPER_SNAKE_CASE at module top
- Config via YAML files parsed with `pydantic` models, not hardcoded values
- Use `pathlib.Path` not string paths
- Imports: stdlib, then third-party, then local, separated by blank lines

## Critical Rules for This Machine

- NEVER hardcode model paths or hyperparameters — always load from config YAML
- NEVER run untested code outside the subprocess sandbox during MCTS — always use `CodeSandbox.execute()` (Docker is not available; sandbox uses subprocess + RLIMIT resource limits)
- ALWAYS set `CUDA_VISIBLE_DEVICES=1` before any GPU operation (GPU 0 is shared and often busy; GPU 1 is typically free)
- ALWAYS run MCTS collection inside `tmux` — it takes 12-18 hours
- ALWAYS save MCTS progress incrementally (JSONL append mode) with periodic checkpoints
- ALWAYS handle model output parsing failures gracefully — return None, log the failure, continue
- Use `bitsandbytes` 4-bit quantization for all model loading on this machine
- NEVER load the model in full bf16 on this machine — that's Machine B's job
- NEVER run `train_dpo.py` on this machine — training happens on Machine B
- Test with `pytest` before committing; use `pytest -x` to stop on first failure
- Push code changes to git so Machine B can pull them

## Iteration Workflow (This Machine's Steps)

1. **Collect** — Run MCTS with current policy in tmux (`scripts/collect.sh`)
2. **Build pairs** — `python -m src.preferences` (CPU only)
3. **Send to B** — `scp` preference data to Machine B (`scripts/sync_to_b.sh`)
4. **Wait** — Machine B trains DPO (2-4 hours)
5. **Receive from B** — `scp` LoRA adapters back (`scripts/sync_from_b.sh`)
6. **Evaluate** — Run zero-shot + MCTS eval on test set (`scripts/evaluate.sh`)
7. **Merge** — `python -m src.merge_lora` to prepare base for next round
8. **Repeat** from step 1 with merged model

## Testing Strategy

- `test_agent.py`: verify prompt formatting and action parsing (including malformed outputs)
- `test_mcts.py`: test UCB1 math, backpropagation updates, tree serialization (mock the LLM, no GPU)
- `test_sandbox.py`: test Docker execution, timeout handling, error capture
- `test_preferences.py`: test blended Q computation, threshold filtering, pair extraction from mock trees

Run `pytest tests/ -v` before every commit. Tests must not require GPU — mock all model calls.

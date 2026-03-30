# CodeQ: MCTS + DPO Code Debugging Agent
# Machine B — DPO Training Node

An autonomous code debugging agent that learns to fix bugs through Monte Carlo Tree Search, AI self-critique, and Direct Preference Optimization. Inspired by Agent Q (Putta et al., 2024).

**This is Machine B.** It handles DPO training with LoRA. The model runs in full bf16 (~14 GB) plus optimizer states and activations (~16-20 GB more). Machine A (separate server) handles MCTS data collection, evaluation, and sends preference data here for training.

## Tech Stack

- Python 3.10, PyTorch 2.x (CUDA 12.1)
- Hugging Face: `transformers`, `accelerate`, `peft`, `trl`, `datasets`
- Base model: `Qwen/Qwen2.5-Coder-7B-Instruct` (loaded in bf16)
- `trl.DPOTrainer` for Direct Preference Optimization
- W&B for training metrics and loss curves

## This Machine's Role

- Run DPO training on preference pairs received from Machine A
- Save LoRA adapter checkpoints after each training round
- Send trained LoRA adapters back to Machine A
- Monitor training metrics via W&B (loss, reward margin, gradient norm)

It does NOT run: MCTS collection, code sandbox execution, evaluation, or preference pair construction. Those happen on Machine A.

## Hardware

- One NVIDIA H100 94GB, **shared with other users**
- ALWAYS run `nvidia-smi` before launching GPU jobs
- ALWAYS prefix GPU commands with `CUDA_VISIBLE_DEVICES=0`
- DPO training uses ~30-35 GB VRAM (model bf16 + LoRA + optimizer + activations)
- Leaves ~60 GB headroom for other users

## Project Structure

```
codeq/
├── CLAUDE.md                # This file
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── agent.py             # Agent prompt templates (needed for tokenizer chat template)
│   ├── mcts.py              # [NOT USED ON THIS MACHINE — runs on Machine A]
│   ├── critic.py            # [NOT USED ON THIS MACHINE — runs on Machine A]
│   ├── sandbox.py           # [NOT USED ON THIS MACHINE — runs on Machine A]
│   ├── preferences.py       # [NOT USED ON THIS MACHINE — runs on Machine A]
│   ├── train_dpo.py         # DPO training with LoRA — PRIMARY MODULE
│   ├── evaluate.py          # [NOT USED ON THIS MACHINE — runs on Machine A]
│   ├── merge_lora.py        # [NOT USED ON THIS MACHINE — runs on Machine A]
│   └── utils.py             # Shared helpers: logging, serialization, config
├── configs/
│   └── train_config.yaml    # DPO training hyperparameters
├── data/
│   └── preferences/         # Preference pairs received from Machine A
├── models/
│   ├── qwen2.5-coder-7b/   # Base model (downloaded once)
│   ├── agentq-round1/       # Trained LoRA adapters (sent to Machine A after training)
│   └── agentq-round2/
├── checkpoints/             # Intermediate training checkpoints
│   ├── round1/
│   └── round2/
├── scripts/
│   ├── train.sh             # Launch DPO training
│   ├── sync_from_a.sh       # Receive preference data from Machine A
│   └── sync_to_a.sh         # Send LoRA adapters to Machine A
└── tests/
    └── test_train.py        # Training config validation, data loading tests
```

## Commands for This Machine

```bash
# Setup
pip install -r requirements.txt
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir models/qwen2.5-coder-7b

# Run tests
pytest tests/ -v
pytest tests/test_train.py -v

# === Receive preference data from Machine A ===
scp user@MACHINE_A_HOST:~/codeq/data/preferences/round1.jsonl data/preferences/

# === PRIMARY WORKFLOW: DPO Training ===
CUDA_VISIBLE_DEVICES=0 python -m src.train_dpo \
  --config configs/train_config.yaml \
  --model models/qwen2.5-coder-7b \
  --preferences data/preferences/round1.jsonl \
  --output models/agentq-round1

# For round 2+, train from merged model or re-apply LoRA to base:
CUDA_VISIBLE_DEVICES=0 python -m src.train_dpo \
  --config configs/train_config.yaml \
  --model models/qwen2.5-coder-7b \
  --preferences data/preferences/round2.jsonl \
  --output models/agentq-round2

# === Send trained LoRA adapters to Machine A ===
scp -r models/agentq-round1/adapter_* user@MACHINE_A_HOST:~/codeq/models/agentq-round1/
scp models/agentq-round1/adapter_config.json user@MACHINE_A_HOST:~/codeq/models/agentq-round1/

# === Quick inspection of preference data (before training) ===
python -c "
import jsonlines
with jsonlines.open('data/preferences/round1.jsonl') as r:
    pairs = list(r)
print(f'Total pairs: {len(pairs)}')
print(f'Avg prompt len: {sum(len(p[\"prompt\"]) for p in pairs)/len(pairs):.0f} chars')
print(f'Sample chosen: {pairs[0][\"chosen\"][:100]}...')
"

# === Monitor GPU usage ===
watch -n 2 nvidia-smi
```

## Architecture: The DPO Training Module (`src/train_dpo.py`)

This is the only module that runs GPU workloads on this machine. It uses HuggingFace TRL's `DPOTrainer`.

### Model Loading

Download once, then load in full bf16 with flash attention for training efficiency:
```python
# Download once: huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir ./models/qwen2.5-coder-7b

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
```

Qwen2.5 uses ChatML-style templates (`<|im_start|>`, `<|im_end|>`). The tokenizer handles this automatically via `apply_chat_template()`. The `DPOTrainer` from TRL will use the tokenizer's chat template when processing preference pairs — no manual formatting needed.

### LoRA Configuration

```python
from peft import LoraConfig
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### DPO Training Config

```python
from trl import DPOConfig
training_args = DPOConfig(
    output_dir=f"./checkpoints/round{iteration}",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,    # effective batch = 16
    learning_rate=5e-6,
    beta=0.1,                         # DPO temperature
    max_length=2048,
    max_prompt_length=1024,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    warmup_ratio=0.1,
    bf16=True,
    gradient_checkpointing=True,
    report_to="wandb",
)
```

### Off-Policy Reference Model Trick

Standard DPO needs a frozen reference model (doubling VRAM). Instead, store log-likelihoods of actions under the data-collection policy in the preference pair data during MCTS on Machine A. During training, load these stored log-probs as the reference, eliminating ~14 GB overhead.

### VRAM Budget

```
Component                     VRAM (approx)
────────────────────────────────────────────
Model weights (bf16)          ~14 GB
LoRA adapters                 ~0.5 GB
Optimizer states (AdamW)      ~1.5 GB
Gradients                     ~1 GB
Activations (batch=2, gc on)  ~8-12 GB
KV cache during forward       ~4-6 GB
────────────────────────────────────────────
TOTAL                         ~30-35 GB
Free on H100 94GB:            ~60 GB
```

### What to Monitor on W&B

- **DPO loss** — should decrease steadily
- **Reward margin** — log-prob gap between chosen/rejected, should increase
- **Chosen reward** — should increase; **Rejected reward** — should decrease
- **Gradient norm** — should stay stable; spikes = reduce learning rate
- If reward margin plateaus early, the preference data may need higher `theta_threshold`

### DPO Loss Formula (Reference)

```
L = -E[log σ(β · log(π_θ(a_w|h) / π_ref(a_w|h)) - β · log(π_θ(a_l|h) / π_ref(a_l|h)))]
```

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
- ALWAYS set `CUDA_VISIBLE_DEVICES=0` before any GPU operation
- ALWAYS load model in full bf16 (not 4-bit) — training requires full precision gradients
- ALWAYS enable gradient checkpointing — saves ~10-15 GB VRAM
- ALWAYS use flash_attention_2 — significantly faster on H100
- ALWAYS log to W&B — training metrics are essential for the portfolio
- NEVER run MCTS, sandbox, or evaluation on this machine — that's Machine A's job
- NEVER load `bitsandbytes` quantized models on this machine — bf16 only
- Save checkpoints every 100 steps in case training is interrupted by other users
- After training, only the `adapter_*` files and `adapter_config.json` need to transfer to Machine A
- Push code changes to git so Machine A can pull them

## Iteration Workflow (This Machine's Steps)

1. **Pull code** — `git pull` to get latest from Machine A's development
2. **Receive data** — `scp` preference pairs from Machine A (`scripts/sync_from_a.sh`)
3. **Inspect data** — Verify pair count, prompt lengths, check for degenerate pairs
4. **Train** — Run DPO training (`scripts/train.sh`), monitor W&B
5. **Send to A** — `scp` LoRA adapter files back to Machine A (`scripts/sync_to_a.sh`)
6. **Wait** — Machine A evaluates and prepares next round's data

## Testing Strategy

- `test_train.py`: validate config loading, preference dataset parsing, LoRA config construction
- Tests should NOT require GPU — mock model loading and trainer initialization
- Verify that preference pair format matches what `DPOTrainer` expects

Run `pytest tests/ -v` before every commit.

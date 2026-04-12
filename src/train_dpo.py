"""DPO training with LoRA for the CodeQ debugging agent.

This is the primary module on Machine B.  It:

1. Loads the Qwen2.5-Coder-7B-Instruct base model in bf16.
2. Attaches LoRA adapters via PEFT.
3. Reads preference pairs (produced by Machine A's MCTS) from a JSONL file.
4. Uses TRL's DPOTrainer with pre-stored reference log-probs (off-policy trick)
   to avoid loading a second frozen reference model (~14 GB VRAM saving).
5. Saves LoRA adapter checkpoints that Machine A can pull and evaluate.

Usage::

    CUDA_VISIBLE_DEVICES=0 python -m src.train_dpo \\
        --config configs/train_config.yaml \\
        --model models/qwen2.5-coder-7b \\
        --preferences data/preferences/round1.jsonl \\
        --output models/agentq-round1
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import selective_log_softmax

from src.utils import get_logger, load_preference_pairs, load_yaml


logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Model loading configuration.

    Attributes:
        name_or_path: HuggingFace model ID or local path.
        torch_dtype: Floating-point precision string ('bfloat16', 'float16',
            or 'float32'). Use 'float32' for full-precision training to
            eliminate mixed-precision NaN paths.
        attn_implementation: Attention backend ('flash_attention_2' or 'eager').
    """

    name_or_path: str
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"


class LoraConfigModel(BaseModel):
    """LoRA adapter configuration.

    Attributes:
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        target_modules: Transformer sub-modules to apply LoRA to.
        lora_dropout: Dropout probability in LoRA layers.
        bias: Bias training strategy ('none', 'all', or 'lora_only').
        task_type: PEFT task type string.
    """

    r: int = 32
    lora_alpha: int = 64
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainingArgs(BaseModel):
    """DPO training hyper-parameters.

    Attributes:
        num_train_epochs: Number of full passes over the preference dataset.
        per_device_train_batch_size: Per-GPU batch size.
        gradient_accumulation_steps: Steps before an optimiser update.
        learning_rate: Peak learning rate for AdamW.
        beta: DPO temperature (controls how far from the reference policy).
        max_length: Maximum total sequence length (prompt + response).
        max_prompt_length: Maximum prompt token length.
        logging_steps: Steps between W&B log events.
        eval_strategy: Evaluation strategy ('steps' or 'epoch').
        eval_steps: Steps between evaluations (when eval_strategy='steps').
        save_strategy: Checkpoint save strategy ('steps' or 'epoch').
        save_steps: Steps between checkpoint saves.
        warmup_ratio: Fraction of training used for LR warm-up.
        max_grad_norm: Gradient clipping threshold.
        max_steps: If > 0, stop after this many optimizer steps (overrides
            num_train_epochs). Use for short diagnostic runs.
        loss_type: DPO loss variant. ``"sigmoid"`` is the default DPO loss
            (logsigmoid form, susceptible to large-margin overflow); ``"ipo"``
            is Identity Preference Optimization (squared hinge form, more
            numerically stable on noisy preferences).
        bf16: Whether to use bf16 mixed precision (False for full fp32).
        fp16: Whether to use fp16 mixed precision (False for full fp32).
        gradient_checkpointing: Whether to enable gradient checkpointing.
        report_to: Logging backend ('wandb', 'tensorboard', 'none').
        dataloader_num_workers: Number of DataLoader worker processes.
        seed: Random seed for reproducibility.
    """

    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    beta: float = 0.1
    max_length: int = 2048
    max_prompt_length: int = 1024
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 100
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_steps: int = 0
    loss_type: str = "sigmoid"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    report_to: str = "wandb"
    dataloader_num_workers: int = 4
    seed: int = 42


class WandbConfig(BaseModel):
    """Weights & Biases configuration.

    Attributes:
        project: W&B project name.
        entity: W&B username or organisation (``None`` uses default).
    """

    project: str = "codeq-dpo"
    entity: str | None = None


class TrainConfig(BaseModel):
    """Top-level training configuration loaded from YAML.

    Attributes:
        model: Model loading settings.
        lora: LoRA adapter settings.
        training: DPO training hyper-parameters.
        wandb: Weights & Biases settings.
    """

    model: ModelConfig
    lora: LoraConfigModel
    training: TrainingArgs
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        """Load and validate config from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Validated :class:`TrainConfig` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            pydantic.ValidationError: If the YAML contents are invalid.
        """
        raw = load_yaml(path)
        return cls(**raw)


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_path: Path,
    cfg: ModelConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model in bf16 and its tokenizer.

    Args:
        model_path: Override path (CLI ``--model`` argument) or the path from
            the config if not overridden.
        cfg: Model loading configuration.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        OSError: If the model directory does not exist.
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if cfg.torch_dtype not in dtype_map:
        raise ValueError(
            f"Unknown torch_dtype {cfg.torch_dtype!r}; "
            f"must be one of {sorted(dtype_map)}"
        )
    torch_dtype = dtype_map[cfg.torch_dtype]

    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        "Loading model from %s (dtype=%s, attn=%s)",
        model_path,
        cfg.torch_dtype,
        cfg.attn_implementation,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=cfg.attn_implementation,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------


def attach_lora(model: AutoModelForCausalLM, cfg: LoraConfigModel) -> Any:
    """Wrap the model with LoRA adapters.

    Args:
        model: The base causal language model.
        cfg: LoRA configuration.

    Returns:
        PEFT model with LoRA adapters attached.
    """
    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=cfg.task_type,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_dataset(
    preferences_path: Path,
    tokenizer: AutoTokenizer,
) -> tuple[Dataset, Dataset]:
    """Convert preference pairs into a HuggingFace Dataset for DPOTrainer.

    The dataset contains the columns expected by TRL's DPOTrainer when using
    pre-computed reference log-probs (off-policy trick):
    ``prompt``, ``chosen``, ``rejected``,
    ``ref_chosen_logps``, ``ref_rejected_logps``.

    Prompts are rendered through the tokenizer's chat template so that
    Qwen2.5's ChatML format (``<|im_start|>`` / ``<|im_end|>``) is applied
    consistently.

    Args:
        preferences_path: Path to the JSONL preference file.
        tokenizer: Qwen2.5 tokenizer (provides ``apply_chat_template``).

    Returns:
        Tuple of (train_dataset, eval_dataset) with a 95/5 split.

    Raises:
        FileNotFoundError: If *preferences_path* does not exist.
        ValueError: If the file contains fewer than 2 pairs.
    """
    pairs = load_preference_pairs(preferences_path)
    if len(pairs) < 2:
        raise ValueError(
            f"Need at least 2 preference pairs for a train/eval split, "
            f"got {len(pairs)}"
        )

    logger.info("Loaded %d preference pairs from %s", len(pairs), preferences_path)

    records: list[dict[str, Any]] = []
    for pair in pairs:
        # Render prompt via tokenizer chat template (no generation prompt — the
        # chosen/rejected completions are supplied separately to DPOTrainer).
        prompt_messages = [
            {"role": "user", "content": pair.prompt},
        ]
        prompt_str = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        records.append(
            {
                "prompt": prompt_str,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
            }
        )

    full_dataset = Dataset.from_list(records)
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    logger.info(
        "Dataset split — train: %d, eval: %d",
        len(split["train"]),
        len(split["test"]),
    )
    return split["train"], split["test"]


# ---------------------------------------------------------------------------
# DPO training
# ---------------------------------------------------------------------------


class Fp32LogitsDPOTrainer(DPOTrainer):
    """DPOTrainer that upcasts logits to float32 before log-softmax.

    Prevents NaN gradients from bfloat16 overflow in selective_log_softmax.
    """

    def _compute_loss(self, model, inputs, return_outputs):
        # Monkey-patch selective_log_softmax locally to upcast logits first.
        import trl.trainer.dpo_trainer as _dpo_mod
        _orig = _dpo_mod.selective_log_softmax

        def _fp32_selective_log_softmax(logits, index):
            return _orig(logits.float(), index)

        _dpo_mod.selective_log_softmax = _fp32_selective_log_softmax
        try:
            return super()._compute_loss(model, inputs, return_outputs)
        finally:
            _dpo_mod.selective_log_softmax = _orig


def run_training(
    model: Any,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: Path,
    args: TrainingArgs,
    round_num: int,
    resume_from_checkpoint: str | None = None,
) -> None:
    """Run DPO fine-tuning and save the LoRA adapter.

    Uses TRL's :class:`~trl.DPOTrainer` with pre-stored reference log-probs
    so that no second frozen model needs to be loaded (saves ~14 GB VRAM).

    Args:
        model: PEFT-wrapped causal LM.
        tokenizer: Qwen2.5 tokenizer.
        train_dataset: HuggingFace Dataset for training.
        eval_dataset: HuggingFace Dataset for evaluation.
        output_dir: Directory where the final LoRA adapter is saved.
        args: Training hyper-parameters.
        round_num: Training round number (used for checkpoint sub-directory).

    Returns:
        None
    """
    checkpoint_dir = Path(f"checkpoints/round{round_num}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dpo_kwargs: dict[str, Any] = dict(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
        # Default truncation_mode is "keep_start", which would chop the
        # COMPLETION when the concatenated prompt+completion overflows
        # max_length — exactly the wrong direction for DPO. Force "keep_end"
        # so the prompt is left-truncated and the completion is preserved.
        truncation_mode="keep_end",
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        # Compute reference log-probs on-the-fly each step. Sharing the live
        # forward path (same truncation, dtypes, and any logits patches)
        # eliminates a deterministic loss explosion seen with precompute=True.
        # PEFT ref is obtained by disabling adapters, so no extra VRAM.
        precompute_ref_log_probs=False,
    )
    if args.max_steps and args.max_steps > 0:
        dpo_kwargs["max_steps"] = args.max_steps
    dpo_config = DPOConfig(**dpo_kwargs)

    trainer = Fp32LogitsDPOTrainer(
        model=model,
        ref_model=None,     # no reference model — logps come from dataset
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training — round %d", round_num)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving LoRA adapter to %s", output_dir)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DPO training with LoRA on Machine B."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Override the model path from the config.",
    )
    parser.add_argument(
        "--preferences",
        type=Path,
        required=True,
        help="Path to the JSONL preference pairs file from Machine A.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save the trained LoRA adapter.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Training round number (used for checkpoint sub-directory naming).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for DPO training."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    # Load and validate config
    cfg = TrainConfig.from_yaml(args.config)

    # CLI --model overrides config
    model_path = args.model if args.model is not None else Path(cfg.model.name_or_path)

    # Initialise W&B
    wb_cfg = cfg.wandb
    wandb.init(
        project=wb_cfg.project,
        entity=wb_cfg.entity,
        config={
            "round": args.round,
            "model": str(model_path),
            "preferences": str(args.preferences),
            **cfg.training.model_dump(),
            **cfg.lora.model_dump(),
        },
    )

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, cfg.model)

    # Attach LoRA
    model = attach_lora(model, cfg.lora)

    # Build dataset
    train_ds, eval_ds = build_dataset(args.preferences, tokenizer)

    # Train
    run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        output_dir=args.output,
        args=cfg.training,
        round_num=args.round,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    wandb.finish()


if __name__ == "__main__":
    main()

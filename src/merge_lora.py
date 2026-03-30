"""Merge LoRA adapters into the base model for the next training round."""

import logging
from pathlib import Path

from src.utils import setup_logging

logger = logging.getLogger(__name__)


def merge_lora(base_path: str, adapter_path: str, output_path: str) -> None:
    """Merge a LoRA adapter into the base model and save the result.

    The merged model is saved in full bf16 precision and can be reloaded
    for the next MCTS round (in 4-bit on this machine).

    Args:
        base_path: Path to the base model directory (e.g. models/qwen2.5-coder-7b).
        adapter_path: Path to the LoRA adapter directory received from Machine B.
        output_path: Path to save the merged model.

    Raises:
        FileNotFoundError: If base_path or adapter_path do not exist.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_path = Path(base_path)
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    logger.info("Loading base model from %s (bf16)", base_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base_path))

    logger.info("Loading LoRA adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    logger.info("Merging LoRA weights into base model")
    model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged model to %s", output_path)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info("Merge complete: %s", output_path)


def main() -> None:
    """CLI entry point for LoRA merging.

    Usage:
        python -m src.merge_lora \\
            --base models/qwen2.5-coder-7b \\
            --adapter models/agentq-round1 \\
            --output models/agentq-round1-merged
    """
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", required=True, help="Base model directory")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Output merged model directory")
    args = parser.parse_args()

    setup_logging()
    merge_lora(args.base, args.adapter, args.output)


if __name__ == "__main__":
    main()

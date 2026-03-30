"""Evaluation harness for zero-shot and MCTS modes."""

import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.agent import ActionParser, SYSTEM_PROMPT, format_observation
from src.sandbox import CodeSandbox
from src.utils import append_jsonl, load_config, load_json, setup_logging

logger = logging.getLogger(__name__)

EvalMode = Literal["zero_shot", "mcts", "full_rewrite"]


class EvalConfig(BaseModel):
    """Evaluation configuration loaded from eval_config.yaml.

    Attributes:
        test_set_path: Path to the held-out evaluation dataset.
        results_dir: Directory to write per-run result files.
        default_mode: Default eval mode if not specified on CLI.
        mcts_rollouts: Number of MCTS rollouts when mode is 'mcts'.
        mcts_depth: Max rollout depth when mode is 'mcts'.
        mcts_k_actions: K actions per expansion when mode is 'mcts'.
        max_new_tokens: Max tokens for generation.
        temperature: Sampling temperature for zero-shot generation.
        log_level: Logging level.
        wandb_project: W&B project name.
    """

    test_set_path: str
    results_dir: str = "results/"
    default_mode: EvalMode = "zero_shot"
    mcts_rollouts: int = 20
    mcts_depth: int = 5
    mcts_k_actions: int = 4
    max_new_tokens: int = 2048
    temperature: float = 0.2
    log_level: str = "INFO"
    wandb_project: str = "codeq-eval"


class Evaluator:
    """Runs evaluation on the held-out test set.

    Supports two modes:
    - zero_shot: A single LLM pass, one action, execute, check.
    - mcts: Full MCTS search with configured rollouts.
    """

    def __init__(
        self,
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
        sandbox: CodeSandbox,
        config: EvalConfig,
    ) -> None:
        """Initialize the Evaluator.

        Args:
            model: Loaded HuggingFace causal LM.
            tokenizer: Corresponding tokenizer.
            sandbox: CodeSandbox for execution.
            config: EvalConfig with evaluation settings.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sandbox = sandbox
        self.config = config
        self._parser = ActionParser()

    def evaluate(self, test_set: list[dict], mode: EvalMode) -> dict:
        """Evaluate the model on the test set.

        Args:
            test_set: List of task dicts with 'id', 'buggy_code', 'test_code'.
            mode: 'zero_shot' or 'mcts'.

        Returns:
            Summary dict with 'total', 'solved', 'pass_rate', and 'results'.
        """
        results = []
        solved_count = 0

        for i, task in enumerate(test_set):
            task_id = task.get("id", str(i))
            logger.info("Evaluating task %d/%d: %s [%s]", i + 1, len(test_set), task_id, mode)

            try:
                if mode == "zero_shot":
                    task_result = self._eval_zero_shot(task)
                elif mode == "full_rewrite":
                    task_result = self._eval_full_rewrite(task)
                else:
                    task_result = self._eval_mcts(task)
            except Exception as exc:
                logger.error("Task %s error: %s", task_id, exc)
                task_result = {"task_id": task_id, "solved": False, "error": str(exc)}

            results.append(task_result)
            if task_result.get("solved"):
                solved_count += 1

        total = len(test_set)
        summary = {
            "total": total,
            "solved": solved_count,
            "pass_rate": solved_count / total if total > 0 else 0.0,
            "mode": mode,
            "results": results,
        }
        logger.info("Pass rate [%s]: %.2f (%d/%d)", mode, summary["pass_rate"], solved_count, total)
        return summary

    def _eval_zero_shot(self, task: dict) -> dict:
        """Evaluate a single task in zero-shot mode.

        One LLM call → apply action → run tests.

        Args:
            task: Task dict.

        Returns:
            Result dict with 'task_id', 'solved', 'final_code', 'test_output'.
        """
        task_id = task.get("id", "unknown")
        code = task["buggy_code"]
        test_code = task["test_code"]

        # Initial test run
        initial = self.sandbox.execute(code, test_code)
        if initial.all_passed:
            return {"task_id": task_id, "solved": True, "final_code": code,
                    "test_output": initial.output}

        observation = format_observation(code, initial.output, step=0)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": observation},
        ]
        response = self._generate(messages)
        if response is None:
            return {"task_id": task_id, "solved": False, "final_code": code,
                    "test_output": initial.output, "parse_failure": True}

        parsed = self._parser.parse(response)
        new_code = None

        if parsed is not None and parsed.action_type:
            from src.mcts import MCTSDebugger, MCTSConfig
            dummy_config = MCTSConfig(
                model_path="", dataset_path="", output_path=""
            )
            debugger = MCTSDebugger(self.model, self.tokenizer, self.sandbox, dummy_config)
            new_state = debugger._apply_action(
                {"code": code, "test_output": initial.output}, parsed
            )
            if new_state is not None:
                new_code = new_state["code"]

        if new_code is None:
            # Fallback: extract a ```python ... ``` block and use it as the
            # full replacement.  Handles models that output fenced code instead
            # of (or alongside) the structured THOUGHT/CODE_ACTION format.
            fence_m = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
            if fence_m is None:
                fence_m = re.search(r"```\s*(.*?)```", response, re.DOTALL)
            if fence_m:
                candidate = fence_m.group(1).strip()
                if candidate:
                    logger.info("Using code-fence fallback for task %s", task_id)
                    new_code = candidate

        if new_code is None:
            return {"task_id": task_id, "solved": False, "final_code": code,
                    "test_output": initial.output, "parse_failure": True}

        result = self.sandbox.execute(new_code, test_code)
        return {
            "task_id": task_id,
            "solved": result.all_passed,
            "final_code": new_code,
            "test_output": result.output,
        }

    def _eval_full_rewrite(self, task: dict) -> dict:
        """Evaluate a single task in full_rewrite zero-shot mode.

        Sends a simple prompt asking for the complete corrected code, extracts
        the code from the response (markdown fences or raw), and runs tests.

        Args:
            task: Task dict with 'buggy_code' and 'test_code'.

        Returns:
            Result dict with 'task_id', 'solved', 'final_code', 'test_output'.
        """
        task_id = task.get("id", "unknown")
        code = task["buggy_code"]
        test_code = task["test_code"]

        prompt = (
            "Here is a buggy Python solution. Return the complete corrected code. "
            "Output only the code, no explanation.\n\n"
            f"```python\n{code}\n```"
        )
        messages = [{"role": "user", "content": prompt}]
        response = self._generate(messages)

        if response is None:
            return {"task_id": task_id, "solved": False, "final_code": code,
                    "parse_failure": True}

        # Extract code: prefer ```python ... ```, then ``` ... ```, then raw.
        fence_m = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if fence_m is None:
            fence_m = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        new_code = fence_m.group(1).strip() if fence_m else response.strip()

        if not new_code:
            return {"task_id": task_id, "solved": False, "final_code": code,
                    "parse_failure": True}

        result = self.sandbox.execute(new_code, test_code)
        return {
            "task_id": task_id,
            "solved": result.all_passed,
            "final_code": new_code,
            "test_output": result.output,
        }

    def _eval_mcts(self, task: dict) -> dict:
        """Evaluate a single task using MCTS search.

        Args:
            task: Task dict.

        Returns:
            Result dict.
        """
        from src.mcts import MCTSConfig, MCTSDebugger

        mcts_config = MCTSConfig(
            model_path="",
            dataset_path="",
            output_path="",
            k_actions=self.config.mcts_k_actions,
            max_rollouts=self.config.mcts_rollouts,
            max_depth=self.config.mcts_depth,
        )
        debugger = MCTSDebugger(self.model, self.tokenizer, self.sandbox, mcts_config)
        trajectory = debugger.search(task)
        return {
            "task_id": trajectory["task_id"],
            "solved": trajectory["solved"],
        }

    def _generate(self, messages: list[dict]) -> str | None:
        """Generate a single LLM response.

        Args:
            messages: Chat messages.

        Returns:
            Generated string or None on failure.
        """
        import torch

        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            encoded = encoded.to(self.model.device)
            input_len = encoded["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][input_len:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        except Exception as exc:
            logger.warning("Generation failed: %s", exc)
            return None


def main() -> None:
    """CLI entry point for evaluation.

    Usage:
        CUDA_VISIBLE_DEVICES=0 python -m src.evaluate \\
            --model models/agentq-round1 \\
            --test-set data/test_set.json \\
            --mode zero_shot
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate the debugging agent")
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--adapter", default=None, help="PEFT adapter path (loaded on top of base model)")
    parser.add_argument("--test-set", required=True, help="Test set JSON path")
    parser.add_argument("--mode", choices=["zero_shot", "mcts", "full_rewrite"], default="zero_shot")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--output", default=None, help="Output JSON path for results")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N tasks")
    parser.add_argument("--mcts-rollouts", type=int, default=None, help="Override mcts_rollouts from config")
    parser.add_argument("--mcts-depth", type=int, default=None, help="Override mcts_depth from config")
    parser.add_argument("--mcts-k", type=int, default=None, help="Override mcts_k_actions from config")
    args = parser.parse_args()

    config = load_config(EvalConfig, Path(args.config))
    if args.mcts_rollouts is not None:
        config.mcts_rollouts = args.mcts_rollouts
    if args.mcts_depth is not None:
        config.mcts_depth = args.mcts_depth
    if args.mcts_k is not None:
        config.mcts_k_actions = args.mcts_k
    setup_logging(config.log_level)

    from src.mcts import load_model

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load_model(args.model)

    if args.adapter:
        from peft import PeftModel
        logger.info("Loading PEFT adapter from %s", args.adapter)
        model = PeftModel.from_pretrained(model, args.adapter)

    sandbox = CodeSandbox()
    evaluator = Evaluator(model, tokenizer, sandbox, config)

    test_set = load_json(Path(args.test_set))

    # Deduplicate by task ID (eval_C had duplicate entries for some tasks)
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for task in test_set:
        tid = task.get("id", "")
        if tid not in seen_ids:
            seen_ids.add(tid)
            deduped.append(task)
    if len(deduped) != len(test_set):
        logger.warning(
            "Removed %d duplicate task IDs from test set (%d -> %d)",
            len(test_set) - len(deduped),
            len(test_set),
            len(deduped),
        )
    test_set = deduped

    if args.limit:
        test_set = test_set[: args.limit]
    summary = evaluator.evaluate(test_set, mode=args.mode)

    adapter_tag = f"_adapter" if args.adapter else ""
    output_path = args.output or f"/home/lamassunobackup/tdebnath/codeQA/results/eval_{args.mode}{adapter_tag}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Results written to %s", output_path)
    print(f"Pass rate [{args.mode}]: {summary['pass_rate']:.2%} ({summary['solved']}/{summary['total']})")


if __name__ == "__main__":
    main()

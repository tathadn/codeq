"""MCTSDebugger: Monte Carlo Tree Search engine for code debugging."""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

import re

from src.agent import (
    MCTS_REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    ActionParser,
    ParsedAction,
    apply_delete,
    apply_edit,
    apply_insert,
    format_observation,
    format_rewrite_prompt,
)
from src.sandbox import CodeSandbox, ExecutionResult
from src.utils import append_jsonl

logger = logging.getLogger(__name__)


class MCTSConfig(BaseModel):
    """MCTS hyperparameters loaded from mcts_config.yaml.

    Attributes:
        model_path: Path to the base (or merged) model directory.
        dataset_path: Path to the training bug dataset JSONL.
        output_path: Path to write MCTS trajectory JSONL.
        k_actions: Number of candidate actions proposed per expansion.
        max_rollouts: Maximum number of MCTS rollouts per task.
        max_depth: Maximum rollout depth (steps before giving up).
        c_exp: UCB1 exploration constant.
        alpha: Blend weight for Q = alpha*Q_mcts + (1-alpha)*Q_ai.
        theta_threshold: Min |Q_chosen - Q_rejected| to create a pair.
        propose_temperature: Sampling temperature for action proposal.
        rank_temperature: Sampling temperature for action ranking.
        checkpoint_every: Append JSONL progress every N tasks.
        log_level: Logging level string.
        wandb_project: W&B project name.
    """

    model_path: str
    dataset_path: str
    output_path: str
    k_actions: int = 4
    max_rollouts: int = 50
    max_depth: int = 8
    c_exp: float = math.sqrt(2)
    alpha: float = 0.5
    theta_threshold: float = 0.2
    propose_temperature: float = 0.8
    rank_temperature: float = 0.2
    checkpoint_every: int = 5
    log_level: str = "INFO"
    wandb_project: str = "codeq-mcts"
    use_rewrite_mode: bool = True


@dataclass
class MCTSNode:
    """A single node in the MCTS tree.

    Attributes:
        state: Dict with 'code' (str) and 'test_output' (str).
        action: The action that produced this state (None for root).
        ai_score: Critic score assigned to this node's action.
        parent: Parent node (None for root).
        children: Child nodes produced by expanding this node.
        visit_count: How many times this node has been visited.
        total_value: Sum of rewards backpropagated through this node.
        depth: Depth in the tree (root = 0).
        is_terminal: True if this node's state has all tests passing.
    """

    state: dict
    action: Optional[str] = None
    ai_score: float = 0.0
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    depth: int = 0
    is_terminal: bool = False

    @property
    def q_value(self) -> float:
        """Mean reward seen through this node (0 if never visited).

        Returns:
            Q-value in [0, 1].
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb1(self, c_exp: float) -> float:
        """UCB1 score for tree-policy selection.

        Formula: Q(h,a) + c_exp * sqrt(log(N(parent)) / (1 + N(self)))

        Args:
            c_exp: Exploration constant.

        Returns:
            UCB1 score (inf for unvisited nodes to force exploration).
        """
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else 1
        return self.q_value + c_exp * math.sqrt(
            math.log(max(parent_visits, 1)) / (1 + self.visit_count)
        )


class MCTSDebugger:
    """MCTS-based code debugging agent.

    Uses Monte Carlo Tree Search to explore the space of possible code
    fixes, guided by an LLM that proposes and ranks actions, and a Docker
    sandbox that evaluates them.
    """

    def __init__(
        self,
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
        sandbox: CodeSandbox,
        config: MCTSConfig,
    ) -> None:
        """Initialize the MCTSDebugger.

        Args:
            model: Loaded HuggingFace causal LM (4-bit quantized).
            tokenizer: Corresponding tokenizer.
            sandbox: CodeSandbox instance for safe code execution.
            config: MCTSConfig hyperparameters.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sandbox = sandbox
        self.config = config
        self._parser = ActionParser()

    def search(self, task: dict) -> dict:
        """Run MCTS on a single debugging task.

        Args:
            task: Dict with keys 'id', 'buggy_code', 'test_code', and
                  optionally 'description'.

        Returns:
            Trajectory dict with 'task_id', 'solved', 'root', and 'nodes'.
        """
        buggy_code: str = task["buggy_code"]
        test_code: str = task["test_code"]
        task_id: str = task.get("id", "unknown")

        # Evaluate initial state
        initial_result = self.sandbox.execute(buggy_code, test_code)
        root_state = {
            "code": buggy_code,
            "test_output": initial_result.output,
            "test_code": test_code,
        }
        root = MCTSNode(
            state=root_state,
            is_terminal=initial_result.all_passed,
        )

        logger.info("Starting MCTS for task %s (initial passed=%s)", task_id, root.is_terminal)

        solved = root.is_terminal
        for rollout_idx in range(self.config.max_rollouts):
            if solved:
                break

            # Selection
            node = self._select(root)

            # Expansion (if not terminal and not at max depth)
            if not node.is_terminal and node.depth < self.config.max_depth:
                self._expand(node, test_code)

            # Rollout from best child (or node itself)
            leaf = self._best_unvisited_child(node) or node
            reward = self._rollout(leaf, test_code)

            # Backpropagation
            path = self._collect_path(leaf)
            self._backpropagate(path, reward)

            if reward == 1.0:
                solved = True
                logger.info("Task %s solved at rollout %d", task_id, rollout_idx)

        return self._serialize_tree(task_id, root, solved)

    def propose_rewrites(self, state: dict) -> list[tuple[str, str, bool]]:
        """Generate K complete code rewrites and immediately test each in the sandbox.

        Uses the same simple prompt as full_rewrite eval mode at depth 0, and
        includes the previous test output at depth > 0 to guide revision.

        Args:
            state: Current state dict with 'code', 'test_output', 'test_code', and 'step'.

        Returns:
            List of (rewritten_code, test_output, all_passed) tuples.
        """
        depth = state.get("step", 0)
        test_code = state.get("test_code", "")
        prompt = format_rewrite_prompt(state["code"], state.get("test_output", ""), depth)
        messages = [
            {"role": "system", "content": MCTS_REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        results: list[tuple[str, str, bool]] = []
        for _ in range(self.config.k_actions):
            response = self._generate(messages, temperature=self.config.propose_temperature)
            if response is None:
                continue
            rewrite = self._extract_code(response)
            if not rewrite:
                continue
            exec_result = self.sandbox.execute(rewrite, test_code)
            results.append((rewrite, exec_result.output, exec_result.all_passed))
        return results

    def rank_rewrites(self, state: dict, codes: list[str], test_outputs: list[str]) -> list[float]:
        """Use the critic to rank proposed code rewrites.

        Args:
            state: Current state dict (provides original buggy code as context).
            codes: Complete code rewrites to rank.
            test_outputs: Test runner output for each rewrite, parallel to codes.

        Returns:
            Normalized AI scores in [0, 1], parallel to codes.
        """
        from src.critic import CriticRanker

        ranker = CriticRanker(self.model, self.tokenizer)
        return ranker.rank_rewrites(
            state["code"], codes, test_outputs, temperature=self.config.rank_temperature
        )

    def propose_actions(self, state: dict, is_first_step: bool = False) -> list[str]:
        """Ask the LLM to propose K candidate actions for the current state.

        Args:
            state: Current state dict with 'code' and 'test_output'.
            is_first_step: True if this is the root node (include PLAN).

        Returns:
            List of up to K raw action strings.
        """
        observation = format_observation(
            state["code"], state["test_output"], step=state.get("step", 0)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": observation},
        ]
        actions: list[str] = []
        for _ in range(self.config.k_actions):
            response = self._generate(messages, temperature=self.config.propose_temperature)
            if response:
                actions.append(response)
        return actions

    def rank_actions(self, state: dict, actions: list[str]) -> list[float]:
        """Use the critic to rank proposed actions.

        Args:
            state: Current state dict.
            actions: Proposed action strings.

        Returns:
            Normalized AI scores in [0, 1], parallel to actions.
        """
        from src.critic import CriticRanker

        ranker = CriticRanker(self.model, self.tokenizer)
        observation = format_observation(state["code"], state["test_output"])
        return ranker.rank(observation, actions, temperature=self.config.rank_temperature)

    def _rollout(self, node: MCTSNode, test_code: str) -> float:
        """Execute a greedy rollout from node, dispatching by mode.

        Args:
            node: Starting node for the rollout.
            test_code: Test file contents for sandbox execution.

        Returns:
            1.0 if a terminal (all-pass) state was reached, else 0.0.
        """
        if self.config.use_rewrite_mode:
            return self._rollout_rewrite(node, test_code)
        return self._rollout_edit(node, test_code)

    def _rollout_rewrite(self, node: MCTSNode, test_code: str) -> float:
        """Greedy rollout using full code rewrites (rewrite mode).

        At each step, generates K rewrites of the current code (using test
        output for context when depth > 0), runs them in the sandbox, and
        greedily selects the best according to the critic.  Returns 1.0 as
        soon as any rewrite passes all tests.

        Args:
            node: Starting node for the rollout.
            test_code: Test file contents for sandbox execution.

        Returns:
            1.0 if any rewrite passes all tests, else 0.0.
        """
        if node.is_terminal:
            return 1.0

        state = dict(node.state)
        state["test_code"] = test_code

        for _ in range(self.config.max_depth - node.depth):
            rewrites_data = self.propose_rewrites(state)
            if not rewrites_data:
                break

            codes = [r[0] for r in rewrites_data]
            test_outputs = [r[1] for r in rewrites_data]
            all_passed_list = [r[2] for r in rewrites_data]

            if any(all_passed_list):
                return 1.0

            scores = self.rank_rewrites(state, codes, test_outputs)
            best_idx = scores.index(max(scores))
            state = {
                "code": codes[best_idx],
                "test_output": test_outputs[best_idx],
                "test_code": test_code,
                "step": state.get("step", 0) + 1,
            }

        return 0.0

    def _rollout_edit(self, node: MCTSNode, test_code: str) -> float:
        """Greedy rollout using structured line-level edits (fallback mode).

        Args:
            node: Starting node for the rollout.
            test_code: Test file contents for sandbox execution.

        Returns:
            1.0 if a terminal (all-pass) state was reached, else 0.0.
        """
        state = dict(node.state)
        state["test_code"] = test_code
        depth = node.depth

        while depth < self.config.max_depth:
            actions = self.propose_actions(state)
            if not actions:
                break

            # Pick greedy best by critic score
            scores = self.rank_actions(state, actions)
            best_action = actions[scores.index(max(scores))]
            parsed = self._parser.parse(best_action)

            if parsed is None:
                break

            new_state = self._apply_action(state, parsed)
            if new_state is None:
                break

            result = self.sandbox.execute(new_state["code"], test_code)
            new_state["test_output"] = result.output
            state = new_state
            depth += 1

            if result.all_passed:
                return 1.0

            if parsed.action_type == "SUBMIT":
                break

        return 0.0

    def _apply_action(self, state: dict, action: ParsedAction) -> Optional[dict]:
        """Apply a parsed action to the current state and return new state.

        Args:
            state: Current state dict.
            action: Parsed action to apply.

        Returns:
            New state dict, or None if the action is invalid.
        """
        try:
            code = state["code"]
            lines = code.splitlines(keepends=True)

            if action.action_type == "EDIT":
                lines = apply_edit(
                    lines,
                    action.action_args["start"],
                    action.action_args["end"],
                    action.action_args["code"],
                )
            elif action.action_type == "INSERT":
                lines = apply_insert(
                    lines,
                    action.action_args["after"],
                    action.action_args["code"],
                )
            elif action.action_type == "DELETE":
                lines = apply_delete(
                    lines,
                    action.action_args["start"],
                    action.action_args["end"],
                )
            elif action.action_type in ("RUN_TESTS", "SUBMIT"):
                # No code change; state carries over as-is
                pass
            else:
                logger.warning("Unknown action type: %s", action.action_type)
                return None

            new_code = "".join(lines)
            return {
                "code": new_code,
                "test_output": state.get("test_output", ""),
                "test_code": state.get("test_code", ""),
                "step": state.get("step", 0) + 1,
            }

        except (IndexError, KeyError) as exc:
            logger.warning("_apply_action failed: %s", exc)
            return None

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Traverse the tree using UCB1 until a leaf is reached.

        Args:
            root: Root node of the search tree.

        Returns:
            Selected leaf node.
        """
        node = root
        while node.children and not node.is_terminal:
            node = max(node.children, key=lambda n: n.ucb1(self.config.c_exp))
        return node

    def _expand(self, node: MCTSNode, test_code: str) -> None:
        """Expand a node, dispatching to rewrite or edit mode.

        Args:
            node: Node to expand.
            test_code: Test file for sandbox execution.
        """
        if self.config.use_rewrite_mode:
            self._expand_rewrite(node, test_code)
        else:
            self._expand_edit(node, test_code)

    def _expand_rewrite(self, node: MCTSNode, test_code: str) -> None:
        """Expand a node by generating K complete rewrites (rewrite mode).

        Each proposed rewrite is tested in the sandbox immediately.  The
        node's ``action`` field stores the full rewritten code string so
        that downstream preference extraction sees complete solutions.

        Args:
            node: Node to expand.
            test_code: Test file for sandbox execution.
        """
        rewrites_data = self.propose_rewrites(node.state)
        if not rewrites_data:
            return

        codes = [r[0] for r in rewrites_data]
        test_outputs = [r[1] for r in rewrites_data]
        all_passed_list = [r[2] for r in rewrites_data]

        scores = self.rank_rewrites(node.state, codes, test_outputs)

        for (rewrite_code, test_output, all_passed), ai_score in zip(rewrites_data, scores):
            new_state = {
                "code": rewrite_code,
                "test_output": test_output,
                "test_code": test_code,
                "step": node.state.get("step", 0) + 1,
            }
            child = MCTSNode(
                state=new_state,
                action=rewrite_code,  # full rewritten code stored as action
                ai_score=ai_score,
                parent=node,
                depth=node.depth + 1,
                is_terminal=all_passed,
            )
            node.children.append(child)

    def _expand_edit(self, node: MCTSNode, test_code: str) -> None:
        """Expand a node by proposing K structured edits (fallback mode).

        Args:
            node: Node to expand.
            test_code: Test file for sandbox execution.
        """
        is_first = node.depth == 0
        actions = self.propose_actions(node.state, is_first_step=is_first)
        if not actions:
            return

        scores = self.rank_actions(node.state, actions)

        for action_str, ai_score in zip(actions, scores):
            parsed = self._parser.parse(action_str)
            if parsed is None:
                continue

            new_state = self._apply_action(node.state, parsed)
            if new_state is None:
                continue

            result = self.sandbox.execute(new_state["code"], test_code)
            new_state["test_output"] = result.output

            child = MCTSNode(
                state=new_state,
                action=action_str,
                ai_score=ai_score,
                parent=node,
                depth=node.depth + 1,
                is_terminal=result.all_passed,
            )
            node.children.append(child)

    @staticmethod
    def _best_unvisited_child(node: MCTSNode) -> Optional[MCTSNode]:
        """Return an unvisited child with the highest ai_score.

        Args:
            node: Parent node.

        Returns:
            Best unvisited child, or None if all children are visited.
        """
        unvisited = [c for c in node.children if c.visit_count == 0]
        if not unvisited:
            return None
        return max(unvisited, key=lambda n: n.ai_score)

    @staticmethod
    def _collect_path(node: MCTSNode) -> list[MCTSNode]:
        """Collect the path from root to node.

        Args:
            node: Leaf node.

        Returns:
            List of nodes from root to node (inclusive).
        """
        path: list[MCTSNode] = []
        current: Optional[MCTSNode] = node
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    @staticmethod
    def _backpropagate(path: list[MCTSNode], reward: float) -> None:
        """Update visit counts and Q-values along the path.

        Formula: Q(h,a) <- (Q(h,a)*N(h,a) + R) / (N(h,a) + 1)

        Args:
            path: List of nodes from root to leaf.
            reward: Terminal reward (0.0 or 1.0).
        """
        for node in path:
            node.total_value += reward
            node.visit_count += 1

    @staticmethod
    def _extract_code(response: str) -> Optional[str]:
        """Extract Python code from an LLM response, stripping markdown fences.

        Args:
            response: Raw LLM output string.

        Returns:
            Extracted code string, or None if the response is empty.
        """
        fence_m = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if fence_m is None:
            fence_m = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        code = fence_m.group(1).strip() if fence_m else response.strip()
        return code if code else None

    def _generate(self, messages: list[dict], temperature: float) -> Optional[str]:
        """Generate a single LLM response.

        Args:
            messages: Chat message list.
            temperature: Sampling temperature.

        Returns:
            Generated text string, or None on failure.
        """
        import torch

        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
            encoded = encoded.to(self.model.device)
            input_len = encoded["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=512,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][input_len:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        except Exception as exc:
            logger.warning("Model generation failed: %s", exc)
            return None

    @staticmethod
    def _serialize_tree(task_id: str, root: MCTSNode, solved: bool) -> dict:
        """Serialize the MCTS tree to a JSON-compatible dict.

        Args:
            task_id: Task identifier.
            root: Root MCTSNode.
            solved: Whether the task was solved.

        Returns:
            Serializable trajectory dict.
        """
        def _node_to_dict(node: MCTSNode) -> dict:
            return {
                "state": {k: v for k, v in node.state.items() if k != "test_code"},
                "action": node.action,
                "ai_score": node.ai_score,
                "visit_count": node.visit_count,
                "q_value": node.q_value,
                "depth": node.depth,
                "is_terminal": node.is_terminal,
                "children": [_node_to_dict(c) for c in node.children],
            }

        return {
            "task_id": task_id,
            "solved": solved,
            "root": _node_to_dict(root),
        }


def load_model(model_path: str):
    """Load the Qwen2.5-Coder model in 4-bit quantization.

    Args:
        model_path: Local path or HuggingFace model ID.

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def main() -> None:
    """CLI entry point for MCTS data collection.

    Usage:
        CUDA_VISIBLE_DEVICES=0 python -m src.mcts \\
            --config configs/mcts_config.yaml \\
            --model models/qwen2.5-coder-7b \\
            --dataset data/debugbench.json \\
            --output trajectories/round1.jsonl
    """
    import argparse

    from src.utils import load_config, load_json, load_jsonl, setup_logging

    parser = argparse.ArgumentParser(description="Run MCTS data collection")
    parser.add_argument("--config", required=True, help="Path to mcts_config.yaml")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Cap total tasks considered from dataset")
    args = parser.parse_args()

    config = load_config(MCTSConfig, Path(args.config))
    # Override with CLI args
    config = config.model_copy(
        update={
            "model_path": args.model,
            "dataset_path": args.dataset,
            "output_path": args.output,
        }
    )

    setup_logging(config.log_level)
    logger.info("Loading model from %s", config.model_path)

    model, tokenizer = load_model(config.model_path)
    sandbox = CodeSandbox()
    debugger = MCTSDebugger(model, tokenizer, sandbox, config)

    dataset = load_json(Path(config.dataset_path))
    if args.limit is not None:
        dataset = dataset[: args.limit]
    output_path = Path(config.output_path)

    # Resume: skip tasks already written to the output file
    completed_ids: set[str] = set()
    if output_path.exists():
        for record in load_jsonl(output_path):
            tid = record.get("task_id")
            if tid:
                completed_ids.add(tid)
        if completed_ids:
            logger.info("Resuming: %d tasks already completed, skipping them", len(completed_ids))

    remaining = [t for t in dataset if t.get("id", "") not in completed_ids]
    logger.info("Starting MCTS collection on %d/%d tasks", len(remaining), len(dataset))

    for i, task in enumerate(remaining):
        logger.info("Task %d/%d: %s", i + 1, len(remaining), task.get("id", i))
        try:
            trajectory = debugger.search(task)
            append_jsonl(trajectory, output_path)
        except Exception as exc:
            logger.error("Task %s failed: %s", task.get("id", i), exc)

    logger.info("MCTS collection complete. Output: %s", output_path)


if __name__ == "__main__":
    main()

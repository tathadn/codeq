"""Preference pair construction from MCTS tree data (Algorithm 1 in paper)."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from pydantic import BaseModel

from src.utils import append_jsonl, load_jsonl

logger = logging.getLogger(__name__)


class PreferenceConfig(BaseModel):
    """Configuration for preference pair construction.

    Attributes:
        alpha: Blending weight for Q = alpha*Q_mcts + (1-alpha)*Q_ai.
        threshold: Min |Q_chosen - Q_rejected| to create a pair.
    """

    alpha: float = 0.5
    threshold: float = 0.2


@dataclass
class PreferencePair:
    """A single DPO preference pair.

    Attributes:
        prompt: The state observation that preceded both actions.
        chosen: The better action string (higher blended Q).
        rejected: The worse action string (lower blended Q).
        q_chosen: Blended Q-value of the chosen action.
        q_rejected: Blended Q-value of the rejected action.
        task_id: Source task identifier.
        depth: Tree depth at which the pair was extracted.
    """

    prompt: str
    chosen: str
    rejected: str
    q_chosen: float
    q_rejected: float
    task_id: str = ""
    depth: int = 0

    def to_dict(self) -> dict:
        """Serialize to a DPO-compatible dict.

        Returns:
            Dict with 'prompt', 'chosen', 'rejected', and metadata.
        """
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "q_chosen": self.q_chosen,
            "q_rejected": self.q_rejected,
            "task_id": self.task_id,
            "depth": self.depth,
        }


def blended_q(q_mcts: float, q_ai: float, alpha: float) -> float:
    """Compute blended Q-value.

    Formula: Q = alpha * Q_mcts + (1 - alpha) * Q_ai

    Args:
        q_mcts: MCTS-derived Q-value (mean reward).
        q_ai: Critic AI score.
        alpha: Blend weight for MCTS component.

    Returns:
        Blended Q in [0, 1].
    """
    return alpha * q_mcts + (1.0 - alpha) * q_ai


def extract_pairs_from_node(
    node: dict, task_id: str, config: PreferenceConfig
) -> Generator[PreferencePair, None, None]:
    """Recursively extract preference pairs from a tree node.

    At each internal node, compare sibling children pairwise: for every
    (child_i, child_j) with child_i having higher blended Q than child_j,
    emit a pair if |Q_i - Q_j| > threshold.

    Args:
        node: Serialized MCTS node dict (from MCTSDebugger._serialize_tree).
        task_id: Task identifier for metadata.
        config: PreferenceConfig with alpha and threshold.

    Yields:
        PreferencePair instances.
    """
    children = node.get("children", [])

    if len(children) >= 2:
        # Build (action, blended_q) for each child
        child_data: list[tuple[dict, float]] = []
        for child in children:
            q_mcts_val = child.get("q_value", 0.0)
            q_ai_val = child.get("ai_score", 0.0)
            q_blend = blended_q(q_mcts_val, q_ai_val, config.alpha)
            child_data.append((child, q_blend))

        # Pairwise comparisons
        state = node.get("state", {})
        prompt = _format_prompt(state)

        for i in range(len(child_data)):
            for j in range(i + 1, len(child_data)):
                child_i, q_i = child_data[i]
                child_j, q_j = child_data[j]

                if abs(q_i - q_j) <= config.threshold:
                    continue

                if q_i >= q_j:
                    chosen_child, q_chosen = child_i, q_i
                    rejected_child, q_rejected = child_j, q_j
                else:
                    chosen_child, q_chosen = child_j, q_j
                    rejected_child, q_rejected = child_i, q_i

                chosen_action = chosen_child.get("action", "")
                rejected_action = rejected_child.get("action", "")

                if chosen_action and rejected_action:
                    yield PreferencePair(
                        prompt=prompt,
                        chosen=chosen_action,
                        rejected=rejected_action,
                        q_chosen=q_chosen,
                        q_rejected=q_rejected,
                        task_id=task_id,
                        depth=node.get("depth", 0),
                    )

    # Recurse into all children
    for child in children:
        yield from extract_pairs_from_node(child, task_id, config)


def _format_prompt(state: dict) -> str:
    """Format a state dict into a prompt string.

    Args:
        state: State dict with 'code' and 'test_output'.

    Returns:
        Formatted prompt string.
    """
    code = state.get("code", "")
    test_output = state.get("test_output", "")
    numbered = "\n".join(
        f"{i + 1:4d} | {line}" for i, line in enumerate(code.splitlines())
    )
    return f"--- Code ---\n{numbered}\n\n--- Test Output ---\n{test_output}"


class PreferenceBuilder:
    """Builds DPO preference pairs from MCTS trajectory files."""

    def __init__(self, config: PreferenceConfig) -> None:
        """Initialize the PreferenceBuilder.

        Args:
            config: PreferenceConfig with alpha and threshold.
        """
        self.config = config

    def build_from_file(self, input_path: Path, output_path: Path) -> int:
        """Process an entire trajectory JSONL and write preference pairs.

        Args:
            input_path: Path to trajectory JSONL (from MCTS collection).
            output_path: Path to write preference pair JSONL.

        Returns:
            Total number of preference pairs written.
        """
        total = 0
        for trajectory in load_jsonl(input_path):
            task_id = trajectory.get("task_id", "unknown")
            root = trajectory.get("root", {})
            for pair in extract_pairs_from_node(root, task_id, self.config):
                append_jsonl(pair.to_dict(), output_path)
                total += 1

        logger.info("Built %d preference pairs -> %s", total, output_path)
        return total


def main() -> None:
    """CLI entry point for preference pair construction.

    Usage:
        python -m src.preferences \\
            --input trajectories/round1.jsonl \\
            --output data/preferences/round1.jsonl \\
            --alpha 0.5 --threshold 0.2
    """
    import argparse

    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Build DPO preference pairs from MCTS trajectories")
    parser.add_argument("--input", required=True, help="Input trajectory JSONL")
    parser.add_argument("--output", required=True, help="Output preference pair JSONL")
    parser.add_argument("--alpha", type=float, default=0.5, help="Q blend weight")
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum Q gap")
    args = parser.parse_args()

    setup_logging()
    config = PreferenceConfig(alpha=args.alpha, threshold=args.threshold)
    builder = PreferenceBuilder(config)
    n = builder.build_from_file(Path(args.input), Path(args.output))
    logger.info("Done. %d pairs written to %s", n, args.output)


if __name__ == "__main__":
    main()

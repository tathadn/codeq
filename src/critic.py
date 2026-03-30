"""AI self-critique: iterative action ranking."""

import logging
from typing import Optional

from src.agent import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are an expert code reviewer. You will be given a buggy code state and a list of proposed fix actions. Your job is to rank these actions from best to worst.

At each round you will see the remaining un-ranked actions and must pick the BEST one by responding with just its number (e.g. "2").

Consider:
- Correctness: Does the action address the root cause of the bug?
- Safety: Is the action minimal and unlikely to introduce new bugs?
- Progress: Does the action move closer to all tests passing?
"""

CRITIC_REWRITE_SYSTEM_PROMPT = """You are an expert code reviewer. You will be shown candidate Python solutions for a bug fix, along with the test output for each.

At each round, pick the BEST remaining solution by responding with only its number (e.g. "1").

Consider:
- Which solution passes more tests (fewer failures)?
- Which is closer to a fully correct implementation?
- Which is cleaner and less likely to introduce new bugs?
"""


class CriticRanker:
    """Ranks K proposed actions using iterative LLM self-critique.

    The LLM is asked repeatedly to pick the best remaining action until
    all are ranked. Returns normalized scores in [0, 1].
    """

    def __init__(self, model: "AutoModelForCausalLM", tokenizer: "AutoTokenizer") -> None:
        """Initialize the CriticRanker.

        Args:
            model: Loaded HuggingFace causal LM.
            tokenizer: Corresponding tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer

    def rank(self, state_observation: str, actions: list[str], temperature: float = 0.2) -> list[float]:
        """Rank actions from best to worst and return normalized scores.

        Scores are assigned as (K - rank_position) / (K - 1) so the best
        action gets 1.0 and the worst gets 0.0. With a single action,
        returns [1.0].

        Args:
            state_observation: Formatted observation of the current state.
            actions: List of proposed action strings.
            temperature: Generation temperature (low = consistent).

        Returns:
            List of normalized scores in [0, 1], parallel to actions.
        """
        k = len(actions)
        if k == 0:
            return []
        if k == 1:
            return [1.0]

        scores: list[float] = [0.0] * k
        remaining_indices = list(range(k))

        for rank_pos in range(k):
            if len(remaining_indices) == 1:
                scores[remaining_indices[0]] = 0.0
                break

            prompt = self._build_ranking_prompt(state_observation, actions, remaining_indices)
            chosen_idx = self._query_model(prompt, remaining_indices, temperature)

            if chosen_idx is None:
                # Fallback: assign remaining scores uniformly
                logger.warning("Critic failed to pick action at rank %d; falling back", rank_pos)
                for i, idx in enumerate(remaining_indices):
                    scores[idx] = (len(remaining_indices) - i - 1) / max(k - 1, 1)
                break

            scores[chosen_idx] = (k - rank_pos) / (k - 1)
            remaining_indices.remove(chosen_idx)

        return scores

    def rank_rewrites(
        self,
        buggy_code: str,
        rewrites: list[str],
        test_outputs: list[str],
        temperature: float = 0.2,
    ) -> list[float]:
        """Rank complete code rewrites using their test outputs as feedback.

        Uses iterative selection: repeatedly asks the LLM to pick the best
        remaining rewrite.  Returns normalized scores in [0, 1].

        Args:
            buggy_code: The original buggy code (shown for context).
            rewrites: Complete code rewrite candidates.
            test_outputs: Test runner output for each rewrite, parallel to rewrites.
            temperature: Sampling temperature (low = consistent).

        Returns:
            Normalized scores in [0, 1], parallel to rewrites.
        """
        k = len(rewrites)
        if k == 0:
            return []
        if k == 1:
            return [1.0]

        scores: list[float] = [0.0] * k
        remaining_indices = list(range(k))

        for rank_pos in range(k):
            if len(remaining_indices) == 1:
                scores[remaining_indices[0]] = 0.0
                break

            prompt = self._build_rewrite_ranking_prompt(
                buggy_code, rewrites, test_outputs, remaining_indices
            )
            chosen_idx = self._query_model(prompt, remaining_indices, temperature)

            if chosen_idx is None:
                logger.warning("Critic failed to pick rewrite at rank %d; falling back", rank_pos)
                for i, idx in enumerate(remaining_indices):
                    scores[idx] = (len(remaining_indices) - i - 1) / max(k - 1, 1)
                break

            scores[chosen_idx] = (k - rank_pos) / (k - 1)
            remaining_indices.remove(chosen_idx)

        return scores

    def _build_rewrite_ranking_prompt(
        self,
        buggy_code: str,
        rewrites: list[str],
        test_outputs: list[str],
        remaining_indices: list[int],
    ) -> list[dict]:
        """Build chat messages for ranking complete code rewrites.

        Args:
            buggy_code: Original buggy code (context).
            rewrites: All proposed rewrite solutions.
            test_outputs: Test output for each rewrite.
            remaining_indices: Indices of not-yet-ranked rewrites.

        Returns:
            List of chat message dicts.
        """
        parts = [f"Original buggy code:\n```python\n{buggy_code}\n```"]
        for idx in remaining_indices:
            parts.append(
                f"Solution {idx + 1}:\n```python\n{rewrites[idx]}\n```\n"
                f"Test output:\n```\n{test_outputs[idx]}\n```"
            )
        user_content = (
            "\n\n".join(parts)
            + "\n\nWhich solution number is BEST? Reply with only the number."
        )
        return [
            {"role": "system", "content": CRITIC_REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _build_ranking_prompt(
        self, state_observation: str, actions: list[str], remaining_indices: list[int]
    ) -> list[dict]:
        """Build the chat messages for the critic.

        Args:
            state_observation: Current state description.
            actions: All proposed actions.
            remaining_indices: Indices of not-yet-ranked actions.

        Returns:
            List of chat message dicts.
        """
        action_list = "\n".join(
            f"{idx + 1}. {actions[idx]}" for idx in remaining_indices
        )
        user_content = (
            f"{state_observation}\n\n"
            f"Remaining actions to rank:\n{action_list}\n\n"
            "Which action number is BEST? Reply with only the number."
        )
        return [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _query_model(
        self, messages: list[dict], valid_indices: list[int], temperature: float
    ) -> Optional[int]:
        """Query the LLM and parse a valid action index from its output.

        Args:
            messages: Chat messages to send.
            valid_indices: Acceptable 0-based action indices.
            temperature: Sampling temperature.

        Returns:
            Chosen 0-based action index, or None on parse failure.
        """
        import torch

        try:
            encoded = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            encoded = encoded.to(self.model.device)
            input_len = encoded["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=8,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][input_len:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Parse first integer from response
            for token in response.split():
                try:
                    num = int(token.strip(".,;:"))
                    zero_based = num - 1
                    if zero_based in valid_indices:
                        return zero_based
                except ValueError:
                    continue

            logger.warning("Critic response not parseable: %r", response)
            return None

        except Exception as exc:
            logger.warning("Critic model query failed: %s", exc)
            return None

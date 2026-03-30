"""Agent prompt templates, action parsing, and observation formatting."""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert software debugger. Your task is to fix bugs in code so that all tests pass.

At each step you will receive:
- The current buggy (or partially fixed) source code
- The output of running the test suite

You must respond with a structured action in the following format:

THOUGHT: <your chain-of-thought reasoning about what is wrong and what to fix>
CODE_ACTION: <one of the following actions>
  EDIT <start_line>-<end_line> <new_code>
  INSERT <after_line> <new_code>
  DELETE <start_line>-<end_line>
  RUN_TESTS
  SUBMIT
EXPLANATION: <why this action will help fix the bug>
STATUS: <CONTINUE or DONE>

On the very first step, also include:
PLAN: <high-level strategy for fixing the bug>

Rules:
- Use EDIT to replace a range of lines with new code.
- Use INSERT to insert new code after a given line number.
- Use DELETE to remove a range of lines.
- Use RUN_TESTS to execute the test suite and observe results.
- Use SUBMIT when all tests pass and you are confident the fix is correct.
- STATUS should be DONE only when using SUBMIT.
- Be precise with line numbers. They are 1-indexed.
"""


@dataclass
class ParsedAction:
    """Represents a parsed agent action from LLM output.

    Attributes:
        plan: High-level fix strategy (first step only).
        thought: Chain-of-thought reasoning.
        code_action: Raw code action string (e.g. "EDIT 3-5 x = 1").
        action_type: One of EDIT, INSERT, DELETE, RUN_TESTS, SUBMIT.
        action_args: Parsed arguments for the action type.
        explanation: Why this action helps.
        status: CONTINUE or DONE.
        raw: Original unparsed LLM output.
    """

    thought: str = ""
    code_action: str = ""
    action_type: str = ""
    action_args: dict = field(default_factory=dict)
    explanation: str = ""
    status: str = "CONTINUE"
    plan: Optional[str] = None
    raw: str = ""


class ActionParser:
    """Parses structured agent actions from LLM text output.

    Field labels may be plain (``THOUGHT:``) or markdown-bold
    (``**THOUGHT:**``) — both are accepted.
    """

    # _F matches an optional leading ** and optional trailing ** before the colon.
    _F = r"\*{0,2}"
    _PLAN_RE = re.compile(
        rf"{_F}PLAN{_F}:\s*(.*?)(?=\n{_F}THOUGHT{_F}:|\n{_F}CODE_ACTION{_F}:|\Z)", re.DOTALL
    )
    _THOUGHT_RE = re.compile(
        rf"{_F}THOUGHT{_F}:\s*(.*?)(?=\n{_F}CODE_ACTION{_F}:|\Z)", re.DOTALL
    )
    _CODE_ACTION_RE = re.compile(
        rf"{_F}CODE_ACTION{_F}:\s*(.*?)(?=\n{_F}EXPLANATION{_F}:|\n{_F}STATUS{_F}:|\Z)", re.DOTALL
    )
    _EXPLANATION_RE = re.compile(
        rf"{_F}EXPLANATION{_F}:\s*(.*?)(?=\n{_F}STATUS{_F}:|\Z)", re.DOTALL
    )
    _STATUS_RE = re.compile(rf"{_F}STATUS{_F}:\s*(CONTINUE|DONE)", re.IGNORECASE)

    _EDIT_RE = re.compile(r"EDIT\s+(\d+)-(\d+)\s+(.*)", re.DOTALL | re.IGNORECASE)
    _EDIT_SPACE_RE = re.compile(r"EDIT\s+(\d+)\s+(\d+)\s+(.*)", re.DOTALL | re.IGNORECASE)
    _INSERT_RE = re.compile(r"INSERT\s+(\d+)(?:-\d+)?\s+(.*)", re.DOTALL | re.IGNORECASE)
    _DELETE_RE = re.compile(r"DELETE\s+(\d+)(?:-(\d+))?", re.IGNORECASE)
    _RUN_TESTS_RE = re.compile(r"RUN_TESTS", re.IGNORECASE)
    _SUBMIT_RE = re.compile(r"SUBMIT", re.IGNORECASE)

    def parse(self, text: str) -> Optional[ParsedAction]:
        """Parse LLM output into a ParsedAction.

        Args:
            text: Raw LLM output string.

        Returns:
            ParsedAction on success, None if parsing fails.
        """
        try:
            action = ParsedAction(raw=text)

            plan_m = self._PLAN_RE.search(text)
            if plan_m:
                action.plan = plan_m.group(1).strip()

            thought_m = self._THOUGHT_RE.search(text)
            if thought_m:
                action.thought = thought_m.group(1).strip()

            code_action_m = self._CODE_ACTION_RE.search(text)
            if not code_action_m:
                logger.warning("No CODE_ACTION found in output")
                return None
            action.code_action = code_action_m.group(1).strip()

            explanation_m = self._EXPLANATION_RE.search(text)
            if explanation_m:
                action.explanation = explanation_m.group(1).strip()

            status_m = self._STATUS_RE.search(text)
            if status_m:
                action.status = status_m.group(1).upper()

            self._parse_code_action(action)
            return action

        except Exception as exc:
            logger.warning("Action parse failure: %s", exc)
            return None

    @staticmethod
    def _normalise_code_action(raw: str) -> str:
        """Strip markdown noise the model sometimes wraps around CODE_ACTION.

        Handles patterns like:
          ** EDIT 3-5 ...      (bold prefix)
          **EDIT 3-5 ...
          ```python\\nEDIT ...  (fenced code block)
          `EDIT 3-5 ...`       (inline backticks)

        Args:
            raw: Raw code_action string from the LLM.

        Returns:
            Cleaned string ready for action-type matching.
        """
        s = raw.strip()
        # Strip leading markdown bold markers and asterisks
        s = re.sub(r"^\*+\s*", "", s)
        # Strip numbered/bullet list prefixes (e.g. "1. EDIT ...", "- EDIT ...")
        s = re.sub(r"^\s*(?:\d+\.\s*|-\s*)", "", s)
        # Strip fenced code block markers (```python, ```, etc.) and inline backticks
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"```\s*$", "", s)
        s = s.strip("`").strip()
        return s

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        """Remove surrounding fenced code blocks and backticks from a code value.

        Args:
            code: Raw code string that may have markdown fencing.

        Returns:
            Clean code string.
        """
        code = re.sub(r"^```[a-zA-Z]*\s*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)
        return code.strip("`").strip()

    def _parse_code_action(self, action: ParsedAction) -> None:
        """Populate action_type and action_args from code_action string.

        Args:
            action: ParsedAction to update in place.
        """
        ca = self._normalise_code_action(action.code_action)
        action.code_action = ca  # store the cleaned version

        edit_m = self._EDIT_RE.match(ca) or self._EDIT_SPACE_RE.match(ca)
        if edit_m:
            action.action_type = "EDIT"
            action.action_args = {
                "start": int(edit_m.group(1)),
                "end": int(edit_m.group(2)),
                "code": self._strip_code_fences(edit_m.group(3).strip()),
            }
            return

        insert_m = self._INSERT_RE.match(ca)
        if insert_m:
            action.action_type = "INSERT"
            action.action_args = {
                "after": int(insert_m.group(1)),
                "code": self._strip_code_fences(insert_m.group(2).strip()),
            }
            return

        delete_m = self._DELETE_RE.match(ca)
        if delete_m:
            action.action_type = "DELETE"
            end = int(delete_m.group(2)) if delete_m.group(2) else int(delete_m.group(1))
            action.action_args = {
                "start": int(delete_m.group(1)),
                "end": end,
            }
            return

        if self._RUN_TESTS_RE.match(ca):
            action.action_type = "RUN_TESTS"
            return

        if self._SUBMIT_RE.match(ca):
            action.action_type = "SUBMIT"
            return

        logger.warning("Unrecognised CODE_ACTION: %s", ca)


def apply_edit(lines: list[str], start: int, end: int, new_code: str) -> list[str]:
    """Replace lines[start-1:end] with new_code lines.

    Args:
        lines: Current source lines (1-indexed semantics).
        start: First line to replace (1-indexed, inclusive).
        end: Last line to replace (1-indexed, inclusive).
        new_code: Replacement source code string.

    Returns:
        Updated list of source lines.
    """
    new_lines = new_code.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    return lines[: start - 1] + new_lines + lines[end:]


def apply_insert(lines: list[str], after: int, new_code: str) -> list[str]:
    """Insert new_code after the given line number.

    Args:
        lines: Current source lines.
        after: Insert after this line number (1-indexed). 0 means prepend.
        new_code: Code to insert.

    Returns:
        Updated list of source lines.
    """
    new_lines = new_code.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    return lines[:after] + new_lines + lines[after:]


def apply_delete(lines: list[str], start: int, end: int) -> list[str]:
    """Delete lines from start to end (inclusive, 1-indexed).

    Args:
        lines: Current source lines.
        start: First line to delete (1-indexed).
        end: Last line to delete (1-indexed).

    Returns:
        Updated list of source lines.
    """
    return lines[: start - 1] + lines[end:]


MCTS_REWRITE_SYSTEM_PROMPT = """You are an expert software debugger. Your task is to fix bugs in Python code so that all tests pass.

You will receive a Python solution and possibly the test output from a previous fix attempt.

Return ONLY the complete corrected Python code. Do not include any explanation, markdown formatting, or code fences — output just the raw Python code.
"""


def format_rewrite_prompt(buggy_code: str, test_output: str, depth: int = 0) -> str:
    """Format a prompt for MCTS full-rewrite mode.

    At depth 0 (first attempt) or when test_output is empty, asks for a
    correction of the buggy code.  At depth > 0 (revision round), also
    includes the test output from the previous attempt.

    Args:
        buggy_code: The current (buggy or partially fixed) code.
        test_output: Test runner output from the previous attempt.
        depth: Revision depth (0 = first attempt, >0 = revision).

    Returns:
        User-turn prompt string.
    """
    if depth == 0 or not test_output:
        return (
            "Here is a buggy Python solution. Return the complete corrected code. "
            "Output only the code, no explanation.\n\n"
            f"```python\n{buggy_code}\n```"
        )
    return (
        "Here is a Python solution that still has failing tests. "
        "Return the complete corrected code. Output only the code, no explanation.\n\n"
        f"```python\n{buggy_code}\n```\n\n"
        f"Test output from the previous attempt:\n```\n{test_output}\n```"
    )


def format_observation(code: str, test_output: str, step: int = 0) -> str:
    """Format the current state as an observation string for the LLM.

    Args:
        code: Current source code.
        test_output: Test runner output.
        step: Current step number (0-indexed).

    Returns:
        Formatted observation string.
    """
    numbered_lines = "\n".join(
        f"{i + 1:4d} | {line}" for i, line in enumerate(code.splitlines())
    )
    return (
        f"=== Step {step} ===\n\n"
        f"--- Current Code ---\n{numbered_lines}\n\n"
        f"--- Test Output ---\n{test_output}\n"
    )

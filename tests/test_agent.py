"""Tests for agent prompt formatting and action parsing."""

import pytest

from src.agent import (
    ActionParser,
    apply_delete,
    apply_edit,
    apply_insert,
    format_observation,
    format_rewrite_prompt,
)


@pytest.fixture
def parser():
    return ActionParser()


# ---------------------------------------------------------------------------
# format_rewrite_prompt
# ---------------------------------------------------------------------------

def test_format_rewrite_prompt_depth_zero_contains_buggy_code():
    prompt = format_rewrite_prompt("x = bug", "", depth=0)
    assert "x = bug" in prompt
    assert "buggy" in prompt.lower()


def test_format_rewrite_prompt_depth_zero_omits_test_output():
    # At depth 0, test_output should not appear even if non-empty
    prompt = format_rewrite_prompt("x = bug", "AssertionError: something", depth=0)
    assert "AssertionError" not in prompt


def test_format_rewrite_prompt_depth_positive_includes_test_output():
    prompt = format_rewrite_prompt("x = bug", "AssertionError: expected 1", depth=1)
    assert "x = bug" in prompt
    assert "AssertionError" in prompt


def test_format_rewrite_prompt_empty_test_output_omits_it():
    # Even at depth > 0, empty test_output should not add a spurious section
    prompt = format_rewrite_prompt("x = bug", "", depth=2)
    assert "Test output" not in prompt


def test_format_rewrite_prompt_asks_for_complete_corrected_code():
    prompt = format_rewrite_prompt("x = bug", "", depth=0)
    assert "complete" in prompt.lower() or "corrected" in prompt.lower()


# ---------------------------------------------------------------------------
# format_observation
# ---------------------------------------------------------------------------

def test_format_observation_includes_code():
    obs = format_observation("x = 1\ny = 2", "1 passed", step=0)
    assert "x = 1" in obs
    assert "y = 2" in obs


def test_format_observation_includes_test_output():
    obs = format_observation("pass", "2 failed, 0 passed", step=1)
    assert "2 failed" in obs


def test_format_observation_line_numbers():
    obs = format_observation("a\nb\nc", "", step=0)
    assert "   1 |" in obs
    assert "   2 |" in obs
    assert "   3 |" in obs


# ---------------------------------------------------------------------------
# ActionParser — valid inputs
# ---------------------------------------------------------------------------

VALID_EDIT = """\
THOUGHT: The variable name is wrong.
CODE_ACTION: EDIT 3-3 result = a + b
EXPLANATION: Renamed variable to result.
STATUS: CONTINUE
"""

VALID_INSERT = """\
THOUGHT: Missing return statement.
CODE_ACTION: INSERT 5 return value
EXPLANATION: Added missing return.
STATUS: CONTINUE
"""

VALID_DELETE = """\
THOUGHT: Duplicate line.
CODE_ACTION: DELETE 7-8
EXPLANATION: Removed duplicate.
STATUS: CONTINUE
"""

VALID_RUN_TESTS = """\
THOUGHT: Let me check tests first.
CODE_ACTION: RUN_TESTS
EXPLANATION: Verify current state.
STATUS: CONTINUE
"""

VALID_SUBMIT = """\
THOUGHT: All tests pass now.
CODE_ACTION: SUBMIT
EXPLANATION: Fix is complete.
STATUS: DONE
"""

VALID_WITH_PLAN = """\
PLAN: Find and fix the off-by-one error in the loop.
THOUGHT: The loop iterates one step too far.
CODE_ACTION: EDIT 4-4 for i in range(len(arr) - 1):
EXPLANATION: Adjusted loop bound.
STATUS: CONTINUE
"""


def test_parse_edit(parser):
    action = parser.parse(VALID_EDIT)
    assert action is not None
    assert action.action_type == "EDIT"
    assert action.action_args["start"] == 3
    assert action.action_args["end"] == 3
    assert "result = a + b" in action.action_args["code"]


def test_parse_insert(parser):
    action = parser.parse(VALID_INSERT)
    assert action is not None
    assert action.action_type == "INSERT"
    assert action.action_args["after"] == 5
    assert "return value" in action.action_args["code"]


def test_parse_delete(parser):
    action = parser.parse(VALID_DELETE)
    assert action is not None
    assert action.action_type == "DELETE"
    assert action.action_args["start"] == 7
    assert action.action_args["end"] == 8


def test_parse_run_tests(parser):
    action = parser.parse(VALID_RUN_TESTS)
    assert action is not None
    assert action.action_type == "RUN_TESTS"


def test_parse_submit(parser):
    action = parser.parse(VALID_SUBMIT)
    assert action is not None
    assert action.action_type == "SUBMIT"
    assert action.status == "DONE"


def test_parse_with_plan(parser):
    action = parser.parse(VALID_WITH_PLAN)
    assert action is not None
    assert action.plan is not None
    assert "off-by-one" in action.plan
    assert action.action_type == "EDIT"


def test_parse_thought(parser):
    action = parser.parse(VALID_EDIT)
    assert action is not None
    assert "variable name" in action.thought.lower()


def test_parse_explanation(parser):
    action = parser.parse(VALID_EDIT)
    assert action is not None
    assert "Renamed" in action.explanation


def test_parse_status_continue(parser):
    action = parser.parse(VALID_EDIT)
    assert action is not None
    assert action.status == "CONTINUE"


# ---------------------------------------------------------------------------
# ActionParser — malformed inputs
# ---------------------------------------------------------------------------

def test_parse_missing_code_action_returns_none(parser):
    bad = "THOUGHT: Something.\nSTATUS: CONTINUE"
    assert parser.parse(bad) is None


def test_parse_unrecognised_action_type(parser):
    bad = "THOUGHT: Hmm.\nCODE_ACTION: UNKNOWN_OP 1 2\nEXPLANATION: x\nSTATUS: CONTINUE"
    action = parser.parse(bad)
    # Should not return None (parse succeeds), but action_type is empty
    assert action is not None
    assert action.action_type == ""


def test_parse_empty_string_returns_none(parser):
    assert parser.parse("") is None


def test_parse_garbage_returns_none(parser):
    assert parser.parse("lkajsdflkajsdf") is None


# ---------------------------------------------------------------------------
# Code mutation helpers
# ---------------------------------------------------------------------------

def _lines(code: str) -> list[str]:
    return code.splitlines(keepends=True)


def test_apply_edit_replaces_range():
    lines = _lines("a = 1\nb = 2\nc = 3\n")
    result = apply_edit(lines, 2, 2, "b = 99")
    assert "".join(result) == "a = 1\nb = 99\nc = 3\n"


def test_apply_edit_multi_line_replacement():
    lines = _lines("a\nb\nc\n")
    result = apply_edit(lines, 1, 2, "x\ny\nz")
    assert "x\n" in result
    assert "z\n" in result
    assert "c\n" in result


def test_apply_insert_after_line():
    lines = _lines("a\nb\nc\n")
    result = apply_insert(lines, 1, "inserted")
    joined = "".join(result)
    assert joined == "a\ninserted\nb\nc\n"


def test_apply_insert_prepend():
    lines = _lines("a\nb\n")
    result = apply_insert(lines, 0, "header")
    assert result[0].strip() == "header"


def test_apply_delete_removes_range():
    lines = _lines("a\nb\nc\nd\n")
    result = apply_delete(lines, 2, 3)
    assert "".join(result) == "a\nd\n"


def test_apply_delete_single_line():
    lines = _lines("a\nb\nc\n")
    result = apply_delete(lines, 2, 2)
    assert "".join(result) == "a\nc\n"

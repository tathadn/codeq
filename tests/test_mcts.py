"""Tests for MCTS logic: UCB1 math, backpropagation, tree serialization.

All model calls are mocked — no GPU required.
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.mcts import MCTSConfig, MCTSDebugger, MCTSNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return MCTSConfig(
        model_path="mock",
        dataset_path="mock",
        output_path="mock",
        k_actions=2,
        max_rollouts=5,
        max_depth=3,
        c_exp=math.sqrt(2),
        alpha=0.5,
        theta_threshold=0.2,
        use_rewrite_mode=False,  # edit mode for backward-compat tests
    )


@pytest.fixture
def rewrite_config():
    return MCTSConfig(
        model_path="mock",
        dataset_path="mock",
        output_path="mock",
        k_actions=2,
        max_rollouts=5,
        max_depth=2,
        c_exp=math.sqrt(2),
        alpha=0.5,
        theta_threshold=0.2,
        use_rewrite_mode=True,
    )


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 0
    tok.apply_chat_template.return_value = MagicMock(shape=(1, 10), __getitem__=MagicMock())
    tok.decode.return_value = ""
    return tok


@pytest.fixture
def mock_sandbox():
    sb = MagicMock()
    sb.execute.return_value = MagicMock(success=True, output="1 failed", all_passed=False)
    return sb


@pytest.fixture
def debugger(mock_model, mock_tokenizer, mock_sandbox, config):
    return MCTSDebugger(mock_model, mock_tokenizer, mock_sandbox, config)


@pytest.fixture
def rewrite_debugger(mock_model, mock_tokenizer, mock_sandbox, rewrite_config):
    return MCTSDebugger(mock_model, mock_tokenizer, mock_sandbox, rewrite_config)


def make_node(q_value: float = 0.0, visit_count: int = 1, ai_score: float = 0.0,
              depth: int = 0, is_terminal: bool = False) -> MCTSNode:
    n = MCTSNode(state={"code": "", "test_output": ""}, depth=depth,
                 is_terminal=is_terminal)
    n.visit_count = visit_count
    n.total_value = q_value * visit_count
    n.ai_score = ai_score
    return n


# ---------------------------------------------------------------------------
# UCB1
# ---------------------------------------------------------------------------

def test_ucb1_unvisited_node_returns_inf():
    node = make_node(visit_count=0)
    assert node.ucb1(c_exp=1.0) == float("inf")


def test_ucb1_formula():
    parent = make_node(visit_count=4)
    child = make_node(q_value=0.5, visit_count=2)
    child.parent = parent

    expected = 0.5 + math.sqrt(2) * math.sqrt(math.log(4) / (1 + 2))
    assert abs(child.ucb1(c_exp=math.sqrt(2)) - expected) < 1e-9


def test_ucb1_prefers_unvisited_over_visited():
    parent = make_node(visit_count=10)
    visited = make_node(q_value=0.9, visit_count=5)
    unvisited = make_node(visit_count=0)
    visited.parent = parent
    unvisited.parent = parent

    assert unvisited.ucb1(1.0) > visited.ucb1(1.0)


def test_ucb1_higher_q_prefers_higher_score():
    parent = make_node(visit_count=10)
    high_q = make_node(q_value=0.9, visit_count=3)
    low_q = make_node(q_value=0.1, visit_count=3)
    high_q.parent = parent
    low_q.parent = parent

    assert high_q.ucb1(math.sqrt(2)) > low_q.ucb1(math.sqrt(2))


# ---------------------------------------------------------------------------
# Q-value property
# ---------------------------------------------------------------------------

def test_q_value_zero_for_unvisited():
    node = MCTSNode(state={})
    assert node.q_value == 0.0


def test_q_value_correct():
    node = MCTSNode(state={})
    node.visit_count = 4
    node.total_value = 3.0
    assert node.q_value == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Backpropagation
# ---------------------------------------------------------------------------

def test_backpropagate_updates_visit_count():
    root = MCTSNode(state={})
    child = MCTSNode(state={}, parent=root)
    path = [root, child]

    MCTSDebugger._backpropagate(path, reward=1.0)

    assert root.visit_count == 1
    assert child.visit_count == 1


def test_backpropagate_accumulates_reward():
    root = MCTSNode(state={})
    child = MCTSNode(state={}, parent=root)
    path = [root, child]

    MCTSDebugger._backpropagate(path, reward=1.0)
    MCTSDebugger._backpropagate(path, reward=0.0)

    # Each node traversed twice; total_value = 1.0 + 0.0 = 1.0
    assert root.visit_count == 2
    assert abs(root.q_value - 0.5) < 1e-9


def test_backpropagate_formula():
    """Q(h,a) = total_value / visit_count (cumulative sum / count)"""
    node = MCTSNode(state={})
    node.visit_count = 3
    node.total_value = 2.0  # q = 2/3

    MCTSDebugger._backpropagate([node], reward=1.0)

    # total_value = 2.0 + 1.0 = 3.0; visit_count = 4; q = 3/4
    expected_q = 3.0 / 4.0
    assert abs(node.q_value - expected_q) < 1e-9


# ---------------------------------------------------------------------------
# Selection (_select)
# ---------------------------------------------------------------------------

def test_select_returns_root_when_no_children(debugger):
    root = MCTSNode(state={"code": "", "test_output": ""})
    selected = debugger._select(root)
    assert selected is root


def test_select_picks_highest_ucb1_child(debugger):
    root = MCTSNode(state={"code": "", "test_output": ""})
    root.visit_count = 10

    child_low = make_node(q_value=0.2, visit_count=5)
    child_high = make_node(q_value=0.8, visit_count=1)
    child_low.parent = root
    child_high.parent = root
    root.children = [child_low, child_high]

    selected = debugger._select(root)
    # child_high has fewer visits → higher UCB1 (and higher q)
    assert selected is child_high


# ---------------------------------------------------------------------------
# _collect_path
# ---------------------------------------------------------------------------

def test_collect_path_root_only():
    root = MCTSNode(state={})
    path = MCTSDebugger._collect_path(root)
    assert path == [root]


def test_collect_path_chain():
    root = MCTSNode(state={})
    child = MCTSNode(state={}, parent=root)
    grandchild = MCTSNode(state={}, parent=child)

    path = MCTSDebugger._collect_path(grandchild)
    assert path == [root, child, grandchild]


# ---------------------------------------------------------------------------
# _apply_action
# ---------------------------------------------------------------------------

def test_apply_action_edit(debugger):
    from src.agent import ParsedAction

    state = {"code": "a = 1\nb = 2\nc = 3\n", "test_output": ""}
    action = ParsedAction(
        action_type="EDIT",
        action_args={"start": 2, "end": 2, "code": "b = 99"},
    )
    new_state = debugger._apply_action(state, action)
    assert new_state is not None
    assert "b = 99" in new_state["code"]


def test_apply_action_delete(debugger):
    from src.agent import ParsedAction

    state = {"code": "a\nb\nc\n", "test_output": ""}
    action = ParsedAction(
        action_type="DELETE",
        action_args={"start": 2, "end": 2},
    )
    new_state = debugger._apply_action(state, action)
    assert new_state is not None
    assert "b" not in new_state["code"]


def test_apply_action_run_tests_no_code_change(debugger):
    from src.agent import ParsedAction

    code = "x = 1\n"
    state = {"code": code, "test_output": ""}
    action = ParsedAction(action_type="RUN_TESTS", action_args={})
    new_state = debugger._apply_action(state, action)
    assert new_state is not None
    assert new_state["code"] == code


def test_apply_action_unknown_type_returns_none(debugger):
    from src.agent import ParsedAction

    state = {"code": "x = 1\n", "test_output": ""}
    action = ParsedAction(action_type="EXPLODE", action_args={})
    assert debugger._apply_action(state, action) is None


# ---------------------------------------------------------------------------
# Tree serialization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rewrite mode: _expand_rewrite
# ---------------------------------------------------------------------------

def test_expand_rewrite_creates_children_with_full_code(rewrite_debugger):
    """Expansion in rewrite mode stores full code as the action field."""
    root = MCTSNode(state={"code": "x = bug", "test_output": "1 failed", "test_code": "assert x == 1"})

    with patch.object(rewrite_debugger, "propose_rewrites", return_value=[
        ("x = 1", "1 passed", True),
        ("x = 2", "1 failed", False),
    ]):
        with patch.object(rewrite_debugger, "rank_rewrites", return_value=[1.0, 0.0]):
            rewrite_debugger._expand_rewrite(root, "assert x == 1")

    assert len(root.children) == 2
    assert root.children[0].action == "x = 1"
    assert root.children[0].state["code"] == "x = 1"
    assert root.children[0].is_terminal is True
    assert root.children[0].depth == 1
    assert root.children[1].action == "x = 2"
    assert root.children[1].is_terminal is False


def test_expand_rewrite_no_proposals_yields_no_children(rewrite_debugger):
    root = MCTSNode(state={"code": "x = bug", "test_output": "", "test_code": "test"})

    with patch.object(rewrite_debugger, "propose_rewrites", return_value=[]):
        rewrite_debugger._expand_rewrite(root, "test")

    assert root.children == []


def test_expand_rewrite_ai_scores_assigned(rewrite_debugger):
    root = MCTSNode(state={"code": "x = bug", "test_output": "", "test_code": "test"})

    with patch.object(rewrite_debugger, "propose_rewrites", return_value=[
        ("x = 1", "ok", False),
        ("x = 2", "ok", False),
    ]):
        with patch.object(rewrite_debugger, "rank_rewrites", return_value=[0.8, 0.3]):
            rewrite_debugger._expand_rewrite(root, "test")

    assert rewrite_debugger._best_unvisited_child(root).ai_score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Rewrite mode: _rollout_rewrite
# ---------------------------------------------------------------------------

def test_rollout_rewrite_terminal_node_returns_1(rewrite_debugger):
    node = MCTSNode(
        state={"code": "x=1", "test_output": "", "test_code": "test"},
        is_terminal=True,
    )
    assert rewrite_debugger._rollout_rewrite(node, "test") == 1.0


def test_rollout_rewrite_returns_1_when_rewrite_passes(rewrite_debugger):
    node = MCTSNode(
        state={"code": "x = bug", "test_output": "1 failed", "test_code": "test", "step": 0},
        depth=0,
    )

    with patch.object(rewrite_debugger, "propose_rewrites", return_value=[
        ("x = 1", "1 passed", True),
    ]):
        reward = rewrite_debugger._rollout_rewrite(node, "test")

    assert reward == 1.0


def test_rollout_rewrite_returns_0_when_all_fail(rewrite_debugger):
    node = MCTSNode(
        state={"code": "x = bug", "test_output": "1 failed", "test_code": "test", "step": 0},
        depth=0,
    )

    with patch.object(rewrite_debugger, "propose_rewrites", return_value=[
        ("x = 2", "1 failed", False),
    ]):
        with patch.object(rewrite_debugger, "rank_rewrites", return_value=[1.0]):
            reward = rewrite_debugger._rollout_rewrite(node, "test")

    assert reward == 0.0


def test_rollout_rewrite_stops_at_max_depth(rewrite_debugger):
    """Rollout starting at depth equal to max_depth does no iterations."""
    node = MCTSNode(
        state={"code": "x = bug", "test_output": "fail", "test_code": "test", "step": 2},
        depth=2,  # == max_depth
    )

    with patch.object(rewrite_debugger, "propose_rewrites", return_value=[
        ("x = 9", "fail", False),
    ]) as mock_propose:
        with patch.object(rewrite_debugger, "rank_rewrites", return_value=[1.0]):
            reward = rewrite_debugger._rollout_rewrite(node, "test")

    # range(max_depth - depth) = range(0) → loop body never executes
    mock_propose.assert_not_called()
    assert reward == 0.0


# ---------------------------------------------------------------------------
# _extract_code
# ---------------------------------------------------------------------------

def test_extract_code_from_python_fence():
    code = MCTSDebugger._extract_code("```python\nx = 1\n```")
    assert code == "x = 1"


def test_extract_code_from_generic_fence():
    code = MCTSDebugger._extract_code("```\nx = 2\n```")
    assert code == "x = 2"


def test_extract_code_raw_response():
    code = MCTSDebugger._extract_code("x = 3")
    assert code == "x = 3"


def test_extract_code_empty_returns_none():
    assert MCTSDebugger._extract_code("") is None


def test_serialize_tree_structure():
    root = MCTSNode(state={"code": "x=1", "test_output": "fail"})
    root.visit_count = 3
    root.total_value = 1.0

    child = MCTSNode(state={"code": "x=2", "test_output": "pass"}, action="EDIT 1-1 x=2",
                     ai_score=0.9, parent=root, depth=1, is_terminal=True)
    root.children = [child]

    result = MCTSDebugger._serialize_tree("task1", root, solved=True)

    assert result["task_id"] == "task1"
    assert result["solved"] is True
    assert result["root"]["visit_count"] == 3
    assert len(result["root"]["children"]) == 1
    assert result["root"]["children"][0]["is_terminal"] is True
    assert result["root"]["children"][0]["action"] == "EDIT 1-1 x=2"

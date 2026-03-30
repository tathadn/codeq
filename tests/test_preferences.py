"""Tests for preference pair construction from MCTS trees."""

import pytest

from src.preferences import (
    PreferenceConfig,
    PreferenceBuilder,
    PreferencePair,
    blended_q,
    extract_pairs_from_node,
)


# ---------------------------------------------------------------------------
# blended_q
# ---------------------------------------------------------------------------

def test_blended_q_midpoint():
    assert blended_q(1.0, 0.0, alpha=0.5) == pytest.approx(0.5)


def test_blended_q_full_mcts():
    assert blended_q(0.8, 0.2, alpha=1.0) == pytest.approx(0.8)


def test_blended_q_full_ai():
    assert blended_q(0.3, 0.9, alpha=0.0) == pytest.approx(0.9)


def test_blended_q_weighted():
    result = blended_q(0.6, 0.4, alpha=0.7)
    expected = 0.7 * 0.6 + 0.3 * 0.4
    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_node(code="x=1", test_output="fail", action=None,
              q_value=0.0, ai_score=0.0, depth=0,
              children=None) -> dict:
    return {
        "state": {"code": code, "test_output": test_output},
        "action": action,
        "q_value": q_value,
        "ai_score": ai_score,
        "depth": depth,
        "children": children or [],
        "is_terminal": False,
    }


# ---------------------------------------------------------------------------
# extract_pairs_from_node — basic cases
# ---------------------------------------------------------------------------

def test_no_children_yields_no_pairs():
    node = make_node()
    config = PreferenceConfig(alpha=0.5, threshold=0.2)
    pairs = list(extract_pairs_from_node(node, "t1", config))
    assert pairs == []


def test_single_child_yields_no_pairs():
    child = make_node(action="EDIT 1-1 x=2", q_value=0.8, ai_score=0.7)
    parent = make_node(children=[child])
    config = PreferenceConfig(alpha=0.5, threshold=0.2)
    pairs = list(extract_pairs_from_node(parent, "t1", config))
    assert pairs == []


def test_two_children_above_threshold_yields_one_pair():
    child_good = make_node(action="EDIT 1-1 x=2", q_value=0.9, ai_score=0.8)
    child_bad = make_node(action="EDIT 1-1 x=0", q_value=0.1, ai_score=0.2)
    parent = make_node(children=[child_good, child_bad])
    config = PreferenceConfig(alpha=0.5, threshold=0.2)

    pairs = list(extract_pairs_from_node(parent, "t1", config))
    assert len(pairs) == 1
    assert pairs[0].chosen == "EDIT 1-1 x=2"
    assert pairs[0].rejected == "EDIT 1-1 x=0"


def test_two_children_below_threshold_yields_no_pairs():
    child_a = make_node(action="EDIT 1-1 x=2", q_value=0.5, ai_score=0.5)
    child_b = make_node(action="EDIT 1-1 x=3", q_value=0.55, ai_score=0.55)
    parent = make_node(children=[child_a, child_b])
    config = PreferenceConfig(alpha=0.5, threshold=0.2)

    pairs = list(extract_pairs_from_node(parent, "t1", config))
    assert pairs == []


def test_blended_q_determines_chosen_rejected():
    # alpha=0.5: blended Q = 0.5*q_mcts + 0.5*q_ai
    # child_a: 0.5*0.8 + 0.5*0.6 = 0.70
    # child_b: 0.5*0.2 + 0.5*0.4 = 0.30
    child_a = make_node(action="ACTION_A", q_value=0.8, ai_score=0.6)
    child_b = make_node(action="ACTION_B", q_value=0.2, ai_score=0.4)
    parent = make_node(children=[child_a, child_b])
    config = PreferenceConfig(alpha=0.5, threshold=0.2)

    pairs = list(extract_pairs_from_node(parent, "t1", config))
    assert len(pairs) == 1
    assert pairs[0].chosen == "ACTION_A"
    assert pairs[0].rejected == "ACTION_B"
    assert pairs[0].q_chosen == pytest.approx(0.70)
    assert pairs[0].q_rejected == pytest.approx(0.30)


def test_three_children_pairwise_comparisons():
    # 3 children → C(3,2) = 3 possible pairs, but only those above threshold
    child_hi = make_node(action="A", q_value=0.9, ai_score=0.9)
    child_mid = make_node(action="B", q_value=0.5, ai_score=0.5)
    child_lo = make_node(action="C", q_value=0.1, ai_score=0.1)
    parent = make_node(children=[child_hi, child_mid, child_lo])
    config = PreferenceConfig(alpha=0.5, threshold=0.2)

    pairs = list(extract_pairs_from_node(parent, "t1", config))
    # hi vs lo: diff 0.8 > 0.2 ✓
    # hi vs mid: diff 0.4 > 0.2 ✓
    # mid vs lo: diff 0.4 > 0.2 ✓
    assert len(pairs) == 3


# ---------------------------------------------------------------------------
# Recursive extraction
# ---------------------------------------------------------------------------

def test_recursive_extraction():
    grandchild_good = make_node(action="EDIT 2-2 x=5", q_value=0.9, ai_score=0.8)
    grandchild_bad = make_node(action="EDIT 2-2 x=0", q_value=0.1, ai_score=0.2)
    child = make_node(action="EDIT 1-1 x=1", q_value=0.5, children=[grandchild_good, grandchild_bad])
    root = make_node(children=[child])  # root has 1 child → no pair at root level

    config = PreferenceConfig(alpha=0.5, threshold=0.2)
    pairs = list(extract_pairs_from_node(root, "t1", config))
    # Only 1 pair, from the grandchildren
    assert len(pairs) == 1
    assert pairs[0].chosen == "EDIT 2-2 x=5"


# ---------------------------------------------------------------------------
# PreferencePair.to_dict
# ---------------------------------------------------------------------------

def test_preference_pair_to_dict():
    pair = PreferencePair(
        prompt="p", chosen="c", rejected="r",
        q_chosen=0.8, q_rejected=0.2, task_id="t1", depth=2
    )
    d = pair.to_dict()
    assert d["prompt"] == "p"
    assert d["chosen"] == "c"
    assert d["rejected"] == "r"
    assert d["q_chosen"] == pytest.approx(0.8)
    assert d["task_id"] == "t1"
    assert d["depth"] == 2


# ---------------------------------------------------------------------------
# PreferenceBuilder (file I/O mocked)
# ---------------------------------------------------------------------------

def test_preference_builder_rewrite_mode(tmp_path):
    """Preference pairs work correctly when action is a full code rewrite."""
    import json

    child_good = make_node(action="def f():\n    return 1\n", q_value=0.9, ai_score=0.8)
    child_bad = make_node(action="def f():\n    return 0\n", q_value=0.1, ai_score=0.2)
    root = make_node(children=[child_good, child_bad])
    trajectory = {"task_id": "rewrite_task", "solved": True, "root": root}

    input_file = tmp_path / "traj.jsonl"
    input_file.write_text(json.dumps(trajectory) + "\n")
    output_file = tmp_path / "prefs.jsonl"

    config = PreferenceConfig(alpha=0.5, threshold=0.2)
    builder = PreferenceBuilder(config)
    n = builder.build_from_file(input_file, output_file)

    assert n == 1
    record = json.loads(output_file.read_text().strip())
    assert "return 1" in record["chosen"]
    assert "return 0" in record["rejected"]
    assert record["task_id"] == "rewrite_task"


def test_preference_builder_counts_pairs(tmp_path):
    import json

    # Write a minimal trajectory file
    child_good = make_node(action="EDIT 1-1 x=2", q_value=0.9, ai_score=0.8)
    child_bad = make_node(action="EDIT 1-1 x=0", q_value=0.1, ai_score=0.2)
    root = make_node(children=[child_good, child_bad])
    trajectory = {"task_id": "t1", "solved": False, "root": root}

    input_file = tmp_path / "traj.jsonl"
    input_file.write_text(json.dumps(trajectory) + "\n")
    output_file = tmp_path / "prefs.jsonl"

    config = PreferenceConfig(alpha=0.5, threshold=0.2)
    builder = PreferenceBuilder(config)
    n = builder.build_from_file(input_file, output_file)

    assert n == 1
    assert output_file.exists()
    lines = output_file.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["chosen"] == "EDIT 1-1 x=2"

"""Prepare DebugBench python3 split: load → synthesize tests via oracle → dedup → split.

For each python3 task in thunlp/DebugBench, parse the LeetCode-style 'examples'
strings into kwargs, exec the oracle Solution to compute expected outputs, and
emit a self-contained pytest file as `test_code` that asserts user solution
matches oracle on every example.

Restricted to ``benchmark/python3_*.json``. Tasks where parsing or oracle
execution fails are dropped with a WARNING (slug + reason). Dedup is sha256
over (buggy_code + '\\0' + test_code). Split is deterministic with seed=42.

Usage:
    python scripts/prepare_data.py --raw-path /tmp/debugbench
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import logging
import random
import re
import signal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Regex captures everything after "Input:" up to (but not including) the
# "Output:" line, or end-of-string for malformed examples.
INPUT_RE = re.compile(r"Input:\s*(.+?)(?=\s*\nOutput:|\Z)", re.DOTALL)

# Common imports / aliases that LeetCode oracle solutions reference without
# importing them. Provided in the exec namespace so most oracle_code samples
# evaluate without modification.
ORACLE_PREAMBLE = (
    "from typing import List, Optional, Tuple, Dict, Set, Any\n"
    "from collections import defaultdict, Counter, deque, OrderedDict\n"
    "from functools import lru_cache, cache, reduce\n"
    "from heapq import heappush, heappop, heapify, nlargest, nsmallest\n"
    "from math import inf, gcd, lcm, ceil, floor, sqrt, log2, log, pi\n"
    "from itertools import combinations, permutations, accumulate, product, chain\n"
    "from bisect import bisect_left, bisect_right, insort\n"
    "import math\n"
    "import string\n"
    "import random\n"
    "import collections\n"
    "import heapq\n"
    "import bisect\n"
    "import itertools\n"
    "import functools\n"
    "import operator\n"
    "from operator import lt, gt, le, ge, eq, ne, add, sub, mul\n"
    "import re\n"
    "import sys\n"
)

# Same imports get embedded into each generated test file so the user solution
# (which may use any of these names without importing them) executes too.
TEST_PREAMBLE_LINES = [
    "from typing import List, Optional, Tuple, Dict, Set, Any",
    "from collections import defaultdict, Counter, deque, OrderedDict",
    "from functools import lru_cache, cache, reduce",
    "from heapq import heappush, heappop, heapify, nlargest, nsmallest",
    "from math import inf, gcd, lcm, ceil, floor, sqrt, log2, log, pi",
    "from itertools import combinations, permutations, accumulate, product, chain",
    "from bisect import bisect_left, bisect_right, insort",
    "import math",
    "import string",
    "from solution import Solution",
]

ORACLE_TIMEOUT_S = 5


# ---------------------------------------------------------------------------
# Example parsing
# ---------------------------------------------------------------------------


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split ``s`` on ``sep`` only at bracket-depth 0 and outside string literals."""
    depth = 0
    in_str: str | None = None
    parts: list[str] = []
    cur: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str is not None:
            cur.append(ch)
            if ch == "\\" and i + 1 < len(s):
                cur.append(s[i + 1])
                i += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
            cur.append(ch)
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    if cur:
        parts.append("".join(cur))
    return parts


def parse_kwargs(input_text: str) -> dict[str, Any]:
    """Parse ``"a = 1, b = [2,3]"`` into ``{'a': 1, 'b': [2,3]}``.

    Each fragment must be of the form ``name = <python literal>``. Anything
    that does not satisfy that (TreeNode literals like ``[1,null,2,3]`` end
    up trying ``ast.literal_eval('null...')`` and raise) is rejected.
    """
    parts = _split_top_level(input_text, ",")
    out: dict[str, Any] = {}
    for raw in parts:
        if "=" not in raw:
            raise ValueError(f"missing '=' in fragment: {raw!r}")
        k, _, v = raw.partition("=")
        key = k.strip()
        if not key.isidentifier():
            raise ValueError(f"invalid kwarg name: {key!r}")
        out[key] = ast.literal_eval(v.strip())
    return out


def extract_input_text(example: str) -> str:
    m = INPUT_RE.search(example)
    if not m:
        raise ValueError("no 'Input:' line")
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# Oracle execution
# ---------------------------------------------------------------------------


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise TimeoutError("oracle execution timed out")


def _candidate_methods_from_source(
    oracle_code: str,
) -> list[tuple[str, set[str]]]:
    """Return [(method_name, set_of_param_names)] for top-level Solution methods."""
    try:
        tree = ast.parse(oracle_code)
    except SyntaxError:
        return []
    out: list[tuple[str, set[str]]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("_"):
                        continue
                    params = {a.arg for a in item.args.args if a.arg != "self"}
                    out.append((item.name, params))
    return out


def _pick_entry_method(
    candidates: list[tuple[str, set[str]]], kwarg_keys: set[str]
) -> str | None:
    """Pick the method whose params match the example kwargs; fall back to last."""
    if not candidates:
        return None
    if kwarg_keys:
        # Exact match wins
        for name, params in candidates:
            if params == kwarg_keys:
                return name
        # Superset (method has extra optional params) is next best
        for name, params in candidates:
            if kwarg_keys.issubset(params):
                return name
    # Fallback: last top-level public method (LeetCode convention is helpers
    # first, entry method last when helpers exist at class scope)
    return candidates[-1][0]


def load_oracle(
    oracle_code: str, kwarg_keys: set[str]
) -> tuple[Any, str]:
    """Exec oracle_code in an isolated namespace and return (instance, method_name)."""
    ns: dict[str, Any] = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(ORACLE_PREAMBLE + oracle_code, ns)  # noqa: S102 - dataset is local
    Solution = ns.get("Solution")
    if Solution is None:
        raise ValueError("no Solution class in oracle_code")
    candidates = _candidate_methods_from_source(oracle_code)
    method_name = _pick_entry_method(candidates, kwarg_keys)
    if method_name is None:
        raise ValueError("no public method on Solution")
    sol = Solution()
    if not hasattr(sol, method_name):
        raise ValueError(f"picked method {method_name!r} not on instance")
    return sol, method_name


def synthesize_test_code(slug: str, examples: list[str], oracle_code: str) -> str:
    """Build a self-contained pytest file that asserts user solution == oracle."""
    # Parse all example inputs upfront so we can use their kwarg names to
    # pick the correct Solution entry method.
    parsed_examples: list[dict[str, Any]] = []
    for example in examples:
        input_text = extract_input_text(example)
        parsed_examples.append(parse_kwargs(input_text))
    if not parsed_examples:
        raise ValueError("no parseable examples")
    kwarg_keys: set[str] = set()
    for kw in parsed_examples:
        kwarg_keys.update(kw.keys())

    sol, method_name = load_oracle(oracle_code, kwarg_keys)
    cases: list[tuple[dict[str, Any], Any]] = []
    for kwargs in parsed_examples:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(ORACLE_TIMEOUT_S)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                expected = getattr(sol, method_name)(**kwargs)
        finally:
            signal.alarm(0)
        # Reject anything we cannot embed as a Python literal (TreeNode, etc.).
        try:
            ast.literal_eval(repr(expected))
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"oracle output not literal-roundtrippable: {exc}"
            )
        cases.append((kwargs, expected))
    if not cases:
        raise ValueError("no parseable examples")

    lines = list(TEST_PREAMBLE_LINES)
    lines.append("")
    lines.append("CASES = [")
    for kw, exp in cases:
        lines.append(f"    ({kw!r}, {exp!r}),")
    lines.append("]")
    lines.append("")
    lines.append("def test_examples():")
    lines.append("    sol = Solution()")
    lines.append("    for kwargs, expected in CASES:")
    lines.append(f"        result = sol.{method_name}(**kwargs)")
    lines.append(
        "        assert result == expected, "
        "f'kwargs={kwargs} expected={expected!r} got={result!r}'"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def load_python3_tasks(raw_path: Path) -> list[dict[str, Any]]:
    bench_dir = raw_path / "benchmark"
    if not bench_dir.is_dir():
        raise FileNotFoundError(f"missing {bench_dir}")
    files = sorted(bench_dir.glob("python3_*.json"))
    if not files:
        raise FileNotFoundError(f"no python3_*.json files in {bench_dir}")
    entries: list[dict[str, Any]] = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            entries.extend(e for e in data if isinstance(e, dict))
    return entries


def build_tasks(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    dropped = 0
    for entry in entries:
        slug = entry.get("slug") or entry.get("id") or "unknown"
        buggy = entry.get("buggy_code")
        oracle = entry.get("oracle_code")
        examples = entry.get("examples") or []
        if not (
            isinstance(buggy, str)
            and isinstance(oracle, str)
            and isinstance(examples, list)
            and examples
        ):
            logger.warning("DROP %s: missing buggy_code/oracle_code/examples", slug)
            dropped += 1
            continue
        try:
            test_code = synthesize_test_code(slug, examples, oracle)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DROP %s: %s", slug, exc)
            dropped += 1
            continue
        out.append({"id": slug, "buggy_code": buggy, "test_code": test_code})
    return out, dropped


def dedup_by_content(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for t in tasks:
        h = hashlib.sha256(
            (t["buggy_code"] + "\0" + t["test_code"]).encode("utf-8")
        ).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        kept.append(t)
    return kept


def split_train_test(
    tasks: list[dict[str, Any]], test_frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_frac))) if shuffled else 0
    return shuffled[n_test:], shuffled[:n_test]


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DebugBench python3 train/test split"
    )
    parser.add_argument(
        "--raw-path",
        required=True,
        type=Path,
        help="Path to a local clone of thunlp/DebugBench",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("data/debugbench.json"),
        help="Output path for MCTS training split",
    )
    parser.add_argument(
        "--test-out",
        type=Path,
        default=Path("data/test_set.json"),
        help="Output path for evaluation holdout",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of deduplicated tasks reserved for the test holdout",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for the split"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw = load_python3_tasks(args.raw_path)
    parsed, dropped = build_tasks(raw)
    deduped = dedup_by_content(parsed)
    train, test = split_train_test(deduped, args.test_frac, args.seed)
    write_json(args.train_out, train)
    write_json(args.test_out, test)

    dup_pct = (
        100.0 * (len(parsed) - len(deduped)) / len(parsed) if parsed else 0.0
    )
    drop_pct = 100.0 * dropped / len(raw) if raw else 0.0
    print("=== DebugBench python3 preparation ===")
    print(f"Raw python3 entries:    {len(raw)}")
    print(f"Successfully parsed:    {len(parsed)}")
    print(f"Dropped:                {dropped}  ({drop_pct:.1f}%)")
    print(f"After dedup (content):  {len(deduped)}  ({dup_pct:.1f}% removed)")
    print(f"Train (MCTS) count:     {len(train)}  -> {args.train_out}")
    print(f"Test  (eval) count:     {len(test)}  -> {args.test_out}")
    print(f"Seed: {args.seed}  Test fraction: {args.test_frac}")


if __name__ == "__main__":
    main()

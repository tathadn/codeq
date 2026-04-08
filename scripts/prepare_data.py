"""Prepare DebugBench data: load raw, dedup by content, split into train/test.

Loads the raw DebugBench dataset from a local clone of thunlp/DebugBench, normalizes
each entry to the {id, buggy_code, test_code} schema that src/mcts.py and
src/evaluate.py expect, deduplicates by content hash (buggy_code + test_code), and
writes a deterministic train/test split.

Usage:
    python scripts/prepare_data.py \\
        --raw-path /path/to/DebugBench \\
        --train-out data/debugbench.json \\
        --test-out data/test_set.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Field name candidates for normalizing heterogeneous raw entries.
ID_FIELDS = ("id", "slug", "task_id", "name")
BUGGY_FIELDS = ("buggy_code", "buggy", "bug_code", "code")
TEST_FIELDS = ("test_code", "test", "tests", "unit_test", "test_case")


def _first_present(entry: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def load_raw(raw_path: Path) -> list[dict[str, Any]]:
    """Recursively load every JSON file under raw_path and flatten into one list.

    Accepts files that contain either a top-level list of entries or a single dict.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"--raw-path does not exist: {raw_path}")

    json_files = sorted(raw_path.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found under {raw_path}")

    entries: list[dict[str, Any]] = []
    for fp in json_files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping unreadable JSON %s: %s", fp, exc)
            continue
        if isinstance(data, list):
            entries.extend(e for e in data if isinstance(e, dict))
        elif isinstance(data, dict):
            entries.append(data)
    return entries


def normalize(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map raw entries to the {id, buggy_code, test_code} schema. Drops incomplete rows."""
    normalized: list[dict[str, Any]] = []
    dropped = 0
    for i, e in enumerate(entries):
        buggy = _first_present(e, BUGGY_FIELDS)
        test = _first_present(e, TEST_FIELDS)
        if not buggy or not test:
            dropped += 1
            continue
        tid = _first_present(e, ID_FIELDS) or f"task_{i}"
        normalized.append({"id": tid, "buggy_code": buggy, "test_code": test})
    if dropped:
        logger.warning("Dropped %d raw entries missing buggy_code or test_code", dropped)
    return normalized


def dedup_by_content(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by sha256(buggy_code + '\\0' + test_code). Keeps first occurrence."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for t in tasks:
        h = hashlib.sha256(
            (t["buggy_code"] + "\0" + t["test_code"]).encode("utf-8")
        ).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(t)
    return out


def split(
    tasks: list[dict[str, Any]], test_frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle deterministically and split into (train, test)."""
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
    parser = argparse.ArgumentParser(description="Prepare DebugBench train/test split")
    parser.add_argument(
        "--raw-path",
        required=True,
        type=Path,
        help="Path to a local clone of thunlp/DebugBench (root of the repo)",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw = load_raw(args.raw_path)
    normalized = normalize(raw)
    deduped = dedup_by_content(normalized)
    train, test = split(deduped, test_frac=args.test_frac, seed=args.seed)

    write_json(args.train_out, train)
    write_json(args.test_out, test)

    dup_pct = (
        100.0 * (len(normalized) - len(deduped)) / len(normalized) if normalized else 0.0
    )
    print("=== DebugBench preparation ===")
    print(f"Raw entries loaded:    {len(raw)}")
    print(f"After normalization:   {len(normalized)}")
    print(f"After dedup (content): {len(deduped)}  ({dup_pct:.1f}% duplicates removed)")
    print(f"Train (MCTS) count:    {len(train)}  -> {args.train_out}")
    print(f"Test  (eval) count:    {len(test)}  -> {args.test_out}")
    print(f"Seed: {args.seed}  Test fraction: {args.test_frac}")


if __name__ == "__main__":
    main()

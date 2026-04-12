"""Filter degenerate (chosen == rejected) preference pairs.

DPO loss on a degenerate pair contributes exactly log(2) with zero gradient
signal. The round-1 dataset has ~38.6% such duplicates per the diagnostic in
reports/scan_all_pairs.json. Drop them up-front so the trainer's effective
batch size reflects only useful signal — and so the filtered file is auditable.

Usage:
    python scripts/filter_degenerate_pairs.py \
        --input data/preferences/round1.jsonl \
        --output data/preferences/round1_filtered.jsonl
"""

import argparse
import json
from pathlib import Path


def filter_pairs(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Stream-filter a preference JSONL, dropping pairs with chosen == rejected.

    Returns:
        (kept, dropped) counts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            record = json.loads(line)
            if record.get("chosen") == record.get("rejected"):
                dropped += 1
                continue
            fout.write(line + "\n")
            kept += 1
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    kept, dropped = filter_pairs(args.input, args.output)
    total = kept + dropped
    pct = (dropped / total * 100) if total else 0.0
    print(f"input:   {args.input}")
    print(f"output:  {args.output}")
    print(f"total:   {total}")
    print(f"kept:    {kept}")
    print(f"dropped: {dropped} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

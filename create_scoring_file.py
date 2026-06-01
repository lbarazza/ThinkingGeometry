"""
Create a to_score.jsonl for LLM-based answer comparison.

Reads a responses JSONL and writes a stripped-down file containing only the
fields needed for scoring: id, question, gold_answer, extracted_answer.
Records where correct is already set (not None) are skipped.

Usage:
    python create_scoring_file.py --responses results/responses/omni_math_l7-8_test_100/responses.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True, help="Responses JSONL file.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: to_score.jsonl alongside the responses file).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or args.responses.parent / "to_score.jsonl"

    records = []
    with args.responses.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    to_score = [
        {
            "id": r["id"],
            "question": r.get("question", ""),
            "gold_answer": r.get("gold_answer", ""),
            "extracted_answer": r.get("extracted_answer", ""),
        }
        for r in records
        if r.get("correct") is None and r.get("extracted_answer", "") != ""
    ]

    with out_path.open("w", encoding="utf-8") as f:
        for rec in to_score:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written {len(to_score)} records to {out_path}")
    skipped = len(records) - len(to_score)
    if skipped:
        print(f"Skipped {skipped} records (already scored or no extracted answer)")


if __name__ == "__main__":
    main()

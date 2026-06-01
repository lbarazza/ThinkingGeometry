"""
Apply LLM scores back to a responses JSONL.

Reads a scored.jsonl (id + correct) and updates the correct field
in the original responses JSONL in-place.

Usage:
    python apply_scores.py --responses results/responses/omni_math_l7-8_test_100/responses.jsonl \
                           --scores   results/responses/omni_math_l7-8_test_100/scored.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True, help="Responses JSONL to update.")
    parser.add_argument("--scores", type=Path, required=True, help="scored.jsonl with id + correct.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scores: dict[str, bool] = {}
    with args.scores.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                scores[rec["id"]] = rec["correct"]

    records = []
    with args.responses.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    updated = 0
    for rec in records:
        if rec["id"] in scores:
            rec["correct"] = scores[rec["id"]]
            updated += 1

    with args.responses.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_correct = sum(r.get("correct") is True for r in records)
    n_total = len(records)
    print(f"Updated {updated} records in {args.responses}")
    print(f"Accuracy: {n_correct}/{n_total} ({100 * n_correct / n_total:.1f}%)" if n_total else "")


if __name__ == "__main__":
    main()

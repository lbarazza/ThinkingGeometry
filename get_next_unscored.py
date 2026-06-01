"""
Return the next unscored record from a responses JSONL as JSON, or nothing if all done.

Reads directly from the responses file and outputs only the fields needed for
scoring (id, question, gold_answer, extracted_answer), skipping records where
correct is already set or extracted_answer is empty.

Used by a scoring agent to process one record at a time without loading the
full file into context. Can resume if interrupted.

Usage:
    python get_next_unscored.py --responses <path> --scored <path>

Prints one JSON line if a record is available, nothing if all are done.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scored_ids: set[str] = set()
    if args.scored.exists():
        with args.scored.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    scored_ids.add(json.loads(line)["id"])

    with args.responses.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["id"] in scored_ids:
                continue
            if rec.get("correct") is not None:
                continue
            if not rec.get("extracted_answer", ""):
                continue
            print(json.dumps({
                "id": rec["id"],
                "question": rec.get("question", ""),
                "gold_answer": rec.get("gold_answer", ""),
                "extracted_answer": rec.get("extracted_answer", ""),
            }, ensure_ascii=False))
            return


if __name__ == "__main__":
    main()

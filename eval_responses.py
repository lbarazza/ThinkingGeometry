from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

FINAL_ANSWER_RE = re.compile(r"####")


def extract_model_answer(model_output: str) -> str:
    """Return the raw text after '####', or '' if not found."""
    if model_output is None:
        return ""
    parts = FINAL_ANSWER_RE.split(model_output, maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def evaluate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach 'extracted_answer' to each record. 'correct' is null for later labeling."""
    results = []
    for rec in records:
        extracted = extract_model_answer(rec.get("model_output", ""))
        results.append({
            **rec,
            "extracted_answer": extracted,
            "correct": None,
        })
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    n = len(results)
    n_no_answer = sum(r["extracted_answer"] == "" for r in results)
    print(f"Total:           {n}")
    print(f"No answer found: {n_no_answer}")


def print_samples(results: list[dict[str, Any]], n: int) -> None:
    sample = random.sample(results, min(n, len(results)))
    sep = "-" * 72
    for r in sample:
        print(sep)
        print(f"ID: {r.get('id', '?')}")
        print(f"Question:  {r.get('question', '')[:200]}")
        print(f"Gold:      {r.get('gold_answer', '')}")
        print(f"Extracted: {r['extracted_answer']}")
        output_preview = (r.get("model_output") or "")[-300:]
        print(f"Output tail:\n{output_preview}")
    print(sep)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model responses: accuracy + sample inspection."
    )
    parser.add_argument("--responses", type=Path, required=True, help="Responses JSONL file.")
    parser.add_argument(
        "--show-n",
        type=int,
        default=0,
        help="Number of sample records to print for inspection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    records = load_jsonl(args.responses)
    results = evaluate_records(records)
    print_summary(results)

    if args.show_n > 0:
        print()
        print_samples(results, args.show_n)


if __name__ == "__main__":
    main()

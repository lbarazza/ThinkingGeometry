from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

FINAL_ANSWER_RE = re.compile(r"####")


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.replace("\\", "")
    answer = answer.rstrip(".")
    answer = re.sub(r"^(\d[\d.]*)\s+[A-Za-z].*", r"\1", answer)
    if "." in answer:
        answer = re.sub(r"\.?0+$", "", answer)
    return answer


def extract_model_answer(model_output: str) -> str:
    """Return the normalized answer extracted after '####', or '' if not found."""
    if model_output is None:
        return ""
    parts = FINAL_ANSWER_RE.split(model_output, maxsplit=1)
    if len(parts) < 2:
        return ""
    raw = parts[1].strip().split("<turn|>")[0].strip()
    first_line = raw.splitlines()[0] if raw.splitlines() else ""
    return normalize_answer(first_line)


def evaluate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach 'extracted_answer' and 'correct' to each record. Returns new list."""
    results = []
    for rec in records:
        extracted = extract_model_answer(rec.get("model_output", ""))
        gold = normalize_answer(rec.get("gold_answer", ""))
        results.append({
            **rec,
            "extracted_answer": extracted,
            "correct": extracted == gold and extracted != "",
        })
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    n_no_answer = sum(r["extracted_answer"] == "" for r in results)
    n_wrong = n - n_correct - n_no_answer
    print(f"Total:   {n}")
    print(f"Correct: {n_correct} ({100 * n_correct / n:.1f}%)" if n else "Correct: 0")
    print(f"Wrong:   {n_wrong}")
    print(f"No answer found: {n_no_answer}")


def print_samples(
    results: list[dict[str, Any]],
    n: int,
    wrong_only: bool = False,
) -> None:
    if wrong_only:
        pool = [r for r in results if not r["correct"]]
        label = "WRONG"
    else:
        pool = results
        label = "SAMPLE"

    sample = random.sample(pool, min(n, len(pool)))
    sep = "-" * 72
    for r in sample:
        ok = "CORRECT" if r["correct"] else "WRONG"
        print(sep)
        print(f"[{ok}] ID: {r.get('id', '?')}")
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
    parser.add_argument(
        "--show-wrong-only",
        action="store_true",
        help="When --show-n is set, only show incorrect examples.",
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
        print_samples(results, args.show_n, wrong_only=args.show_wrong_only)


if __name__ == "__main__":
    main()

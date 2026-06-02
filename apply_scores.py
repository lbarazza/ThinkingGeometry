"""
Apply LLM scores to a responses JSONL, writing a new *_scored.jsonl file.

Reads scored.jsonl (id + correct) and writes a new file to --out directory
(default: postprocessing/<slug>) with '_scored' (or '_scored_<run>') appended
to the stem. The original file is never modified.

Usage:
    python apply_scores.py --responses results/responses/omni_math_l7-8_test_100/responses.jsonl \
                           --scores   postprocessing/omni_math_l7-8_test_100/scored.jsonl \
                           --out      postprocessing/omni_math_l7-8_test_100

    # named run:
    python apply_scores.py --responses ... --scores postprocessing/.../scored_v2.jsonl \
                           --out postprocessing/... --run v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True, help="Responses JSONL to update.")
    parser.add_argument("--scores", type=Path, required=True, help="scored.jsonl with id + correct.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: postprocessing/<slug>).")
    parser.add_argument("--run", type=str, default=None, help="Run name; appended to output stem as '_scored_<run>'.")
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

    out_dir = args.out if args.out is not None else Path("postprocessing") / args.responses.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_scored_{args.run}" if args.run else "_scored"
    out_path = out_dir / (args.responses.stem + suffix + args.responses.suffix)

    updated = 0
    for rec in records:
        if rec["id"] in scores:
            rec["correct"] = scores[rec["id"]]
            updated += 1

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_correct = sum(r.get("correct") is True for r in records)
    n_total = len(records)
    print(f"Written {updated} scored records to {out_path}")
    print(f"Accuracy: {n_correct}/{n_total} ({100 * n_correct / n_total:.1f}%)" if n_total else "")


if __name__ == "__main__":
    main()

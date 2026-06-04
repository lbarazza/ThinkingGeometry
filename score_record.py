"""
Score a single record by calling Claude Haiku to judge answer equivalence.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anthropic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--extracted", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prompt = (
        "Judge whether gold_answer and extracted_answer are equivalent. "
        "Consider them equivalent if they represent the same mathematical answer, "
        "value, expression, or concept regardless of how they are written — "
        "including differences in notation, formatting, simplification, LaTeX vs "
        "plain text, symbolic vs numeric form, or any other representational "
        'difference. Reply with only a JSON object: {"correct": true} or {"correct": false}\n\n'
        f"Question: {args.question}\n"
        f"Gold answer: {args.gold}\n"
        f"Extracted answer: {args.extracted}"
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    match = re.search(r'\{[^}]+\}', text)
    if match:
        result = json.loads(match.group())
        correct = bool(result.get("correct", False))
    else:
        correct = "true" in text.lower() and "false" not in text.lower()

    record = {"id": args.id, "correct": correct}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(f"{args.id}: correct={correct}")


if __name__ == "__main__":
    main()

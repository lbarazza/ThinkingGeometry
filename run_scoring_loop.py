"""
Orchestrate scoring loop: repeatedly call get_next_unscored and score via Claude Haiku.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import anthropic

RESPONSES = "remote_results/responses/omni_math_l7-8_test_500/responses.jsonl"
SCORED = "postprocessing/omni_math_l7-8_test_500/scored.jsonl"
OUT_DIR = Path("postprocessing/omni_math_l7-8_test_500")

client = anthropic.Anthropic()


def get_next() -> dict | None:
    result = subprocess.run(
        ["conda", "run", "-n", "thinking-geometry", "python", "get_next_unscored.py",
         "--responses", RESPONSES, "--scored", SCORED],
        capture_output=True, text=True, cwd="/Users/leo/Documents/projects/ThinkingGeometry"
    )
    out = result.stdout.strip()
    if not out:
        return None
    return json.loads(out)


def score_record(rec: dict) -> bool:
    prompt = (
        "Judge whether gold_answer and extracted_answer are equivalent. "
        "Consider them equivalent if they represent the same mathematical answer, "
        "value, expression, or concept regardless of how they are written — "
        "including differences in notation, formatting, simplification, LaTeX vs "
        "plain text, symbolic vs numeric form, or any other representational "
        'difference. Reply with only a JSON object: {"correct": true} or {"correct": false}\n\n'
        f"Question: {rec['question']}\n"
        f"Gold answer: {rec['gold_answer']}\n"
        f"Extracted answer: {rec['extracted_answer']}"
    )

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    match = re.search(r'\{[^}]+\}', text)
    if match:
        result = json.loads(match.group())
        return bool(result.get("correct", False))
    return "true" in text.lower() and "false" not in text.lower()


def write_score(record_id: str, correct: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "scored.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": record_id, "correct": correct}) + "\n")


def main() -> None:
    count = 0
    while True:
        rec = get_next()
        if rec is None:
            print(f"\nAll records scored. Total scored this run: {count}")
            break
        try:
            correct = score_record(rec)
            write_score(rec["id"], correct)
            count += 1
            print(f"[{count}] {rec['id']}: correct={correct}", flush=True)
        except Exception as e:
            print(f"ERROR on {rec['id']}: {e}", file=sys.stderr)
            # Write as null to skip or retry? Skip for now by writing False
            # Actually don't write anything so it can be retried
            break

    # Run apply_scores
    print("\nRunning apply_scores.py...")
    result = subprocess.run(
        ["conda", "run", "-n", "thinking-geometry", "python", "apply_scores.py",
         "--responses", RESPONSES,
         "--scores", SCORED,
         "--out", str(OUT_DIR)],
        cwd="/Users/leo/Documents/projects/ThinkingGeometry",
        capture_output=False, text=True
    )


if __name__ == "__main__":
    main()

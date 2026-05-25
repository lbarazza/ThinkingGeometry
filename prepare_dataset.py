# src/prepare_gsm8k.py

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


def normalize_answer(answer: str) -> str:
    """
    Normalize GSM8K-style numeric answers.

    Examples:
        "1,234" -> "1234"
        "$16."  -> "16"
    """
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.rstrip(".")
    return answer


def extract_final_answer(gsm8k_answer: str) -> str:
    """
    GSM8K's `answer` field contains reasoning plus the final answer.

    Example:
        "... Therefore the answer is 16. #### 16"

    This function extracts the part after ####.
    """
    if "####" not in gsm8k_answer:
        raise ValueError(f"No final answer marker found in: {gsm8k_answer}")

    final_answer = gsm8k_answer.split("####")[-1]
    return normalize_answer(final_answer)


def build_prompt(question: str) -> str:
    """
    Prompt used for the controlled step-by-step reproduction.
    """
    return f"""Solve the following math problem step by step.

Use exactly this format, no exceptions:
Step 1: ...
Step 2: ...
Step 3: ...
...
#### final answer

Problem: {question}
"""


def prepare_gsm8k(
    split: str,
    n_examples: int | None,
    seed: int,
    output_path: Path,
) -> None:
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    if n_examples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(n_examples, len(dataset))))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            question = example["question"]
            gold_reasoning = example["answer"]
            gold_answer = extract_final_answer(gold_reasoning)

            record = {
                "id": f"gsm8k_{split}_{i:05d}",
                "dataset": "gsm8k",
                "split": split,
                "question": question,
                "gold_reasoning": gold_reasoning,
                "gold_answer": gold_answer,
                "prompt": build_prompt(question),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(dataset)} examples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GSM8K prompts for Gemma reasoning-trajectory experiments."
    )

    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="GSM8K split to use.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of examples to use. Use -1 for the full split.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/prompts/gsm8k_test_200.jsonl"),
        help="Output JSONL path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n_examples = None if args.n == -1 else args.n

    prepare_gsm8k(
        split=args.split,
        n_examples=n_examples,
        seed=args.seed,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset



# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_prompt(question: str) -> str:
    return f"""You are a helpful assistant that solves problems step by step with each step signified by "Step [step_number]: ".
Always provide your final answer after #### at the end.

Question: {question}

Please solve this step by step, putting each step after "Step [step_number]: " and always provide your final answer after ####.

Solution:

"""


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def _normalize_gsm8k_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.rstrip(".")
    return answer


def _extract_gsm8k_final_answer(raw: str) -> str:
    if "####" not in raw:
        raise ValueError(f"No final answer marker in: {raw}")
    return _normalize_gsm8k_answer(raw.split("####")[-1])


def build_gsm8k_records(split: str, n_examples: int | None, seed: int) -> list[dict[str, Any]]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if n_examples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(n_examples, len(dataset))))

    records = []
    for i, example in enumerate(dataset):
        question = example["question"]
        gold_reasoning = example["answer"]
        gold_answer = _extract_gsm8k_final_answer(gold_reasoning)
        records.append({
            "id": f"gsm8k_{split}_{i:05d}",
            "dataset": "gsm8k",
            "split": split,
            "question": question,
            "gold_reasoning": gold_reasoning,
            "gold_answer": gold_answer,
            "prompt": build_prompt(question),
        })
    return records


# ---------------------------------------------------------------------------
# MATH (hendrycks/competition_math)
# ---------------------------------------------------------------------------

def _extract_boxed_answer(solution: str) -> str:
    """Extract the last \\boxed{...} from a MATH solution (handles one level of nested braces)."""
    matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", solution)
    if not matches:
        raise ValueError(f"No \\boxed{{}} found in: {solution[:120]}")
    return matches[-1].strip()


def build_math_records(
    split: str,
    n_examples: int | None,
    seed: int,
    math_levels: list[int] | None = None,
    math_type: str | None = None,
) -> list[dict[str, Any]]:
    dataset = load_dataset("chiayewken/competition_math", split=split)

    # Filter by level ("Level 1" ... "Level 5")
    if math_levels is not None:
        level_strs = {f"Level {l}" for l in math_levels}
        dataset = dataset.filter(lambda ex: ex["level"] in level_strs)

    # Filter by problem type (e.g. "Algebra", "Number Theory")
    if math_type is not None:
        dataset = dataset.filter(lambda ex: ex["type"] == math_type)

    if n_examples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(n_examples, len(dataset))))

    level_tag = "l" + "-".join(str(l) for l in sorted(math_levels)) if math_levels else "all"
    type_tag = f"_{math_type.lower().replace(' ', '_')}" if math_type else ""

    records = []
    for i, example in enumerate(dataset):
        question = example["problem"]
        solution = example["solution"]
        try:
            gold_answer = _extract_boxed_answer(solution)
        except ValueError:
            gold_answer = ""

        records.append({
            "id": f"math_{level_tag}{type_tag}_{split}_{i:05d}",
            "dataset": "math",
            "math_level": example["level"],
            "math_type": example["type"],
            "split": split,
            "question": question,
            "gold_reasoning": solution,
            "gold_answer": gold_answer,
            "prompt": build_prompt(question),
        })
    return records


# ---------------------------------------------------------------------------
# nlile/hendrycks-MATH-benchmark
# level is int (1-5), answer is pre-extracted, subject instead of type
# ---------------------------------------------------------------------------

def build_nlile_math_records(
    split: str,
    n_examples: int | None,
    seed: int,
    math_levels: list[int] | None = None,
    math_type: str | None = None,
) -> list[dict[str, Any]]:
    dataset = load_dataset("nlile/hendrycks-MATH-benchmark", split=split)

    if math_levels is not None:
        dataset = dataset.filter(lambda ex: ex["level"] in math_levels)

    if math_type is not None:
        dataset = dataset.filter(lambda ex: ex["subject"] == math_type)

    if n_examples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(n_examples, len(dataset))))

    level_tag = "l" + "-".join(str(l) for l in sorted(math_levels)) if math_levels else "all"
    type_tag = f"_{math_type.lower().replace(' ', '_')}" if math_type else ""

    records = []
    for i, example in enumerate(dataset):
        records.append({
            "id": f"nlile_math_{level_tag}{type_tag}_{split}_{i:05d}",
            "dataset": "nlile_math",
            "math_level": example["level"],
            "math_type": example["subject"],
            "split": split,
            "question": example["problem"],
            "gold_reasoning": example["solution"],
            "gold_answer": example["answer"],
            "prompt": build_prompt(example["problem"]),
        })
    return records


# ---------------------------------------------------------------------------
# Omni-MATH (KbsdJames/Omni-MATH)
# difficulty is a float ~1–10; split is typically "train" (single split)
# ---------------------------------------------------------------------------

def build_omni_math_records(
    split: str,
    n_examples: int | None,
    seed: int,
    difficulty_min: float = 7.0,
    difficulty_max: float = 8.0,
) -> list[dict[str, Any]]:
    dataset = load_dataset("KbsdJames/Omni-MATH", split=split)

    dataset = dataset.filter(
        lambda ex: difficulty_min <= float(ex["difficulty"]) < difficulty_max
    )

    if n_examples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(n_examples, len(dataset))))

    level_tag = f"l{int(difficulty_min)}-{int(difficulty_max)}"

    records = []
    for i, example in enumerate(dataset):
        records.append({
            "id": f"omni_math_{level_tag}_{split}_{i:05d}",
            "dataset": "omni_math",
            "difficulty": example["difficulty"],
            "source": example.get("source", ""),
            "split": split,
            "question": example["problem"],
            "gold_reasoning": example.get("solution", ""),
            "gold_answer": example.get("answer", ""),
            "prompt": build_prompt(example["problem"]),
        })
    return records


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_records(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} examples to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GSM8K or MATH prompts for Gemma reasoning-trajectory experiments."
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "math", "nlile_math", "omni_math"],
        default="gsm8k",
        help="Dataset to prepare.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of examples. Use -1 for all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to data/prompts/<dataset>_<split>_<n>.jsonl",
    )
    # MATH-specific
    parser.add_argument(
        "--math-level",
        type=int,
        nargs="+",
        dest="math_levels",
        default=None,
        help="MATH difficulty levels to include (e.g. --math-level 3 4). Default: all.",
    )
    parser.add_argument(
        "--math-type",
        type=str,
        default=None,
        help="MATH problem type to include (e.g. 'Algebra'). Default: all.",
    )
    # Omni-MATH-specific
    parser.add_argument(
        "--omni-difficulty-min",
        type=float,
        default=7.0,
        dest="omni_difficulty_min",
        help="Omni-MATH difficulty lower bound (inclusive). Default: 7.0.",
    )
    parser.add_argument(
        "--omni-difficulty-max",
        type=float,
        default=8.0,
        dest="omni_difficulty_max",
        help="Omni-MATH difficulty upper bound (exclusive). Default: 8.0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_examples = None if args.n == -1 else args.n

    if args.dataset == "gsm8k":
        records = build_gsm8k_records(args.split, n_examples, args.seed)
        default_out = Path(f"data/prompts/gsm8k_{args.split}_{args.n}.jsonl")
    elif args.dataset == "math":
        records = build_math_records(
            args.split, n_examples, args.seed, args.math_levels, args.math_type
        )
        level_tag = "l" + "-".join(str(l) for l in sorted(args.math_levels)) if args.math_levels else "all"
        default_out = Path(f"data/prompts/math_{level_tag}_{args.split}_{args.n}.jsonl")
    elif args.dataset == "nlile_math":
        records = build_nlile_math_records(
            args.split, n_examples, args.seed, args.math_levels, args.math_type
        )
        level_tag = "l" + "-".join(str(l) for l in sorted(args.math_levels)) if args.math_levels else "all"
        default_out = Path(f"data/prompts/nlile_math_{level_tag}_{args.split}_{args.n}.jsonl")
    else:
        records = build_omni_math_records(
            args.split, n_examples, args.seed, args.omni_difficulty_min, args.omni_difficulty_max
        )
        level_tag = f"l{int(args.omni_difficulty_min)}-{int(args.omni_difficulty_max)}"
        default_out = Path(f"data/prompts/omni_math_{level_tag}_{args.split}_{args.n}.jsonl")

    out = args.out or default_out
    write_records(records, out)


if __name__ == "__main__":
    main()

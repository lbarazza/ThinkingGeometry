"""
Print 10 samples from step_positions.jsonl with extracted tokens highlighted.
Usage: python inspect_step_positions.py \
    --input postprocessing/omni_math_l7-8_test_500/responses_scored.jsonl \
    --annotations data/annotations/step_positions/gemma-4-31b-it/omni_math_l7-8_test_500.jsonl \
    --model-id google/gemma-4-E4B-it \
    --n 10
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

RESET  = "\033[0m"
BOLD   = "\033[1m"
YELLOW = "\033[1;93m"
CYAN   = "\033[1;96m"


def print_sample(rid, ann, resp, tokenizer, show_prompt=False):
    n_prompt = len(resp["prompt_token_ids"])
    output_ids = resp["output_token_ids"]
    targets = {s["full_token_index"]: s["manual_step_number"] for s in ann["steps"]}

    correct = resp.get("correct")
    correct_str = "✓" if correct else "✗" if correct is False else "?"

    print(BOLD + "=" * 72 + RESET)
    print(BOLD + f"  {rid}   steps={len(ann['steps'])}   correct={correct_str}" + RESET)
    print(BOLD + "=" * 72 + RESET)

    parts = []

    if show_prompt:
        for i, tid in enumerate(resp["prompt_token_ids"]):
            text = tokenizer.decode([tid])
            if i in targets:
                step_num = targets[i]
                label = "prompt" if step_num == -2 else "final" if step_num == -1 else f"step{step_num}"
                parts.append(f"{YELLOW}[{label}: {repr(text)[1:-1]}]{RESET}")
            else:
                parts.append(f"{CYAN}{text}{RESET}")

    for i, tid in enumerate(output_ids):
        abs_idx = n_prompt + i
        text = tokenizer.decode([tid])
        if abs_idx in targets:
            step_num = targets[abs_idx]
            label = "prompt" if step_num == -2 else "final" if step_num == -1 else f"step{step_num}"
            parts.append(f"{YELLOW}[{label}: {repr(text)[1:-1]}]{RESET}")
        else:
            parts.append(text)

    print("".join(parts))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--show-prompt", action="store_true", help="Show full prompt + response instead of response only")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    annotations = {}
    with open(args.annotations) as f:
        for line in f:
            r = json.loads(line)
            if r["steps"]:
                annotations[r["id"]] = r

    responses = {}
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in annotations:
                responses[r["id"]] = r

    sample_ids = list(annotations.keys())[: args.n]
    print(f"Showing {len(sample_ids)} samples\n")

    for rid in sample_ids:
        print_sample(rid, annotations[rid], responses[rid], tokenizer, show_prompt=args.show_prompt)


if __name__ == "__main__":
    main()

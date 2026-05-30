"""
Batch sweep: generate responses for multiple datasets with a single model,
evaluate accuracy on the fly, and save everything to results/.

Edit SWEEP_COMBOS and MODEL_CONFIG at the top to configure the run.

Usage:
    conda activate thinking-geometry
    python run_sweep.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_responses import evaluate_records, normalize_answer
from generate_gemma import build_chat_prompt, generate_one
from prepare_dataset import build_gsm8k_records, build_math_records

# ---------------------------------------------------------------------------
# Configuration — edit here
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    "model_id": "google/gemma-4-31b-it",
    "device": "auto",   # "auto" detects cuda > mps > cpu
    "dtype": "bf16",
    "max_new_tokens": 1024,
    "thinking": False,
}

SWEEP_COMBOS = [
    {
        "slug": "gsm8k_test_100",
        "dataset": "gsm8k",
        "kwargs": {"split": "test", "n_examples": 100, "seed": 0},
    },
    {
        "slug": "math_l3_test_100",
        "dataset": "math",
        "kwargs": {"split": "test", "n_examples": 100, "seed": 0, "math_levels": [3]},
    },
    {
        "slug": "math_l4_test_100",
        "dataset": "math",
        "kwargs": {"split": "test", "n_examples": 100, "seed": 0, "math_levels": [4]},
    },
    {
        "slug": "math_l3-4_test_100",
        "dataset": "math",
        "kwargs": {"split": "test", "n_examples": 100, "seed": 0, "math_levels": [3, 4]},
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(dtype_arg: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_arg]


def load_prompts(dataset: str, kwargs: dict) -> list[dict]:
    if dataset == "gsm8k":
        return build_gsm8k_records(**kwargs)
    if dataset == "math":
        return build_math_records(**kwargs)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep() -> None:
    device = get_device(MODEL_CONFIG["device"])
    dtype = get_dtype(MODEL_CONFIG["dtype"])
    model_id = MODEL_CONFIG["model_id"]

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model: {model_id}  device={device}  dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    results_dir = Path("results")
    summary_rows: list[dict] = []

    for combo in SWEEP_COMBOS:
        slug = combo["slug"]
        print(f"\n{'='*60}")
        print(f"Combo: {slug}")
        print(f"{'='*60}")

        prompts = load_prompts(combo["dataset"], combo["kwargs"])
        print(f"Loaded {len(prompts)} prompts")

        out_dir = results_dir / "responses" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "responses.jsonl"

        # Resume support
        done_ids: set[str] = set()
        existing_records: list[dict] = []
        if out_path.exists():
            with out_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        done_ids.add(rec["id"])
                        existing_records.append(rec)

        to_do = [p for p in prompts if p["id"] not in done_ids]
        print(f"Already done: {len(done_ids)}  Remaining: {len(to_do)}")

        with out_path.open("a", encoding="utf-8") as fout:
            for prompt_rec in tqdm(to_do, desc=slug):
                try:
                    gen = generate_one(
                        model=model,
                        tokenizer=tokenizer,
                        user_prompt=prompt_rec["prompt"],
                        device=device,
                        max_new_tokens=MODEL_CONFIG["max_new_tokens"],
                        enable_thinking=MODEL_CONFIG["thinking"],
                    )
                    output_rec = {
                        **prompt_rec,
                        "model_id": model_id,
                        "device": device,
                        "dtype": MODEL_CONFIG["dtype"],
                        "thinking_mode": MODEL_CONFIG["thinking"],
                        "generation_config": {
                            "do_sample": False,
                            "max_new_tokens": MODEL_CONFIG["max_new_tokens"],
                        },
                        **gen,
                        "error": None,
                    }
                except Exception as e:
                    output_rec = {
                        **prompt_rec,
                        "model_id": model_id,
                        "device": device,
                        "dtype": MODEL_CONFIG["dtype"],
                        "thinking_mode": MODEL_CONFIG["thinking"],
                        "generation_config": {
                            "do_sample": False,
                            "max_new_tokens": MODEL_CONFIG["max_new_tokens"],
                        },
                        "chat_prompt": None,
                        "model_output": None,
                        "num_prompt_tokens": None,
                        "num_output_tokens": None,
                        "generation_time_seconds": None,
                        "error": repr(e),
                    }

                fout.write(json.dumps(output_rec, ensure_ascii=False) + "\n")
                fout.flush()

        # Evaluate
        all_records = existing_records + to_do  # to_do already flushed; reload from file
        with out_path.open(encoding="utf-8") as f:
            all_records = [json.loads(l) for l in f if l.strip()]

        results = evaluate_records(all_records)
        n = len(results)
        n_correct = sum(r["correct"] for r in results)
        n_errors = sum(r.get("error") is not None for r in results)
        accuracy = round(100 * n_correct / n, 1) if n else 0.0

        print(f"  Accuracy: {n_correct}/{n} ({accuracy}%)  Errors: {n_errors}")

        # Write evaluated records back (add correct + extracted_answer fields)
        eval_path = out_dir / "responses.jsonl"
        with eval_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary_rows.append({
            "model": model_id,
            "dataset_slug": slug,
            "n": n,
            "n_correct": n_correct,
            "accuracy_pct": accuracy,
            "n_errors": n_errors,
        })

    # Save summary
    summary_path = results_dir / "sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    # Print markdown table
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'model':<22} {'dataset':<25} {'n':>5} {'accuracy':>10} {'errors':>7}")
    print("-" * 72)
    for row in summary_rows:
        print(
            f"{row['model']:<22} {row['dataset_slug']:<25} {row['n']:>5} "
            f"{row['accuracy_pct']:>9.1f}% {row['n_errors']:>7}"
        )
    print(f"\nFull results saved to: {results_dir}/")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_sweep()

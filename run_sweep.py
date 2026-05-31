"""
Batch sweep: generate responses for multiple datasets with a single model,
evaluate accuracy on the fly, and save everything to results/.

Usage:
    python run_sweep.py --mode local   # quick test with gemma-4-E4B-it, 3 examples per dataset
    python run_sweep.py --mode full    # full run with gemma-4-31b-it, 100 examples per dataset
    python run_sweep.py --mode full --auto-shutdown              # shut down when done (2 min grace)
    python run_sweep.py --mode full --auto-shutdown --shutdown-delay-minutes 5
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_responses import evaluate_records
from generate_gemma import build_chat_prompt, generate_one
from prepare_dataset import build_gsm8k_records, build_math_records, build_nlile_math_records

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "local": {
        "model_config": {
            "model_id": "google/gemma-4-E4B-it",
            "device": "auto",
            "dtype": "bf16",
            "max_new_tokens": 512,
            "thinking": False,
        },
        "combos": [
            {"slug": "gsm8k_test_3",           "dataset": "gsm8k",      "kwargs": {"split": "test", "n_examples": 3, "seed": 0}},
            {"slug": "math_l3_test_3",         "dataset": "math",       "kwargs": {"split": "test", "n_examples": 3, "seed": 0, "math_levels": [3]}},
            {"slug": "math_l4_test_3",         "dataset": "math",       "kwargs": {"split": "test", "n_examples": 3, "seed": 0, "math_levels": [4]}},
            {"slug": "math_l3-4_test_3",       "dataset": "math",       "kwargs": {"split": "test", "n_examples": 3, "seed": 0, "math_levels": [3, 4]}},
            {"slug": "nlile_math_l3_test_3",   "dataset": "nlile_math", "kwargs": {"split": "test", "n_examples": 3, "seed": 0, "math_levels": [3]}},
            {"slug": "nlile_math_l4_test_3",   "dataset": "nlile_math", "kwargs": {"split": "test", "n_examples": 3, "seed": 0, "math_levels": [4]}},
            {"slug": "nlile_math_l3-4_test_3", "dataset": "nlile_math", "kwargs": {"split": "test", "n_examples": 3, "seed": 0, "math_levels": [3, 4]}},
        ],
    },
    "full": {
        "model_config": {
            "model_id": "google/gemma-4-31b-it",
            "device": "auto",
            "dtype": "bf16",
            "max_new_tokens": 2048,
            "thinking": False,
        },
        "combos": [
            {"slug": "gsm8k_test_100",           "dataset": "gsm8k",      "kwargs": {"split": "test", "n_examples": 100, "seed": 0}},
            {"slug": "math_l3_test_100",         "dataset": "math",       "kwargs": {"split": "test", "n_examples": 100, "seed": 0, "math_levels": [3]}},
            {"slug": "math_l4_test_100",         "dataset": "math",       "kwargs": {"split": "test", "n_examples": 100, "seed": 0, "math_levels": [4]}},
            {"slug": "math_l3-4_test_100",       "dataset": "math",       "kwargs": {"split": "test", "n_examples": 100, "seed": 0, "math_levels": [3, 4]}},
            {"slug": "nlile_math_l3_test_100",   "dataset": "nlile_math", "kwargs": {"split": "test", "n_examples": 100, "seed": 0, "math_levels": [3]}},
            {"slug": "nlile_math_l4_test_100",   "dataset": "nlile_math", "kwargs": {"split": "test", "n_examples": 100, "seed": 0, "math_levels": [4]}},
            {"slug": "nlile_math_l3-4_test_100", "dataset": "nlile_math", "kwargs": {"split": "test", "n_examples": 100, "seed": 0, "math_levels": [3, 4]}},
        ],
    },
}

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
    if dataset == "nlile_math":
        return build_nlile_math_records(**kwargs)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(model_config: dict, combos: list[dict]) -> None:
    device = get_device(model_config["device"])
    dtype = get_dtype(model_config["dtype"])
    model_id = model_config["model_id"]

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model: {model_id}  device={device}  dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    results_dir = Path("results")
    summary_rows: list[dict] = []

    for combo in combos:
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
                        max_new_tokens=model_config["max_new_tokens"],
                        enable_thinking=model_config["thinking"],
                    )
                    output_rec = {
                        **prompt_rec,
                        "model_id": model_id,
                        "device": device,
                        "dtype": model_config["dtype"],
                        "thinking_mode": model_config["thinking"],
                        "generation_config": {
                            "do_sample": False,
                            "max_new_tokens": model_config["max_new_tokens"],
                        },
                        **gen,
                        "error": None,
                    }
                except Exception as e:
                    output_rec = {
                        **prompt_rec,
                        "model_id": model_id,
                        "device": device,
                        "dtype": model_config["dtype"],
                        "thinking_mode": model_config["thinking"],
                        "generation_config": {
                            "do_sample": False,
                            "max_new_tokens": model_config["max_new_tokens"],
                        },
                        "chat_prompt": None,
                        "model_output": None,
                        "num_prompt_tokens": None,
                        "num_output_tokens": None,
                        "generation_time_seconds": None,
                        "error": repr(e),
                    }

                output_rec = evaluate_records([output_rec])[0]
                fout.write(json.dumps(output_rec, ensure_ascii=False) + "\n")
                fout.flush()

        # Tally accuracy across all records (existing + newly generated)
        with out_path.open(encoding="utf-8") as f:
            all_records = [json.loads(l) for l in f if l.strip()]

        n = len(all_records)
        n_correct = sum(r.get("correct", False) for r in all_records)
        n_errors = sum(r.get("error") is not None for r in all_records)
        accuracy = round(100 * n_correct / n, 1) if n else 0.0

        print(f"  Accuracy: {n_correct}/{n} ({accuracy}%)  Errors: {n_errors}")

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


def _auto_shutdown(delay_minutes: int) -> None:
    total_seconds = delay_minutes * 60
    print("\n" + "!" * 60)
    print("WARNING: --auto-shutdown is set.")
    print("The machine will shut down in {} minute(s).".format(delay_minutes))
    print("Rented GPU instances keep billing until the shutdown completes.")
    print("Press Ctrl-C to cancel.")
    print("!" * 60)

    elapsed = 0
    interval = 30
    while elapsed < total_seconds:
        remaining = total_seconds - elapsed
        print(f"Shutting down in {remaining}s ...  (Ctrl-C to cancel)")
        time.sleep(min(interval, remaining))
        elapsed += interval

    print("Shutting down now.")
    os.system("sudo shutdown -h now")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["local", "full"],
        default="local",
        help="'local' = quick test with E4B model (3 examples); 'full' = 31B model (100 examples).",
    )
    parser.add_argument(
        "--auto-shutdown",
        action="store_true",
        default=False,
        help="Shut down the machine after the sweep completes (Linux only). Default: off.",
    )
    parser.add_argument(
        "--shutdown-delay-minutes",
        type=int,
        default=2,
        metavar="MINUTES",
        help="Grace period before shutdown when --auto-shutdown is set (default: 2).",
    )
    args = parser.parse_args()
    cfg = CONFIGS[args.mode]
    print(f"Mode: {args.mode}  |  Model: {cfg['model_config']['model_id']}")
    run_sweep(cfg["model_config"], cfg["combos"])
    if args.auto_shutdown:
        _auto_shutdown(args.shutdown_delay_minutes)

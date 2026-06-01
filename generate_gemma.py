from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_existing_ids(path: Path) -> set[str]:
    """
    Allows generation to resume if the script crashes.
    """
    if not path.exists():
        return set()

    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    ids.add(record["id"])
                except Exception:
                    pass
    return ids


def get_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "fp32":
        return torch.float32

    raise ValueError(f"Unknown dtype: {dtype_arg}")


def build_chat_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    enable_thinking: bool,
) -> str:
    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    device: str,
    max_new_tokens: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    chat_prompt = build_chat_prompt(
        tokenizer=tokenizer,
        user_prompt=user_prompt,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - start_time

    generated_ids = output_ids[0][input_len:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    )

    return {
        "chat_prompt": chat_prompt,
        "model_output": generated_text,
        "num_prompt_tokens": int(input_len),
        "num_output_tokens": int(generated_ids.shape[0]),
        "generation_time_seconds": elapsed,
        "prompt_token_ids": output_ids[0][:input_len].tolist(),
        "output_token_ids": generated_ids.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Gemma 4 answers for prepared GSM8K prompts."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/prompts/gsm8k_test_50.jsonl"),
        help="Input JSONL file from prepare_gsm8k.py.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/responses/gemma4_e4b_it/gsm8k_test_50_responses.jsonl"),
        help="Output JSONL file.",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-4-E4B-it",
        help="Hugging Face model id.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, mps, or cpu.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Torch dtype.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of generated tokens.",
    )

    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Gemma 4 thinking mode.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file instead of resuming.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Loading model: {args.model_id}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    )

    model = model.to(device)
    model.eval()

    records = load_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and args.output.exists():
        args.output.unlink()

    existing_ids = load_existing_ids(args.output)

    print(f"Loaded {len(records)} input records")
    print(f"Already completed {len(existing_ids)} records")
    print(f"Writing to {args.output}")

    with args.output.open("a", encoding="utf-8") as fout:
        for record in tqdm(records):
            example_id = record["id"]

            if example_id in existing_ids:
                continue

            user_prompt = record["prompt"]

            try:
                generation = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    user_prompt=user_prompt,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    enable_thinking=args.thinking,
                )

                output_record = {
                    **record,
                    "model_id": args.model_id,
                    "device": device,
                    "dtype": args.dtype,
                    "thinking_mode": args.thinking,
                    "generation_config": {
                        "do_sample": False,
                        "max_new_tokens": args.max_new_tokens,
                    },
                    **generation,
                    "error": None,
                }

            except Exception as e:
                output_record = {
                    **record,
                    "model_id": args.model_id,
                    "device": device,
                    "dtype": args.dtype,
                    "thinking_mode": args.thinking,
                    "generation_config": {
                        "do_sample": False,
                        "max_new_tokens": args.max_new_tokens,
                    },
                    "chat_prompt": None,
                    "model_output": None,
                    "num_prompt_tokens": None,
                    "num_output_tokens": None,
                    "generation_time_seconds": None,
                    "prompt_token_ids": None,
                    "output_token_ids": None,
                    "error": repr(e),
                }

            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
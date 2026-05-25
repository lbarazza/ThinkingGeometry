from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
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
    return torch.float32


def extract_for_record(
    record: dict[str, Any],
    annotation: dict[str, Any],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: list[int],
    steps: set[int] | None,
    device: str,
) -> dict[str, Any]:
    chat_prompt: str = record["chat_prompt"]
    model_output: str = record["model_output"]
    full_text = chat_prompt + model_output
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    actual_len = input_ids.shape[1]

    step_positions = [
        sp for sp in (annotation.get("step_positions") or [])
        if sp.get("manual_step_number") is not None
        and sp.get("full_token_index") is not None
        and (steps is None or sp["manual_step_number"] in steps)
    ]

    if not step_positions:
        return {"id": record["id"], "skipped": True, "reason": "no matching steps"}

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # hidden_states: tuple of (num_layers+1) tensors, each [1, seq_len, hidden_dim]
    # index 0 = embeddings, index L+1 = after transformer block L
    hidden_states = outputs.hidden_states

    manual_step_nums = []
    token_idxs = []
    activations = []

    for sp in step_positions:
        tok_idx = sp["full_token_index"]
        if tok_idx >= actual_len:
            print(
                f"  [warn] {record['id']} step {sp['manual_step_number']}: "
                f"token index {tok_idx} out of range ({actual_len})"
            )
            continue

        layer_vecs = []
        for layer in layers:
            # hidden_states[layer + 1] = output of transformer block `layer`
            hs_idx = layer + 1
            vec = hidden_states[hs_idx][0, tok_idx, :].float().cpu().numpy()
            layer_vecs.append(vec)

        manual_step_nums.append(sp["manual_step_number"])
        token_idxs.append(tok_idx)
        activations.append(layer_vecs)  # [num_layers, hidden_dim]

    del outputs, hidden_states

    return {
        "id": record["id"],
        "skipped": False,
        "activations": np.array(activations, dtype=np.float32),  # [S, L, H]
        "manual_step_numbers": np.array(manual_step_nums, dtype=np.int32),
        "token_indices": np.array(token_idxs, dtype=np.int32),
        "layer_indices": np.array(layers, dtype=np.int32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract residual-stream activations at step-marker token positions."
    )
    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="JSONL responses file (from generate_gemma.py).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="JSONL step-positions annotation file (from extract_step_positions.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated/hidden_states/gemma4_e4b_it"),
        help="Directory to write per-example .npz files.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 8, 16, 24, 33],
        help="0-indexed transformer block indices to extract (e.g. 0 16 33).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "0-indexed manual step numbers to extract (sequential order of detected steps). "
            "Use -1 for the final answer marker. Omit to extract all detected steps."
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-4-E4B-it",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many examples (for debugging).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process examples whose .npz already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    steps: set[int] | None = set(args.steps) if args.steps else None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Loading model: {args.model_id} ({args.dtype}, {device})")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    print(f"Loading responses: {args.responses}")
    records = load_jsonl(args.responses)

    print(f"Loading annotations: {args.annotations}")
    annotations = load_jsonl(args.annotations)
    ann_by_id = {a["id"]: a for a in annotations}

    if args.limit:
        records = records[: args.limit]

    print(
        f"Layers: {args.layers}  |  Manual steps filter: {sorted(steps) if steps else 'all'}  |  "
        f"Examples: {len(records)}"
    )

    skipped = 0
    errors = 0

    for record in tqdm(records, desc="Extracting"):
        ex_id = record["id"]
        out_path = args.output_dir / f"{ex_id}.npz"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        if record.get("error"):
            errors += 1
            continue

        annotation = ann_by_id.get(ex_id)
        if annotation is None or annotation.get("error"):
            errors += 1
            continue

        result = extract_for_record(
            record, annotation, model, tokenizer, args.layers, steps, device
        )

        if result.get("skipped"):
            skipped += 1
            continue

        np.savez(
            out_path,
            activations=result["activations"],
            manual_step_numbers=result["manual_step_numbers"],
            token_indices=result["token_indices"],
            layer_indices=result["layer_indices"],
        )

    print(f"Done. Skipped: {skipped}  Errors: {errors}")


if __name__ == "__main__":
    main()

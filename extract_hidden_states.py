"""
For each annotated record, run a forward pass and extract residual-stream
activations at the step-boundary token positions for the requested layers.
Saves one .npz per record to the output directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_annotations(annotations_path):
    annotations = {}
    with open(annotations_path) as f:
        for line in f:
            rec = json.loads(line)
            annotations[rec["id"]] = rec
    return annotations


def load_responses(responses_path):
    responses = {}
    with open(responses_path) as f:
        for line in f:
            rec = json.loads(line)
            responses[rec["id"]] = rec
    return responses


def extract_for_record(annotation, response, model, layers, device):
    steps = annotation["steps"]
    if not steps:
        return None

    prompt_ids = response["prompt_token_ids"]
    output_ids = response["output_token_ids"]
    full_ids = prompt_ids + output_ids

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    # hidden_states: tuple of (num_layers+1) tensors, each [1, seq_len, hidden_dim]
    # index 0 = embedding output; index i+1 = output of transformer block i (0-indexed)
    hidden_states = out.hidden_states

    token_indices = [s["full_token_index"] for s in steps]
    manual_step_numbers = [s["manual_step_number"] for s in steps]

    S = len(steps)
    L = len(layers)
    H = hidden_states[0].shape[-1]

    activations = np.empty((S, L, H), dtype=np.float32)
    for li, layer_idx in enumerate(layers):
        layer_hs = hidden_states[layer_idx + 1]  # output of transformer block layer_idx
        for si, tok_idx in enumerate(token_indices):
            activations[si, li, :] = layer_hs[0, tok_idx, :].float().cpu().numpy()

    return {
        "activations": activations,
        "manual_step_numbers": np.array(manual_step_numbers, dtype=np.int32),
        "token_indices": np.array(token_indices, dtype=np.int32),
        "layer_indices": np.array(layers, dtype=np.int32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to responses_scored.jsonl")
    parser.add_argument("--annotations", required=True, help="Path to step_positions.jsonl")
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices (e.g. 18,27,35)")
    parser.add_argument("--output-dir", default="data/generated/hidden_states")
    parser.add_argument("--model-id", default="google/gemma-4-31B-it")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    model_slug = args.model_id.split("/")[-1].lower()
    run_slug = Path(args.input).parent.name
    out_dir = Path(args.output_dir) / model_slug / run_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {args.dtype}")

    print(f"Loading model {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype, device_map=device)
    model.eval()

    annotations = load_annotations(args.annotations)
    responses = load_responses(args.input)

    ids = [rid for rid in annotations if rid in responses]
    print(f"Processing {len(ids)} records, layers {layers}")

    done = skipped = 0
    for rid in ids:
        out_path = out_dir / f"{rid}.npz"
        if out_path.exists():
            done += 1
            continue

        annotation = annotations[rid]
        response = responses[rid]

        if not annotation["steps"] or response.get("error"):
            skipped += 1
            continue

        arrays = extract_for_record(annotation, response, model, layers, device)
        if arrays is None:
            skipped += 1
            continue

        np.savez(out_path, **arrays)
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(ids)} done")

    print(f"Done. {done} saved, {skipped} skipped.")


if __name__ == "__main__":
    main()

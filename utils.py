"""Shared helpers."""

from pathlib import Path

import numpy as np


def build_prompt(question, tokenizer):
    ...


def find_step_boundaries(full_token_ids, tokenizer):
    ...


def is_correct(generated_text, ground_truth):
    ...


def load_all_examples(data_dir):
    examples = []
    for path in sorted(Path(data_dir).glob("*.npz")):
        d = np.load(path)
        examples.append({
            "id": path.stem,
            "activations": d["activations"],
            "manual_step_numbers": d["manual_step_numbers"],
            "token_indices": d["token_indices"],
            "layer_indices": d["layer_indices"],
        })
    return examples

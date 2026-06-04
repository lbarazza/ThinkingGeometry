# %%
"""Token-length distribution of model outputs, split by result class."""
import json
import sys
from pathlib import Path

import numpy as np

# %%
def load_records(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def classify(rec: dict) -> str:
    if not rec.get("extracted_answer", ""):
        return "incomplete"
    if rec.get("correct") is True:
        return "correct"
    if rec.get("correct") is False:
        return "wrong"
    return "unscored"


def print_stats(label: str, lengths: list[int]) -> None:
    if not lengths:
        print(f"  {label}: (none)")
        return
    a = np.array(lengths)
    print(
        f"  {label} (n={len(a)}): "
        f"min={a.min()}  p25={np.percentile(a,25):.0f}  "
        f"median={np.median(a):.0f}  p75={np.percentile(a,75):.0f}  "
        f"p95={np.percentile(a,95):.0f}  max={a.max()}  mean={a.mean():.1f}"
    )

# %%
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("remote_results/responses/omni_math_l7-8_test_100/responses.jsonl")
records = load_records(path)
print(f"Loaded {len(records)} records from {path}\n")

classes: dict[str, list[int]] = {"correct": [], "wrong": [], "incomplete": [], "unscored": []}
for rec in records:
    cls = classify(rec)
    n = rec.get("num_output_tokens")
    if n is not None:
        classes[cls].append(n)

print("Output token length distribution by class:")
for cls in ["correct", "wrong", "incomplete", "unscored"]:
    print_stats(cls, classes[cls])

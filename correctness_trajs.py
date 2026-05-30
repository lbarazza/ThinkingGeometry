# %% Imports and paths
import json
import re
from pathlib import Path

RESPONSES_PATH = Path("data/responses/gemma4_e4b_it/gsm8k_train_2000_responses.jsonl")
ANNOTATIONS_PATH = Path("data/annotations/step_positions/gemma4_e4b_it/gsm8k_train_2000_step_positions.jsonl")

EXPECTED_STEP_NUMS = {-1, 0, 1, 2, 3}
FINAL_ANSWER_RE = re.compile(r"####\s*final\s+answer", re.IGNORECASE)


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.replace("\\", "")
    answer = answer.rstrip(".")
    # strip trailing unit words: "1200 meters" → "1200"
    answer = re.sub(r"^(\d[\d.]*)\s+[A-Za-z].*", r"\1", answer)
    # strip trailing decimal zeros: "72.00" → "72", "3.50" → "3.5"
    if "." in answer:
        answer = re.sub(r"\.?0+$", "", answer)
    return answer


# %% Load responses and annotations
responses: dict[str, dict] = {}
with RESPONSES_PATH.open() as f:
    for line in f:
        rec = json.loads(line)
        responses[rec["id"]] = rec

annotations: dict[str, dict] = {}
with ANNOTATIONS_PATH.open() as f:
    for line in f:
        ann = json.loads(line)
        annotations[ann["id"]] = ann

total_loaded = len(responses)

# %% Filter: exactly steps {0, 1, 2, 3, -1}
n_missing_annotation = 0
n_annotation_error = 0
kept: list[dict] = []

for id_, rec in responses.items():
    ann = annotations.get(id_)
    if ann is None:
        print(f"WARNING: no annotation for {id_}")
        n_missing_annotation += 1
        continue
    if ann.get("error") is not None:
        print(f"WARNING: annotation error for {id_}: {ann['error']}")
        n_annotation_error += 1
        continue

    nums = {s["manual_step_number"] for s in ann["step_positions"]}
    if nums == EXPECTED_STEP_NUMS:
        kept.append(rec)

n_excluded = total_loaded - n_missing_annotation - n_annotation_error - len(kept)

# %% Extract model answer and classify correctness
results: list[dict] = []

for rec in kept:
    model_output = rec.get("model_output", "")
    parts = FINAL_ANSWER_RE.split(model_output, maxsplit=1)

    if len(parts) < 2:
        results.append({"rec": rec, "raw": "", "extracted": "", "gold": rec["gold_answer"], "correct": False})
        continue

    raw = parts[1].strip().split("<turn|>")[0].strip()
    lines = raw.splitlines()
    if not lines:
        results.append({"rec": rec, "raw": raw, "extracted": "", "gold": rec["gold_answer"], "correct": False})
        continue

    extracted = normalize_answer(lines[0])
    gold = rec["gold_answer"]
    results.append({"rec": rec, "raw": raw, "extracted": extracted, "gold": gold, "correct": extracted == gold})

# %% Print first 100
RAW_W, NORM_W, GOLD_W = 35, 15, 15
header = f"{'#':>4}  {'Raw model output':<{RAW_W}}  {'Normalized':<{NORM_W}}  {'Gold':<{GOLD_W}}  OK?"
print(header)
print("-" * len(header))
for i, r in enumerate(results[:100]):
    raw_cell = r["raw"].replace("\n", " ")[:RAW_W]
    norm_cell = r["extracted"][:NORM_W]
    gold_cell = r["gold"][:GOLD_W]
    ok = "✓" if r["correct"] else "✗"
    print(f"{i+1:>4}  {raw_cell:<{RAW_W}}  {norm_cell:<{NORM_W}}  {gold_cell:<{GOLD_W}}  {ok}")

# %% Summary report
n_kept = len(kept)
n_correct = sum(r["correct"] for r in results)
n_incorrect = n_kept - n_correct

print()
print(f"Total loaded:        {total_loaded}")
if n_missing_annotation:
    print(f"Missing annotation:  {n_missing_annotation}")
if n_annotation_error:
    print(f"Annotation error:    {n_annotation_error}")
print(f"Excluded (≠5 steps): {n_excluded}")
print(f"Remaining (=5 steps):{n_kept}")
print(f"  Correct:   {n_correct} ({100 * n_correct / n_kept:.1f}%)" if n_kept else "  Correct:   0")
print(f"  Incorrect: {n_incorrect} ({100 * n_incorrect / n_kept:.1f}%)" if n_kept else "  Incorrect: 0")

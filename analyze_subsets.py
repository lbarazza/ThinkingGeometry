# %%
import json
from collections import defaultdict

records = [json.loads(l) for l in open("postprocessing/omni_math_l7-8_test_100/responses_scored.jsonl")]

def subset(rec):
    if rec.get("correct") is True:
        return "correct"
    if not rec.get("extracted_answer"):
        return "incomplete"
    return "wrong"

# Build counts: difficulty -> subset -> count
counts = defaultdict(lambda: defaultdict(int))
totals = defaultdict(int)

for rec in records:
    diff = rec.get("difficulty")
    sub = subset(rec)
    counts[diff][sub] += 1
    totals[diff] += 1

subsets = ["correct", "wrong", "incomplete"]
difficulties = sorted(counts)

# Header
print(f"{'Difficulty':>12}  {'N':>4}  " + "  ".join(f"{s:>10}" for s in subsets))
print("-" * (12 + 4 + 3 + len(subsets) * 13))

for diff in difficulties:
    n = totals[diff]
    row = "  ".join(f"{100 * counts[diff][s] / n:>9.1f}%" for s in subsets)
    print(f"{diff:>12}  {n:>4}  {row}")

# Overall
n = len(records)
overall = defaultdict(int)
for rec in records:
    overall[subset(rec)] += 1
row = "  ".join(f"{100 * overall[s] / n:>9.1f}%" for s in subsets)
print("-" * (12 + 4 + 3 + len(subsets) * 13))
print(f"{'Overall':>12}  {n:>4}  {row}")

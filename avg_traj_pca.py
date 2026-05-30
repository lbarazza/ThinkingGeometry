# %% Imports and paths
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

RESPONSES_PATH = Path("data/responses/gemma4_e4b_it/gsm8k_train_2000_responses.jsonl")
ANNOTATIONS_PATH = Path("data/annotations/step_positions/gemma4_e4b_it/gsm8k_train_2000_step_positions.jsonl")
DATA_DIR = Path("data/generated/hidden_states/gemma4_e4b_it/gsm8k_train_2000")
FIGURES_DIR = Path("results/figures")

EXPECTED_STEP_NUMS = {-1, 0, 1, 2, 3}  # annotation convention
NPZ_STEP_NUMS = {-1, 1, 2, 3}          # npz convention (1-indexed thinking steps)
STEPS_ORDERED = [1, 2, 3, -1]
STEP_LABELS = {1: "step 1", 2: "step 2", 3: "step 3", -1: "final ans"}
LAYER = 33
FINAL_ANSWER_RE = re.compile(r"####\s*final\s+answer", re.IGNORECASE)


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.replace("\\", "")
    answer = answer.rstrip(".")
    answer = re.sub(r"^(\d[\d.]*)\s+[A-Za-z].*", r"\1", answer)
    if "." in answer:
        answer = re.sub(r"\.?0+$", "", answer)
    return answer


# %% Build correct/incorrect ID sets from correctness_trajs logic
responses = {}
with RESPONSES_PATH.open() as f:
    for line in f:
        rec = json.loads(line)
        responses[rec["id"]] = rec

annotations = {}
with ANNOTATIONS_PATH.open() as f:
    for line in f:
        ann = json.loads(line)
        annotations[ann["id"]] = ann

correct_ids: set[str] = set()
incorrect_ids: set[str] = set()

for id_, rec in responses.items():
    ann = annotations.get(id_)
    if ann is None or ann.get("error") is not None:
        continue
    nums = {s["manual_step_number"] for s in ann["step_positions"]}
    if nums != EXPECTED_STEP_NUMS:
        continue

    model_output = rec.get("model_output", "")
    parts = FINAL_ANSWER_RE.split(model_output, maxsplit=1)
    if len(parts) < 2:
        incorrect_ids.add(id_)
        continue
    raw = parts[1].strip().split("<turn|>")[0].strip()
    lines = raw.splitlines()
    if not lines:
        incorrect_ids.add(id_)
        continue
    extracted = normalize_answer(lines[0])
    if extracted == rec["gold_answer"]:
        correct_ids.add(id_)
    else:
        incorrect_ids.add(id_)

print(f"Correct IDs: {len(correct_ids)}  |  Incorrect IDs: {len(incorrect_ids)}")

# %% Load hidden states at layer 33
per_step_vecs: dict[str, dict[int, list]] = {
    "correct": defaultdict(list),
    "incorrect": defaultdict(list),
}
all_vecs: dict[str, list] = {"correct": [], "incorrect": []}

n_loaded = {"correct": 0, "incorrect": 0}
n_missing = 0
n_wrong_steps = 0

for label, id_set in [("correct", correct_ids), ("incorrect", incorrect_ids)]:
    for id_ in id_set:
        npz_path = DATA_DIR / f"{id_}.npz"
        if not npz_path.exists():
            n_missing += 1
            continue
        d = np.load(npz_path)
        layer_indices = d["layer_indices"].tolist()
        if LAYER not in layer_indices:
            n_wrong_steps += 1
            continue
        manual_step_numbers = d["manual_step_numbers"].tolist()
        if set(manual_step_numbers) != NPZ_STEP_NUMS:
            n_wrong_steps += 1
            continue
        pos = layer_indices.index(LAYER)
        acts = d["activations"][:, pos, :]  # (S, 2560)
        for step_num, vec in zip(manual_step_numbers, acts):
            per_step_vecs[label][step_num].append(vec)
            all_vecs[label].append(vec)
        n_loaded[label] += 1

print(f"Loaded — correct: {n_loaded['correct']}  incorrect: {n_loaded['incorrect']}")
print(f"Skipped — missing npz: {n_missing}  wrong step pattern: {n_wrong_steps}")

# %% Compute mean trajectories
mean_traj: dict[str, np.ndarray] = {}
for label in ("correct", "incorrect"):
    means = []
    for step in STEPS_ORDERED:
        vecs = per_step_vecs[label][step]
        means.append(np.mean(vecs, axis=0))
    mean_traj[label] = np.stack(means)  # (4, 2560)

# %% Fit PCA on all individual vectors
X_correct = np.stack(all_vecs["correct"]).astype(np.float32)
X_incorrect = np.stack(all_vecs["incorrect"]).astype(np.float32)
X_all = np.concatenate([X_correct, X_incorrect], axis=0)

pca = PCA(n_components=2, random_state=0)
pca.fit(X_all)

var = pca.explained_variance_ratio_
print(f"PCA variance explained — PC1: {var[0]:.1%}  PC2: {var[1]:.1%}  total: {sum(var):.1%}")

X_correct_2d = pca.transform(X_correct)
X_incorrect_2d = pca.transform(X_incorrect)
mean_correct_2d = pca.transform(mean_traj["correct"])
mean_incorrect_2d = pca.transform(mean_traj["incorrect"])

# %% Plot
step_hover = [STEP_LABELS[s] for s in STEPS_ORDERED]

traces = [
    go.Scatter(
        x=X_correct_2d[:, 0], y=X_correct_2d[:, 1],
        mode="markers",
        marker=dict(size=4, color="steelblue", opacity=0.35),
        name="correct (all steps)",
    ),
    go.Scatter(
        x=X_incorrect_2d[:, 0], y=X_incorrect_2d[:, 1],
        mode="markers",
        marker=dict(size=4, color="tomato", opacity=0.35),
        name="incorrect (all steps)",
    ),
    go.Scatter(
        x=mean_correct_2d[:, 0], y=mean_correct_2d[:, 1],
        mode="lines+markers+text",
        marker=dict(size=12, color="steelblue", symbol="circle"),
        line=dict(width=3, color="steelblue"),
        text=step_hover,
        textposition="top right",
        textfont=dict(size=11, color="steelblue"),
        name="avg correct trajectory",
    ),
    go.Scatter(
        x=mean_incorrect_2d[:, 0], y=mean_incorrect_2d[:, 1],
        mode="lines+markers+text",
        marker=dict(size=12, color="tomato", symbol="circle"),
        line=dict(width=3, color="tomato"),
        text=step_hover,
        textposition="bottom right",
        textfont=dict(size=11, color="tomato"),
        name="avg incorrect trajectory",
    ),
]

fig = go.Figure(data=traces)
fig.update_layout(
    title=(f"PCA of hidden states at layer {LAYER} — correct vs incorrect trajectories<br>"
           f"n_correct={n_loaded['correct']}, n_incorrect={n_loaded['incorrect']}, "
           f"PC1+PC2={sum(var):.1%}"),
    template="plotly_dark",
    xaxis_title=f"PC1 ({var[0]:.1%})",
    yaxis_title=f"PC2 ({var[1]:.1%})",
    legend=dict(itemsizing="constant"),
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out_path = str(FIGURES_DIR / f"avg_traj_pca_layer{LAYER}.html")
fig.write_html(out_path)
print(f"Saved: {out_path}")
os.system(f"open '{out_path}'")

# %%

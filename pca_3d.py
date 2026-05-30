# %% Import libraries and set constants
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils import load_all_examples

DATA_DIR = "data/generated/hidden_states/gemma4_e4b_it/gsm8k_train_2000"
FIGURES_DIR = "results/figures"

# %% Load data

examples = load_all_examples(DATA_DIR)
print(f"Loaded {len(examples)} examples")

# %% PCA 3D visualization

LAYER = 24  # transformer block index; must be in layer_indices (e.g. [0, 8, 16, 24, 33])

vecs = []
labels = []
example_ids = []

for i, ex in enumerate(examples):
    layer_indices = ex["layer_indices"].tolist()
    if LAYER not in layer_indices:
        continue
    pos = layer_indices.index(LAYER)
    acts = ex["activations"][:, pos, :]

    # uncomment for raw activations:
    vecs.append(acts)
    # uncomment for within-example centering (removes problem-content signal):
    # vecs.append(acts - acts.mean(axis=0, keepdims=True))

    labels.append(ex["manual_step_numbers"])
    example_ids.append(np.full(len(ex["manual_step_numbers"]), i))

STEPS = [-1, 1, 2, 3, 4]  # set to e.g. [1, 2, 3] to include only those step numbers; None = all steps (-1 = final answer)
DIMS = 2  # 2 or 3

X = np.concatenate(vecs, axis=0).astype(np.float32)
y = np.concatenate(labels, axis=0)
ex_ids = np.concatenate(example_ids, axis=0)

if STEPS is not None:
    mask = np.isin(y, STEPS)
    X, y, ex_ids = X[mask], y[mask], ex_ids[mask]

print(f"Total vectors: {X.shape[0]}  |  layer {LAYER}  |  steps {STEPS or 'all'}  |  hidden dim {X.shape[1]}")

# uncomment for PCA (maximises variance, ignores labels):
X_red = PCA(n_components=DIMS, random_state=0).fit_transform(X)
# uncomment for LDA (maximises class separation between steps):
# X_red = LinearDiscriminantAnalysis(n_components=DIMS).fit_transform(X, y)

_steps_sorted = sorted(s for s in set(y.tolist()) if s != -1) + ([-1] if -1 in y else [])
step_labels = {s: (f"step {s}" if s != -1 else "final answer") for s in _steps_sorted}
colors = [f"hsl({int(i * 360 / len(step_labels))},70%,55%)" for i in range(len(step_labels))]

traces = []
for (s, label), color in zip(step_labels.items(), colors):
    mask = y == s
    if DIMS == 3:
        traces.append(go.Scatter3d(
            x=X_red[mask, 0], y=X_red[mask, 1], z=X_red[mask, 2],
            mode="markers",
            marker=dict(size=2, color=color, opacity=0.5),
            name=label,
        ))
    else:
        traces.append(go.Scatter(
            x=X_red[mask, 0], y=X_red[mask, 1],
            mode="markers",
            marker=dict(size=4, color=color, opacity=0.5),
            name=label,
        ))

fig = go.Figure(data=traces)
if DIMS == 3:
    fig.update_layout(
        title=f"Hidden states at layer {LAYER} — {DIMS}D",
        template="plotly_dark",
        scene=dict(xaxis_title="dim 1", yaxis_title="dim 2", zaxis_title="dim 3"),
        legend=dict(itemsizing="constant"),
    )
else:
    fig.update_layout(
        title=f"Hidden states at layer {LAYER} — {DIMS}D",
        template="plotly_dark",
        xaxis_title="dim 1",
        yaxis_title="dim 2",
        legend=dict(itemsizing="constant"),
    )

Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
out_path = f"{FIGURES_DIR}/layer{LAYER}_{DIMS}d.html"
fig.write_html(out_path)
print(f"Saved: {out_path}")
import os
os.system(f"open '{out_path}'")

# %% Step mean collinearity analysis

sorted_steps = sorted((s for s in step_labels if s != -1)) + ([-1] if -1 in step_labels else [])
consecutive_pairs = list(zip(sorted_steps[:-1], sorted_steps[1:]))

# Per-example difference vectors: for each example, compute vec(step_i+1) - vec(step_i)
# then average across examples to get the mean transition direction
# Find examples that have all steps present
complete_ex_ids = set(np.unique(ex_ids))
for s in sorted_steps:
    complete_ex_ids &= set(ex_ids[y == s].tolist())
print(f"Examples with all steps present: {len(complete_ex_ids)}")

per_example_diffs = {pair: [] for pair in consecutive_pairs}
for ex_id in sorted(complete_ex_ids):
    mask_ex = ex_ids == ex_id
    ex_X, ex_y = X[mask_ex], y[mask_ex]  # original activation space
    for s1, s2 in consecutive_pairs:
        per_example_diffs[(s1, s2)].append(ex_X[ex_y == s2].mean(0) - ex_X[ex_y == s1].mean(0))

mean_diffs = {}
for pair, diffs in per_example_diffs.items():
    if diffs:
        mean_diffs[pair] = np.mean(diffs, axis=0)

print("Mean per-example transition vectors in activation space (averaged over examples):")
for (s1, s2), d in mean_diffs.items():
    print(f"  {step_labels[s1]} → {step_labels[s2]}: norm={np.linalg.norm(d):.4f}  (n={len(per_example_diffs[(s1,s2)])})")

diff_vecs = list(mean_diffs.values())
pairs = list(mean_diffs.keys())

# (1) cosine of mean diffs — do average trajectories point the same direction?
print("\nCosine similarities between consecutive MEAN transition vectors (mean-then-cosine):")
for i in range(len(diff_vecs) - 1):
    a, b = diff_vecs[i], diff_vecs[i + 1]
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    (s1a, s2a), (s1b, s2b) = pairs[i], pairs[i + 1]
    print(f"  ({step_labels[s1a]}→{step_labels[s2a]}) · ({step_labels[s1b]}→{step_labels[s2b]}): {cos:+.4f}")

# (2) per-example cosine then average — are individual trajectories smooth?
print("\nMean per-example cosine similarities between consecutive transition vectors (cosine-then-mean):")
for i in range(len(pairs) - 1):
    (s1a, s2a), (s1b, s2b) = pairs[i], pairs[i + 1]
    per_ex_cos = []
    for a, b in zip(per_example_diffs[(s1a, s2a)], per_example_diffs[(s1b, s2b)]):
        per_ex_cos.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    per_ex_cos = np.array(per_ex_cos)
    print(f"  ({step_labels[s1a]}→{step_labels[s2a]}) · ({step_labels[s1b]}→{step_labels[s2b]}): "
          f"mean={per_ex_cos.mean():+.4f}  std={per_ex_cos.std():.4f}")

# Global linearity: PCA on cluster means restricted to complete examples
complete_mask = np.isin(ex_ids, list(complete_ex_ids))
print(f"\nPCA of cluster means (complete examples only):")

means_complete = np.array([X[complete_mask & (y == s)].mean(0) for s in sorted_steps])
pca_means = PCA().fit(means_complete)
print(f"Variance explained by PCs of cluster means (complete examples only):")
for i, var in enumerate(pca_means.explained_variance_ratio_):
    print(f"  PC{i+1}: {var*100:.1f}%")
print(f"  → {'means are nearly collinear' if pca_means.explained_variance_ratio_[0] > 0.99 else 'means deviate from a line'} "
      f"(PC1={pca_means.explained_variance_ratio_[0]*100:.1f}%)")

# %%

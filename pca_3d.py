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

for ex in examples:
    layer_indices = ex["layer_indices"].tolist()
    if LAYER not in layer_indices:
        continue
    pos = layer_indices.index(LAYER)
    acts = ex["activations"][:, pos, :]
    
    # uncomment for raw activations:
    # vecs.append(acts)
    # uncomment for within-example centering (removes problem-content signal):
    vecs.append(acts - acts.mean(axis=0, keepdims=True))


    labels.append(ex["manual_step_numbers"])

X = np.concatenate(vecs, axis=0).astype(np.float32)
y = np.concatenate(labels, axis=0)
print(f"Total vectors: {X.shape[0]}  |  layer {LAYER}  |  hidden dim {X.shape[1]}")

# uncomment for PCA (maximises variance, ignores labels):
X_3d = PCA(n_components=3, random_state=0).fit_transform(X)
# uncomment for LDA (maximises class separation between steps):
# X_3d = LinearDiscriminantAnalysis(n_components=3).fit_transform(X, y)

step_labels = {s: (f"step {s}" if s != -1 else "final answer") for s in sorted(set(y.tolist()))}
colors = [f"hsl({int(i * 360 / len(step_labels))},70%,55%)" for i in range(len(step_labels))]

traces = []
for (s, label), color in zip(step_labels.items(), colors):
    mask = y == s
    traces.append(go.Scatter3d(
        x=X_3d[mask, 0], y=X_3d[mask, 1], z=X_3d[mask, 2],
        mode="markers",
        marker=dict(size=2, color=color, opacity=0.5),
        name=label,
    ))

fig = go.Figure(data=traces)
fig.update_layout(
    title=f"PCA 3D of hidden states at layer {LAYER}",
    template="plotly_dark",
    scene=dict(xaxis_title="PC 1", yaxis_title="PC 2", zaxis_title="PC 3"),
    legend=dict(itemsizing="constant"),
)

Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
out_path = f"{FIGURES_DIR}/pca3d_layer{LAYER}.html"
fig.write_html(out_path)
print(f"Saved: {out_path}")
import os
os.system(f"open '{out_path}'")

# %%

# %% Import libraries and set constants
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import load_all_examples

DATA_DIR = "data/generated/hidden_states/gemma4_e4b_it/gsm8k_train_2000"
FIGURES_DIR = "results/figures"

# %% Load data

examples = load_all_examples(DATA_DIR)
print(f"Loaded {len(examples)} examples")

# %% Train linear probes per layer

# %% Plot separability curve

# %% t-SNE visualization

LAYER = 33  # transformer block index; must be in layer_indices (e.g. [0, 8, 16, 24, 33])

vecs = []
labels = []

for ex in examples:
    layer_indices = ex["layer_indices"].tolist()
    if LAYER not in layer_indices:
        continue
    pos = layer_indices.index(LAYER)
    # activations: (S, L, H)
    vecs.append(ex["activations"][:, pos, :])        # (S, H)
    labels.append(ex["manual_step_numbers"])          # (S,)

X = np.concatenate(vecs, axis=0).astype(np.float32)  # (N, H)
y = np.concatenate(labels, axis=0)                   # (N,)
print(f"Total vectors: {X.shape[0]}  |  layer {LAYER}  |  hidden dim {X.shape[1]}")

# PCA → 50 dims, then t-SNE → 2 dims
# X_pca = PCA(n_components=2, random_state=0).fit_transform(X)
# X_2d = TSNE(n_components=2, perplexity=100, random_state=0).fit_transform(X_pca)
X_2d = PCA(n_components=2, random_state=0).fit_transform(X)


# Plot
step_labels = {s: (f"step {s}" if s != -1 else "final answer") for s in sorted(set(y.tolist()))}
cmap = plt.get_cmap("tab10")
colors = {s: cmap(i) for i, s in enumerate(sorted(step_labels))}

fig, ax = plt.subplots(figsize=(9, 7))
for s, label in step_labels.items():
    mask = y == s
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=4, alpha=0.4, color=colors[s], label=label)

ax.legend(markerscale=3, loc="best")
ax.set_title(f"t-SNE of hidden states at layer {LAYER}")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_xticks([])
ax.set_yticks([])

Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
out_path = f"{FIGURES_DIR}/tsne_layer{LAYER}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()

# %%

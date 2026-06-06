# %%
"""
PCA of last-layer hidden states at step positions, colored by step number.
Run after syncing remote_results/hidden_states/ from RunPod.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-states-dir", default="remote_results/hidden_states/gemma-4-31b-it/omni_math_l7-8_test_500")
    parser.add_argument("--layer", type=int, default=59)
    parser.add_argument("--output", default="results/figures/pca_steps.html")
    args = parser.parse_args()

    npz_paths = sorted(Path(args.hidden_states_dir).glob("*.npz"))
    print(f"Loading {len(npz_paths)} files...")

    all_acts = []
    all_step_nums = []

    for path in npz_paths:
        data = np.load(path)
        layer_indices = data["layer_indices"]
        if args.layer not in layer_indices:
            continue
        li = np.where(layer_indices == args.layer)[0][0]
        all_acts.append(data["activations"][:, li, :])  # [S, H]
        all_step_nums.append(data["manual_step_numbers"])  # [S]

    X = np.concatenate(all_acts, axis=0)       # [N, H]
    step_nums = np.concatenate(all_step_nums)  # [N]

    print(f"Running PCA on {X.shape[0]} points (dim={X.shape[1]})...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    print(f"Explained variance: {pca.explained_variance_ratio_}")

    # Keep only steps 1-5
    keep = np.isin(step_nums, [1, 2, 3, 4, 5])
    coords = coords[keep]
    step_nums = step_nums[keep]

    # Build one trace per unique step number
    unique_steps = sorted(set(step_nums.tolist()))
    label_map = {}

    fig = go.Figure()
    for step in unique_steps:
        mask = step_nums == step
        label = label_map.get(step, f"step {step}")
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            marker=dict(size=4, opacity=0.6),
            name=label,
        ))

    fig.update_layout(
        title=f"PCA of layer {args.layer} activations at step positions",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        legend_title="Step",
        width=900,
        height=700,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"Saved to {out_path}")

    import webbrowser
    webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()

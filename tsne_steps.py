# %%
"""
t-SNE of last-layer hidden states at step positions, colored by step number.
PCA to 32 dims first, then t-SNE to 2D.
Run after syncing hidden states from RunPod.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go



def _extract_steps_arg(default="-1,0,1,2,3,4,5"):
    """Extract --steps value before argparse sees it (handles negative numbers like -1)."""
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--steps="):
            val = arg[len("--steps="):]
            sys.argv.pop(i)
            return val
        if arg == "--steps" and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            sys.argv.pop(i)
            sys.argv.pop(i)
            return val
    return default


def main():
    steps_to_include = [int(x) for x in _extract_steps_arg().split(",")]

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-states-dir", default="remote_results/gemma-4-31b-it/omni_math_l7-8_test_500")
    parser.add_argument("--layer", type=int, default=59)
    parser.add_argument("--output", default="results/figures/tsne_steps.html")
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
        acts = data["activations"][:, li, :]      # [S, H]
        steps = data["manual_step_numbers"]        # [S]

        mask = np.isin(steps, steps_to_include)
        if mask.any():
            all_acts.append(acts[mask])
            all_step_nums.append(steps[mask])

    X = np.concatenate(all_acts, axis=0).astype(np.float32)   # [N, H]
    step_nums = np.concatenate(all_step_nums)                  # [N]
    print(f"{X.shape[0]} points, dim={X.shape[1]}")

    print("PCA to 32 dims...")
    pca = PCA(n_components=64)
    X32 = pca.fit_transform(X)
    print(f"PCA explained variance (32 components): {pca.explained_variance_ratio_.sum():.1%}")

    print("t-SNE to 2D...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1)
    coords = tsne.fit_transform(X32)

    label_map = {-1: "answer (-1)", 0: "step 0"}
    fig = go.Figure()
    for step in sorted(set(step_nums.tolist())):
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
        title=f"t-SNE of layer {args.layer} activations (steps {steps_to_include})",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
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

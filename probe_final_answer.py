# %% Imports and config
from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_DIR = "remote_results/gemma-4-31b-it/omni_math_l7-8_test_500"
SCORED_PATH = "postprocessing/omni_math_l7-8_test_500/responses_scored_v2.jsonl"
FIGURES_DIR = "results/figures"
PCA_DIMS = 32

# %% Load labels
labels = {}
with open(SCORED_PATH) as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("extracted_answer", ""):
            labels[rec["id"]] = bool(rec["correct"])

print(f"Answered samples with labels: {len(labels)}")
print(f"  Correct: {sum(labels.values())}  Wrong: {sum(not v for v in labels.values())}")

# %% Load hidden states — select manual_step_number == -1 and the step just before it
npz_paths = sorted(Path(DATA_DIR).glob("*.npz"))

X_list = []      # final-answer activation
X_prev_list = [] # activation of the last numbered step before the final marker
y_list = []
layer_indices = None

for path in npz_paths:
    rid = path.stem
    if rid not in labels:
        continue

    data = np.load(path)
    msn = data["manual_step_numbers"]  # (S,)
    activations = data["activations"]  # (S, L, H)

    final_idx = np.where(msn == -1)[0]
    if len(final_idx) == 0:
        continue

    # last numbered step: highest manual_step_number among steps >= 0
    numbered = np.where(msn >= 0)[0]
    if len(numbered) == 0:
        continue

    pos_final = final_idx[0]
    pos_prev = numbered[np.argmax(msn[numbered])]

    X_list.append(activations[pos_final])   # (L, H)
    X_prev_list.append(activations[pos_prev])  # (L, H)
    y_list.append(labels[rid])

    if layer_indices is None:
        layer_indices = data["layer_indices"].tolist()

if not X_list:
    raise RuntimeError("No examples loaded — check DATA_DIR and SCORED_PATH")

X = np.stack(X_list, axis=0).astype(np.float32)       # (N, L, H)
X_prev = np.stack(X_prev_list, axis=0).astype(np.float32)  # (N, L, H)
y = np.array(y_list, dtype=np.int32)                  # (N,)

print(f"\nLoaded {len(X)} examples  |  layers {layer_indices}  |  hidden dim {X.shape[2]}")
print(f"Correct: {y.sum()}  Wrong: {(y == 0).sum()}")

# %% 80/20 stratified split
idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
)
X_train, X_test = X[idx_train], X[idx_test]
X_prev_train, X_prev_test = X_prev[idx_train], X_prev[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

print(f"Train: {len(y_train)}  (correct {y_train.sum()})")
print(f"Test:  {len(y_test)}   (correct {y_test.sum()})")

# %% Train probes — method A: final-answer activation only
results_A = []

for l_pos, layer_idx in enumerate(layer_indices):
    X_tr = X_train[:, l_pos, :]
    X_te = X_test[:, l_pos, :]

    pca = PCA(n_components=PCA_DIMS, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_tr_pca, y_train)

    y_pred = clf.predict(X_te_pca)
    y_prob = clf.predict_proba(X_te_pca)[:, 1]

    results_A.append({
        "layer": layer_idx,
        "accuracy": (y_pred == y_test).mean(),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    })

# %% Train probes — method C: (final - prev) diff only
results_C = []

for l_pos, layer_idx in enumerate(layer_indices):
    feat_tr = X_train[:, l_pos, :] - X_prev_train[:, l_pos, :]
    feat_te = X_test[:, l_pos, :]  - X_prev_test[:, l_pos, :]

    pca = PCA(n_components=PCA_DIMS, random_state=42)
    feat_tr_pca = pca.fit_transform(feat_tr)
    feat_te_pca = pca.transform(feat_te)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(feat_tr_pca, y_train)

    y_pred = clf.predict(feat_te_pca)
    y_prob = clf.predict_proba(feat_te_pca)[:, 1]

    results_C.append({
        "layer": layer_idx,
        "accuracy": (y_pred == y_test).mean(),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    })

# %% Train probes — method B: [final | final - prev] concatenated
results_B = []

for l_pos, layer_idx in enumerate(layer_indices):
    act_final_tr = X_train[:, l_pos, :]
    act_prev_tr  = X_prev_train[:, l_pos, :]
    act_final_te = X_test[:, l_pos, :]
    act_prev_te  = X_prev_test[:, l_pos, :]

    feat_tr = np.concatenate([act_final_tr, act_final_tr - act_prev_tr], axis=1)  # (N_tr, 2H)
    feat_te = np.concatenate([act_final_te, act_final_te - act_prev_te], axis=1)  # (N_te, 2H)

    pca = PCA(n_components=PCA_DIMS, random_state=42)
    feat_tr_pca = pca.fit_transform(feat_tr)
    feat_te_pca = pca.transform(feat_te)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(feat_tr_pca, y_train)

    y_pred = clf.predict(feat_te_pca)
    y_prob = clf.predict_proba(feat_te_pca)[:, 1]

    results_B.append({
        "layer": layer_idx,
        "accuracy": (y_pred == y_test).mean(),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    })

# %% Train probes — method D: [final | prev] concatenated
results_D = []

for l_pos, layer_idx in enumerate(layer_indices):
    act_final_tr = X_train[:, l_pos, :]
    act_prev_tr  = X_prev_train[:, l_pos, :]
    act_final_te = X_test[:, l_pos, :]
    act_prev_te  = X_prev_test[:, l_pos, :]

    feat_tr = np.concatenate([act_final_tr, act_prev_tr], axis=1)  # (N_tr, 2H)
    feat_te = np.concatenate([act_final_te, act_prev_te], axis=1)  # (N_te, 2H)

    pca = PCA(n_components=PCA_DIMS, random_state=42)
    feat_tr_pca = pca.fit_transform(feat_tr)
    feat_te_pca = pca.transform(feat_te)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(feat_tr_pca, y_train)

    y_pred = clf.predict(feat_te_pca)
    y_prob = clf.predict_proba(feat_te_pca)[:, 1]

    results_D.append({
        "layer": layer_idx,
        "accuracy": (y_pred == y_test).mean(),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    })

# %% Train probes — method B2: [PCA(final) | PCA(final - prev)] independently, PCA_DIMS//2 each
results_B2 = []

for l_pos, layer_idx in enumerate(layer_indices):
    act_final_tr = X_train[:, l_pos, :]
    act_prev_tr  = X_prev_train[:, l_pos, :]
    act_final_te = X_test[:, l_pos, :]
    act_prev_te  = X_prev_test[:, l_pos, :]

    half = PCA_DIMS // 2
    pca_f = PCA(n_components=half, random_state=42)
    pca_d = PCA(n_components=half, random_state=42)

    f_tr = pca_f.fit_transform(act_final_tr)
    f_te = pca_f.transform(act_final_te)
    d_tr = pca_d.fit_transform(act_final_tr - act_prev_tr)
    d_te = pca_d.transform(act_final_te - act_prev_te)

    feat_tr = np.concatenate([f_tr, d_tr], axis=1)
    feat_te = np.concatenate([f_te, d_te], axis=1)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(feat_tr, y_train)

    y_pred = clf.predict(feat_te)
    y_prob = clf.predict_proba(feat_te)[:, 1]

    results_B2.append({
        "layer": layer_idx,
        "accuracy": (y_pred == y_test).mean(),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    })

# %% Train probes — method D2: [PCA(final) | PCA(prev)] independently, PCA_DIMS//2 each
results_D2 = []

for l_pos, layer_idx in enumerate(layer_indices):
    act_final_tr = X_train[:, l_pos, :]
    act_prev_tr  = X_prev_train[:, l_pos, :]
    act_final_te = X_test[:, l_pos, :]
    act_prev_te  = X_prev_test[:, l_pos, :]

    half = PCA_DIMS // 2
    pca_f = PCA(n_components=half, random_state=42)
    pca_p = PCA(n_components=half, random_state=42)

    f_tr = pca_f.fit_transform(act_final_tr)
    f_te = pca_f.transform(act_final_te)
    p_tr = pca_p.fit_transform(act_prev_tr)
    p_te = pca_p.transform(act_prev_te)

    feat_tr = np.concatenate([f_tr, p_tr], axis=1)
    feat_te = np.concatenate([f_te, p_te], axis=1)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(feat_tr, y_train)

    y_pred = clf.predict(feat_te)
    y_prob = clf.predict_proba(feat_te)[:, 1]

    results_D2.append({
        "layer": layer_idx,
        "accuracy": (y_pred == y_test).mean(),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    })

# %% Summary table — ROC-AUC comparison
print(f"\n{'Layer':>6} | {'A: final':>10} | {'C: diff':>10} | {'B: f+diff':>10} | {'B2: f+diff*':>12} | {'D: f+prev':>10} | {'D2: f+prev*':>12}")
print("-" * 82)
for a, c, b, b2, d, d2 in zip(results_A, results_C, results_B, results_B2, results_D, results_D2):
    print(f"{a['layer']:>6} | {a['roc_auc']:>10.3f} | {c['roc_auc']:>10.3f} | {b['roc_auc']:>10.3f} | {b2['roc_auc']:>12.3f} | {d['roc_auc']:>10.3f} | {d2['roc_auc']:>12.3f}")
print("  * = independent PCA per part (PCA_DIMS//2 each)")

# %% Plot ROC-AUC comparison across layers
layers_x = [r["layer"] for r in results_A]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(layers_x, [r["roc_auc"] for r in results_A], marker="o", label=f"Final only (PCA {PCA_DIMS})")
ax.plot(layers_x, [r["roc_auc"] for r in results_C], marker="^", label=f"Diff only (PCA {PCA_DIMS})")
ax.plot(layers_x, [r["roc_auc"] for r in results_B], marker="s", label=f"Final + diff (PCA {PCA_DIMS})")
ax.plot(layers_x, [r["roc_auc"] for r in results_B2], marker="s", linestyle="--", label=f"Final + diff, indep. PCA ({PCA_DIMS//2}+{PCA_DIMS//2})")
ax.plot(layers_x, [r["roc_auc"] for r in results_D], marker="D", label=f"Final + prev (PCA {PCA_DIMS})")
ax.plot(layers_x, [r["roc_auc"] for r in results_D2], marker="D", linestyle="--", label=f"Final + prev, indep. PCA ({PCA_DIMS//2}+{PCA_DIMS//2})")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
ax.set_xlabel("Layer")
ax.set_ylabel("ROC-AUC")
ax.set_title("Linear probe — ROC-AUC per layer")
ax.set_xticks(layers_x)
ax.legend()
ax.grid(True, alpha=0.3)

Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
out_path = f"{FIGURES_DIR}/probe_final_answer.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.show()

# %%

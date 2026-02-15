# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 21:42:11 2026

@author: lukai
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, roc_auc_score


# ====== Top imports for export functionality ======
import json
import pickle
from pathlib import Path

seed = 1613
# ========== Read and preprocess data ========== #
df = pd.read_csv("data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("__", "_")

# ====== [Classification modification] Convert to binary classification: 0=Normal/Benign, 1=EIN/EC ======
if df["Label"].max() > 1:
    df["Label"] = df["Label"].apply(lambda x: 0 if x in [0, 1] else 1)

node_ids = df["Plot_ID"].tolist()
df_label_max = df.groupby("Plot_ID")["Label"].max()
node_labels = df_label_max.to_dict()

# Feature columns selection
feature_cols = ["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"]
df_features = df.set_index("Plot_ID")[feature_cols].fillna(0)

# Standardize features and compute cosine similarity
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)
similarity_matrix = cosine_similarity(X_scaled)
similarity_matrix = (similarity_matrix + 1) / 2  # Scale to [0, 1]

# Create export directory
OUT_DIR = Path("export_graph_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Add suffix to distinguish multiple rows from the same patient
_cnt = defaultdict(int)
sample_ids = []
for pid in df_features.index:
    _cnt[pid] += 1
    sample_ids.append(f"{pid}#{_cnt[pid]}")

# Save sample-level similarity matrix
sim_df = pd.DataFrame(similarity_matrix, index=sample_ids, columns=sample_ids)
sim_df.to_csv(OUT_DIR / "similarity_matrix_raw.csv", index=True)
np.save(OUT_DIR / "similarity_matrix_raw.npy", similarity_matrix)

# ========== Build patient-level similarity matrix (cross-sample average) ========== #
patient_to_indices = defaultdict(list)
for idx, pid in enumerate(df_features.index):
    patient_to_indices[pid].append(idx)

unique_patients = list(patient_to_indices.keys())
P = len(unique_patients)
patient_mat = np.full((P, P), np.nan, dtype=float)

for i, pid1 in enumerate(unique_patients):
    patient_mat[i, i] = 1.0  # Self-similarity
    idxs1 = patient_to_indices[pid1]
    for j in range(i + 1, P):
        pid2 = unique_patients[j]
        idxs2 = patient_to_indices[pid2]
        sims = [similarity_matrix[m, n] for m in idxs1 for n in idxs2]
        if len(sims) > 0:
            avg_sim = float(sum(sims) / len(sims))
            patient_mat[i, j] = avg_sim
            patient_mat[j, i] = avg_sim

# Save patient-level average similarity matrix
patient_sim_df = pd.DataFrame(patient_mat, index=unique_patients, columns=unique_patients)
patient_sim_df.to_csv(OUT_DIR / "patient_similarity_matrix_avg.csv", index=True)
np.save(OUT_DIR / "patient_similarity_matrix_avg.npy", patient_mat)

# (Optional) Save thresholded adjacency matrix for reference
threshold = 0.5
A = (patient_sim_df.values >= threshold).astype(int)
np.fill_diagonal(A, 0)
adj_df = pd.DataFrame(A, index=unique_patients, columns=unique_patients)
adj_df.to_csv(OUT_DIR / "patient_adjacency_thresholded.csv", index=True)

# ========== Build patient-level similarity matrix (pairwise average) ========== #
patient_to_indices = defaultdict(list)
for idx, pid in enumerate(df_features.index):
    patient_to_indices[pid].append(idx)

unique_patients = list(patient_to_indices.keys())
patient_sim_matrix = dict()
for i, pid1 in enumerate(unique_patients):
    for j in range(i + 1, len(unique_patients)):
        pid2 = unique_patients[j]
        idxs1 = patient_to_indices[pid1]
        idxs2 = patient_to_indices[pid2]
        sims = [similarity_matrix[m, n] for m in idxs1 for n in idxs2]
        if sims:
            avg_sim = sum(sims) / len(sims)
            patient_sim_matrix[(pid1, pid2)] = avg_sim

# ========== Build graph structure with layout (multiple edges averaged) ========== #
threshold = 0.5

# 1) Aggregate edges: multiple candidate weights for the same endpoint pair
edge_bag = defaultdict(list)
for (a, b), sim in patient_sim_matrix.items():
    if sim >= threshold and a != b:
        u, v = sorted((a, b))  # Undirected graph: unified key
        edge_bag[(u, v)].append(sim)

# Export edge bag statistics
with open(OUT_DIR / "edge_bag.pkl", "wb") as f:
    pickle.dump(edge_bag, f)

rows = []
for (u, v), ws in edge_bag.items():
    if not ws:
        continue
    rows.append({
        "u": u,
        "v": v,
        "n": len(ws),
        "mean": float(sum(ws) / len(ws)),
        "min": float(min(ws)),
        "max": float(max(ws)),
        "values_json": json.dumps([float(x) for x in ws]),
    })
pd.DataFrame(rows).to_csv(OUT_DIR / "edge_bag_summary.csv", index=False)

# 2) Build graph: average weights for each endpoint pair
G = nx.Graph()
for pid in unique_patients:
    G.add_node(pid, Label=node_labels[pid])

for (u, v), ws in edge_bag.items():
    avg_sim = sum(ws) / len(ws)
    G.add_edge(u, v, weight=avg_sim)

# Export final graph edges
edges_df = nx.to_pandas_edgelist(G)  # columns: source, target, weight
edges_df.to_csv(OUT_DIR / "graph_edges_after_threshold.csv", index=False)

# 3) Layout and extract coordinates
pos_2d = nx.spring_layout(G, dim=2, seed=seed)
node_positions = np.array([pos_2d[node] for node in G.nodes()])
node_ids_list = list(G.nodes())
node_labels_list = [G.nodes[node]["Label"] for node in node_ids_list]

# Save final 2D coordinates for each node
coords_rows = []
for node in G.nodes():
    x, y = pos_2d[node]
    coords_rows.append({
        "Plot_ID": node,
        "x": float(x),
        "y": float(y),
        "Label": G.nodes[node]["Label"]
    })

df = pd.DataFrame(coords_rows)

C = 1                               # Logistic regression regularization strength (larger = weaker regularization)
THRESHOLD_MODE = "acc"              # "fixed" | "acc" | "youden"
FIXED_THRESHOLD = 0.5               # Used only when THRESHOLD_MODE = "fixed"
USE_EQUAL_AXIS = True               # Whether to enforce equal axis scaling

X = df[["x", "y"]].to_numpy(float)
x = X[:, 0]; y = X[:, 1]
labels_raw = df["Label"].to_numpy(int)

# Binary conversion as required:
# 0/1 -> 0 (Benign), 2/3/4/5 -> 1 (Malignant)
# ========== Generate binary ground-truth labels y_true (compatible with Label=0/1 or 0–5) ==========
uniq = np.unique(labels_raw)
print("[DEBUG] Unique labels in CSV:", uniq)

if np.array_equal(uniq, [0]) or np.array_equal(uniq, [1]) or np.array_equal(uniq, [0, 1]):
    # CSV already contains binary labels 0/1
    y_true = labels_raw.astype(int)
else:
    # CSV contains multi-class labels 0–5: 0/1->0, 2–5->1
    y_true = (labels_raw >= 2).astype(int)

# Check whether both classes are present
vals, cnts = np.unique(y_true, return_counts=True)
print("[DEBUG] y_true class counts:", dict(zip(vals, cnts)))

if vals.size < 2:
    raise ValueError(
        f"y_true contains only one class {vals[0]}, cannot train LogisticRegression. "
        f"Please check whether patient_layout_coords.csv contains only one label, "
        f"or whether the binarization rule is correct."
    )


# ================ Train logistic regression ================
clf = LogisticRegression(C=C, solver="lbfgs", max_iter=10000)
clf.fit(X, y_true)

# Predicted probability (positive class = 1)
proba = clf.predict_proba(X)[:, 1]

# ================ Threshold selection ================
def pick_threshold(y_true, proba, mode="youden", fixed_t=0.5):
    if mode == "fixed":
        return float(fixed_t)
    # Candidate thresholds from ROC curve
    fpr, tpr, thr = roc_curve(y_true, proba)
    if mode == "youden":
        # Maximize Youden's J = TPR - FPR
        i = np.argmax(tpr - fpr)
        return float(thr[i])
    elif mode == "acc":
        # Use midpoints between unique probabilities to avoid equality issues
        p = np.sort(np.unique(proba))
        if p.size == 1:
            cand = np.array([max(1e-6, min(1-1e-6, float(p[0])))] )
        else:
            mids = (p[:-1] + p[1:]) / 2.0
            left  = max(1e-6, float(p[0] / 2.0))
            right = min(1-1e-6, float((1.0 + p[-1]) / 2.0))
            cand = np.r_[left, mids, right]

        acc_best, t_best = -1, 0.5
        for t in cand:
            pred = (proba >= t).astype(int)
            acc = accuracy_score(y_true, pred)
            if acc > acc_best:
                acc_best, t_best = acc, float(t)
        return t_best
    else:
        raise ValueError("THRESHOLD_MODE must be 'fixed' | 'acc' | 'youden'")

t_star = pick_threshold(y_true, proba, THRESHOLD_MODE, FIXED_THRESHOLD)

# ================ Compute metrics and decision boundary parameters ================
def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn)
    return sens, spec, acc, (tn, fp, fn, tp)

y_pred = (proba >= t_star).astype(int)
sens, spec, acc, cm = metrics(y_true, y_pred)

# Decision boundary:
# w1*x + w2*y + b0 = logit(t)  ⇒  y = -(w1/w2)*x + (logit(t) - b0)/w2
w1, w2 = clf.coef_[0]
b0 = clf.intercept_[0]
eps = 1e-12
def logit(p): 
    p = np.clip(p, 1e-9, 1-1e-9)
    return np.log(p/(1-p))

if abs(w2) > eps:
    a = -w1 / w2
    b = (logit(t_star) - b0) / w2
    vertical = False
else:
    # Degenerate case: vertical line x = (logit(t) - b0) / w1
    x0 = (logit(t_star) - b0) / w1
    a = None; b = None
    vertical = True

print("=== Logistic Regression ===")
print(f"C = {C}")
print(f"Threshold mode = {THRESHOLD_MODE}, t* = {t_star:.6f}")
print(f"Metrics:  ACC={acc:.3f}, Sens={sens:.3f}, Spec={spec:.3f}, CM(tn,fp,fn,tp)={cm}")
if not vertical:
    print(f"Decision boundary:  y = a*x + b  with  a={a:.12f}, b={b:.12f}")
else:
    print(f"Decision boundary:  vertical line  x = {x0:.12f}")

# ================ Plot 1: Logistic boundary ================
plt.figure(figsize=(8,6))

# Marker and color styles for binary classification
marker_map = {0:('o',True), 1:('x',True)}
color_map  = {0:'blue',   1:'red'}

# Display names for legend
label_display_names = {
    0: "Normal/Benign",
    1: "EC/EIN"
}

# Track legend handles
legend_handles = {}

# Plot each class
for lbl in [0, 1]:
    if lbl in np.unique(labels_raw):
        m = labels_raw == lbl
        mk, filled = marker_map[lbl]
        color = color_map[lbl]
        
        if mk == 'x':
            sc = plt.scatter(x[m], y[m], marker='x', s=80, color=color, linewidths=2)
        else:
            edge = 'black' if mk == 'o' else color
            face = color if filled else 'none'
            sc = plt.scatter(
                x[m], y[m], marker=mk, s=80,
                edgecolors=edge, facecolors=face, linewidths=1.8
            )
        
        legend_handles[label_display_names[lbl]] = sc

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 400)

if not vertical:
    yy = a * xx + b
    (line_plot,) = plt.plot(
        xx, yy, '--', lw=2.0, color='tab:green',
        label=("Logistic boundary")
    )
else:
    (line_plot,) = plt.plot(
        [x0, x0], [ylim[0], ylim[1]], '--', lw=2.0,
        color='tab:green',
        label="Logistic boundary"
    )

# Restore axis limits
ax.set_xlim(xlim)
ax.set_ylim(ylim)
if USE_EQUAL_AXIS:
    plt.axis('equal')

plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

handles = list(legend_handles.values()) + [line_plot]
labels = list(legend_handles.keys()) + [line_plot.get_label()]

# ✅ Legend moved to upper-left
plt.legend(
    handles, labels,
    loc='upper left',
    frameon=True,
    framealpha=0.0,
    prop={'size': 14}   # ← change this number
)

plt.tight_layout()
plt.savefig(f"logistic_boundary_seed_{seed}.png", dpi=300)
plt.show()


# ================ Plot 2: ROC curve ================
fpr, tpr, _ = roc_curve(y_true, proba)
auc = roc_auc_score(y_true, proba)

plt.figure(figsize=(6, 5))
ax_roc = plt.gca()

# ROC curve (shown in legend
ax_roc.plot(
    fpr, tpr,
    lw=3,
    color='orange',
    label=f"ROC Curve (AUC = {auc:.3f})"
)

# Random reference line (not included in legend)
ax_roc.plot(
    [0, 1], [0, 1],
    linestyle='--',
    color='black',
    lw=3,
    label="_nolegend_"
)

# Axis limits
ax_roc.set_xlim(-0.04, 1.04)
ax_roc.set_ylim(-0.04, 1.04)

ax_roc.set_xlabel("Specificity", fontsize=22)
ax_roc.set_ylabel("Sensitivity", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# ax_roc.set_title("ROC Curve")

# Legend (ONLY call legend once, after all plots)
ax_roc.legend(loc="lower right", prop={'size': 16})

# Remove top and right spines
ax_roc.spines['right'].set_visible(False)
ax_roc.spines['top'].set_visible(False)

# ax_roc.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"logistic_roc_seed_{seed}.png", dpi=300)
plt.show()


# ================ Plot 3: Confusion matrix ================
# y_true: 0 = Normal/Benign, 1 = EIN/EC
class_names = ["Normal/Benign", "EC/EIN"]

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

plt.figure(figsize=(5, 4))
ax_cm = plt.gca()

# Use Blues colormap
im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')

tick_marks = np.arange(len(class_names))

# X-axis labels
ax_cm.set_xticks(tick_marks)
ax_cm.set_xticklabels(class_names, rotation=0, fontsize=14)

# Y-axis labels (rotated to align with axis)
ax_cm.set_yticks(tick_marks)
ax_cm.set_yticklabels(class_names, rotation=90, va="center", fontsize=14)

ax_cm.set_ylabel("True label", rotation=90, fontsize=18)
ax_cm.set_xlabel("Predicted label", fontsize=18)

# Annotate each cell with count only
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        ax_cm.text(
            j, i, f"{count}",
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=20
        )

plt.tight_layout()
plt.savefig(f"logistic_confusion_matrix_seed_{seed}.png", dpi=300)
plt.show()


# coords_df = pd.DataFrame(coords_rows)
# coords_df.to_csv("patient_layout_coords.csv", index=False, encoding="utf-8-sig")
# print(f"Saved coordinates for {len(coords_df)} nodes to patient_layout_coords.csv")


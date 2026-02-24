# -*- coding: utf-8 -*-
"""
LOOCV Logistic Regression on Graph Layout Coordinates
Outputs:
- Confusion matrix (out-of-sample)
- Accuracy / Sensitivity / Specificity
- ROC curve and AUC (LOOCV-based)
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import LeaveOneOut

seed = 1613

# =====================================================
# ========== Read and preprocess data ================
# =====================================================
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("__", "_")

# Convert to binary if needed
if df["Label"].max() > 1:
    df["Label"] = df["Label"].apply(lambda x: 0 if x in [0, 1] else 1)

df_label_max = df.groupby("Plot_ID")["Label"].max()
node_labels = df_label_max.to_dict()

feature_cols = ["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"]
df_features = df.set_index("Plot_ID")[feature_cols].fillna(0)

# =====================================================
# ========== Similarity matrix ========================
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)
similarity_matrix = cosine_similarity(X_scaled)
similarity_matrix = (similarity_matrix + 1) / 2

# =====================================================
# ========== Patient-level similarity matrix ==========
# =====================================================
patient_to_indices = defaultdict(list)
for idx, pid in enumerate(df_features.index):
    patient_to_indices[pid].append(idx)

unique_patients = list(patient_to_indices.keys())
P = len(unique_patients)
patient_mat = np.full((P, P), np.nan)

for i, pid1 in enumerate(unique_patients):
    patient_mat[i, i] = 1.0
    idxs1 = patient_to_indices[pid1]
    for j in range(i + 1, P):
        pid2 = unique_patients[j]
        idxs2 = patient_to_indices[pid2]
        sims = [similarity_matrix[m, n] for m in idxs1 for n in idxs2]
        if sims:
            avg_sim = float(sum(sims) / len(sims))
            patient_mat[i, j] = avg_sim
            patient_mat[j, i] = avg_sim

# =====================================================
# ========== Graph layout =============================
# =====================================================
G = nx.Graph()
for pid in unique_patients:
    G.add_node(pid, Label=node_labels[pid])

threshold = 0.5
for i in range(P):
    for j in range(i + 1, P):
        if patient_mat[i, j] >= threshold:
            G.add_edge(unique_patients[i], unique_patients[j], weight=patient_mat[i, j])

pos_2d = nx.spring_layout(G, dim=2, seed=seed)

coords_rows = []
for node in G.nodes():
    x, y = pos_2d[node]
    coords_rows.append({
        "Plot_ID": node,
        "x": float(x),
        "y": float(y),
        "Label": G.nodes[node]["Label"]
    })

layout_df = pd.DataFrame(coords_rows)

# =====================================================
# ========== Prepare classification ===================
# =====================================================
X = layout_df[["x", "y"]].to_numpy()
labels_raw = layout_df["Label"].to_numpy(int)

uniq = np.unique(labels_raw)
if np.array_equal(uniq, [0, 1]):
    y_true = labels_raw
else:
    y_true = (labels_raw >= 2).astype(int)

# =====================================================
# ========== Leave-One-Out Cross Validation ===========
# =====================================================
loo = LeaveOneOut()

y_pred_all = []
y_prob_all = []
y_true_all = []

print(f"Running LOOCV on {len(y_true)} patients...")

for train_idx, test_idx in loo.split(X):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_true[train_idx], y_true[test_idx]

    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train)
    X_test_scaled = scaler_fold.transform(X_test)

    clf = LogisticRegression(C=1, solver="lbfgs", max_iter=10000)
    clf.fit(X_train_scaled, y_train)

    proba_test = clf.predict_proba(X_test_scaled)[0, 1]

    y_prob_all.append(proba_test)
    y_pred_all.append(1 if proba_test >= 0.5 else 0)
    y_true_all.append(y_test[0])

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)
y_prob_all = np.array(y_prob_all)

# =====================================================
# ========== Metrics ==================================
# =====================================================
cm = confusion_matrix(y_true_all, y_pred_all)
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(y_true_all, y_pred_all)
sens = tp / (tp + fn) if (tp + fn) else 0
spec = tn / (tn + fp) if (tn + fp) else 0
auc_value = roc_auc_score(y_true_all, y_prob_all)

print("\n=== LOOCV Results ===")
print(f"Accuracy:    {acc:.3f}")
print(f"Sensitivity: {sens:.3f}")
print(f"Specificity: {spec:.3f}")
print(f"AUC:         {auc_value:.3f}")
print(f"Confusion Matrix (tn,fp,fn,tp): {(tn,fp,fn,tp)}")

# =====================================================
# ========== Plot Confusion Matrix ====================
# =====================================================
class_names = ["Normal/Benign", "EC/EIN"]

plt.figure(figsize=(5, 4))
ax = plt.gca()
im = ax.imshow(cm, cmap="Blues")

tick_marks = np.arange(len(class_names))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(class_names, fontsize=12)
ax.set_yticklabels(class_names, rotation=90, va="center" , fontsize=12)

ax.set_xlabel("Predicted label", fontsize=14)
ax.set_ylabel("True label", fontsize=14)

thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=22)

plt.tight_layout()
plt.savefig("LOOCV_confusion_matrix.png", dpi=300)
plt.show()

# =====================================================
# ========== Plot ROC Curve ===========================
# =====================================================
fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)

plt.figure(figsize=(6, 5))
ax_roc = plt.gca()

ax_roc.plot(
    fpr, tpr,
    lw=3,
    color='orange',
    label=f"ROC Curve (AUC = {auc_value:.3f})"
)

ax_roc.plot(
    [0, 1], [0, 1],
    linestyle='--',
    color='black',
    lw=2,
    label="_nolegend_"
)

ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(-0.02, 1.02)
ax_roc.set_xlabel("False Positive Rate", fontsize=16)
ax_roc.set_ylabel("True Positive Rate", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax_roc.legend(loc="lower right", fontsize=16)

ax_roc.spines['right'].set_visible(False)
ax_roc.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig("LOOCV_ROC_curve.png", dpi=300)
plt.show()
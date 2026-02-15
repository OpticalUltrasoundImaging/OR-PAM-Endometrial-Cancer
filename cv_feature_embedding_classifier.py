# -*- coding: utf-8 -*-
"""
5-fold CV version (CORRECTED)

Fixes:
• consistent prediction aggregation
• correct global metrics
• stores out-of-fold predictions
• reports mean ± std metrics
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

seed = 1613

# ================= READ DATA =================
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("__", "_")

# Binary labels
if df["Label"].max() > 1:
    df["Label"] = df["Label"].apply(lambda x: 0 if x in [0,1] else 1)

node_ids = df["Plot_ID"].tolist()
df_label_max = df.groupby("Plot_ID")["Label"].max()
node_labels = df_label_max.to_dict()

# ================= FEATURE MATRIX =================
feature_cols = ["Feature_1","Feature_2","Feature_3","Feature_4","Feature_5"]
df_features = df.set_index("Plot_ID")[feature_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)
similarity_matrix = cosine_similarity(X_scaled)
similarity_matrix = (similarity_matrix + 1)/2

# ================= PATIENT SIM MATRIX =================
patient_to_indices = defaultdict(list)
for idx,pid in enumerate(df_features.index):
    patient_to_indices[pid].append(idx)

unique_patients=list(patient_to_indices.keys())
P=len(unique_patients)

patient_mat=np.full((P,P),np.nan)

for i,p1 in enumerate(unique_patients):
    patient_mat[i,i]=1
    for j in range(i+1,P):
        p2=unique_patients[j]
        sims=[similarity_matrix[m,n] for m in patient_to_indices[p1] for n in patient_to_indices[p2]]
        if sims:
            val=sum(sims)/len(sims)
            patient_mat[i,j]=val
            patient_mat[j,i]=val

# ================= GRAPH BUILD =================
threshold=0.5
G=nx.Graph()

for pid in unique_patients:
    G.add_node(pid,Label=node_labels[pid])

for i,p1 in enumerate(unique_patients):
    for j,p2 in enumerate(unique_patients):
        if j>i and patient_mat[i,j]>=threshold:
            G.add_edge(p1,p2,weight=patient_mat[i,j])

# Layout
pos=nx.spring_layout(G,seed=seed)

coords=[]
for n in G.nodes():
    x,y=pos[n]
    coords.append({"Plot_ID":n,"x":x,"y":y,"Label":G.nodes[n]["Label"]})

df=pd.DataFrame(coords)

X=df[["x","y"]].to_numpy()
labels_raw=df["Label"].to_numpy()

# Binary labels safe
if set(np.unique(labels_raw)).issubset({0,1}):
    y_true=labels_raw
else:
    y_true=(labels_raw>=2).astype(int)

# ================= 5-FOLD CV =================
kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)

all_probs=[]
all_true=[]
all_preds=[]
cms=[]

accs=[]
sens_list=[]
spec_list=[]

for fold,(train_idx,test_idx) in enumerate(kf.split(X,y_true),1):

    Xtr,Xte=X[train_idx],X[test_idx]
    ytr,yte=y_true[train_idx],y_true[test_idx]

    clf=LogisticRegression(max_iter=10000)
    clf.fit(Xtr,ytr)

    prob=clf.predict_proba(Xte)[:,1]

    # fold-specific optimal threshold
    fpr, tpr, thr = roc_curve(yte, prob)
    best = np.argmax(tpr - fpr)
    t_star = thr[best]

    pred = (prob >= t_star).astype(int)

    cm=confusion_matrix(yte,pred)
    cms.append(cm)

    all_probs.extend(prob)
    all_true.extend(yte)
    all_preds.extend(pred)

    tn,fp,fn,tp=cm.ravel()

    acc=(tp+tn)/(tp+tn+fp+fn)
    sens=tp/(tp+fn) if (tp+fn) else 0
    spec=tn/(tn+fp) if (tn+fp) else 0

    accs.append(acc)
    sens_list.append(sens)
    spec_list.append(spec)

    print(f"\nFOLD {fold}")
    print(cm)
    print(f"ACC={acc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")

# ================= OVERALL METRICS =================
all_probs=np.array(all_probs)
all_true=np.array(all_true)
all_preds=np.array(all_preds)

auc=roc_auc_score(all_true,all_probs)

tn,fp,fn,tp=confusion_matrix(all_true,all_preds).ravel()
acc=(tp+tn)/(tp+tn+fp+fn)
sens=tp/(tp+fn)
spec=tn/(tn+fp)

print("\n===== CROSS-VALIDATION RESULTS =====")
print(f"AUC  = {auc:.3f}")
print(f"ACC  = {acc:.3f}  ± {np.std(accs):.3f}")
print(f"Sens = {sens:.3f} ± {np.std(sens_list):.3f}")
print(f"Spec = {spec:.3f} ± {np.std(spec_list):.3f}")

# ================= PLOT ROC =================
fpr,tpr,_=roc_curve(all_true,all_probs)

plt.figure(figsize=(6,5))
plt.plot(fpr,tpr,lw=3,label=f"AUC = {auc:.3f}",color="orange")
plt.plot([0,1],[0,1],'k--',lw=2)
plt.xlabel("Specificity", fontsize=22)
plt.ylabel("Sensitivity", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.legend()
plt.legend(loc="lower right", fontsize=16)
plt.tight_layout()
plt.show()

# ================= CONFUSION MATRICES PER FOLD =================
for i,cm in enumerate(cms,1):
    plt.figure(figsize=(4,3))
    plt.imshow(cm,cmap="Blues")
    plt.title(f"Fold {i}")
    plt.xticks([0,1],["Normal/Benign","EC/EIN"])
    plt.yticks([0,1],["Normal/Benign","EC/EIN"])

    for a in range(2):
        for b in range(2):
            plt.text(b,a,str(cm[a,b]),ha="center",va="center",fontsize=16)

    plt.tight_layout()
    plt.show()

# ================= FINAL CONFUSION MATRIX =================
cm=confusion_matrix(all_true,all_preds)

plt.figure(figsize=(4,3))
plt.imshow(cm,cmap="Blues")

plt.xticks([0,1],["Normal/Benign","EC/EIN"])
plt.yticks([0,1],["Normal/Benign","EC/EIN"])
plt.xticks(fontsize=10)
plt.yticks(rotation=90, fontsize=10)


ax = plt.gca()
ax.set_ylabel("True label", fontsize=12)
ax.set_xlabel("Predicted label", fontsize=12)

for a in range(2):
    for b in range(2):
        plt.text(b,a,str(cm[a,b]),ha="center",va="center",fontsize=16)

plt.tight_layout()
plt.show()

# ================= CONFUSION MATRICES PER FOLD =================
class_names = ["Normal/Benign", "EC/EIN"]

for i, cm in enumerate(cms, 1):

    plt.figure(figsize=(4,3))
    ax = plt.gca()

    im = ax.imshow(cm, interpolation='nearest', cmap="Blues")

    tick_marks = np.arange(len(class_names))

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=10)

    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, rotation=90, va="center", fontsize=10)

    ax.set_title(f"Fold {i}", fontsize=14)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)

    # dynamic text color
    thresh = cm.max() / 2.0

    for a in range(cm.shape[0]):
        for b in range(cm.shape[1]):
            ax.text(
                b, a, str(cm[a, b]),
                ha="center", va="center",
                color="white" if cm[a, b] > thresh else "black",
                fontsize=14
            )

    plt.tight_layout()
    plt.show()



# ================= FINAL CONFUSION MATRIX =================
cm = confusion_matrix(all_true, all_preds)

plt.figure(figsize=(4,3))
ax = plt.gca()

im = ax.imshow(cm, interpolation='nearest', cmap="Blues")

tick_marks = np.arange(len(class_names))

ax.set_xticks(tick_marks)
ax.set_xticklabels(class_names, fontsize=10)

ax.set_yticks(tick_marks)
ax.set_yticklabels(class_names, rotation=90, va="center", fontsize=10)

ax.set_ylabel("True label", fontsize=12)
ax.set_xlabel("Predicted label", fontsize=12)

# dynamic text color
thresh = cm.max() / 2.0

for a in range(cm.shape[0]):
    for b in range(cm.shape[1]):
        ax.text(
            b, a, str(cm[a, b]),
            ha="center", va="center",
            color="white" if cm[a, b] > thresh else "black",
            fontsize=16
        )

plt.tight_layout()
plt.show()

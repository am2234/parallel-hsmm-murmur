import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import utils

labels_df = utils.read_training_spreadsheet()
labels_df.index = labels_df.index.astype(str)

df = pd.read_csv("final_model/recordings.csv", index_col=0)
df.index = df.index.str.split("_", expand=True)
df["diff"] = df["holo_HSMM"] - df["healthy_HSMM"]
df["max_conf"] = df[["healthy_HSMM", "holo_HSMM"]].max(axis=1)


patient_df = df.groupby(level=0).max()


for index, rows in df.groupby(level=0):
    val = None
    if (rows["diff"] > 0).any():
        val = "Present"
    elif (rows["max_conf"] < 0.65).any():
        val = "Unknown"
    else:
        val = "Absent"

    patient_df.loc[index, "murmur_pred"] = val
patient_df = patient_df.join(labels_df)

mat = sklearn.metrics.confusion_matrix(
    patient_df.rec_murmur_label, patient_df.murmur_pred, labels=["Present", "Unknown", "Absent"]
).T
print(mat)


ORDER = ["Present", "Unknown", "Absent"]
binary_predictions = np.zeros((len(patient_df), 3), dtype=int)
binary_labels = np.zeros((len(patient_df), 3), dtype=int)
for i, (_, row) in enumerate(patient_df.iterrows()):
    binary_predictions[i, ORDER.index(row["murmur_pred"])] = 1
    binary_labels[i, ORDER.index(row["rec_murmur_label"])] = 1

F_macro, F_scores = utils.compute_f_measure(binary_labels, binary_predictions)
print("F_macro", F_macro)
print("F scores", F_scores)

accuracy = np.trace(mat) / np.sum(mat)
print("Accuracy", accuracy)

weighted_acc = utils.compute_weighted_accuracy(
    binary_labels, binary_predictions, ["Present", "Unknown", "Absent"]
)
print("Weighted accuracy", weighted_acc)

result = sklearn.metrics.classification_report(
    patient_df.rec_murmur_label,
    patient_df.murmur_pred,
    output_dict=True,
)

row_order = ["Present", "Unknown", "Absent"]
rows = [result[k] for k in row_order]
df = pd.DataFrame(rows, index=row_order)
df["precision"] = (df["precision"] * 100).round(1)
df["recall"] = (df["recall"] * 100).round(1)
df["f1-score"] = df["f1-score"].round(3)

df = df[["support", "recall", "precision", "f1-score"]]
df["support"] = df["support"].astype(int)

df = df.rename(
    index={"Present": "Murmur present", "Absent": "Murmur absent"},
    columns={
        "support": "Cases",
        "recall": "Sensitivity (%)",
        "precision": "PPV (%)",
        "f1-score": "F1 Score",
    },
)


print(df)


font = {"size": 9, "family": "Arial"}
plt.rc("font", **font)

fig, axes = plt.subplots(figsize=(4.5, 4), dpi=500)
axes.invert_xaxis()
axes.set_aspect("equal")
axes.set_facecolor("white")
fig.set_facecolor("white")
axes.set_ylabel("Sensitivity")
axes.set_xlabel("Specificity")
axes.set_xlim(1, 0)
axes.set_ylim(0, 1)

clean_df = patient_df[patient_df["rec_murmur_label"] != "Unknown"]

fpr, tpr, _ = sklearn.metrics.roc_curve(
    clean_df["rec_murmur_label"] == "Present",
    clean_df["diff"],
)

auc = sklearn.metrics.auc(fpr, tpr)

fpr_index = np.linspace(0, 1, 201)

rng = np.random.default_rng(seed=1)

all_tpr = []
all_auc = []
for i in range(1000):
    sampled_df = clean_df.sample(frac=1, replace=True, random_state=rng)
    bootstrap_fpr, bootstrap_tpr, _ = sklearn.metrics.roc_curve(
        sampled_df["rec_murmur_label"] == "Present",
        sampled_df["diff"],
    )

    tpr_interp = np.interp(fpr_index, bootstrap_fpr, bootstrap_tpr)
    all_tpr.append(tpr_interp)
    all_auc.append(sklearn.metrics.auc(bootstrap_fpr, bootstrap_tpr))

low_tpr, high_tpr = np.quantile(np.stack(all_tpr), [0.025, 0.975], axis=0)
low_auc, high_auc = np.quantile(all_auc, [0.025, 0.975])

axes.grid(c="#ededed")
axes.plot([0, 1], [1, 0], label="Random guess", ls="--", c="#d3d3d3")
axes.fill_between(
    1 - fpr_index,
    low_tpr,
    high_tpr,
    facecolor="#d3d3d3",
    label=f"95% CI (AUC={low_auc:.3f}-{high_auc:.3f})",
)
axes.plot(1 - fpr, tpr, label=f"Parallel HSMM (AUC={auc:.3f})", c=[0.2, 0.2, 0.2])


tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
    clean_df["rec_murmur_label"] == "Present", clean_df["diff"] > 0
).ravel()
se = tp / (tp + fn)
sp = tn / (tn + fp)

axes.plot(
    sp,
    se,
    c=[0.2, 0.2, 0.2],
    marker="o",
    ls="none",
    label=f"Operating point (Se={se*100:.1f}%, Sp={sp*100:.1f}%)",
)

print(len(clean_df))
axes.legend(loc="lower right")

fig.tight_layout()
plt.savefig("results/figures/murmur_roc.png")
plt.savefig("results/figures/murmur_roc.tif")

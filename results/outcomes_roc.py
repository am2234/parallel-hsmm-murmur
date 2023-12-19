import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from utils import outcome_cost, read_training_spreadsheet


def conf_matrix_calc(se, sp, prevalence, num_patients):
    """Reverse engineer a confusion matrix using sensitivity, specificity, prevalence, N"""
    tn = sp * num_patients * (1 - prevalence)
    tp = num_patients * prevalence * se
    fp = -num_patients * (1 - prevalence) * (sp - 1)
    fn = num_patients * prevalence * (1 - se)
    return tp, tn, fp, fn


def outcome_cost_from_metrics(se, sp, prevalence, num_patients):
    tp, tn, fp, fn = conf_matrix_calc(se, sp, prevalence, num_patients)
    return outcome_cost(tp, tn, fp, fn)


def main():
    df = read_training_spreadsheet()
    df_no_unk = df.copy()
    df_no_unk["Murmur"] = df_no_unk["Murmur"].map(
        {"Present": True, "Absent": False, "Unknown": True}
    )
    df_no_unk["Outcome"] = df_no_unk["Outcome"].map({"Abnormal": True, "Normal": False})

    conf_matrix = sklearn.metrics.confusion_matrix(df_no_unk["Outcome"], df_no_unk["Murmur"])
    tn, fp, fn, tp = conf_matrix.ravel()
    gp_se = tp / (tp + fn)
    gp_sp = tn / (tn + fp)
    gp_pr = tp / (tp + fp)

    print("### Detection of any murmur ###")
    print(f"Sensitivity = {gp_se:.3f}, Specificity = {gp_sp:.3f}, Precision = {gp_pr:.3f}")
    outcome_cost(tp, tn, fp, fn, printl=True)

    print("All correct:")
    outcome_cost(413, 461, 0, 0, printl=True)
    print("None correct:")
    outcome_cost(0, 0, 413, 461, printl=True)
    print("All rejected:")
    outcome_cost(0, 461, 0, 413, printl=True)
    print("All referred:")
    outcome_cost(413, 0, 461, 0, printl=True)

    PREVALENCE = sum(df.Outcome == "Abnormal") / len(df) * 1
    print("Prevalence in training set", PREVALENCE)

    se_range = np.linspace(0, 1, 100)
    sp_range = np.linspace(1, 0, 100)
    SE, SP = np.meshgrid(se_range, sp_range)
    out = [
        outcome_cost_from_metrics(x, y, PREVALENCE, len(df))
        for x, y in zip(SE.ravel(), SP.ravel())
    ]
    out = np.array(out).reshape(SE.shape)

    font = {"size": 8, "family": "Arial"}
    plt.rc("font", **font)
    fig, axes = plt.subplots(figsize=(6, 4.5), dpi=300)
    axes.invert_xaxis()
    axes.set_aspect("equal")
    axes.set_facecolor("white")
    fig.set_facecolor("white")
    axes.set_ylabel("Sensitivity")
    axes.set_xlabel("Specificity")
    axes.set_xlim(1, 0)
    axes.set_ylim(0, 1)

    plt.pcolormesh(SP, SE, out)
    plt.colorbar(
        label="Clinical outcome cost",
        ticks=[out.min()] + list(np.arange(7500, 22501, 2500)) + [out.max()],
    )

    model_folder = pathlib.Path("final_model/").resolve()

    outcome_df = pd.read_csv(model_folder / "outcome_predictions.csv", index_col=0)
    outcome_df["abnormal_pred"] = (
        outcome_df["probabilities"].str[1:-1].str.split(" ", expand=True)[0].astype(float)
    )

    fpr, tpr, _ = sklearn.metrics.roc_curve(
        outcome_df["label"] == "Abnormal", outcome_df["abnormal_pred"]
    )

    axes.plot([0, 1], [1, 0], lw=1, color="white", ls="--", label="Random guess", alpha=0.5)
    axes.plot(1 - fpr, tpr, lw=1, c="white", label="CatBoost ROC")

    axes.plot(0.31, 0.84, "o", color="white", mew=2, label="CatBoost operating point")
    axes.text(0.31 + 0.03, 0.84, "11040", va="center", ha="right", c="white")

    axes.plot(0.7757, 0.5526, "*", color="white", mew=2, label="Murmur algorithm")
    axes.text(0.77 + 0.125, 0.55 - 0.001, "13681", va="center", c="white")

    axes.plot(gp_sp, gp_se, "x", color="white", mew=2, label="Clinician murmur label")
    axes.text(gp_sp - 0.04, gp_se - 0.001, "16083", va="center", c="white")

    axes.plot(0.217, 0.783, "d", color="white", mew=2, label="Best random classifier")
    axes.text(0.217 - 0.02, 0.783 - 0.001, "12579", va="center", c="white")

    axes.legend(loc="lower right", facecolor="lightgray", labelcolor="black", edgecolor="black")

    plt.savefig("results/figures/outcomes_roc.png", bbox_inches="tight")


if __name__ == "__main__":
    main()

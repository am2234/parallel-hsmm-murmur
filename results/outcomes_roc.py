import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from utils import outcome_cost, read_training_spreadsheet

font = {"size": 9, "family": "Arial"}
plt.rc("font", **font)


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


def cost_and_metrics(prediction, target):
    conf_matrix_murmur = sklearn.metrics.confusion_matrix(target, prediction)
    tn, fp, fn, tp = conf_matrix_murmur.ravel()
    murmur_se = tp / (tp + fn)
    murmur_sp = tn / (tn + fp)
    ppv = tp / (tp + fp)
    murmur_cost = outcome_cost(tp, tn, fp, fn, printl=False)
    return {
        "sensitivity": murmur_se,
        "specificity": murmur_sp,
        "PPV": ppv,
        "F1": (2 * ppv * murmur_se) / (ppv + murmur_se),
        "cost": round(murmur_cost),
    }


def random_cost_and_metrics(prevalence, num_patients):
    rows = []
    for i in np.arange(0, 1.001, 0.001):
        se = i
        sp = 1 - i
        cost = outcome_cost_from_metrics(se, sp, prevalence, num_patients)
        rows.append([se, sp, cost])
    df_cost = pd.DataFrame(rows, columns=["se", "sp", "cost"])
    best_idx = df_cost["cost"].idxmin()
    best_row = df_cost.loc[best_idx]
    se, sp = best_row[["se", "sp"]]
    cost = outcome_cost_from_metrics(se, sp, prevalence, num_patients)

    ppv = (se * prevalence) / ((se * prevalence) + ((1 - prevalence) * (1 - sp)))
    return {
        "sensitivity": se,
        "specificity": sp,
        "PPV": ppv,
        "F1": (2 * ppv * se) / (ppv + se),
        "cost": round(cost),
    }


def load_murmur_predictions():
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

    patient_df.index = patient_df.index.astype(int)

    return patient_df


def main():
    df = read_training_spreadsheet()
    PREVALENCE = sum(df.Outcome == "Abnormal") / len(df) * 1
    print("Prevalence in training set", PREVALENCE)
    model_folder = pathlib.Path("final_model/").resolve()

    df_prediction = df[["Outcome"]].copy()
    df_prediction["Outcome"] = df_prediction["Outcome"].map({"Abnormal": True, "Normal": False})
    df_prediction["Clinician"] = df["Murmur"].map(
        {"Present": True, "Absent": False, "Unknown": True}
    )

    outcome_df = pd.read_csv(model_folder / "outcome_predictions.csv", index_col=0)
    outcome_df["abnormal_pred"] = (
        outcome_df["probabilities"].str[1:-1].str.split(" ", expand=True)[0].astype(float)
    )
    df_prediction["CatBoost"] = outcome_df["abnormal_pred"] > 0.4738

    murmur_df = load_murmur_predictions()
    murmur_df["outcome_pred"] = murmur_df["murmur_pred"].map(
        {"Present": True, "Unknown": True, "Absent": False}
    )
    df_prediction["Murmur"] = murmur_df["outcome_pred"]

    assert not pd.isna(df_prediction).any().any()

    rows = []
    for predictor in ["CatBoost", "Murmur", "Clinician"]:
        result = cost_and_metrics(df_prediction[predictor], df_prediction["Outcome"])
        result["point"] = predictor
        rows.append(result)

    random_result = random_cost_and_metrics(PREVALENCE, len(df))
    random_result["point"] = "Random"
    rows.append(random_result)

    df_result = pd.DataFrame.from_records(rows, index="point")

    fig, axes = plt.subplots(figsize=(6, 4.5), dpi=500)
    axes.invert_xaxis()
    axes.set_aspect("equal")
    axes.set_facecolor("white")
    fig.set_facecolor("white")
    axes.set_ylabel("Sensitivity")
    axes.set_xlabel("Specificity")
    axes.set_xlim(1, 0)
    axes.set_ylim(0, 1)

    SE, SP, out = calculate_cost_vs_se_sp(df, PREVALENCE)
    plt.pcolormesh(SP, SE, out)
    plt.colorbar(
        label="Clinical outcome cost",
        ticks=[out.min()] + list(np.arange(7500, 22501, 2500)) + [out.max()],
    )

    fpr, tpr, _ = sklearn.metrics.roc_curve(
        outcome_df["label"] == "Abnormal", outcome_df["abnormal_pred"]
    )
    auc = sklearn.metrics.roc_auc_score(
        outcome_df["label"] == "Abnormal", outcome_df["abnormal_pred"]
    )

    markers = {
        "CatBoost": ["CatBoost operating point", "o", 0.125],
        "Murmur": ["Murmur algorithm", "*", 0.125],
        "Clinician": ["Clinician murmur label", "x", -0.03],
        "Random": ["Best random classifier", "d", -0.03],
    }

    axes.plot([0, 1], [1, 0], lw=1, color="white", ls="--", label="Random guess", alpha=0.5)
    axes.plot(1 - fpr, tpr, lw=1, c="white", label=f"CatBoost ROC (AUC={auc:.3f})")

    for predictor, options in markers.items():
        label, marker, offset = options
        row = df_result.loc[predictor]
        sp, se = row["specificity"], row["sensitivity"]
        axes.plot(sp, se, marker, color="white", mew=2, label=label)
        axes.text(sp + offset, se, f"{row['cost']:.0f}", va="center", c="white")

    axes.legend(loc="lower right", facecolor="lightgray", labelcolor="black", edgecolor="black")
    plt.savefig("results/figures/outcomes_roc.png", bbox_inches="tight")
    plt.savefig("results/figures/outcomes_roc.tif", bbox_inches="tight")

    df_result[["sensitivity", "specificity", "PPV"]] = (
        df_result[["sensitivity", "specificity", "PPV"]] * 100
    ).round(1)
    df_result["F1"] = df_result["F1"].round(3)

    print(df_result)

    micro_accuracy = (df_prediction["CatBoost"] == df_prediction["Outcome"]).mean()
    print("Algorithm accuracy", micro_accuracy)


def calculate_cost_vs_se_sp(df, PREVALENCE):
    se_range = np.linspace(0, 1, 1000)
    sp_range = np.linspace(1, 0, 1000)
    SE, SP = np.meshgrid(se_range, sp_range)
    out = [
        outcome_cost_from_metrics(x, y, PREVALENCE, len(df)) for x, y in zip(SE.ravel(), SP.ravel())
    ]
    out = np.array(out).reshape(SE.shape)
    return SE, SP, out


if __name__ == "__main__":
    main()

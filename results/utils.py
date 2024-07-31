import pathlib
import os

import scipy.io.wavfile
import pandas as pd
import numpy as np

C_ALGORITHM = 10
C_TREATMENT = 10_000
C_ERROR = 50_000


MAIN_FOLDER = pathlib.Path(os.environ["CIRCOR_FOLDER"])
DATA_FOLDER = MAIN_FOLDER / "training_data"
SPREADSHEET_PATH = MAIN_FOLDER / "training_data.csv"


def C_EXPERT(s, t):
    s_t = s / t
    return t * (25 + (397 * s_t) - (1718 * s_t**2) + (11296 * s_t**4))


def outcome_cost(tp, tn, fp, fn, printl=False):
    num_patients = tp + tn + fp + fn

    c_algo = C_ALGORITHM * num_patients
    c_exp = C_EXPERT(tp + fp, num_patients)
    c_treat = C_TREATMENT * tp
    c_err = C_ERROR * fn
    total_cost = c_algo + c_exp + c_treat + c_err
    result = total_cost / num_patients

    if printl:
        print(
            f"C_ALGO = {c_algo/num_patients:.0f}, C_EXP = {c_exp/num_patients:4.0f},"
            f"C_TREAT = {c_treat/num_patients:4.0f}, C_ERR = {c_err/num_patients:5.0f},"
            f"C_TOTAL = {result:5.0f}"
        )

    return result


def read_training_spreadsheet():
    df = pd.read_csv(SPREADSHEET_PATH, index_col=0)
    return df


def load_recording(filename):
    filepath = DATA_FOLDER / filename
    fs, data = scipy.io.wavfile.read(filepath)
    return data, fs


def compute_f_measure(labels, outputs):
    # copied from evaluate_model.py 
    num_patients, num_classes = np.shape(labels)

    A = compute_one_vs_rest_confusion_matrix(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float("nan")

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float("nan")

    return macro_f_measure, f_measure

# Compute binary one-vs-rest confusion matrices, where the columns are the expert labels and the rows are the classifier labels.
def compute_one_vs_rest_confusion_matrix(labels, outputs):
    assert np.shape(labels) == np.shape(outputs)
    assert all(value in (0, 1, True, False) for value in np.unique(labels))
    assert all(value in (0, 1, True, False) for value in np.unique(outputs))

    num_patients, num_classes = np.shape(labels)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_patients):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                A[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                A[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                A[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                A[j, 1, 1] += 1

    return A


# Compute accuracy.
def compute_weighted_accuracy(labels, outputs, classes):
    # Define constants.
    if classes == ["Present", "Unknown", "Absent"]:
        weights = np.array([[5, 3, 1], [5, 3, 1], [5, 3, 1]])
    elif classes == ["Abnormal", "Normal"]:
        weights = np.array([[5, 1], [5, 1]])
    else:
        raise NotImplementedError(
            "Weighted accuracy undefined for classes {}".format(", ".join(classes))
        )

    # Compute confusion matrix.
    assert np.shape(labels) == np.shape(outputs)
    A = compute_confusion_matrix(labels, outputs)

    # Multiply the confusion matrix by the weight matrix.
    assert np.shape(A) == np.shape(weights)
    B = weights * A

    # Compute weighted_accuracy.
    if np.sum(B) > 0:
        weighted_accuracy = np.trace(B) / np.sum(B)
    else:
        weighted_accuracy = float("nan")

    return weighted_accuracy



# Compute a binary confusion matrix, where the columns are the expert labels and the rows are the classifier labels.
def compute_confusion_matrix(labels, outputs):
    assert np.shape(labels)[0] == np.shape(outputs)[0]
    assert all(value in (0, 1, True, False) for value in np.unique(labels))
    assert all(value in (0, 1, True, False) for value in np.unique(outputs))

    num_patients = np.shape(labels)[0]
    num_label_classes = np.shape(labels)[1]
    num_output_classes = np.shape(outputs)[1]

    A = np.zeros((num_output_classes, num_label_classes))
    for k in range(num_patients):
        for i in range(num_output_classes):
            for j in range(num_label_classes):
                if outputs[k, i] == 1 and labels[k, j] == 1:
                    A[i, j] += 1

    return A

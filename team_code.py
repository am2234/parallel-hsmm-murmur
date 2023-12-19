#!/usr/bin/env python
# team_code.py is © 2022, University of Cambridge
#
# team_code.py is published and distributed under the GAP Available Source License v1.0 (ASL).
#
# team_code.py is is distributed in the hope that it will be useful for non-commercial academic
# research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.
#
# You should have received a copy of the ASL along with this program; if not, write to
# am2234 (at) cam (dot) ac (dot) uk.

import os
import pathlib
import re
import json
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import torch

from helper_code import *

from src import segmenter, decision_tree, neural_networks

USE_MURMUR_DECISION_TREE = False

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    return train_challenge_model_full(data_folder, model_folder, verbose)


# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    with (pathlib.Path(model_folder) / "settings.json").open("r") as f:
        settings = json.load(f)

    if "rnn_hidden_size" not in settings:
        # Assume default model sizes as in challenge
        settings = {
            **settings,
            "rnn_hidden_size": 60,
            "ann_hidden_size": [60, 40],
            "rnn_num_layers": 3,
            "rnn_dropout": 0.1,
            "ann_dropout": 0.1,
        }

    return [
        {
            "network": neural_networks.load_single_network_fold(model_folder, fold, settings),
            "tree_murmur": decision_tree.load_catboost_model(model_folder, fold, "murmur_label")
            if USE_MURMUR_DECISION_TREE
            else None,
            "tree_outcome": decision_tree.load_catboost_model(model_folder, fold, "outcome_label"),
            "outcome_threshold": settings["threshold"],
        }
        for fold in range(5)
    ]


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    num_locations = get_num_locations(data)
    recording_information = data.split("\n")[1 : num_locations + 1]
    recording_name = [r.split(" ")[0] for r in recording_information]
    fs = get_frequency(data)

    for model_fold in model:
        model_fold["network"].cuda()

    location_predictions = defaultdict(list)
    location_signal_quals = defaultdict(list)
    for recording, name in zip(recordings, recording_name):
        posteriors = predict_single_recording(recording, fs, model)
        _, _, healthy_conf, murmur_conf, murmur_timing = segmenter.double_duration_viterbi(
            posteriors, model[0]["network"].output_fs
        )
        location_predictions[name].append(murmur_conf - healthy_conf)
        location_signal_quals[name].append(max(murmur_conf, healthy_conf))

    full_features = {f"conf_difference_{k}": np.mean(v) for k, v in location_predictions.items()}
    for k, v in location_signal_quals.items():
        full_features[f"signal_qual_{k}"] = np.mean(v)
    for k, v in functions.items():
        full_features[k] = v(data)
    full_features["num_rec"] = len(recordings)

    if USE_MURMUR_DECISION_TREE:
        ordered_array = [
            full_features.get(k, None) for k in model[0]["tree_murmur"].feature_names_
        ]
        probabilities = []
        for model_fold in model:
            probabilities.append(
                model_fold["tree_murmur"].predict(ordered_array, prediction_type="Probability")
            )
        murmur_probabilities = np.mean(probabilities, axis=0)

        # Choose label with higher probability.
        murmur_labels = np.zeros(len(murmur_probabilities), dtype=np.int_)
        idx = np.argmax(murmur_probabilities)
        murmur_labels[idx] = 1
    else:
        prediction = decide_murmur_outcome(full_features)
        murmur_probabilities = np.zeros(3)
        prediction_to_index = {"Present": 0, "Unknown": 1, "Absent": 2}
        murmur_probabilities[prediction_to_index[prediction]] = 1
        murmur_labels = np.zeros(3, dtype=np.int_)
        murmur_labels[prediction_to_index[prediction]] = 1

    ordered_array = []
    for k in model[0]["tree_outcome"].feature_names_:
        if k not in full_features:
            # Note no filling of NaN values
            val = None
        else:
            val = full_features[k]
        ordered_array.append(val)

    probabilities = []
    for model_fold in model:
        probabilities.append(
            model_fold["tree_outcome"].predict(ordered_array, prediction_type="Probability")
        )
    outcome_probabilities = np.mean(probabilities, axis=0)

    # Choose label with higher probability.
    outcome_labels = np.zeros(len(outcome_probabilities), dtype=np.int_)
    outcome_class_order = list(model_fold["tree_outcome"].classes_)
    abnormal_idx = outcome_class_order.index("Abnormal")
    idx = (
        abnormal_idx
        if outcome_probabilities[abnormal_idx] > model_fold["outcome_threshold"]
        else 1 - abnormal_idx
    )
    outcome_labels[idx] = 1

    if USE_MURMUR_DECISION_TREE:
        class_order = list(model_fold["tree_murmur"].classes_) + outcome_class_order
    else:
        class_order = ["Present", "Unknown", "Absent"] + outcome_class_order
    labels = list(murmur_labels) + list(outcome_labels)
    probabilities = list(murmur_probabilities) + list(outcome_probabilities)
    return class_order, labels, probabilities


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


def predict_single_recording(recording, fs, model):
    # Load features.
    recording = torch.as_tensor(recording.copy())
    features, fs_features = neural_networks.calculate_features(recording, fs)
    features = features.unsqueeze(0)
    lengths = torch.as_tensor([features.shape[-1]])

    # Predict using each cross validated model, but now average before segmentation
    fold_predictions = []
    for model_fold in model:
        results = neural_networks.predict_single_model(
            model_fold["network"], features, lengths, gpu=True
        )
        fold_predictions.append(results[0])
    posteriors = torch.mean(torch.stack(fold_predictions, dim=0), dim=0)
    posteriors = posteriors.T  # [C, T] to [T, C]
    posteriors = posteriors.numpy()
    return posteriors


functions = {
    "age": get_age,
    "pregnant": get_pregnancy_status,
    "height": get_height,
    "weight": get_weight,
    "sex": get_sex,
}


def train_challenge_model_full(
    data_folder, model_folder, verbose, hparams=None, load_old_file=False, gpu=True, quick=False
):
    if hparams is None:
        # Use same default parameters CUED_Acoustics PhysioNet entry
        hparams = {
            "rnn_hidden_size": 60,
            "rnn_num_layers": 3,
            "rnn_dropout": 0.1,
            "ann_hidden_size": [60, 40],
            "ann_dropout": 0.1,
            "batch_size": 32,
            "lr": 0.0001,
        }

    # Create a folder for the model if it does not already exist.
    model_folder = pathlib.Path(model_folder)

    # Load patient information from files and assign stratified folds
    patient_df = load_patient_files(data_folder, stop_frac=0.1 if quick else 1)
    print(f"Training with {len(patient_df)} patients")
    patient_df = create_folds(patient_df)

    # ### Part 1 ###
    # Train cross-validated neural network to predict heart sound category at each timestep

    # Get individual recording files and murmur labels from each patient
    recording_rows = []
    for row in patient_df.itertuples():
        for recording_path in row.recordings:
            # Get recording location from filename
            recording_loc = re.split("[_.]", recording_path)[1]
            if recording_loc not in {"AV", "MV", "PV", "TV", "Phc"}:
                raise ValueError(f"Recording loc for {recording_path} is {recording_loc}")

            # Assign murmur label to specific recording
            assert row.murmur_label in {"Present", "Absent", "Unknown"}
            if row.murmur_label == "Present" and recording_loc not in row.murmur_locations:
                rec_murmur_label = "Absent"
            else:
                rec_murmur_label = row.murmur_label

            recording_rows.append(
                {
                    "recording": recording_path.replace(".wav", ""),
                    "murmur_timing": row.systolic_timing,
                    "rec_murmur_label": rec_murmur_label,
                    "patient_murmur_label": row.murmur_label,
                    "val_fold": row.val_fold,
                }
            )
    recording_df = pd.DataFrame.from_records(recording_rows, index="recording")

    shared_feature_cache = {}
    val_results = defaultdict(list)
    fold_names = sorted(patient_df["val_fold"].unique())

    location_specific_murmur_label = True
    murmur_label_col = (
        "rec_murmur_label" if location_specific_murmur_label else "patient_murmur_label"
    )

    # We only want to train the murmur segmentation on recordings that have a murmur label
    # (i.e. not those recordings labelled as 'Unknown')
    recording_df_gq = recording_df[recording_df["patient_murmur_label"] != "Unknown"]
    recording_df_bq = recording_df[recording_df["patient_murmur_label"] == "Unknown"]

    val_losses = []
    if not load_old_file:
        os.makedirs(model_folder, exist_ok=True)
        assert len(fold_names) == 5
        for fold in fold_names:
            if verbose >= 1:
                print(f"****** Fold {fold + 1} of {len(fold_names)} ******")

            train_recording_df = recording_df_gq[recording_df_gq["val_fold"] != fold]
            val_recording_df = recording_df_gq[recording_df_gq["val_fold"] == fold]

            model, val_loss, val_fold_results = neural_networks.train_and_validate_model(
                model_folder=model_folder,
                fold=fold,
                data_folder=data_folder,
                train_files=train_recording_df.index,
                train_labels=train_recording_df[murmur_label_col],
                train_timings=train_recording_df.murmur_timing,
                val_files=val_recording_df.index,
                val_labels=val_recording_df[murmur_label_col],
                val_timings=val_recording_df.murmur_timing,
                shared_feature_cache=shared_feature_cache,
                verbose=verbose,
                gpu=gpu,
                quick=quick,
                hparams=hparams,
            )
            val_losses.append(val_loss.cpu().item())
            assert len(val_fold_results.keys() & val_results.keys()) == 0
            for k, v in val_fold_results.items():
                val_results[k].append(v)

            # Make a prediction for the bad quality "unknown" labelled signals
            # because we will need these predictions to train the decision tree
            unknown_posteriors = neural_networks.predict_files(
                model=model,
                data_folder=data_folder,
                files=recording_df_bq.index,
                labels=recording_df_bq[murmur_label_col],
                timings=recording_df_bq.murmur_timing,
                cache=shared_feature_cache,
                gpu=gpu,
            )
            for k, v in unknown_posteriors.items():
                val_results[k].append(v)

        # ### Part 2 ###
        # Segment recordings using neural network predictions as observation probabilities for
        # two HSMMs

        num_examples = len(val_results)
        results = {}
        for i, (k, posteriors) in enumerate(val_results.items()):
            print(f"\rSegmenting {i + 1:03d} of {num_examples:03d} ", end="")
            posteriors = torch.stack(posteriors).mean(dim=0)  # mean predictions on 'Unknown' recs
            posteriors = posteriors.T  # [C, T] to [T, C]
            posteriors = posteriors.numpy()
            _, _, healthy_conf, murmur_conf, murmur_timing = segmenter.double_duration_viterbi(
                posteriors, 50
            )

            results[k] = {
                "healthy_HSMM": healthy_conf,
                "holo_HSMM": murmur_conf,
                "murmur_timing": murmur_timing,
            }
        print("\nSegmenting complete.")

        rec_predictions_df = pd.DataFrame.from_dict(results, orient="index")
        recording_df = recording_df.merge(rec_predictions_df, left_index=True, right_index=True)
        recording_df.to_csv(model_folder / "recordings.csv")

        ### Part 3 ###
        # Train cross-validated gradient boosted decision trees to use segmentation confidences
        # and other patient biometrics to predict final class

        df = prepare_tree_df(rec_predictions_df, patient_df)
        for index, row in df.iterrows():
            df.at[index, "num_rec"] = len(row.recordings)
        df["num_rec"] = df["num_rec"].astype(int)
        df.to_csv(model_folder / "tree_inputs.csv")
    else:
        df = pd.read_csv(model_folder / "tree_inputs.csv", index_col=0)
        df.index = df.index.astype(str)
        for col in ["age", "sex", "pregnant"]:
            df.loc[pd.isna(df[col]), col] = "None"

    val_murmur_predictions = {}
    val_outcome_predictions = {}

    for index, row in df.iterrows():
        prediction = decide_murmur_outcome(row.to_dict())
        val_murmur_predictions[index] = {
            "prediction": prediction,
            "probabilities": [],
            "label": row["murmur_label"],
            "fold": row.val_fold,
        }
        val_outcome_predictions[index] = {
            "prediction": "Normal" if prediction == "Absent" else "Abnormal",
            "probabilities": [],
            "label": row["outcome_label"],
            "fold": row.val_fold,
        }

    print("WITHOUT DECISION TREE")
    score = compute_cross_val_weighted_murmur_accuracy(val_murmur_predictions)
    print(f"Weighted murmur accuracy = {score:.3f}")

    outcome_score = compute_cross_val_outcome_score(val_outcome_predictions)
    print(f"Outcome score = {outcome_score:.0f}")

    print("WITH DECISION TREE")

    val_murmur_predictions = {}
    for fold in fold_names:
        val_fold_predictions = decision_tree.train_and_validate_model(
            train_df=df.loc[df["val_fold"] != fold],
            val_df=df.loc[df["val_fold"] == fold],
            model_folder=model_folder,
            fold=fold,
            target_name="murmur_label",
            class_weights={"Present": 20, "Unknown": 5, "Absent": 1},
        )
        for k, v in val_fold_predictions.items():
            val_murmur_predictions[k] = {**v, "fold": fold}

    decision_score = compute_cross_val_weighted_murmur_accuracy(val_murmur_predictions)
    print(f"Weighted murmur accuracy = {decision_score:.3f}")

    val_outcome_predictions = {}
    for fold in fold_names:
        val_fold_outcome_predictions = decision_tree.train_and_validate_model(
            train_df=df.loc[df["val_fold"] != fold],
            val_df=df.loc[df["val_fold"] == fold],
            model_folder=model_folder,
            fold=fold,
            target_name="outcome_label",
            class_weights={"Abnormal": 1.8, "Normal": 1},
        )
        for k, v in val_fold_outcome_predictions.items():
            val_outcome_predictions[k] = {**v, "fold": fold}

    df = pd.DataFrame.from_dict(val_outcome_predictions, orient="index")
    df.to_csv(model_folder / "outcome_predictions.csv")

    outcome_score, threshold = choose_outcome_threshold(df)
    print(f"Outcome score = {outcome_score:.0f} (threshold = {threshold:.03f})")

    settings = {"threshold": threshold, **hparams}
    with (model_folder / "settings.json").open("w") as f:
        json.dump(settings, f)

    if verbose >= 1:
        print("Done.")

    return score, decision_score, outcome_score, np.mean(val_losses)


def choose_outcome_threshold(df):
    df["first_prob"] = [p[0] for p in df["probabilities"]]
    df["label_int"] = df["label"].map({"Abnormal": 1, "Normal": 0})

    rows = []
    for threshold in sorted(df["first_prob"].unique())[:-1]:
        cost = compute_outcome_score(df["first_prob"] > threshold, df["label_int"])
        rows.append({"cost": cost, "threshold": threshold})
    rows_df = pd.DataFrame.from_records(rows)

    min_row = rows_df.iloc[rows_df.cost.idxmin()]
    compute_outcome_score(df["first_prob"] > min_row.threshold, df["label_int"], print_matrix=True)

    return min_row.cost, min_row.threshold


def decide_murmur_outcome(features: Dict):
    conf_differences = [v for k, v in features.items() if k.startswith("conf_difference_")]
    signal_quals = [v for k, v in features.items() if k.startswith("signal_qual_")]

    if np.nanmax(conf_differences) > 0:
        return "Present"
    elif np.nanmin(signal_quals) < 0.65:
        return "Unknown"
    else:
        return "Absent"


def get_murmur_locations(data):
    locations = set()
    for l in data.split("\n"):
        if l.startswith("#Murmur locations:"):
            try:
                locations = l.split(": ")[1].strip().split("+")
            except:
                pass
    if locations == ["nan"]:
        locations = []
    return set(locations)


# Get pregnancy status from patient data.
def get_murmur_timing(data):
    timing = None
    for l in data.split("\n"):
        if l.startswith("#Systolic murmur timing:"):
            try:
                timing = l.split(": ")[1].strip()
            except:
                pass
    return timing


def load_patient_files(data_folder, start_frac=0, stop_frac=1):
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)
    if num_patient_files == 0:
        raise Exception("No data was provided.")

    stop_index = int(stop_frac * num_patient_files)
    start_index = int(start_frac * num_patient_files)
    patient_files = patient_files[start_index : stop_index + 1]
    num_patient_files = len(patient_files)

    rows = {}
    for i in range(num_patient_files):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        num_locations = get_num_locations(current_patient_data)
        recording_information = current_patient_data.split("\n")[1 : num_locations + 1]

        rec_files = []
        for i in range(num_locations):
            recording_wav = recording_information[i].split(" ")[2]
            if recording_wav in ["50782_MV_1.wav"]:  # no segmentation
                continue
            rec_files.append(recording_wav)

        rows[get_patient_id(current_patient_data)] = {
            "murmur_label": get_murmur(current_patient_data),
            "outcome_label": get_outcome(current_patient_data),
            "systolic_timing": get_murmur_timing(current_patient_data),
            **{k: v(current_patient_data) for k, v in functions.items()},
            "murmur_locations": get_murmur_locations(current_patient_data),
            "recordings": rec_files,
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def create_folds(patient_df: pd.DataFrame):
    # NB: set random_state=1 for earlier version
    skf = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    folds = skf.split(np.zeros(len(patient_df)), patient_df.murmur_label)

    patient_df["val_fold"] = np.nan
    for fold, (train, test) in enumerate(folds):
        patient_df.iloc[test, patient_df.columns.get_loc("val_fold")] = fold
    patient_df["val_fold"] = patient_df["val_fold"].astype(int)

    return patient_df


def print_confusion_matrix(matrix, classes):
    max_name = max(len(c) for c in classes)
    print(" " * max_name + "True Class".center(len(classes) + max_name * (len(classes))))
    print("Predict".ljust(8) + " ".join([c.rjust(8) for c in classes]))
    for row, c in zip(matrix, classes):
        row = c.ljust(8) + " ".join([f"{i:8d}" for i in row])
        print(row)


def compute_cross_val_weighted_murmur_accuracy(val_predictions, print=True):
    df = pd.DataFrame.from_dict(val_predictions, orient="index")

    # Define order of classes in confusion matrix (and corresponding score weights)
    class_order = ["Present", "Unknown", "Absent"]
    matrix_weights = np.asarray([[5, 3, 1], [5, 3, 1], [5, 3, 1]]).astype(float)

    conf_matrix = sklearn.metrics.confusion_matrix(df.label, df.prediction, labels=class_order).T

    if print:
        print_confusion_matrix(conf_matrix, class_order)
    weighted_conf = matrix_weights * conf_matrix
    if print:
        print_confusion_matrix(weighted_conf.astype(int), class_order)

    # print(sklearn.metrics.classification_report(df.label, df.prediction, labels=class_order))

    # from evaluate_model import compute_auc
    # MAP = {"Present": 0, "Unknown": 1, "Absent": 2}
    # labels = np.zeros((len(df), 3))
    # for i in range(len(labels)):
    #     labels[i, MAP[df.iloc[i].label]] = 1
    # output = np.stack(df.probabilities)
    # print(compute_auc(labels, output))

    return np.trace(weighted_conf) / np.sum(weighted_conf)


def compute_outcome_score(prediction, label, print_matrix=False):
    conf_matrix = sklearn.metrics.confusion_matrix(label, prediction, labels=[1, 0]).T
    tp, fp, fn, tn = conf_matrix.ravel()
    if print_matrix:
        print_confusion_matrix(conf_matrix, ["Abnormal", "Normal"])

        print("Se", tp / (tp + fn), "Sp", tn / (tn + fp))

    return outcome_cost(tp, tn, fp, fn)


def compute_cross_val_outcome_score(val_predictions):
    df = pd.DataFrame.from_dict(val_predictions, orient="index")

    class_order = ["Abnormal", "Normal"]
    conf_matrix = sklearn.metrics.confusion_matrix(df.label, df.prediction, labels=class_order).T
    print_confusion_matrix(conf_matrix, class_order)

    tp, fp, fn, tn = conf_matrix.ravel()

    cost = outcome_cost(tp, tn, fp, fn)
    return cost


def C_EXPERT(s, t):
    s_t = s / t
    return t * (25 + (397 * s_t) - (1718 * s_t**2) + (11296 * s_t**4))


def outcome_cost(tp, tn, fp, fn, printl=False):
    num_patients = tp + tn + fp + fn

    C_ALGORITHM = 10
    C_TREATMENT = 10_000
    C_ERROR = 50_000

    c_algo = C_ALGORITHM * num_patients
    c_exp = C_EXPERT(tp + fp, num_patients)
    c_treat = C_TREATMENT * tp
    c_err = C_ERROR * fn
    total_cost = c_algo + c_exp + c_treat + c_err
    result = total_cost / num_patients

    if printl:
        print(
            f"C_ALGO = {c_algo/num_patients:.0f}, C_EXP = {c_exp/num_patients:4.0f}, C_TREAT = {c_treat/num_patients:4.0f}, C_ERR = {c_err/num_patients:5.0f}, C_TOTAL = {result:5.0f}"
        )

    return result


def prepare_tree_df(pred_df, patient_df):
    pred_df.index = pred_df.index.str.split("_", expand=True)

    # Calculate difference
    pred_df["conf_difference"] = pred_df["holo_HSMM"] - pred_df["healthy_HSMM"]
    pred_df["signal_qual"] = pred_df[["holo_HSMM", "healthy_HSMM"]].max(axis=1)

    # Average predictions for recordings at same site
    pred_df = pred_df.groupby(level=[0, 1]).mean()
    pred_df = pred_df.unstack()

    # Drop PhC values as not enough of them to train
    pred_df = pred_df.drop(("conf_difference", "Phc"), axis=1, errors="ignore")
    pred_df.columns = [
        "_".join(col) if len(col[1]) > 0 else col[0] for col in pred_df.columns.values
    ]

    combined_df = pd.concat([pred_df, patient_df], axis=1)
    return combined_df

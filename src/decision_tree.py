# decision_tree.py is © 2022, University of Cambridge
#
# decision_tree.py is published and distributed under the GAP Available Source License v1.0 (ASL).
#
# decision_tree.py is distributed in the hope that it will be useful for non-commercial academic
# research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.
#
# You should have received a copy of the ASL along with this program; if not, write to
# am2234 (at) cam (dot) ac (dot) uk.
import pathlib

import catboost as cb
import numpy as np
import pandas as pd


TREE_VARIABLES = []
for a in ["conf_difference", "signal_qual"]:
    for l in ["AV", "MV", "TV", "PV"]:
        TREE_VARIABLES.append(a + "_" + l)

for a in ["age", "pregnant", "num_rec"]:
    TREE_VARIABLES.append(a)

CAT_FEATURES = sorted(list({"age", "pregnant", "sex"} & set(TREE_VARIABLES)))


def train_and_validate_model(train_df, val_df, model_folder, fold, target_name, class_weights):
    tree_model = train_catboost_model(
        train_df=train_df[TREE_VARIABLES],
        train_target=train_df[target_name],
        val_df=val_df[TREE_VARIABLES],
        val_target=val_df[target_name],
        class_weights=class_weights,
    )

    save_catboost_model(model_folder, tree_model, fold, target_name)

    return predict_catboost(tree_model, val_df[TREE_VARIABLES], val_df[target_name])


def train_catboost_model(train_df, train_target, val_df, val_target, class_weights):
    assert train_df.columns.equals(
        val_df.columns
    ), "Train and validation should have same variables"

    train_pool = _create_pool(train_df, TREE_VARIABLES, train_target, CAT_FEATURES, class_weights)
    val_pool = _create_pool(val_df, TREE_VARIABLES, val_target, CAT_FEATURES, class_weights)

    model = cb.CatBoostClassifier(
        loss_function="MultiClass",
        iterations=None,
        depth=9,
        use_best_model=True,
        early_stopping_rounds=100,
    )
    model.fit(train_pool, verbose=False, eval_set=val_pool)
    return model


def save_catboost_model(model_folder, model, fold, target_name):
    model_path = pathlib.Path(model_folder) / f"cb_model_{fold}_{target_name}.cbm"
    model.save_model(model_path, format="cbm")


def load_catboost_model(model_folder, fold, target_name):
    tree_model = cb.CatBoostClassifier()
    tree_model = tree_model.load_model(
        pathlib.Path(model_folder) / f"cb_model_{fold}_{target_name}.cbm", format="cbm"
    )
    return tree_model


def predict_catboost(model: cb.CatBoostClassifier, df: pd.DataFrame, target: pd.Series):
    pool = _create_pool(df, TREE_VARIABLES, target, CAT_FEATURES)

    val_fold_predictions = model.predict(pool, prediction_type="Probability")

    predictions = {}
    for prediction, id, label in zip(val_fold_predictions, df.index, target):
        probabilities = np.asarray(prediction.round(4))
        predicted_class = model.classes_[probabilities.argmax()]

        predictions[id] = {
            "prediction": predicted_class,
            "probabilities": probabilities,
            "label": label,
        }
    return predictions


def _get_class_weights(target, weight_vals=None):
    weight = np.ones(len(target))
    if weight_vals is None:
        return weight
    for k, v in weight_vals.items():
        weight[target == k] = v
    return weight


def _create_pool(df, in_variables, target, cat_features, class_weights=None):
    # Note, no replacement of NaN values in this version of code.
    return cb.Pool(
        df[in_variables],
        target,
        cat_features=cat_features,
        weight=_get_class_weights(target, class_weights),
    )

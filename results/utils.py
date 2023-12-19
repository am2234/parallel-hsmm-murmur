import pathlib
import os

import scipy.io.wavfile
import pandas as pd

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

# segmenter.py is © 2022, University of Cambridge
#
# segmenter.py is published and distributed under the GAP Available Source License v1.0 (ASL).
#
# segmenter.py is distributed in the hope that it will be useful for non-commercial academic
# research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.
#
# You should have received a copy of the ASL along with this program; if not, write to
# am2234 (at) cam (dot) ac (dot) uk.
import copy

import numpy as np
import scipy.stats as sci_stat

import pyximport

pyximport.install()
import viterbi_hmm


def get_systolic_interval(posterior, fs, heart_rate, min_duration=150):
    heart_cycle_samples = (60 / heart_rate) * fs

    max_systolic_duration = int(
        heart_cycle_samples / 2
    )  # systole can at most be half of the heart cycle
    min_systolic_duration = round(min_duration * 1e-3 * fs)

    mhs_posterior = np.sum(posterior[:, [0, 2]], axis=1)

    mhs_acf = np.correlate(mhs_posterior, mhs_posterior, mode="full")
    mhs_acf = mhs_acf[len(mhs_acf) // 2 :]
    mhs_acf = mhs_acf / mhs_acf[0]

    valid_acf = mhs_acf[min_systolic_duration : max_systolic_duration + 1]

    try:
        peak = np.argmax(valid_acf)
        absolute_peak = min_systolic_duration + peak
        return absolute_peak / fs
    except ValueError:
        print("Attempt to get argmax of empty systolic interval..")
        print(
            "HR:{}, min/max:{}/{}".format(heart_rate, min_systolic_duration, max_systolic_duration)
        )
        return min_systolic_duration / fs


def get_duration_distributions(heart_rate, systolic_interval, fs):
    distrib_S1 = sci_stat.norm(loc=0.1163 * fs, scale=0.0196 * fs)
    distrib_S2 = sci_stat.norm(loc=0.1032 * fs, scale=0.0195 * fs)

    mean_sys = (systolic_interval * fs) - (0.1279 * fs)
    mean_sys = max(mean_sys, 0.07 * fs)
    std_sys = 0.025 * fs

    mean_dia = (((60 / heart_rate) - systolic_interval) * fs) - (0.1053 * fs)
    mean_dia = max(mean_dia, 0.1 * fs)
    std_diastole = 0.050 * fs

    distrib_sys = sci_stat.norm(loc=mean_sys, scale=std_sys)
    distrib_dia = sci_stat.norm(loc=mean_dia, scale=std_diastole)

    return distrib_S1, distrib_sys, distrib_S2, distrib_dia


def get_heart_rate(posterior, fs, min_val=30, max_val=150, states=[1]):
    systolic_posterior = np.sum(posterior[:, states], axis=1)

    acf = np.correlate(systolic_posterior, systolic_posterior, mode="full")
    acf = acf[len(acf) // 2 :]
    acf = acf / acf[0]

    min_index = round((60 / max_val) * fs)  # min length of heart cycle in samples
    max_index = round((60 / min_val) * fs)  # max length

    valid_acf = acf[min_index : max_index + 1]

    rel_peak_loc = np.argmax(valid_acf)
    absolute_peak_loc = min_index + rel_peak_loc
    heart_cycle_time = absolute_peak_loc / fs

    return 60 / heart_cycle_time


def get_duration_matrix(d, distributions):
    duration_vectors = [distrib.pdf(d) for distrib in distributions]
    duration_matrix = np.stack(duration_vectors).T
    duration_matrix = duration_matrix / np.sum(
        duration_matrix, axis=0
    )  # normalise each column to sum to 1
    return duration_matrix


def compute_segmentation(posteriors, duration_dists, max_duration, transition_matrix):
    d = np.arange(1, max_duration + 1)
    duration_matrix = get_duration_matrix(d, duration_dists)

    # Run through HSMM using fast Cython implementation
    states = viterbi_hmm.hsmm_viterbi(
        posteriors,
        duration_matrix.astype(np.float32),
        max_duration,
        transition_matrix.astype(np.float32),
    )
    return states, compute_segmentation_confidence(posteriors, states)


def compute_segmentation_confidence(posteriors, segmentation):
    return posteriors[np.arange(len(posteriors)), segmentation].mean()


def segment_healthy_signal(posteriors, duration_distributions, max_duration):
    transition_matrix = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])

    # Now go from 5-state to 4-state
    # First model assumes no murmur
    new_posteriors = np.zeros((posteriors.shape[0], 4), dtype=posteriors.dtype)
    new_posteriors[:, 0] = posteriors[:, 0]
    new_posteriors[:, 1] = posteriors[:, 1]
    new_posteriors[:, 2] = posteriors[:, 2]
    new_posteriors[:, 3] = posteriors[:, 3]

    return compute_segmentation(
        new_posteriors, duration_distributions, max_duration, transition_matrix
    )


def segment_holosystolic_murmur(posteriors, duration_distributions, max_duration):
    transition_matrix = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])

    # Now go from 5-state to 4-state
    # Assume murmur posterior for all of systole
    new_posteriors = np.zeros((posteriors.shape[0], 4), dtype=posteriors.dtype)
    new_posteriors[:, 0] = posteriors[:, 0]
    new_posteriors[:, 1] = posteriors[:, 4]
    new_posteriors[:, 2] = posteriors[:, 2]
    new_posteriors[:, 3] = posteriors[:, 3]

    return compute_segmentation(
        new_posteriors, duration_distributions, max_duration, transition_matrix
    )


def segment_early_systolic_murmur(posteriors, duration_distributions, max_duration):
    transition_matrix = np.array(
        [
            [0, 0, 0, 0, 1],  # S1 -> murmur
            [0, 0, 1, 0, 0],  # systole -> S2
            [0, 0, 0, 1, 0],  # S2 - > diastole
            [1, 0, 0, 0, 0],  # diastole -> S1
            [0, 1, 0, 0, 0],
        ]
    )  # murmur -> systole

    orig_sys_distrib = duration_distributions[1]
    distrib_sys_half = sci_stat.norm(orig_sys_distrib.mean() / 2, orig_sys_distrib.std() / 2)
    new_duration_distribs = list(copy.copy(duration_distributions))
    new_duration_distribs[1] = distrib_sys_half
    new_duration_distribs.append(distrib_sys_half)  # murmur uses same distribution as sys

    return compute_segmentation(posteriors, new_duration_distribs, max_duration, transition_matrix)


def segment_mid_systolic_murmur(posteriors, duration_distributions, max_duration):
    transition_matrix = np.array(
        [
            [0, 1, 0, 0, 1],  # S1 -> systole
            [0, 0, 1, 0, 1],  # systole -> murmur or S2
            [0, 0, 0, 1, 0],  # S2 - > diastole
            [1, 0, 0, 0, 0],  # diastole -> S1
            [0, 1, 0, 0, 0],
        ]
    )  # murmur -> systole

    orig_sys_distrib = duration_distributions[1]
    distrib_sys_half = sci_stat.norm(orig_sys_distrib.mean() / 2, orig_sys_distrib.std() / 2)
    distrib_sys_quarter = sci_stat.norm(orig_sys_distrib.mean() / 4, orig_sys_distrib.std() / 4)
    new_duration_distribs = list(copy.copy(duration_distributions))
    new_duration_distribs[1] = distrib_sys_quarter
    new_duration_distribs.append(distrib_sys_half)  # murmur uses same distribution as sys

    return compute_segmentation(posteriors, new_duration_distribs, max_duration, transition_matrix)


def double_duration_viterbi(
    posteriors,
    fs,
    min_hr=30,
    max_hr=180,
    min_systole=150,
    max_duration=1,
    murmur_models=("Holosystolic", "Early-systolic", "Mid-systolic"),
):
    # posteriors shape [T, C]

    hr_states = [0, 1, 2, 4]  # S1 + systole + S2 + murmur
    heart_rate = get_heart_rate(posteriors, fs, min_hr, max_hr, hr_states)
    systolic_interval = get_systolic_interval(posteriors, fs, heart_rate, min_systole)
    duration_distributions = get_duration_distributions(heart_rate, systolic_interval, fs)

    max_duration = int((60 / heart_rate) * fs * max_duration)
    healthy_seg, healthy_conf = segment_healthy_signal(
        posteriors, duration_distributions, max_duration
    )

    murmur_functions = {
        "Holosystolic": segment_holosystolic_murmur,
        "Early-systolic": segment_early_systolic_murmur,
        "Mid-systolic": segment_mid_systolic_murmur,
    }

    best_murmur_confidence = -1
    best_murmur_model = None
    best_murmur_states = None
    for model in murmur_models:
        seg, conf = murmur_functions[model](posteriors, duration_distributions, max_duration)
        if conf > best_murmur_confidence:
            best_murmur_confidence = conf
            best_murmur_model = model
            best_murmur_states = seg

    return healthy_seg, best_murmur_states, healthy_conf, best_murmur_confidence, best_murmur_model

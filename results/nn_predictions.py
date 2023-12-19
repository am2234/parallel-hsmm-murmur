import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("")

from src.neural_networks import calculate_features
import utils

font = {"size": 8, "family": "Arial"}
plt.rc("font", **font)

rec, fs = utils.load_recording("50260_MV.wav")

T0 = 1.4
T1 = 3.7
rec = rec[int(T0 * fs) : int(T1 * fs)]

# Load posteriors for recording, generated through predict_single_recording() with model
model_folder = pathlib.Path("final_model/").resolve()
posteriors = np.loadtxt(model_folder / "50260_MV_posteriors.csv", delimiter=",")

fig, axes = plt.subplots(3, 1, figsize=(5.2, 4), sharex="all", dpi=300)

t_rec = np.arange(len(rec)) / fs
axes[0].plot(t_rec, rec, lw=1, c=[0.2, 0.2, 0.2])
axes[0].set_xlabel("")
axes[0].set_ylabel("Amplitude (a.u.)")
axes[0].set_yticks([])

features, features_fs = calculate_features(torch.as_tensor(rec), fs, norm=True)
x = np.linspace(0, T1 - T0, features.shape[1])
y = np.arange(0, 800, 20)
axes[1].pcolormesh(x, y, features, vmin=-1.5, vmax=2.5)
axes[1].set_ylim(0, 800)
axes[1].set_ylabel("Frequency (Hz)")

t_pos = np.arange(posteriors.shape[0]) / 50
axes[2].plot(t_pos, posteriors[:, 0], lw=1, c="k", label="S1")
axes[2].plot(t_pos, posteriors[:, 2], lw=1, c="b", ls="--", label="S2")
axes[2].plot(t_pos, posteriors[:, 4], lw=1, c="r", ls=":", label="Murmur")
axes[2].set_ylabel("State probability")
axes[2].set_xlabel("Time (seconds)")
axes[2].legend()
axes[2].set_xlim(0, T1 - T0 - 0.1)
axes[2].set_ylim(0, 1)

fig.align_ylabels()
fig.tight_layout(h_pad=0.4)

plt.savefig("results/figures/nn_predictions.png")

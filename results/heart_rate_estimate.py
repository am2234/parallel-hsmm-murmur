import pathlib

import matplotlib.pyplot as plt
import numpy as np
import utils

font = {"size": 8, "family": "Arial"}
plt.rc("font", **font)
COLOR = "#545454"

FOLDER_SPRINGER = pathlib.Path("/home/am2234/Software/springer_hsmm/")
envelope = np.loadtxt(FOLDER_SPRINGER / "out_envelope.csv", delimiter=",")

series, fs = utils.load_recording("85203_AV.wav")

fig, axes = plt.subplots(
    3, 2, figsize=(7.5, 6), dpi=300, gridspec_kw={"hspace": 0.4, "wspace": 0.3}
)

T0 = 6
T1 = 9
t_series = np.arange(len(series)) / fs
for a in axes[0, :]:
    a.plot(t_series, series, c=COLOR)
    a.set_xlim(T0, T1)
    a.set_xlabel("Time (seconds)")
    a.set_ylabel("Amplitude (a.u.)")
    a.set_yticks([])
    a.set_ylim(-5000, 5000)


axes[1, 0].plot(t_series, envelope, c=COLOR)
axes[1, 0].set_xlim(T0, T1)
axes[1, 0].set_xlabel("Time (seconds)")
axes[1, 0].set_ylabel("Homomorphic envelope")
axes[1, 0].set_ylim(None, 0.03)


corr = np.correlate(envelope, envelope, mode="full")
corr = corr[len(envelope) :]
corr = corr / corr[0]

t_corr = np.arange(len(corr)) / fs
axes[2, 0].plot(t_corr, corr, c=COLOR)
axes[2, 0].set_xlim(0, 3)
axes[2, 0].set_xlabel("Lag (seconds)")
axes[2, 0].axvline(60 / 180, ls=":", c=COLOR)
axes[2, 0].axvline(60 / 30, ls=":", c=COLOR)
axes[2, 0].set_ylabel("Normalised autocorrelation")
axes[2, 0].set_ylim(None, 1)


min_index = int(60 / 180 * fs)
max_index = int(60 / 30 * fs)
peak_index = min_index + np.argmax(corr[min_index:max_index])

axes[2, 0].plot(
    t_corr[peak_index], corr[peak_index], marker="o", ls="none", c=COLOR, label="Selected peak"
)
print(60 * fs / peak_index)
axes[2, 0].legend()


pos = np.loadtxt("final_model/85203_AV_posterior.csv", delimiter=",")
t_pos = np.arange(pos.shape[0]) / 50

axes[1, 1].set_ylim(0, 1.01)
axes[1, 1].set_ylabel("Non-diastolic probability")

non_dia_pos = pos[:, 0] + pos[:, 1] + pos[:, 2] + pos[:, 4]
axes[1, 1].plot(t_pos, non_dia_pos, c=COLOR)
axes[1, 1].set_xlim(T0, T1)
axes[1, 1].set_xlabel("Time (seconds)")

corr2 = np.correlate(non_dia_pos, non_dia_pos, mode="full")
corr2 = corr2[len(non_dia_pos) :]
corr2 /= corr2[0]
t_corr2 = np.arange(len(corr2)) / 50
axes[2, 1].plot(t_corr2, corr2, c=COLOR)
axes[2, 1].set_xlim(0, 3)
axes[2, 1].axvline(60 / 180, ls=":", c=COLOR)
axes[2, 1].axvline(60 / 30, ls=":", c=COLOR)
axes[2, 1].set_ylim(0.5, 1)
axes[2, 1].set_xlabel("Lag (seconds)")
axes[2, 1].set_ylabel("Normalised autocorrelation")

min_index = int(60 / 180 * 50)
max_index = int(60 / 30 * 50)
peak_index2 = min_index + np.argmax(corr2[min_index:max_index])
axes[2, 1].plot(
    t_corr2[peak_index2], corr2[peak_index2], marker="o", c=COLOR, ls="none", label="Selected peak"
)
print((60 * 50 / peak_index2))
axes[2, 1].legend()

fig.align_ylabels()

plt.savefig("results/figures/heart_rate_estimate.png", bbox_inches="tight")

import matplotlib.pyplot as plt
import numpy as np
import utils

font = {"size": 8, "family": "Arial"}
plt.rc("font", **font)

sseries = [
    ["49628_MV.wav", 8, 11],
    ["50111_PV.wav", 1.9, 4.9],
    ["85294_MV.wav", 9.4, 12.4],
    ["85196_TV.wav", 4, 7],
]
fig, axes = plt.subplots(
    len(sseries), 1, figsize=(5.2, 3), dpi=300, gridspec_kw={"hspace": 0}, sharex="all"
)

LETTERS = "abcd"
for i, (ax, (series, t0, t1)) in enumerate(zip(axes, sseries)):
    rec, fs = utils.load_recording(series)
    rec = (rec - np.mean(rec)) / np.max(np.abs(rec))
    rec = rec[int(t0 * fs) : int(t1 * fs)]

    t_series = np.arange(len(rec)) / fs
    ax.plot(t_series, rec, c="#545454", lw=1)
    ax.set_xlim(0, t_series[-1])
    ax.set_yticks([])
    ax.set_ylabel("Amplitude")
    ax.text(2.99, np.max(rec) * 0.99, series[:-4], size=6, va="top", ha="right")
    ax.text(0.01, np.max(rec) * 0.99, f"({LETTERS[i]})", size=6, va="top", ha="left")

axes[-1].set_xlabel("Time (seconds)")

fig.tight_layout()
plt.savefig("results/figures/sample_recordings.png")

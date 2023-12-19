import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

font = {"size": 8, "family": "Arial"}
plt.rc("font", **font)

model_folder = pathlib.Path("final_model/").resolve()
recordings_df = pd.read_csv(model_folder / "recordings.csv", index_col=0)
recordings_df["diff"] = recordings_df["holo_HSMM"] - recordings_df["healthy_HSMM"]

bins = np.arange(-0.125, 0.3 + 0.025, 0.025)
recordings_df["diff_bin"] = pd.cut(recordings_df["diff"], bins)

r = []
for group, _df in recordings_df.groupby("diff_bin"):
    _df = _df[_df["rec_murmur_label"] != "Unknown"]
    sens = (_df["rec_murmur_label"] == "Present").mean()
    r.append(sens)

fig, axes = plt.subplots(figsize=(3, 2.5), dpi=300)

axes.bar(bins[:-1], r, edgecolor="k", width=0.025, align="edge", facecolor="#6983ac")
axes.set_ylim(0, 1)
axes.set_xlabel("$C^{(M-N)}$")
axes.set_ylabel("Relative frequency of murmurs")

axes.set_xlim(bins[0], bins[-1])
axes.plot([-0.075, 0.27], (0, 1), c="k", ls="--")

plt.savefig("results/figures/reliability_diagram.png", bbox_inches="tight")

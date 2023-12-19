import pathlib

import matplotlib.pyplot as plt
import pandas as pd

model_folder = pathlib.Path("final_model/").resolve()

recordings_df = pd.read_csv(model_folder / "recordings.csv", index_col=0)
recordings_df["diff"] = recordings_df["holo_HSMM"] - recordings_df["healthy_HSMM"]
recordings_df["max"] = recordings_df[["holo_HSMM", "healthy_HSMM"]].max(axis=1)
recordings_df.index = recordings_df.index.str.split("_", expand=True)

font = {"size": 8, "family": "Arial"}
plt.rc("font", **font)

plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",  # i need upright \micro symbols, but you need...
    r"\sisetup{detect-all}",  # ...this to force siunitx to actually use your fonts
    r"\usepackage{helvet}",  # set the normal font here
    r"\usepackage{sansmath}",  # load up the sansmath so that math -> helvet
    r"\sansmath",  # <- tricky! -- gotta actually tell tex to use!
]

fig, axes = plt.subplots(figsize=(5.2, 4), dpi=300)
axes.set_facecolor("white")
fig.set_facecolor("white")
style = {
    "Absent": ["#8cb464", "x", "Murmur absent"],
    "Present": ["#ff392e", "o", "Murmur present"],
    "Unknown": ["gray", "d", "Unknown (poor signal qual.)"],
}

for group in ["Absent", "Present", "Unknown"]:
    sub_df = recordings_df[recordings_df["rec_murmur_label"] == group]
    print(group, len(sub_df))
    axes.scatter(
        sub_df["max"],
        sub_df["diff"],
        color=style[group][0],
        label=style[group][2],
        marker=style[group][1],
        s=5,
        alpha=1,
    )

MURMUR_THRESHOLD = 0.0  # 36842
SQ_THRESHOLD = 0.65

axes.axhline(MURMUR_THRESHOLD, ls="--", c="gray", lw=1, label="Murmur threshold")
axes.axvline(SQ_THRESHOLD, ls=":", c="gray", lw=1, label="Signal quality threshold")
axes.set_xlabel("$C^{(\hat{\omega})}$")
axes.set_ylabel("$C^{(M-N)}$")

axes.set_xlim(0.3, 1)
axes.set_ylim(-0.15, 0.3)
axes.legend(loc="upper left")

patient_results = {}
for index, sub_df in recordings_df.groupby(level=0):
    unknown_recs = sub_df["max"] < SQ_THRESHOLD

    # result = None
    murmur_recs = sub_df["diff"] > MURMUR_THRESHOLD
    if murmur_recs.any():
        result = "Present"
    elif unknown_recs.any():
        result = "Unknown"
    else:
        result = "Absent"

    patient_results[index] = {"prediction": result, "label": sub_df.patient_murmur_label[0]}

axes.grid(alpha=0.2)

fig.tight_layout()
plt.savefig("results/figures/confidences_2D.png")

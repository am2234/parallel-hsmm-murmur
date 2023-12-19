import pandas as pd
import matplotlib.pyplot as plt

font = {"size": 8, "family": "Arial"}
plt.rc("font", **font)

outcome_df = pd.read_csv("results/official_outcome_scores.tsv", sep="\t", index_col=0)

fig, axes = plt.subplots(figsize=(4, 4), dpi=300)
axes.plot(
    outcome_df.iloc[1:]["Cost on Training Set"],
    outcome_df.iloc[1:]["Cost on Test Set"],
    "x",
    ms=3,
    mew=1.5,
    c="gray",
    label="Other entries",
)
axes.plot(
    outcome_df.iloc[0]["Cost on Training Set"],
    outcome_df.iloc[0]["Cost on Test Set"],
    "o",
    ms=4,
    c="gray",
    label="Parallel HSMM",
)
axes.axhline(13004, ls=":", c="gray", label="Optimum random classifier")
axes.set_xlim(5000, 17000)
axes.set_ylim(5000, 17000)
axes.plot([5000, 17000], [5000, 17000], ls="--", c="gray", label="Equal training and test score")
axes.legend()

axes.set_xlabel("Training score")
axes.set_ylabel("Test score")
axes.arrow(8200, 8800, -2000, 2000, width=30, head_width=200, facecolor="k", edgecolor="none")
axes.text(5500, 8500, "Increasing\noverfitting")

axes.arrow(14000, 12000, 0, -2000, width=30, head_width=200, facecolor="k", edgecolor="none")
axes.text(14300, 10800, "Improving\ntest score")

axes.grid(alpha=0.2)

plt.savefig("results/figures/outcome_scores.png", bbox_inches="tight")

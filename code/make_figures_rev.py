"""Revision figures: calibration-only interval validation and scaffold dependence."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = sys.argv[1]; HERE = sys.argv[2]
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8883"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})

# ---- scaffold dependence ----
res = json.load(open(f"{HERE}/decision_sims.json"))["C_scaffold"]
sd = [r["cluster_sd"] for r in res]
fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.9), sharey=True)
for ax, key, title in zip(axes, ["same clusters", "new clusters"], ["calibration set drawn from\nthe screening library", "calibration set from\nother scaffolds"]):
    for out, col, mk, lab in [("false leads", ORANGE, "o", "false leads"), ("missed actives", BLUE, "s", "missed in-class"), ("deferred", GRAY, "^", "deferred")]:
        ax.plot(sd, [r[key][out] for r in res], color=col, marker=mk, ms=4, lw=1.6, label=lab)
    ax.axhline(0.95, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_title(title, fontsize=7.5, color=INK); ax.set_xticks(sd); ax.set_ylim(0.55, 1.0)
    ax.set_xlabel("scaffold effect (SD)")
axes[0].set_ylabel("interval coverage")
axes[1].legend(fontsize=6.3, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_scaffold.pdf"); fig.savefig(f"{OUT}/fig5_scaffold.png", dpi=300)

# ---- calibration-only intervals ----
try:
    co = json.load(open(f"{HERE}/calib_only.json"))
except FileNotFoundError:
    sys.exit(0)
labels = ["correct rejection", "false lead", "deferred (out-class)", "confident hit", "missed in-class", "deferred (in-class)"]
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.3), sharey=True, gridspec_kw={"width_ratios": [1, 1.3]})
y = np.arange(len(labels))
for j, (o, col, mk) in enumerate(zip(co, [AQUA, BLUE], ["s", "o"])):
    axes[0].scatter([r["coverage_class_conditional"] for r in o["rows"]], y + (j - 0.5) * 0.25, color=col, marker=mk, s=18,
                    edgecolors="white", linewidths=0.6, zorder=3, label=f"n = {o['n_cal']}")
axes[0].axvline(0.95, color=INK2, lw=0.8, ls=(0, (3, 2))); axes[0].set_xlim(0.90, 1.005)
axes[0].set_xticks([0.90, 0.95, 1.0]); axes[0].set_xlabel("coverage of 95% intervals (per class)")
axes[0].set_yticks(y); axes[0].set_yticklabels(labels); axes[0].invert_yaxis()
axes[0].legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.2)
o = co[-1]
wc = [r["width_calibration_only_mixed"] for r in o["rows"]]; ws = [r["width_split_budget"] for r in o["rows"]]
axes[1].barh(y - 0.17, wc, height=0.32, color=BLUE, label="all labels for calibration")
axes[1].barh(y + 0.17, ws, height=0.32, color=GRAY, label="half for a test set")
axes[1].set_xlabel(f"width per 1000, mixed batch ({o['n_cal']} labels)"); axes[1].set_xlim(0, 150)
axes[1].legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3)
fig.subplots_adjust(left=0.20, right=0.98, top=0.86, bottom=0.2, wspace=0.18); fig.savefig(f"{OUT}/fig4_calibration_only.pdf"); fig.savefig(f"{OUT}/fig4_calibration_only.png", dpi=300)
print("ok")

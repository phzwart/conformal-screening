"""Main-text figures 1 and 3. Usage: python3 -I make_figures.py FIGDIR validation.json"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta
OUT = sys.argv[1]
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b5b3ad"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})

# ---------- Figure 1: thresholds -> three prediction sets ----------
u = np.linspace(0, 1, 600); pi = 0.15
f0 = (1 - pi) * beta.pdf(u, 2, 7); f1 = pi * beta.pdf(u, 9, 3)
lo, hi = 0.27, 0.58          # 1 - tau_1 and tau_0
fig, ax = plt.subplots(figsize=(3.35, 1.9))
for a, b, lab in [(0, lo, "{out-class}"), (lo, hi, "both labels"), (hi, 1, "{in-class}")]:
    ax.axvspan(a, b, color={"{out-class}": "#e8f0fb", "both labels": "#efeeea", "{in-class}": "#fcebe3"}[lab], lw=0)
    ax.text((a + b) / 2, 3.12, lab, ha="center", va="top", fontsize=6.2, color=INK)
ax.plot(u, f0, color=BLUE, lw=1.6); ax.plot(u, f1, color=ORANGE, lw=1.6)
ax.text(0.40, 1.45, "out-class", color=INK, fontsize=7)
ax.text(0.83, 0.75, "in-class", color=INK, fontsize=7, ha="center")
for x, t in [(lo, r"$1-\tau_1$"), (hi, r"$\tau_0$")]:
    ax.axvline(x, color=INK2, lw=0.8, ls=(0, (3, 2))); ax.text(x + 0.01, 2.55, t, ha="left", va="center", fontsize=7, color=INK2)
ax.set_xlim(0, 1); ax.set_ylim(0, 3.2); ax.set_yticks([])
ax.set_xlabel("model probability that the compound is in-class")
ax.set_ylabel("density × prevalence")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_prediction_sets.pdf"); fig.savefig(f"{OUT}/fig1_prediction_sets.png", dpi=300)

# ---------- Figure 3 (main text Fig. 2): interval validation ----------
res = json.load(open(sys.argv[2]))
kpis = res[0]["kpis"]
PURPLE = "#8a5cc2"
fig, axes = plt.subplots(1, len(res), figsize=(7.0, 2.1), sharey=True)
meth = [("exact", "audit interval (Beta-Binomial)", BLUE, "o"), ("bayes1", "Bayesian, uniform prior", AQUA, "s"),
        ("cpxm", "m × Clopper–Pearson", PURPLE, "D")]   # plug-in Binomial (44-93%) reported in the text and ESI only
for ax, r in zip(axes, res):
    ax.axvline(0.95, color=INK2, lw=0.8, ls=(0, (3, 2)))
    for j, (key, lab, col, mk) in enumerate(meth):
        y = np.arange(len(kpis)) + (j - 1) * 0.22
        ax.scatter(r[key + "_coverage"], y, color=col, marker=mk, s=14, label=lab, zorder=3,
                   edgecolors="white", linewidths=0.5)
    ax.set_title(f"audit set n = {r['n_test']}", fontsize=8, color=INK)
    ax.set_xlim(0.84, 1.0); ax.set_xticks([0.85, 0.90, 0.95, 1.0])
    ax.set_xlabel("coverage of future-batch count")
LAB = {"single-label": "single label (any)", "single-label error": "false lead +\nmissed in-class", "both labels": "deferred:\nboth labels",
       "empty set": "deferred:\nempty set", "miscoverage": "true label\nnot in set"}
axes[0].set_yticks(range(len(kpis))); axes[0].set_yticklabels([LAB.get(k, k) for k in kpis], fontsize=6.5); axes[0].invert_yaxis()
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=6.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.0))
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(f"{OUT}/fig3_interval_validation.pdf"); fig.savefig(f"{OUT}/fig3_interval_validation.png", dpi=300)
print("ok")

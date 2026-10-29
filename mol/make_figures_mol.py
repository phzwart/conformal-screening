"""Molecular figures: Tox21 outcome tables (Fig. 6) and solubility planning (Fig. 7)."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
RES, FIG = sys.argv[1], sys.argv[2]
BLUE, ORANGE, AQUA, GRAY, LIGHT, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#b5b3ad", "#dcdad4", "#6250d6"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
# ---------------- Fig. 6: Tox21 ----------------
import os
if os.path.exists(f"{RES}/tox21_results_v2.json"):
    T = json.load(open(f"{RES}/tox21_results_v2.json"))
    E = {t: v["random"]["batch"] for t, v in T["endpoints"].items() if "random" in v}
    order = sorted(E, key=lambda t: -E[t]["single_fraction"]["median"])
    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    segs = [("correct_rejection", BLUE, "correct rejection"), ("confident_hit", AQUA, "confident hit"), ("false_lead", ORANGE, "false lead"),
            ("missed", VIOLET, "missed in-class"), ("deferred_both", GRAY, "deferred: both labels"), ("empty", "#f2c94c", "deferred: empty set")]
    for i, t in enumerate(order):
        left = 0
        for key, col, lab in segs:
            v = E[t][key]["median"]
            ax.barh(i, max(v - 1.5, 0), left=left + 0.75, color=col, height=0.62, label=lab if i == 0 else None); left += v
        sf = E[t]["single_fraction"]; ax.plot([1000 * sf["p5"], 1000 * sf["p95"]], [i - 0.42, i - 0.42], color=INK, lw=0.9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([t.replace("gamma", "γ") for t in order]); ax.invert_yaxis(); ax.set_xlim(0, 1000)
    ax.set_xlabel("compounds per 1000 (median of splits)")
    fig.legend(ncol=3, fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.0), handlelength=1, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(f"{FIG}/fig6_tox21.pdf"); fig.savefig(f"{FIG}/fig6_tox21.png", dpi=300)
# ---------------- Fig. 7: solubility (sweep + scenario maps); ESI: cost map, CALiSol scenario maps ----------------
try: S = json.load(open(f"{RES}/aqsol_results.json"))
except FileNotFoundError: sys.exit(0)
SM = json.load(open(f"{RES}/scenario_map.json")) if os.path.exists(f"{RES}/scenario_map.json") else None
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
GREEN = "#1baf7a"
def scenario_panel(ax, ds, sname, title, show_legend=False):
    D_ = SM[ds]; al = np.array(SM["alphas"]); G = {k: {q: np.array(v) for q, v in d.items()} for k, d in D_["grid"].items()}
    sc_ = D_["scenarios"][sname]; ok = np.array(sc_["feasible_mask"])
    im = ax.pcolormesh(al, al, G["false lead"]["point"].T, cmap="Greys", shading="nearest", vmin=0, vmax=max(250, G["false lead"]["point"].max()))
    ax.pcolormesh(al, al, np.ma.masked_where(ok.T == 0, ok.T), cmap=ListedColormap([BLUE]), alpha=0.6, shading="nearest")
    styles = {"deferred": (ORANGE, "-"), "confirmed in-class": (GREEN, "-"), "missed in-class": (VIOLET, "--")}
    for k, (side, v) in sc_["limits"].items():
        col, ls = styles[k]; ax.contour(al, al, G[k]["bound"].T, levels=[v], colors=[col], linestyles=[ls], linewidths=1.3)
    b = sc_["best"]
    if b: ax.plot([b["alpha0"]], [b["alpha1"]], marker="*", ms=11, color="#f2c94c", mec=INK, mew=0.6)
    lim = sc_["limits"]
    ax.set_title(f"{title}: assays ≤ {lim['deferred'][1]}\nconfirmed ≥ {lim['confirmed in-class'][1]}, missed ≤ {lim['missed in-class'][1]}", fontsize=6.6)
    ax.set_xlabel("out-class request $\\alpha_0$"); ax.set_ylabel("in-class request $\\alpha_1$")
    if show_legend:
        ax.figure.legend(handles=[Line2D([], [], color=ORANGE, lw=1.3, label="assay limit"), Line2D([], [], color=GREEN, lw=1.3, label="yield limit"),
                           Line2D([], [], color=VIOLET, lw=1.3, ls="--", label="miss limit"), Patch(color=BLUE, alpha=0.45, label="feasible"),
                           Line2D([], [], marker="*", ls="", color="#f2c94c", mec=INK, ms=9, label="fewest false leads")],
                  fontsize=6, frameon=False, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    return im
pts = S["all_points"]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.15, 1, 1]})
ax = axes[0]
sc = ax.scatter([p["missed"] for p in pts], [p["deferred"] for p in pts], c=[p["false_lead"] for p in pts], cmap="Blues", s=12,
                edgecolors=INK2, linewidths=0.3, vmin=0)
for key, lab in (("setting_A_fewest_missed", "A"), ("setting_B_fewest_deferred", "B")):
    s_ = S[key]; x, y = s_["missed"]["test_per1000"], s_["deferred_insoluble"]["test_per1000"] + s_["deferred_soluble"]["test_per1000"]
    ax.scatter([x], [y], s=55, facecolors="none", edgecolors=ORANGE, linewidths=1.4); ax.annotate(lab, (x, y), xytext=(5, 4), textcoords="offset points", color=INK, fontsize=8)
cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.set_label("false leads per 1000", fontsize=6.5)
ax.set_xlabel("missed soluble per 1000 (audit)"); ax.set_ylabel("sent to measurement per 1000"); ax.set_title("audited consequences of every setting", fontsize=6.8)
if SM:
    scenario_panel(axes[1], "aqsol", "loose", "scenario 1", show_legend=True)
    im = scenario_panel(axes[2], "aqsol", "tight", "scenario 2")
    cb = fig.colorbar(im, ax=axes[2], fraction=0.05, pad=0.02); cb.set_label("expected false leads per 1000", fontsize=6.5)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f"{FIG}/fig7_solubility.pdf"); fig.savefig(f"{FIG}/fig7_solubility.png", dpi=300)
# ESI: cost map
fig, ax = plt.subplots(figsize=(3.6, 2.7))
C = S["costmap"]; lam, rho = np.array(C["lambda"]), np.array(C["rho"]); B = np.array(C["best_rule_index"])
defer = np.array([r["deferred"] for r in C["rules"]])
im = ax.pcolormesh(lam, rho, defer[B], cmap="Greys", shading="nearest", vmin=0, vmax=1000)
ok_pt, ok_rb = np.array(C["convention_optimal_point"]), np.array(C["convention_optimal_robust"])
ax.pcolormesh(lam, rho, np.ma.masked_where(ok_pt == 0, ok_pt), cmap=ListedColormap([BLUE]), alpha=0.35, shading="nearest")
if ok_rb.any(): ax.pcolormesh(lam, rho, np.ma.masked_where(ok_rb == 0, ok_rb), cmap=ListedColormap([ORANGE]), alpha=0.9, shading="nearest")
ax.set_xscale("log"); ax.set_xlabel("cost of a missed soluble compound\n(relative to a wasted follow-up)"); ax.set_ylabel("cost of a measurement\n(relative to a wasted follow-up)")
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("deferred per 1000 (cheapest setting)", fontsize=7)
ax.legend(handles=[Patch(color=BLUE, alpha=0.35, label="acting on the sets is expected-cost\noptimal (audited within-set fractions)"),
                   Patch(color=ORANGE, alpha=0.9, label="also for all fractions in\n95% audit intervals")], fontsize=5.6, frameon=True, framealpha=0.95, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/figS2_costmap.pdf"); fig.savefig(f"{FIG}/figS2_costmap.png", dpi=300)
# ESI: CALiSol scenario maps
if SM:
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.0))
    scenario_panel(axes[0], "calisol", "loose", "electrolytes, scenario 1", show_legend=True)
    im = scenario_panel(axes[1], "calisol", "tight", "electrolytes, scenario 2")
    cb = fig.colorbar(im, ax=axes[1], fraction=0.05, pad=0.02); cb.set_label("expected false leads per 1000", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(f"{FIG}/figS3_calisol_scenarios.pdf"); fig.savefig(f"{FIG}/figS3_calisol_scenarios.png", dpi=300)
print("ok")

# ---------------- Fig. 9: sizing a round ----------------
if os.path.exists(f"{RES}/sizing.json"):
    S = json.load(open(f"{RES}/sizing.json"))
    rows = S["required"]["20"]
    rates = [o["rate"] for o in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    ax.plot(rates, [o["expectation"] for o in rows], ls="--", color=GRAY, lw=1.6,
            marker="o", ms=3, label="expectation, target/rate")
    for n, col in zip((49, 100, 384, 1000), (ORANGE, VIOLET, BLUE, AQUA)):
        ax.plot(rates, [o[f"n{n}"] for o in rows], color=col, lw=1.6, marker="o", ms=3,
                label=f"audit $n={n}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("in-class rate of the audited level set")
    ax.set_ylabel("candidates needed for 20 hits")
    ax.set_xticks(rates); ax.set_xticklabels([f"{100*r:g}%" for r in rates])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_title("guaranteed yield needs a larger batch\nthan the expectation suggests",
                 fontsize=8, color=INK)

    ax = axes[1]
    for m, col, lab in ((50, ORANGE, "batch 50"), (500, BLUE, "batch 500")):
        v = S["variance_share"][str(m)]
        ax.plot([o["audit"] for o in v], [100 * o["epistemic_share"] for o in v],
                color=col, lw=1.6, marker="o", ms=3, label=lab)
        ax.axvline(m, color=col, ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("audit size $n$")
    ax.set_ylabel("share of interval variance\nfrom the audit (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_title("beyond $n\\approx m$ the batch, not the audit,\nlimits the interval",
                 fontsize=8, color=INK)

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig9_sizing.pdf"); fig.savefig(f"{FIG}/fig9_sizing.png", dpi=300)
    print("wrote fig9_sizing")

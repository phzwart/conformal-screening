"""Presentation slide: the model looks fine in every split, and the guarantee does not.

Draws the library holdout experiment as one viewgraph.  Left, how often the predicted
outcome counts actually contained what the batch delivered, by split design, against the
nominal level.  Right, the model's own diagnostics for the same runs: its ranking of the
batch, and whether the enrichment of its confident calls over the base rate reached
significance.  The point of putting them side by side is that the right-hand panel is what a
practitioner normally inspects, and it looks healthy everywhere.

Usage: python3 -I slide_holdout.py RUNS.jsonl OUTSTEM
"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS, STEM = sys.argv[1], sys.argv[2]
CATS = ("confident hit", "false lead", "missed binder", "deferred binder",
        "correct rejection", "deferred non-binder")
ORDER = [("random", "random\nassignment"), ("block3", "withhold\nposition 3"),
         ("block2", "withhold\nposition 2"), ("block1", "withhold\nposition 1")]
GREEN, ORANGE, RED, INK, INK2, GRAY = "#1baf7a", "#eb8c34", "#d6453a", "#0b0b0b", "#52514e", "#b5b3ad"
COL = {"random": GREEN, "block3": ORANGE, "block2": ORANGE, "block1": RED}

plt.rcParams.update({"font.size": 13, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


def main():
    rows = [json.loads(l) for l in open(RUNS)]
    rows = [r for r in rows if not r.get("skipped") and r["label"] == "activity"
            and r["score"] == "model"]
    data = {}
    for key, _ in ORDER:
        v = [r for r in rows if r["design"] == key]
        ins = tot = 0
        for r in v:
            g = [r["audit_intervals"][c]["inside"] for c in CATS]
            ins += sum(g); tot += len(g)
        data[key] = {"n": len(v), "cov": 100.0 * ins / tot, "ins": ins, "tot": tot,
                     "auroc": [r["auroc_future"] for r in v],
                     "sig": sum(1 for r in v if r.get("significant"))}

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), gridspec_kw={"width_ratios": [1, 1]})
    labels = [lab for _, lab in ORDER]
    x = np.arange(len(ORDER))

    ax = axes[0]
    vals = [data[k]["cov"] for k, _ in ORDER]
    ax.bar(x, vals, color=[COL[k] for k, _ in ORDER], width=0.62)
    ax.axhline(95, color=INK, ls="--", lw=1.6)
    ax.text(3.46, 96.5, "nominal 95%", ha="right", fontsize=12, color=INK)
    for i, (k, _) in enumerate(ORDER):
        d = data[k]
        ax.text(i, d["cov"] + 2.5, f"{d['cov']:.0f}%", ha="center", fontsize=15,
                color=INK, fontweight="bold")
        ax.text(i, 3, f"{d['ins']}/{d['tot']}", ha="center", fontsize=11, color="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 108); ax.set_ylabel("predicted outcome counts that\ncontained the batch")
    ax.set_title("the guarantee", fontsize=16, color=INK, pad=12)

    ax = axes[1]
    rng = np.random.default_rng(0)
    for i, (k, _) in enumerate(ORDER):
        a = np.array(data[k]["auroc"])
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(a)), a, s=46, color=COL[k],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.hlines(a.mean(), i - 0.3, i + 0.3, color=INK, lw=2.2, zorder=4)
        ax.text(i, 1.012, f"{a.min():.2f}–{a.max():.2f}", ha="center", fontsize=11, color=INK2)
    ax.axhline(0.5, color=GRAY, ls=":", lw=1.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0.45, 1.05); ax.set_ylabel("model ranking of the batch (AUROC)")
    ax.set_title("what the model reports", fontsize=16, color=INK, pad=12)
    nsig = sum(data[k]["sig"] for k, _ in ORDER); nrun = sum(data[k]["n"] for k, _ in ORDER)
    ax.text(0.5, 0.06, f"enrichment over base rate significant in {nsig} of {nrun} runs",
            transform=ax.transAxes, ha="center", fontsize=12.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="#f2f1ee", ec=GRAY, lw=0.8))

    fig.suptitle("A batch built on unseen building blocks is priced wrongly, "
                 "and the model does not say so", fontsize=17.5, color=INK, y=0.98)
    fig.text(0.5, 0.015, "one position-1 batch was promised 578–723 confirmed hits, "
             "at AUROC 0.89, and returned 119",
             ha="center", fontsize=13, color=RED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.935))
    fig.savefig(f"{STEM}.png", dpi=200); fig.savefig(f"{STEM}.pdf")
    print(f"wrote {STEM}.png and {STEM}.pdf")
    for k, lab in ORDER:
        d = data[k]
        print(f"  {k:8} n={d['n']:>2}  coverage {d['cov']:.0f}%  "
              f"AUROC {min(d['auroc']):.3f}-{max(d['auroc']):.3f}  sig {d['sig']}/{d['n']}")


if __name__ == "__main__":
    main()

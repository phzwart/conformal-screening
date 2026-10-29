"""Presentation slide in three layers: the library, the split, and what breaks.

The holdout result is hard to follow without knowing how a DNA-encoded library is built,
because the thing being withheld is not a set of molecules but a building block, and every
molecule carrying it leaves with it.  This draws the three layers in order.  First the
molecule: one block at each of three positions, assembled combinatorially and read out
through its DNA tag.  Then the split: assigning molecules at random scatters every block
across all four roles, while withholding a block takes a whole row of the library out at
once, so the batch is built on chemistry the calibration and audit sets never held.  Then
the consequence, with the model's own diagnostics beside it, since those are what a
practitioner would otherwise consult.

Usage: python3 -I slide_layers.py RUNS.jsonl OUTSTEM
"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, RegularPolygon, FancyArrowPatch

RUNS, STEM = sys.argv[1], sys.argv[2]
CATS = ("confident hit", "false lead", "missed binder", "deferred binder",
        "correct rejection", "deferred non-binder")
ORDER = [("random", "random\nassignment"), ("block3", "withhold\nposition 3"),
         ("block2", "withhold\nposition 2"), ("block1", "withhold\nposition 1")]
GREEN, ORANGE, RED, BLUE, VIOLET = "#1baf7a", "#eb8c34", "#d6453a", "#2a78d6", "#6250d6"
INK, INK2, GRAY, LIGHT = "#0b0b0b", "#52514e", "#b5b3ad", "#e8e6e1"
COL = {"random": GREEN, "block3": ORANGE, "block2": ORANGE, "block1": RED}
ROLE = [("train", "#9fb9d9"), ("calibrate", "#86cdb4"), ("audit", "#f3b77e"), ("batch", "#d98b85")]

plt.rcParams.update({"font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


def draw_molecule(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off"); ax.set_aspect("equal")
    ax.plot([0.5, 3.1], [1.6, 1.6], color=GRAY, lw=3, solid_capstyle="round", zorder=1)
    for x in np.linspace(0.7, 2.9, 9):
        ax.plot([x, x], [1.38, 1.82], color=GRAY, lw=1.4, zorder=1)
    ax.text(1.8, 2.25, "DNA tag", ha="center", fontsize=11, color=INK2, style="italic")
    ax.add_patch(RegularPolygon((4.0, 1.6), 6, radius=0.45, orientation=np.pi / 6,
                                fc="white", ec=INK, lw=1.8, zorder=3))
    ax.plot([3.1, 3.55], [1.6, 1.6], color=INK, lw=1.8, zorder=2)
    spots = [((5.6, 2.55), BLUE, "block at\nposition 1", 306),
             ((5.6, 1.6), VIOLET, "block at\nposition 2", 754),
             ((5.6, 0.65), GREEN, "block at\nposition 3", 935)]
    for (xy, col, lab, n) in spots:
        ax.add_patch(FancyArrowPatch((4.45, 1.6), (xy[0] - 0.42, xy[1]),
                                     arrowstyle="-", color=INK, lw=1.6,
                                     connectionstyle="arc3,rad=0.12", zorder=2))
        ax.add_patch(Circle(xy, 0.4, fc=col, ec="white", lw=2, zorder=4))
        ax.text(xy[0] + 0.62, xy[1], f"{lab}   ({n} choices)", va="center",
                fontsize=11.5, color=INK)
    ax.text(9.9, 0.08, "306 × 754 × 935 ≈ 2 × 10⁸ molecules, one assay read per molecule",
            ha="right", fontsize=11, color=INK2)


def draw_grid(ax, mode, title, rng):
    nrow, ncol = 9, 22
    if mode == "random":
        roles = rng.integers(0, 4, size=(nrow, ncol))
    else:
        roles = np.repeat(rng.permutation(np.array([0, 0, 0, 0, 1, 1, 2, 3, 3]))[:, None],
                          ncol, axis=1)
    for i in range(nrow):
        for j in range(ncol):
            ax.add_patch(FancyBboxPatch((j, nrow - 1 - i), 0.86, 0.86,
                                        boxstyle="round,pad=0.0,rounding_size=0.12",
                                        fc=ROLE[roles[i, j]][1], ec="white", lw=0.8))
    ax.set_xlim(-0.3, ncol + 0.2); ax.set_ylim(-1.5, nrow + 0.9); ax.axis("off")
    ax.text(ncol / 2, nrow + 0.25, title, ha="center", fontsize=13.5, color=INK)
    ax.text(-0.2, nrow / 2, "blocks at\nposition 1", rotation=90, va="center", ha="right",
            fontsize=10.5, color=INK2)
    ax.text(ncol / 2, -0.75, "every pairing of the other two positions", ha="center",
            fontsize=10.5, color=INK2)


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

    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(14.5, 13.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.78, 1.0, 1.25], hspace=0.30, wspace=0.22,
                          left=0.07, right=0.97, top=0.925, bottom=0.075)

    ax = fig.add_subplot(gs[0, :]); draw_molecule(ax)
    ax.set_title("1.  what a molecule in the library is", loc="left", fontsize=16,
                 color=INK, pad=6)

    axl = fig.add_subplot(gs[1, 0]); draw_grid(axl, "random", "assign molecules at random", rng)
    axl.set_title("2.  how the library is divided", loc="left", fontsize=16, color=INK, pad=6)
    axr = fig.add_subplot(gs[1, 1]); draw_grid(axr, "block", "withhold whole blocks", rng)
    handles = [plt.Line2D([], [], marker="s", ls="", ms=12, mfc=c, mec="white", label=n)
               for n, c in ROLE]
    axr.legend(handles=handles, frameon=False, fontsize=11.5, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.30))
    axl.text(11, -1.35, "each block appears in all four roles", ha="center",
             fontsize=11.5, color=GREEN)
    axr.text(11, -1.35, "the batch holds chemistry the audit never saw", ha="center",
             fontsize=11.5, color=RED)

    x = np.arange(len(ORDER)); labels = [lab for _, lab in ORDER]
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(x, [data[k]["cov"] for k, _ in ORDER], color=[COL[k] for k, _ in ORDER], width=0.62)
    ax.axhline(95, color=INK, ls="--", lw=1.6)
    ax.text(3.46, 97, "nominal 95%", ha="right", fontsize=11.5, color=INK)
    for i, (k, _) in enumerate(ORDER):
        d = data[k]
        ax.text(i, d["cov"] + 2.5, f"{d['cov']:.0f}%", ha="center", fontsize=15,
                color=INK, fontweight="bold")
        ax.text(i, 3, f"{d['ins']}/{d['tot']}", ha="center", fontsize=10.5, color="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylim(0, 110); ax.set_ylabel("predicted counts that\ncontained the batch")
    ax.set_title("3.  what breaks — and what does not", loc="left", fontsize=16,
                 color=INK, pad=6)

    ax = fig.add_subplot(gs[2, 1])
    for i, (k, _) in enumerate(ORDER):
        a = np.array(data[k]["auroc"])
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(a)), a, s=46, color=COL[k],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.hlines(a.mean(), i - 0.3, i + 0.3, color=INK, lw=2.2, zorder=4)
        ax.text(i, 1.015, f"{a.min():.2f}–{a.max():.2f}", ha="center", fontsize=11, color=INK2)
    ax.axhline(0.5, color=GRAY, ls=":", lw=1.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylim(0.45, 1.06); ax.set_ylabel("model ranking of the batch (AUROC)")
    nsig = sum(data[k]["sig"] for k, _ in ORDER); nrun = sum(data[k]["n"] for k, _ in ORDER)
    ax.text(0.5, 0.05, f"enrichment over base rate significant in {nsig} of {nrun} runs",
            transform=ax.transAxes, ha="center", fontsize=12, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f2f1ee", ec=GRAY, lw=0.8))

    fig.suptitle("Withhold a building block and the guarantee fails, "
                 "while every model diagnostic stays healthy",
                 fontsize=19, color=INK, y=0.975)
    fig.text(0.5, 0.018, "one position-1 batch was promised 578–723 confirmed hits, "
             "at AUROC 0.89, and returned 119", ha="center", fontsize=13, color=RED)
    fig.savefig(f"{STEM}.png", dpi=170); fig.savefig(f"{STEM}.pdf")
    print(f"wrote {STEM}.png and {STEM}.pdf")


if __name__ == "__main__":
    main()

"""Summary, table rows and figure for the CA-DEL screen.
Usage: python3 -I cadel_summary.py RUNS.json OUT.json FIGDIR TABLEDIR"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1]))
runs = [r for r in R["runs"] if not r.get("skipped")]
ISO = list(R["isoforms"]); CUT = R["info"]["cutoffs"]; DES = R["info"]["designs"]
CORE = [d for d in DES if not d.startswith("N")]


def pb(ps):
    pmf = np.array([1.0])
    for p in ps: pmf = np.convolve(pmf, [1 - p, p])
    return pmf


def med(X, f):
    v = [f(r) for r in X]; v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


S = {"info": R["info"], "isoforms": R["isoforms"], "cells": {}}
for iso in ISO:
    for c in CUT:
        for des in DES:
            X = [r for r in runs if r["isoform"] == iso and r["cutoff"] == c and r["design"] == des]
            nsk = sum(1 for r in R["runs"] if r["isoform"] == iso and r["cutoff"] == c
                      and r["design"] == des and r.get("skipped"))
            o = {"runs": len(X), "skipped": nsk}
            if X:
                o["auroc_median"] = med(X, lambda r: r["auroc_audit"])
                o["prevalence_cal"] = float(np.mean([r["prevalence_cal"] for r in X]))
                o["prevalence_audit"] = float(np.mean([r["prevalence_audit"] for r in X]))
                o["n_cal_median"] = med(X, lambda r: sum(r["n_cal"]))
                o["n_audit_median"] = med(X, lambda r: sum(r["n_audit"]))
                o["n_future_median"] = med(X, lambda r: sum(r["n_future"]))
                for y, nm in ((0, "nonbinder"), (1, "binder")):
                    ps = [r["default"]["p_viol"][y] for r in X]
                    obs = int(sum(r["default"]["viol"][y] for r in X)); pmf = pb(ps)
                    o[f"violations_{nm}"] = {"observed": obs, "expected": float(sum(ps)),
                                             "p_upper": float(pmf[obs:].sum()),
                                             "cov_mean": float(np.mean([r["default"]["cov"][y] for r in X]))}
                    o[f"infeasible_{nm}"] = float(np.mean([not r["default"]["feasible"][y] for r in X]))
                keys = list(X[0]["audit_default"])
                o["audit_default"] = {k: float(np.mean([r["audit_default"][k] for r in X])) for k in keys}
                o["audit_sweep_pointwise"] = sum(r["audit_sweep"][0] for r in X) / sum(r["audit_sweep"][1] for r in X)
                o["bonferroni_all"] = float(np.mean([r["bonf_all"] for r in X]))
                o["default_per1000"] = {
                    k: med(X, lambda r, k=k: 1000 * r["audit_intervals"][k]["audit_count"] / sum(r["n_audit"]))
                    for k in ("confident hit", "false lead", "missed binder", "deferred")}
                o["plate_intervals"] = {
                    k: [med(X, lambda r, k=k: r["audit_intervals"][k]["per_plate"][0]),
                        med(X, lambda r, k=k: r["audit_intervals"][k]["per_plate"][1])]
                    for k in ("confident hit", "false lead", "missed binder", "deferred")}
                cap = [cc for r in X for cc in r["capacity"]]
                o["capacity"] = {"chosen": len(cap),
                                 "exceeded": float(np.mean([cc[2] for cc in cap])) if cap else None,
                                 "upper_median": float(np.median([cc[0] for cc in cap])) if cap else None}
                o["distinct_rules"] = [int(min(r["distinct_rules"] for r in X)), int(max(r["distinct_rules"] for r in X))]
                o["pareto"] = [int(min(r["pareto"] for r in X)), int(max(r["pareto"] for r in X))]
                if des == "D":
                    o["del_label_rule"] = X[0].get("del_label_rule")
            S["cells"][f"{iso}|c{c}|{des}"] = o

# primary cut-off: c = 8 unless its calibration in-class rate in design P is outside [5%, 50%]
prev_P = {c: float(np.mean([S["cells"][f"{iso}|c{c}|P"]["prevalence_cal"]
                            for iso in ISO if S["cells"][f"{iso}|c{c}|P"]["runs"]])) for c in CUT}
if 0.05 <= prev_P[8] <= 0.50:
    PRIMARY = 8; why = f"c=8 (Ki <= 10 nM): mean in-class rate in the P calibration sets {prev_P[8]:.3f}, inside [0.05, 0.50]"
else:
    PRIMARY = min(CUT, key=lambda c: abs(prev_P[c] - 0.25))
    why = (f"c=8 rejected (in-class rate {prev_P[8]:.3f} outside [0.05, 0.50]); "
           f"c={PRIMARY} chosen as the cut-off closest to 25% prevalence ({prev_P[PRIMARY]:.3f})")
S["primary_cutoff"] = PRIMARY; S["primary_cutoff_reason"] = why; S["prevalence_cal_P_by_cutoff"] = prev_P
json.dump(S, open(sys.argv[2], "w"), indent=1)
print(json.dumps({"primary_cutoff": PRIMARY, "reason": why, "prevalence": prev_P}, indent=1))
for key, o in S["cells"].items():
    if o["runs"]:
        print(key, "auroc", None if o["auroc_median"] is None else round(o["auroc_median"], 3),
              "viol nb/b", f"{o['violations_nonbinder']['observed']}/{o['violations_nonbinder']['expected']:.1f}",
              f"{o['violations_binder']['observed']}/{o['violations_binder']['expected']:.1f}",
              "audit", f"{100 * min(o['audit_default'].values()):.0f}-{100 * max(o['audit_default'].values()):.0f}%",
              "sweep", f"{100 * o['audit_sweep_pointwise']:.1f}%")
    else:
        print(key, "all", o["skipped"], "repeats skipped")

# table rows: one row per isoform x design at the primary cut-off
rows = [f"% primary cut-off c = {PRIMARY}: {why}"]
LAB = {"P": "physical pilot", "D": "DEL calibration", "S": "scaffold series"}
for iso in ISO:
    for des in CORE:
        o = S["cells"][f"{iso}|c{PRIMARY}|{des}"]
        if not o["runs"]:
            rows.append(f"{iso} & {LAB[des]} & \\multicolumn{{6}}{{c}}{{all repeats skipped}} \\\\"); continue
        vb, vn = o["violations_binder"], o["violations_nonbinder"]
        rows.append(f"{iso} & {LAB[des]} & {o['auroc_median']:.2f} & {o['prevalence_audit']:.2f} & "
                    f"{vn['observed']}/{vn['expected']:.1f} & {vb['observed']}/{vb['expected']:.1f} & "
                    f"{100 * min(o['audit_default'].values()):.0f}--{100 * max(o['audit_default'].values()):.0f} & "
                    f"{100 * o['audit_sweep_pointwise']:.1f} \\\\")
open(f"{sys.argv[4]}/tab_cadel_rows.tex", "w").write("\n".join(rows) + "\n")

# figure: per-class violation fraction (observed vs exact law) and audit-interval coverage
BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
keys = [(iso, des) for iso in ISO for des in CORE]
x = np.arange(len(keys))
for j, (nm, col, lab) in enumerate((("nonbinder", BLUE, "non-binder class"), ("binder", ORANGE, "binder class"))):
    obs, exp = [], []
    for iso, des in keys:
        o = S["cells"][f"{iso}|c{PRIMARY}|{des}"]
        obs.append(o[f"violations_{nm}"]["observed"] / o["runs"] if o["runs"] else np.nan)
        exp.append(o[f"violations_{nm}"]["expected"] / o["runs"] if o["runs"] else np.nan)
    axes[0].bar(x + (j - 0.5) * 0.38, obs, width=0.36, color=col, label=f"{lab}, observed")
    axes[0].scatter(x + (j - 0.5) * 0.38, exp, color="k", marker="_", s=110, zorder=3,
                    label="exact law" if j else None)
axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("repeats with coverage < 90%")
axes[0].set_xticks(x); axes[0].set_xticklabels([f"{i}\n{d}" for i, d in keys], fontsize=6)
axes[0].legend(fontsize=6.3, frameon=False, loc="upper left")
axes[0].set_title(f"class-conditional coverage, c = {PRIMARY}", fontsize=7)
for j, (nm, col) in enumerate((("pointwise", BLUE), ("joint (Bonferroni)", ORANGE))):
    v = []
    for iso, des in keys:
        o = S["cells"][f"{iso}|c{PRIMARY}|{des}"]
        if not o["runs"]: v.append(np.nan); continue
        v.append(o["audit_sweep_pointwise"] if nm == "pointwise" else o["bonferroni_all"])
    axes[1].bar(x + (j - 0.5) * 0.38, v, width=0.36, color=col, label=nm)
axes[1].axhline(0.95, color=INK, lw=0.9, ls="--")
axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("audit intervals covering the future batch")
axes[1].set_xticks(x); axes[1].set_xticklabels([f"{i}\n{d}" for i, d in keys], fontsize=6)
axes[1].legend(fontsize=6.3, frameon=False, loc="lower left")
axes[1].set_title("sweep of coverage requests", fontsize=7)
fig.tight_layout(); fig.savefig(f"{sys.argv[3]}/figS4_cadel.pdf"); fig.savefig(f"{sys.argv[3]}/figS4_cadel.png", dpi=300)
print("ok")

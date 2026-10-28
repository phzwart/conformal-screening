"""Summary, table rows and figure for the CALiSol-23 electrolyte screen.
Usage: python3 -I calisol_summary.py RUNS.json OUT.json FIGDIR TABLEDIR"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1])); runs = [r for r in R["runs"] if not r.get("skipped")]
def pb(ps):
    pmf = np.array([1.0])
    for p in ps: pmf = np.convolve(pmf, [1 - p, p])
    return pmf
S = {"info": R["info"], "designs": {}}
for des in dict.fromkeys(r["design"] for r in runs):
    X = [r for r in runs if r["design"] == des]; o = {"runs": len(X), "skipped": sum(1 for r in R["runs"] if r["design"] == des and r.get("skipped"))}
    o["auroc_median"] = float(np.median([r["auroc_audit"] for r in X if r["auroc_audit"] is not None])); o["prevalence_audit"] = float(np.mean([r["prevalence_audit"] for r in X]))
    o["n_cal_good_median"] = float(np.median([r["n_cal"][1] for r in X])); o["n_audit_median"] = float(np.median([sum(r["n_audit"]) for r in X]))
    for y, name in ((0, "poor"), (1, "good")):
        ps = [r["default"]["p_viol"][y] for r in X]; obs = int(sum(r["default"]["viol"][y] for r in X)); pmf = pb(ps)
        o[f"violations_{name}"] = {"observed": obs, "expected": float(sum(ps)), "p_upper": float(pmf[obs:].sum())}
        o[f"infeasible_{name}"] = float(np.mean([not r["default"]["feasible"][y] for r in X]))
    keys = list(X[0]["audit_default"]); o["audit_default"] = {k: float(np.mean([r["audit_default"][k] for r in X])) for k in keys}
    o["audit_sweep_pointwise"] = sum(r["audit_sweep"][0] for r in X) / sum(r["audit_sweep"][1] for r in X)
    o["bonferroni_all"] = float(np.mean([r["bonf_all"] for r in X]))
    cap = [c for r in X for c in r["capacity"]]
    o["capacity"] = {"chosen": len(cap), "exceeded": float(np.mean([c[2] for c in cap])) if cap else None,
                     "upper_median": float(np.median([c[0] for c in cap])) if cap else None, "realized_median": float(np.median([c[1] for c in cap])) if cap else None}
    o["distinct_rules"] = [int(min(r["distinct_rules"] for r in X)), int(max(r["distinct_rules"] for r in X))]
    o["pareto"] = [int(min(r["pareto"] for r in X)), int(max(r["pareto"] for r in X))]
    o["calonly"] = sum(r["calonly"][0] for r in X) / sum(r["calonly"][1] for r in X)
    o["cov_good_mean"] = float(np.mean([r["default"]["cov"][1] for r in X])); o["cov_poor_mean"] = float(np.mean([r["default"]["cov"][0] for r in X]))
    S["designs"][des] = o
json.dump(S, open(sys.argv[2], "w"), indent=1); print(json.dumps(S, indent=1))
# table rows: one row per design
lab = {"random": "random formulations", "system": "new salt--solvent systems", "salt": "new salt, calibrated on others", "salt_pilot": "new salt, pilot calibration"}
rows = []
for des, o in S["designs"].items():
    vg, vp = o["violations_good"], o["violations_poor"]
    rows.append(f"{lab[des]} & {o['auroc_median']:.2f} & {vp['observed']}/{vp['expected']:.1f} & {vg['observed']}/{vg['expected']:.1f} & "
                f"{100 * min(o['audit_default'].values()):.1f}--{100 * max(o['audit_default'].values()):.1f} & {100 * o['audit_sweep_pointwise']:.1f} & {100 * o['bonferroni_all']:.1f} \\\\")
open(f"{sys.argv[4]}/tab_calisol_rows.tex", "w").write("\n".join(rows) + "\n")
# figure: audited sweep of one random split, and per-class coverage of the good class by design
BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.2, 1]})
r0 = next(r for r in runs if r["design"] == "random"); P = np.array(r0["sweep_points"])
sc = axes[0].scatter(P[:, 0], P[:, 2], c=P[:, 1], cmap="Blues", s=14, edgecolors=INK2, linewidths=0.3, vmin=0)
cb = fig.colorbar(sc, ax=axes[0], fraction=0.05, pad=0.02); cb.set_label("false leads per 1000", fontsize=7)
axes[0].set_xlabel("missed good electrolytes per 1000 (audit set)"); axes[0].set_ylabel("sent to measurement per 1000")
des = list(S["designs"]); x = np.arange(len(des))
for j, (key, col, nm) in enumerate((("poor", BLUE, "poor class"), ("good", ORANGE, "good class"))):
    obs = [S["designs"][d][f"violations_{key}"]["observed"] / S["designs"][d]["runs"] for d in des]
    exp = [S["designs"][d][f"violations_{key}"]["expected"] / S["designs"][d]["runs"] for d in des]
    axes[1].bar(x + (j - 0.5) * 0.38, obs, width=0.36, color=col, label=f"{nm}, observed")
    axes[1].scatter(x + (j - 0.5) * 0.38, exp, color="k", marker="_", s=120, zorder=3, label="exact law" if j else None)
for i, d_ in enumerate(des):
    axes[1].text(i, 1.06, f"audit\n{100 * S['designs'][d_]['audit_sweep_pointwise']:.0f}%", ha="center", va="bottom", fontsize=6.5, color=INK2)
axes[1].set_ylim(0, 1.25); axes[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[1].set_xticks(x); axes[1].set_xticklabels(["random", "new\nsystems", "new salt", "new salt,\npilot"], fontsize=7)
axes[1].set_ylabel("splits with coverage < 90%"); axes[1].legend(fontsize=6.3, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
fig.tight_layout(); fig.savefig(f"{sys.argv[3]}/fig8_calisol.pdf"); fig.savefig(f"{sys.argv[3]}/fig8_calisol.png", dpi=300)

"""Write ESI LaTeX tables for the cost grid and the calibration-only validation."""
import json, sys
HERE, OUT = sys.argv[1], sys.argv[2]
a = json.load(open(f"{HERE}/decision_sims.json"))["A_cost"]
rhos = sorted({r["rho"] for r in a}); geoms = []
for r in a:
    if r["geometry"] not in geoms: geoms.append(r["geometry"])
L = [r"\begin{table}[ht]", r"\centering",
     r"\caption{Expected cost relative to Chow's optimal rule (\%), constrained (coverage $\ge0.90$ for both classes) / default ($\alpha=0.10$ for both classes). $\lambda$: cost of a miss; $\rho$: cost of an assay; both relative to a false lead.}",
     r"\label{tab:cost}", r"\small", r"\begin{tabular}{ll" + "r" * len(rhos) + "}", r"\toprule",
     "Scores & $\\lambda$ & " + " & ".join(f"$\\rho={x}$" for x in rhos) + r" \\", r"\midrule"]
for g in geoms:
    for lam in sorted({r["lambda"] for r in a}):
        cells = []
        for rho in rhos:
            r = next(x for x in a if x["geometry"] == g and x["lambda"] == lam and x["rho"] == rho)
            cells.append(f"{r['excess_constrained_pct']:.0f} / {r['excess_default_pct']:.0f}")
        L.append(f"{g if lam == 2 else ''} & {lam} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule" if g != geoms[-1] else r"\bottomrule")
L += [r"\end{tabular}", r"\end{table}"]
open(f"{OUT}/tab_cost.tex", "w").write("\n".join(L))
# tab_calonly.tex is written by make_tables_v2.py

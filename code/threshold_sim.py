"""Threshold-correction simulation (main text Table 2, ESI Table S2).
Calibration scores and a future batch of m = 100 scores are drawn from |Cauchy|; for each
calibration size n and each threshold rule (standard, batch correction, one-sided DKW) we
record how often the fraction of future scores at or below the threshold falls below 1 - alpha.
Writes a JSON file and the two LaTeX tables.
Usage: python3 -I threshold_sim.py DRAWS OUT.json TABLE_DIR
       python3 -I threshold_sim.py tables OUT.json TABLE_DIR   (tables only)"""
import numpy as np, sys, os, json
from scipy.stats import beta, betabinom
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mol"))
from conformal import k_standard, k_batch, k_batchfree, k_dkw

A, D, M = 0.10, 0.10, 100
NS = (50, 75, 100, 150, 200, 250, 300, 500)

def simulate(n, ks, draws, rng, chunk=20000):
    x = int(np.ceil(round((1 - A) * M, 9))); bad = np.zeros(len(ks), dtype=np.int64); done = 0
    while done < draws:
        c = min(chunk, draws - done)
        cal = np.abs(rng.standard_cauchy((c, n))); fut = np.abs(rng.standard_cauchy((c, M)))
        part = np.partition(cal, sorted({k - 1 for k in ks}), axis=1)
        for i, k in enumerate(ks):
            covered = np.sum(fut <= part[:, k - 1:k], axis=1); bad[i] += np.sum(covered < x)
        done += c
    return bad / draws

def main(draws, out, tdir):
    rng = np.random.default_rng(20240501); res = []
    x = int(np.ceil(round((1 - A) * M, 9)))
    for n in NS:
        rules = {"standard": k_standard(n, A), "batch correction": k_batch(n, A, D, M), "DKW (one-sided)": k_dkw(n, A, D),
                 "batch-free correction": k_batchfree(n, A, D)}
        names = list(rules); obs = simulate(n, [rules[r] for r in names], draws, rng)
        for r, o in zip(names, obs):
            k = rules[r]; u = n + 1 - k
            res.append({"n": n, "rule": r, "u": u, "level": round(u / (n + 1), 4), "observed": float(o),
                        "theory_batch": float(betabinom.cdf(x - 1, M, k, n + 1 - k)), "theory_batchfree": float(beta.cdf(1 - A, k, n + 1 - k))})
        print(n, [(r["rule"], r["u"], round(r["observed"], 4)) for r in res[-4:]], flush=True)
    json.dump({"draws": draws, "m": M, "alpha": A, "delta": D, "rows": res}, open(out, "w"), indent=1)
    write_tables(res, draws, tdir)

def write_tables(res, draws, tdir):
    f3 = lambda v: f"{v:.3f}"
    lines = []
    for n in NS:
        first = True
        for r in [r for r in res if r["n"] == n and r["rule"] != "batch-free correction"]:
            lab = {"batch correction": "correction"}.get(r["rule"], r["rule"])
            lines.append(f"{n if first else '':>3} & {lab} & {r['u']} & {f3(r['level'])} & {f3(r['observed'])} & {f3(r['theory_batch'])} & {f3(r['theory_batchfree'])} \\\\")
            first = False
    open(os.path.join(tdir, "tab_grid_rows.tex"), "w").write("\n".join(lines) + "\n")
    main_rows = []
    for n in (50, 100, 500):
        sel = ["standard", "batch correction"] + (["batch-free correction"] if n in (100, 500) else [])
        first = True
        for r in [r for r in res if r["n"] == n and r["rule"] in sel]:
            main_rows.append(f"{n if first else '':<3} & {r['rule']} & {f3(r['observed'])} & {f3(r['theory_batch'])} \\\\"); first = False
    open(os.path.join(tdir, "tab_correction_rows.tex"), "w").write("\n".join(main_rows) + "\n")

if __name__ == "__main__":
    if sys.argv[1] == "tables":            # regenerate the tables from a stored JSON: tables OUT.json TABLE_DIR
        d = json.load(open(sys.argv[2])); write_tables(d["rows"], d["draws"], sys.argv[3])
    else:
        main(int(sys.argv[1]), sys.argv[2], sys.argv[3])

"""Tox21 analysis of the revision run (fresh seeds, randomized tie-breaking, batch correction
with class counts planned from the calibration prevalence). One fixed model throughout.
Class 1 = active (flagged toxic). Reports:
  - outcome tables per endpoint (Fig. 6) and a median-split representative table (Table 4);
  - per-class violation rates against their exact expectation, with Poisson-binomial tests;
  - calibration-only interval coverage on real data, per endpoint and pooled, class-conditional
    and for the mixed test batch;
  - widths of calibration-only against test-set intervals on real data;
  - pilot samples: actives, feasibility of the request, violations against the exact law, sizing rule;
  - a monotonicity diagnostic for the probability of activity as a function of the score;
  - identical fingerprints shared between calibration and test sets.
Usage: python3 -I analyze_tox21_v2.py RUNDIR OUTDIR"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import betabinom, binom, kendalltau
import rowint, intervals
from functools import lru_cache
rowint.thatcher = lru_cache(maxsize=None)(rowint.thatcher); rowint.rank_law = lru_cache(maxsize=None)(rowint.rank_law)   # exact limits, memoized
from rowint import calonly_class, calonly_mixed
thatcher = rowint.thatcher
RUN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
runs = [json.loads(l) for l in open(f"{RUN}/tox21_runs_v2.jsonl")]
tasks = list(dict.fromkeys(r["task"] for r in runs)); METHODS = ["standard", "batchfree", "batch", "dkw"]
A = D = 0.10; G = 0.05
q = lambda v, p: float(np.percentile(v, p)) if len(v) else None
DEF0, DEF1 = ("0_both", "0_empty"), ("1_both", "1_empty")

def per1000(rec, meth):
    t = rec["methods"][meth]["test"]; n = rec["m0"] + rec["m1"]
    return {"correct_rejection": 1000 * t["0_s0"] / n, "confident_hit": 1000 * t["1_s1"] / n, "false_lead": 1000 * t["0_s1"] / n,
            "missed": 1000 * t["1_s0"] / n, "deferred_both": 1000 * (t["0_both"] + t["1_both"]) / n,
            "empty": 1000 * (t["0_empty"] + t["1_empty"]) / n, "single_fraction": (t["0_s0"] + t["0_s1"] + t["1_s0"] + t["1_s1"]) / n}
def coverage(rec, meth):
    t = rec["methods"][meth]["test"]; return (t["0_s0"] + t["0_both"]) / rec["m0"], (t["1_s1"] + t["1_both"]) / rec["m1"]
def p_violation(n, k, m):
    """exact probability that coverage of m future compounds of the class falls below 1 - A, threshold = k-th of n"""
    if m <= 0: return 0.0
    x = int(np.ceil(round((1 - A) * m, 9))); return float(betabinom.cdf(x - 1, m, k, n + 1 - k))
def feasible(n, m):
    x = int(np.ceil(round((1 - A) * m, 9))); return bool(betabinom.sf(x - 1, m, n, 1) >= 1 - D)
def poisson_binomial(ps):
    pmf = np.array([1.0])
    for p in ps: pmf = np.convolve(pmf, [1 - p, p])
    return pmf
def viol_test(recs, meth):
    out = {}
    for y, (nk, kk, mk) in enumerate((("n0", "k0", "m0"), ("n1", "k1", "m1"))):
        ps = [p_violation(r[nk], r["methods"][meth][kk], r[mk]) for r in recs]
        obs = int(sum(coverage(r, meth)[y] < 0.9 for r in recs)); pmf = poisson_binomial(ps)
        out["inactive" if y == 0 else "active"] = {"n": len(recs), "observed": obs, "observed_rate": obs / max(1, len(recs)),
            "expected": float(sum(ps)), "expected_rate": float(np.mean(ps)) if ps else None,
            "p_upper": float(pmf[obs:].sum()), "p_lower": float(pmf[:obs + 1].sum())}
    return out
def calcov(recs, meth="batch"):
    hit = tot = 0; hm = tm = 0
    for rec in recs:
        b = rec["methods"][meth]
        for y, n, m in ((0, rec["n0"], rec["m0"]), (1, rec["n1"], rec["m1"])):
            for k, (lo, hi) in calonly_class(b, y, n, m).items(): hit += lo <= b["test"][k] <= hi; tot += 1
        for k, (lo, hi) in calonly_mixed(b, rec["n0"], rec["n1"], rec["m0"] + rec["m1"]).items(): hm += lo <= b["test"][k] <= hi; tm += 1
    return {"cells": tot, "coverage_class_conditional": hit / tot if tot else None, "coverage_mixed": hm / tm if tm else None}
CELLROWS = (("correct rejection", ["0_s0"]), ("false lead", ["0_s1"]), ("deferred inactive", list(DEF0)),
            ("confident hit", ["1_s1"]), ("missed active", ["1_s0"]), ("deferred active", list(DEF1)))
def table(rec, meth="batch", M=1000):
    b = rec["methods"][meth]; n = rec["m0"] + rec["m1"]; ci = calonly_mixed(b, rec["n0"], rec["n1"], M); rows = {}
    for name, keys in CELLROWS:
        k = sum(b["test"][x] for x in keys)
        rows[name] = {"test_per1000": round(1000 * k / n), "testset_interval": [int(v) for v in thatcher(k, n, M, G)],
                      "calonly_interval": [int(sum(ci.get(x, (0, 0))[0] for x in keys)), int(sum(ci.get(x, (0, 0))[1] for x in keys))]}
    return rows

res = {"n_runs": len(runs), "endpoints": {}}
rand = [r for r in runs if r["split"] == "random"]; scaf = [r for r in runs if r["split"] == "scaffold"]
for task in tasks:
    E = {}
    for split in ("random", "scaffold"):
        R = [r for r in runs if r["task"] == task and r["split"] == split]
        if not R: continue
        e = {"reps": len(R), "auroc_median": q([r["auroc_test"] for r in R], 50), "tied_fraction_median": q([r["tied_fraction"] for r in R], 50),
             "prevalence_test": float(np.mean([r["record"]["m1"] / (r["record"]["m0"] + r["record"]["m1"]) for r in R])),
             "n_cal_actives_median": q([r["record"]["n1"] for r in R], 50), "n_cal_median": q([r["record"]["n0"] + r["record"]["n1"] for r in R], 50),
             "n_test_median": q([r["record"]["m0"] + r["record"]["m1"] for r in R], 50)}
        for meth in METHODS:
            P = [per1000(r["record"], meth) for r in R]
            e[meth] = {k: {"median": q([p[k] for p in P], 50), "p5": q([p[k] for p in P], 5), "p95": q([p[k] for p in P], 95)} for k in P[0]}
            k = sum(r["record"]["methods"][meth]["test"]["1_s1"] for r in R); n = k + sum(r["record"]["methods"][meth]["test"]["0_s1"] for r in R)
            e[meth]["hit_rate_active_calls"] = k / n if n else None; e[meth]["enrichment"] = (k / n) / e["prevalence_test"] if n else None
        e["violations_batch"] = viol_test([r["record"] for r in R], "batch")
        e["calonly"] = calcov([r["record"] for r in R])
        E[split] = e
    res["endpoints"][task] = E
# pooled violations by method (random splits) with exact expectations
res["violations_random"] = {meth: viol_test([r["record"] for r in rand], meth) for meth in METHODS}
res["violations_scaffold_other"] = viol_test([r["record"] for r in scaf], "batch")
res["calonly_random"] = calcov([r["record"] for r in rand]); res["calonly_scaffold_other"] = calcov([r["record"] for r in scaf])
# pilots
pil = {}
for npil in ("200", "500", "1000"):
    P = [(r, r["pilot"].get(npil)) for r in scaf if npil in r.get("pilot", {})]
    recs = [p for _, p in P if p is not None]
    if not P: continue
    feas = [feasible(p["n1"], p["planned_m"][1]) for p in recs]
    pil[npil] = {"pilots": len(P), "too_few_actives_skipped": sum(p is None for _, p in P), "median_actives": q([p["n1"] for p in recs], 50),
                 "request_infeasible_fraction": float(1 - np.mean(feas)) if recs else None,
                 "violations": viol_test(recs, "batch"), "calonly": calcov(recs),
                 "violations_feasible_only": viol_test([p for p, f in zip(recs, feas) if f], "batch")}
res["pilots"] = pil
# sizing rule: smallest number of calibration compounds of a class for which (alpha, delta, m) is feasible
res["sizing"] = {str(m): next(n for n in range(1, 2000) if feasible(n, m)) for m in (20, 50, 100, 200, 500, 1000, 5000)}
res["sizing"]["batch_free"] = next(n for n in range(1, 2000) if 1 - (1 - A) ** n >= 1 - D)
# widths on real data: calibration-only vs test-set intervals, batch of 1000, random splits
W = {}
for task in tasks:
    R = [r for r in rand if r["task"] == task]
    if not R: continue
    W[task] = {}
    TB = [table(r["record"]) for r in R]
    for name, _ in CELLROWS:
        tw = [np.diff(tb[name]["testset_interval"])[0] for tb in TB]; cw = [np.diff(tb[name]["calonly_interval"])[0] for tb in TB]
        W[task][name] = {"testset_median": q(tw, 50), "calonly_median": q(cw, 50), "ratio_median": q([c / max(t, 1) for c, t in zip(cw, tw)], 50)}
res["widths_random"] = W
res["width_ratio_summary"] = {name: {"min": min(W[t][name]["ratio_median"] for t in W), "max": max(W[t][name]["ratio_median"] for t in W)} for name, _ in CELLROWS}
# representative endpoints, median split by single-label fraction
sf = {t: res["endpoints"][t]["random"]["batch"]["single_fraction"]["median"] for t in tasks if "random" in res["endpoints"][t]}
hi_t, lo_t = max(sf, key=sf.get), min(sf, key=sf.get); res["representative"] = {}
for t in (hi_t, lo_t):
    R = sorted([r for r in rand if r["task"] == t], key=lambda r: per1000(r["record"], "batch")["single_fraction"])
    r = R[(len(R) - 1) // 2]; rec = r["record"]
    res["representative"][t] = {"rep": r["rep"], "rank": f"{(len(R) - 1) // 2 + 1} of {len(R)}", "batch": table(rec), "n_cal": [rec["n0"], rec["n1"]], "n_test": [rec["m0"], rec["m1"]]}
# monotonicity of P(active | u): pooled test scores over random splits, 10 equal-count bins
mono = {}
for task in tasks:
    U, Y = [], []
    for r in rand:
        if r["task"] != task: continue
        f = f"{RUN}/scores/{task}_random_{r['rep']}.npz"
        if os.path.exists(f): z = np.load(f); U.append(z["u_test"]); Y.append(z["y_test"])
    if not U: continue
    U, Y = np.concatenate(U), np.concatenate(Y); edges = np.unique(np.quantile(U, np.linspace(0, 1, 11)))
    b = np.clip(np.searchsorted(edges, U, side="right") - 1, 0, len(edges) - 2)
    rate = np.array([Y[b == i].mean() for i in range(len(edges) - 1)]); cnt = np.array([np.sum(b == i) for i in range(len(edges) - 1)])
    drops = [(i, float(rate[i] - rate[i + 1])) for i in range(len(rate) - 1) if rate[i + 1] < rate[i]]
    # largest drop in standard errors
    z_drop = max([(rate[i] - rate[i + 1]) / np.sqrt(rate[i] * (1 - rate[i]) / cnt[i] + rate[i + 1] * (1 - rate[i + 1]) / cnt[i + 1] + 1e-12) for i, _ in drops], default=0.0)
    mono[task] = {"bins": len(rate), "active_fraction_by_bin": [round(float(x), 4) for x in rate], "decreases": len(drops),
                  "largest_decrease_z": float(z_drop), "kendall_tau": float(kendalltau(np.arange(len(rate)), rate)[0])}
res["monotonicity"] = mono
# identical fingerprints shared between calibration and test (random splits)
try:
    X = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tox21_features.npz"), allow_pickle=True)["X"]
    h = np.array([hash(np.packbits(x.astype(np.uint8)).tobytes()) for x in X]); shared = []
    for r in rand:
        f = f"{RUN}/scores/{r['task']}_random_{r['rep']}.npz"
        if not os.path.exists(f): continue
        z = np.load(f); hc = set(h[z["cal"]].tolist()); shared.append(np.mean([v in hc for v in h[z["test"]]]))
    res["shared_fingerprints_test_in_cal"] = {"median": q(shared, 50), "max": q(shared, 100)}
except FileNotFoundError:
    pass
json.dump(res, open(f"{OUT}/tox21_results_v2.json", "w"), indent=1)
show = {k: v for k, v in res.items() if k not in ("endpoints", "widths_random", "monotonicity", "representative")}
print(json.dumps(show, indent=1))
print("single-label fraction:", {t: round(v, 3) for t, v in sf.items()})
print("monotonicity:", {t: (m["decreases"], round(m["largest_decrease_z"], 2)) for t, m in mono.items()})

"""Real-data check of audit-set (test-set) prediction intervals, and of choosing a rule on
the audit set. The labelled compounds not used for training or calibration are split at
random, repeatedly, into an audit half and a future half. Thresholds come from the
calibration set; intervals for the future-half counts come from the audit half.
  Tox21: batch-correction thresholds of every random split, and of every scaffold split
         (calibration on other scaffolds; audit and future halves from the screened scaffolds).
  AqSolDB: every rule of the coverage-request sweep (log P > 3.5), pointwise and with a
         Bonferroni correction over the 144 requests; and a rule chosen on the audit half
         (fewest audit errors whose Bonferroni upper limit for deferrals is within capacity),
         checked on the future half.
Usage: python3 -I audit_check.py RUNDIR aqsol_runs.npz OUT.json HALVINGS"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import lru_cache
import intervals
from conformal import thresholds, cells, break_ties, k_batch
TH = lru_cache(maxsize=None)(intervals.thatcher)
RUN, NPZ, OUT, H = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); G = 0.05
GROUPS = {"correct rejection": ["0_s0"], "false lead": ["0_s1"], "deferred inactive": ["0_both", "0_empty"],
          "confident hit": ["1_s1"], "missed active": ["1_s0"], "deferred active": ["1_both", "1_empty"],
          "single label": ["0_s0", "0_s1", "1_s0", "1_s1"], "deferred": ["0_both", "0_empty", "1_both", "1_empty"]}
def grouped(c): return {g: sum(c[k] for k in ks) for g, ks in GROUPS.items()}
def check(ua, ya, uf, yf, t0, t1, g=G):
    a, f = grouped(cells(ua, ya, t0, t1)), grouped(cells(uf, yf, t0, t1)); na, nf = len(ya), len(yf)
    return {k: TH(a[k], na, nf, g)[0] <= f[k] <= TH(a[k], na, nf, g)[1] for k in GROUPS}
res = {"halvings_per_split": H}
# ---------------- Tox21 ----------------
runs = [json.loads(l) for l in open(f"{RUN}/tox21_runs_v2.jsonl")]
for split in ("random", "scaffold"):
    hit = {k: 0 for k in GROUPS}; joint = 0; tot = 0; per_task = {}
    for r in runs:
        if r["split"] != split: continue
        z = np.load(f"{RUN}/scores/{r['task']}_{split}_{r['rep']}.npz"); b = r["record"]["methods"]["batch"]
        rng = np.random.default_rng(r["seed"] + 77)
        uc, ut = break_ties(rng, z["u_cal"], z["u_test"]); t0, t1 = thresholds(uc, z["y_cal"], b["k0"], b["k1"]); yt = z["y_test"]
        for h in range(H):
            p = rng.permutation(len(yt)); A, F = p[: len(p) // 2], p[len(p) // 2:]
            c = check(ut[A], yt[A], ut[F], yt[F], t0, t1)
            for k, v in c.items(): hit[k] += v
            joint += all(c.values()); tot += 1
            pt = per_task.setdefault(r["task"], [0, 0]); pt[0] += sum(c.values()); pt[1] += len(c)
    res[f"tox21_{split}"] = {"checks": tot, "coverage": {k: v / tot for k, v in hit.items()}, "all_outcomes_together": joint / tot,
                             "per_task": {t: v[0] / v[1] for t, v in per_task.items()}}
    print(split, json.dumps(res[f"tox21_{split}"]), flush=True)
# ---------------- AqSolDB ----------------
D = np.load(NPZ); R = len([k for k in D.files if k.endswith("_cal_u")]); ALPHAS = np.round(np.arange(0.025, 0.301, 0.025), 3); J = len(ALPHAS) ** 2
DEF = ["0_both", "0_empty", "1_both", "1_empty"]
pt_hit = pt_tot = 0; bonf_all = bonf_tot = 0; cap_rows = []
for r in range(R):
    rng = np.random.default_rng(7000 + r); jc, jt = break_ties(rng, D[f"r{r}_cal_u"], D[f"r{r}_test_u"])
    sc, st = D[f"r{r}_cal_logP"] > 3.5, D[f"r{r}_test_logP"] > 3.5
    uc, yc, ut, yt = jc[sc], D[f"r{r}_cal_y"][sc], jt[st], D[f"r{r}_test_y"][st]
    n0, n1 = int(np.sum(yc == 0)), int(np.sum(yc == 1)); m1 = int(round(1000 * n1 / (n0 + n1))); m0 = 1000 - m1
    rules = {}
    for a0 in ALPHAS:
        for a1 in ALPHAS:
            key = (k_batch(n0, a0, 0.10, m0), k_batch(n1, a1, 0.10, m1))
            if key not in rules: rules[key] = thresholds(uc, yc, *key)
    rng2 = np.random.default_rng(9000 + r)
    for h in range(H):
        p = rng2.permutation(len(yt)); A, F = p[: len(p) // 2], p[len(p) // 2:]; nA, nF = len(A), len(F)
        allok = True; cand = []
        for (t0, t1) in rules.values():
            c = check(ut[A], yt[A], ut[F], yt[F], t0, t1); pt_hit += sum(c.values()); pt_tot += len(c)
            cb = check(ut[A], yt[A], ut[F], yt[F], t0, t1, G / J); allok &= all(cb.values())
            ca, cf = cells(ut[A], yt[A], t0, t1), cells(ut[F], yt[F], t0, t1)
            dA = sum(ca[k] for k in DEF); up = TH(dA, nA, nF, 2 * G / J)[1]          # one-sided upper limit at G/J
            if up <= 0.5 * nF: cand.append((ca["1_s0"] + ca["0_s1"], up, sum(cf[k] for k in DEF), cf["1_s0"] + cf["0_s1"]))
        bonf_all += allok; bonf_tot += 1
        if cand:
            e, up, dF, eF = min(cand); cap_rows.append({"upper": up, "future_deferred": dF, "exceeded": dF > 0.5 * nF, "future_errors": eF, "n_future": nF})
res["aqsol"] = {"pointwise_coverage": pt_hit / pt_tot, "cell_checks": pt_tot, "bonferroni_all_rules_all_outcomes": bonf_all / bonf_tot, "halvings": bonf_tot,
                "capacity_rule": {"found": len(cap_rows), "exceeded_fraction": float(np.mean([c["exceeded"] for c in cap_rows])) if cap_rows else None,
                                  "future_deferred_per1000_median": float(np.median([1000 * c["future_deferred"] / c["n_future"] for c in cap_rows])) if cap_rows else None,
                                  "upper_per1000_median": float(np.median([1000 * c["upper"] / c["n_future"] for c in cap_rows])) if cap_rows else None}}
print("aqsol", json.dumps(res["aqsol"]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)

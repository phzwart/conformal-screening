"""Calibration-only intervals, row-simultaneous version (as in the paper and molecular
analysis): for each class, the own-class rank-law interval and the cross-class interval are
each computed at level gamma/2, and every cell of that class's row is bounded by arithmetic
on the two, whichever regime the thresholds fall in. Mixed batches add an interval for the
number of future actives (gamma/2) and take the union over it.
Usage: python3 -I calib_only_rowsim.py REPS OUT.json"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_intervals import correction_index, draw
from rowint import calonly_class, calonly_mixed
from intervals import thatcher
CELLS = ["0_s0", "0_s1", "0_mid", "1_s1", "1_s0", "1_mid"]
NAMES = {"0_s0": "inactive {0} (correct rejection)", "0_s1": "inactive {1} (false lead)", "0_mid": "inactive deferred",
         "1_s1": "active {1} (confident hit)", "1_s0": "active {0} (missed active)", "1_mid": "active deferred"}
def counts(y, u, t0, t1):
    in0 = u <= t0; in1 = u >= 1 - t1; out = {}
    for c in (0, 1):
        s = y == c
        out[f"{c}_s0"] = int(np.sum(s & in0 & ~in1)); out[f"{c}_s1"] = int(np.sum(s & in1 & ~in0))
        out[f"{c}_both"] = int(np.sum(s & in0 & in1)); out[f"{c}_empty"] = int(np.sum(s & ~in0 & ~in1))
    return out
def key(k): return k[:2] + "mid" if k.endswith(("both", "empty")) else k
def run(n_cal, m, R, seed, g=0.05):
    rng = np.random.default_rng(seed); hc = dict.fromkeys(CELLS, 0); hm = dict.fromkeys(CELLS, 0)
    wm = dict.fromkeys(CELLS, 0.0); ws = dict.fromkeys(CELLS, 0.0); hs = dict.fromkeys(CELLS, 0); reps = 0; near = 0; skipped = 0; joint = 0
    for r in range(R):
        F = (rng.uniform(0.05, 0.5), rng.uniform(2, 10), rng.uniform(2, 4), rng.uniform(1.5, 3), rng.uniform(5, 9))
        yc, uc = draw(rng, F, n_cal); s0 = np.sort(uc[yc == 0]); s1 = np.sort(1 - uc[yc == 1]); n0, n1 = len(s0), len(s1)
        if n0 < 20 or n1 < 20: skipped += 1; continue
        k0, k1 = correction_index(n0), correction_index(n1); t0, t1 = s0[k0 - 1], s1[k1 - 1]
        rec = {"k0": k0, "k1": k1, "two_label": bool(t0 + t1 >= 1),
               "c0": int(np.sum((yc == 0) & (uc >= 1 - t1))), "c1": int(np.sum((yc == 1) & (uc <= t0)))}
        near += abs(t0 + t1 - 1) < 0.05
        yf, uf = draw(rng, F, m); fut = counts(yf, uf, t0, t1); mf = {0: int(np.sum(yf == 0)), 1: int(np.sum(yf == 1))}
        for y, n in ((0, n0), (1, n1)):
            for k, (lo, hi) in calonly_class(rec, y, n, mf[y]).items(): hc[key(k)] += lo <= fut[k] <= hi
        allok = True
        for k, (lo, hi) in calonly_mixed(rec, n0, n1, m).items(): ok = lo <= fut[k] <= hi; hm[key(k)] += ok; wm[key(k)] += hi - lo; allok &= ok
        joint += allok
        h = n_cal // 2; ya, ua, yb, ub = yc[:h], uc[:h], yc[h:], uc[h:]
        v0 = np.sort(ua[ya == 0]); v1 = np.sort(1 - ua[ya == 1]); q0 = v0[correction_index(len(v0)) - 1]; q1 = v1[correction_index(len(v1)) - 1]
        tt = counts(yb, ub, q0, q1); tf = counts(yf, uf, q0, q1)
        for c in (0, 1):
            for kk, ks in ((f"{c}_s0", [f"{c}_s0"]), (f"{c}_s1", [f"{c}_s1"]), (f"{c}_mid", [f"{c}_both", f"{c}_empty"])):
                lo, hi = thatcher(sum(tt[x] for x in ks), len(yb), m, g); ws[kk] += hi - lo; hs[kk] += lo <= sum(tf[x] for x in ks) <= hi
        reps += 1
    rows = [{"cell": NAMES[k], "coverage_class_conditional": round(hc[k] / reps, 4), "coverage_mixed_batch": round(hm[k] / reps, 4),
             "width_calibration_only_mixed": round(wm[k] / reps, 1), "width_split_budget": round(ws[k] / reps, 1),
             "coverage_split_budget": round(hs[k] / reps, 4)} for k in CELLS]
    return {"n_cal": n_cal, "m": m, "reps": reps, "skipped_fewer_than_20_per_class": skipped, "mixed_joint_all_cells": round(joint / reps, 4), "near_boundary_fraction": round(near / reps, 3), "rows": rows}
if __name__ == "__main__":
    R = int(sys.argv[1]); out = []
    for n_cal in (300, 1000):
        o = run(n_cal, 1000, R, seed=n_cal); out.append(o); print(json.dumps(o), flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)

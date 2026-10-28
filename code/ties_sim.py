"""Tied scores and the calibration-only intervals. Inactive scores u ~ Beta(2,7), except that a
fraction t of them share one value placed at the 0.92 quantile (a duplicate-fingerprint cluster);
actives ~ Beta(6,3). Coverage of the class-conditional row intervals for inactive and active
cells, with ties kept and with randomized tie-breaking (jitter applied jointly to calibration and
future scores). Usage: python3 -I ties_sim.py REPS OUT.json"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mol"))
from scipy.stats import beta
from rowint import calonly_class
from conformal import k_standard, thresholds, cross_counts, cells, break_ties
def draw(rng, n, pi, t):
    y = (rng.random(n) < pi).astype(int); u = np.where(y == 1, rng.beta(6, 3, n), rng.beta(2, 7, n))
    tie = (y == 0) & (rng.random(n) < t); u[tie] = beta.ppf(0.92, 2, 7); return y, u
def run(n, m, t, R, seed, pi=0.2):
    rng = np.random.default_rng(seed); out = {}
    for mode in ("ties kept", "randomized tie-breaking"):
        cell_hit = {}; cell_n = {}; row_hit = {0: 0, 1: 0}
        for r in range(R):
            yc, uc = draw(rng, n, pi, t); yf, uf = draw(rng, m, pi, t)
            if mode != "ties kept": uc, uf = break_ties(rng, uc, uf)
            n0, n1 = int(np.sum(yc == 0)), int(np.sum(yc == 1)); k0, k1 = k_standard(n0, 0.10), k_standard(n1, 0.10)
            t0, t1 = thresholds(uc, yc, k0, k1); c0, c1 = cross_counts(uc, yc, t0, t1)
            rec = {"k0": k0, "k1": k1, "t0": t0, "t1": t1, "two_label": bool(t0 + t1 >= 1), "c0": c0, "c1": c1}
            fut = cells(uf, yf, t0, t1)
            for y, ny in ((0, n0), (1, n1)):
                iv = calonly_class(rec, y, ny, int(np.sum(yf == y))); allok = True
                for k, (lo, hi) in iv.items():
                    ok = lo <= fut[k] <= hi; cell_hit[k] = cell_hit.get(k, 0) + ok; cell_n[k] = cell_n.get(k, 0) + 1; allok &= ok
                row_hit[y] += allok
        out[mode] = {"cells": {k: round(v / cell_n[k], 4) for k, v in cell_hit.items()}, "cell_n": cell_n, "row_joint": {str(k): round(v / R, 4) for k, v in row_hit.items()}}
    return out
if __name__ == "__main__":
    R = int(sys.argv[1]); res = []
    for t in (0.0, 0.04, 0.06, 0.10):
        o = {"tied_fraction": t, "n_cal": 1000, "m": 1000, **run(1000, 1000, t, R, seed=int(100 * t) + 1)}; res.append(o); print(json.dumps(o), flush=True)
    json.dump(res, open(sys.argv[2], "w"), indent=1)

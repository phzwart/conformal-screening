"""Class-conditional split conformal thresholds and outcome-table counts for binary
classifiers with probability scores u = p_hat(class 1 | x).
0 in C(x) iff u <= tau0 (tau0 = k0-th smallest of inactive calibration u);
1 in C(x) iff u >= 1 - tau1 (tau1 = k1-th smallest of active calibration 1-u)."""
import numpy as np
from scipy.stats import beta, betabinom

def k_standard(n, a):
    return int(min(n, np.ceil((n + 1) * (1 - a))))

def k_batchfree(n, a, d):
    us = np.arange(1, n + 1); ok = beta.sf(1 - a, n + 1 - us, us) >= 1 - d
    return n if not ok.any() else int(n + 1 - us[ok].max())

def k_batch(n, a, d, m):
    """least conservative k with P(batch coverage >= 1-a) >= 1-d, batch of m compounds of the class"""
    if m <= 0: return k_batchfree(n, a, d)
    x = int(np.ceil(round((1 - a) * m, 9)))          # integer arithmetic for ceil((1-a) m)
    us = np.arange(1, n + 1); ok = betabinom.sf(x - 1, m, n + 1 - us, us) >= 1 - d
    return n if not ok.any() else int(n + 1 - us[ok].max())

def k_dkw(n, a, d):
    eps = np.sqrt(np.log(1 / d) / (2 * n)); a2 = a - eps
    return n if a2 <= 0 else k_standard(n, a2)

def thresholds(u_cal, y_cal, k0, k1):
    s0 = np.sort(u_cal[y_cal == 0]); s1 = np.sort(1 - u_cal[y_cal == 1])
    return float(s0[k0 - 1]), float(s1[k1 - 1])

def cells(u, y, t0, t1):
    """counts of the 8 outcome cells: (class, set) with set in {0},{1},both,empty"""
    in0 = u <= t0; in1 = u >= 1 - t1
    out = {}
    for c in (0, 1):
        s = y == c
        out[f"{c}_s0"] = int(np.sum(s & in0 & ~in1)); out[f"{c}_s1"] = int(np.sum(s & in1 & ~in0))
        out[f"{c}_both"] = int(np.sum(s & in0 & in1)); out[f"{c}_empty"] = int(np.sum(s & ~in0 & ~in1))
    return out

def cross_counts(u_cal, y_cal, t0, t1):
    """calibration compounds of each class in the cross-class event:
       c0 = # inactive with u >= 1 - t1 ; c1 = # active with u <= t0"""
    return int(np.sum((y_cal == 0) & (u_cal >= 1 - t1))), int(np.sum((y_cal == 1) & (u_cal <= t0)))

def record(u_cal, y_cal, u_test, y_test, a=0.10, d=0.10):
    """all sufficient statistics for every threshold method"""
    n0, n1 = int(np.sum(y_cal == 0)), int(np.sum(y_cal == 1))
    m0, m1 = int(np.sum(y_test == 0)), int(np.sum(y_test == 1))
    ks = {"standard": (k_standard(n0, a), k_standard(n1, a)),
          "batchfree": (k_batchfree(n0, a, d), k_batchfree(n1, a, d)),
          "batch": (k_batch(n0, a, d, m0), k_batch(n1, a, d, m1)),
          "dkw": (k_dkw(n0, a, d), k_dkw(n1, a, d))}
    rec = {"n0": n0, "n1": n1, "m0": m0, "m1": m1, "methods": {}}
    for name, (k0, k1) in ks.items():
        t0, t1 = thresholds(u_cal, y_cal, k0, k1)
        c0, c1 = cross_counts(u_cal, y_cal, t0, t1)
        rec["methods"][name] = {"k0": k0, "k1": k1, "t0": t0, "t1": t1, "two_label": bool(t0 + t1 >= 1),
                                "c0": c0, "c1": c1, "test": cells(u_test, y_test, t0, t1)}
    return rec

def break_ties(rng, *arrays):
    """Randomized tie-breaking: add independent uniform jitter smaller than the smallest
    positive gap between distinct scores (over all arrays), so that all scores are distinct
    and every ordering of tied scores is equally likely. Makes the rank laws exact."""
    allu = np.concatenate(arrays); d = np.diff(np.unique(allu))
    eps = 1e-3 * (d[d > 0].min() if np.any(d > 0) else 1e-9)
    return tuple(a + eps * rng.random(len(a)) for a in arrays)

def planned_counts(n0, n1, m):
    """class counts of a planned batch of m compounds at the calibration prevalence"""
    m1 = int(round(m * n1 / (n0 + n1))); return m - m1, m1

def record_planned(u_cal, y_cal, u_test, y_test, a=0.10, d=0.10):
    """as record(), but the batch correction uses class counts planned from the calibration
    prevalence and the known batch size (not the realized test-set class counts)"""
    n0, n1 = int(np.sum(y_cal == 0)), int(np.sum(y_cal == 1)); m = len(y_test)
    p0, p1 = planned_counts(n0, n1, m)
    ks = {"standard": (k_standard(n0, a), k_standard(n1, a)),
          "batchfree": (k_batchfree(n0, a, d), k_batchfree(n1, a, d)),
          "batch": (k_batch(n0, a, d, p0), k_batch(n1, a, d, p1)),
          "dkw": (k_dkw(n0, a, d), k_dkw(n1, a, d))}
    rec = {"n0": n0, "n1": n1, "m0": int(np.sum(y_test == 0)), "m1": int(np.sum(y_test == 1)), "planned_m": [p0, p1], "methods": {}}
    for name, (k0, k1) in ks.items():
        t0, t1 = thresholds(u_cal, y_cal, k0, k1); c0, c1 = cross_counts(u_cal, y_cal, t0, t1)
        rec["methods"][name] = {"k0": k0, "k1": k1, "t0": t0, "t1": t1, "two_label": bool(t0 + t1 >= 1),
                                "c0": c0, "c1": c1, "test": cells(u_test, y_test, t0, t1)}
    return rec

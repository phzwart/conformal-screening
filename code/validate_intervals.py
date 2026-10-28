"""Validation of audit-set prediction intervals for future-batch counts.

Data are exchangeable but not i.i.d.: each replicate draws its own data-generating
distribution F (prevalence and class-conditional score laws), then calibration,
test and future-batch compounds are drawn i.i.d. from F (de Finetti mixture).
Class-conditional (Mondrian) split conformal thresholds use the finite-sample
coverage correction (Vovk 2012) at alpha = delta = 0.10.

Interval methods for the count K_m in a future batch of m compounds, given the
audit-set count k out of n:
  exact   : hypergeometric inversion = Beta-Binomial(m; k, n-k+1) lower,
            Beta-Binomial(m; k+1, n-k) upper   (Thatcher 1964)
  bayes1  : Beta-Binomial(m; k+1, n-k+1) equal-tailed (previous manuscript)
  plugin  : Binomial(m, k/n) quantiles (ignores the uncertainty of the audited rate)
  cpxm    : m times the Clopper-Pearson interval for the rate, rounded outward
            (ignores the sampling variation of the future batch)
"""
import numpy as np, json, sys
from scipy.stats import betabinom, binom, beta

G = 0.05          # 95% intervals
ALPHA = DELTA = 0.10

def correction_index(n, alpha=ALPHA, delta=DELTA):
    """largest u with P(Beta(n+1-u,u) >= 1-alpha) >= 1-delta; returns k=n+1-u"""
    us = np.arange(1, n + 1)
    ok = beta.sf(1 - alpha, n + 1 - us, us) >= 1 - delta
    if not ok.any():
        return n
    return n + 1 - us[ok].max()

def interval_tables(n, m):
    k = np.arange(n + 1)
    T = {}
    lo = np.where(k > 0, betabinom.ppf(G / 2, m, np.maximum(k, 1), n - k + 1), 0)
    hi = np.where(k < n, betabinom.ppf(1 - G / 2, m, k + 1, np.maximum(n - k, 1)), m)
    T["exact"] = (lo, hi)
    T["bayes1"] = (betabinom.ppf(G / 2, m, k + 1, n - k + 1),
                   betabinom.ppf(1 - G / 2, m, k + 1, n - k + 1))
    T["plugin"] = (binom.ppf(G / 2, m, k / n), binom.ppf(1 - G / 2, m, k / n))
    cl = np.where(k > 0, beta.ppf(G / 2, np.maximum(k, 1), n - k + 1), 0.0)
    cu = np.where(k < n, beta.ppf(1 - G / 2, k + 1, np.maximum(n - k, 1)), 1.0)
    T["cpxm"] = (np.floor(m * cl), np.ceil(m * cu))
    return T

KPIS = ["single-label", "single-label error", "both labels", "empty set", "miscoverage"]

def draw(rng, F, size):
    pi, a1, b1, a0, b0 = F
    y = (rng.random(size) < pi).astype(int)
    u = np.where(y == 1, rng.beta(a1, b1, size), rng.beta(a0, b0, size))  # P(class 1 | x)
    return y, u

def outcomes(y, u, t0, t1):
    in0 = u <= t0            # s(x,0) = u   (= 1 - P0)
    in1 = (1 - u) <= t1      # s(x,1) = 1 - u
    size = in0.astype(int) + in1
    single = size == 1
    covered = np.where(y == 1, in1, in0)
    return np.stack([single, single & ~covered, size == 2, size == 0, ~covered])

def longrun(F, t0, t1):
    pi, a1, b1, a0, b0 = F
    lo, hi = 1 - t1, t0
    def cells(a, b):
        cdf = lambda x: beta.cdf(np.clip(x, 0, 1), a, b)
        both = max(cdf(hi) - cdf(lo), 0.0)
        only0 = cdf(min(hi, lo))
        only1 = 1 - cdf(max(hi, lo))
        empty = max(cdf(lo) - cdf(hi), 0.0)
        return only0, only1, both, empty
    o0, o1, bt, em = cells(a0, b0)   # y = 0
    p0, p1, bt1, em1 = cells(a1, b1) # y = 1
    w0, w1 = 1 - pi, pi
    single = w0 * (o0 + o1) + w1 * (p0 + p1)
    err = w0 * o1 + w1 * p0
    both = w0 * bt + w1 * bt1
    empty = w0 * em + w1 * em1
    mis = w0 * (o1 + em) + w1 * (p0 + em1)
    return np.array([single, err, both, empty, mis])

def run(n_cal, n_test, m, R, seed):
    rng = np.random.default_rng(seed)
    T = interval_tables(n_test, m)
    hit = {k: np.zeros(len(KPIS)) for k in T}
    width = {k: np.zeros(len(KPIS)) for k in T}
    cp_hit = np.zeros(len(KPIS)); cp_cnt = 0
    for r in range(R):
        F = (rng.uniform(0.05, 0.5), rng.uniform(2, 10), rng.uniform(2, 4),
             rng.uniform(1.5, 3), rng.uniform(5, 9))
        yc, uc = draw(rng, F, n_cal)
        s0 = np.sort(uc[yc == 0]); s1 = np.sort(1 - uc[yc == 1])
        if len(s0) < 30 or len(s1) < 30:
            continue
        t0 = s0[correction_index(len(s0)) - 1]; t1 = s1[correction_index(len(s1)) - 1]
        yt, ut = draw(rng, F, n_test); yf, uf = draw(rng, F, m)
        k = outcomes(yt, ut, t0, t1).sum(1); km = outcomes(yf, uf, t0, t1).sum(1)
        for name, (lo, hi) in T.items():
            hit[name] += (km >= lo[k]) & (km <= hi[k])
            width[name] += hi[k] - lo[k]
        p = longrun(F, t0, t1)
        cl = np.where(k > 0, beta.ppf(G / 2, np.maximum(k, 1), n_test - k + 1), 0)
        cu = np.where(k < n_test, beta.ppf(1 - G / 2, k + 1, np.maximum(n_test - k, 1)), 1)
        cp_hit += (p >= cl) & (p <= cu); cp_cnt += 1
    out = {"n_cal": n_cal, "n_test": n_test, "m": m, "reps": cp_cnt, "kpis": KPIS}
    for name in T:
        out[name + "_coverage"] = (hit[name] / cp_cnt).round(4).tolist()
        out[name + "_mean_width"] = (width[name] / cp_cnt).round(2).tolist()
    out["clopper_pearson_longrun_coverage"] = (cp_hit / cp_cnt).round(4).tolist()
    return out

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    res = [run(1000, n, 1000, R, seed=n) for n in (100, 300, 1000)]
    json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "validation.json", "w"), indent=1)
    for o in res:
        print(o["n_test"], o["reps"])
        for i, kp in enumerate(KPIS):
            print(f"  {kp:20s} exact {o['exact_coverage'][i]:.3f} (w {o['exact_mean_width'][i]:6.1f})"
                  f"  bayes1 {o['bayes1_coverage'][i]:.3f} (w {o['bayes1_mean_width'][i]:6.1f})"
                  f"  plugin {o['plugin_coverage'][i]:.3f}  m*CP {o['cpxm_coverage'][i]:.3f} (w {o['cpxm_mean_width'][i]:6.1f})  CP-longrun {o['clopper_pearson_longrun_coverage'][i]:.3f}")

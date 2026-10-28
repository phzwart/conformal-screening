"""Finite-library check: exchangeable data that are NOT a mixture of i.i.d. draws.

For each of L fixed libraries of N = n_cal + n_test + m compounds (each library
drawn once from its own random distribution and then held fixed), we repeatedly
shuffle the library and split it into calibration / test / future batch, i.e.
sampling without replacement from a fixed finite population. Coverage of the
95% intervals for the future-batch count is computed per library (conditional on
the library), which isolates finite exchangeability from any i.i.d. structure."""
import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_intervals import correction_index, interval_tables, outcomes, KPIS, draw

def run(n_cal, n_test, m, L, P, seed):
    rng = np.random.default_rng(seed); N = n_cal + n_test + m
    T = interval_tables(n_test, m)
    per_lib = {k: [] for k in T}
    for l in range(L):
        F = (rng.uniform(0.05, 0.5), rng.uniform(2, 10), rng.uniform(2, 4),
             rng.uniform(1.5, 3), rng.uniform(5, 9))
        y, u = draw(rng, F, N)                     # the fixed library
        hit = {k: np.zeros(len(KPIS)) for k in T}
        for p in range(P):
            idx = rng.permutation(N); c, t, f = idx[:n_cal], idx[n_cal:n_cal+n_test], idx[n_cal+n_test:]
            s0 = np.sort(u[c][y[c] == 0]); s1 = np.sort(1 - u[c][y[c] == 1])
            t0 = s0[correction_index(len(s0)) - 1]; t1 = s1[correction_index(len(s1)) - 1]
            k = outcomes(y[t], u[t], t0, t1).sum(1); km = outcomes(y[f], u[f], t0, t1).sum(1)
            for name, (lo, hi) in T.items():
                hit[name] += (km >= lo[k]) & (km <= hi[k])
        for name in T: per_lib[name].append((hit[name] / P).tolist())
    out = {"n_cal": n_cal, "n_test": n_test, "m": m, "libraries": L, "shuffles": P, "kpis": KPIS}
    for name in T:
        a = np.array(per_lib[name])            # L x KPI
        out[name + "_mean"] = a.mean(0).round(4).tolist()
        out[name + "_min_over_libraries"] = a.min(0).round(4).tolist()
    return out

if __name__ == "__main__":
    L, P = int(sys.argv[1]), int(sys.argv[2])
    res = [run(1000, n, 1000, L, P, seed=10 + n) for n in (100, 300)]
    json.dump(res, open(sys.argv[3], "w"), indent=1)
    for o in res:
        print("n_test", o["n_test"])
        for i, kp in enumerate(KPIS):
            print(f"  {kp:20s} exact mean {o['exact_mean'][i]:.3f} min {o['exact_min_over_libraries'][i]:.3f} | "
                  f"bayes1 mean {o['bayes1_mean'][i]:.3f} min {o['bayes1_min_over_libraries'][i]:.3f} | "
                  f"plugin mean {o['plugin_mean'][i]:.3f}")

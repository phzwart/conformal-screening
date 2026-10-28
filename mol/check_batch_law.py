"""Real-score check of the batch coverage law on Tox21.
One CatBoost model per endpoint (random half for training). The other half is re-split
5000 times into calibration and test halves with class counts held fixed (the batch law
is conditional on class counts). Batch-correction thresholds (alpha = delta = 0.10) are
then fixed order statistics; we record how often per-class test coverage falls below 90%
and compare with the exact law (Poisson-binomial reduces to binomial here)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catboost import CatBoostClassifier
from scipy.stats import betabinom, binom
from conformal import k_batch
d = np.load("tox21_features.npz", allow_pickle=True); X, Y, tasks = d["X"], d["Y"], list(d["tasks"])
REPS = int(sys.argv[1]); out = {}
for j, t in enumerate(tasks):
    m = ~np.isnan(Y[:, j]); Xj, yj = X[m], Y[m, j].astype(int); rng = np.random.default_rng(42 + j)
    p = rng.permutation(len(yj)); h = len(p) // 2; tr, pool = p[:h], p[h:]
    clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0, thread_count=2, random_seed=42 + j).fit(Xj[tr], yj[tr])
    u = clf.predict_proba(Xj[pool])[:, 1]; y = yj[pool]; res = {}
    for c in (0, 1):
        s = u[y == c] if c == 0 else 1 - u[y == c]; N = len(s); n = N // 2; mm = N - n
        k = k_batch(n, 0.10, 0.10, mm); x = int(np.ceil(round(0.9 * mm, 9))); theory = float(betabinom.cdf(x - 1, mm, k, n + 1 - k))
        v = 0
        for rep in range(REPS):
            q = rng.permutation(N); cal = np.sort(s[q[:n]]); test = s[q[n:]]
            v += np.sum(test <= cal[k - 1]) < x
        res[f"class{c}"] = {"n_cal": n, "m_test": mm, "k": k, "observed": v / REPS, "theory": theory,
                            "p_excess": float(binom.sf(v - 1, REPS, theory)), "tied_fraction": float(1 - len(np.unique(s)) / N)}
    out[t] = res; print(t, json.dumps(res), flush=True)
    json.dump(out, open("results/batch_law_check.json", "w"), indent=1)

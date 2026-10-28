"""AqSolDB: binary soluble (log S >= -4) vs insoluble, CatBoost on Morgan fingerprints.
Stratified random split 70/15/15 (train/calibration/test); fixed hyperparameters, no early
stopping (so no tuning set). The model is trained on all training compounds; predicted
probabilities for every calibration and test compound are saved with log P, so that the
lipophilic restriction (log P > 3.5) and all sweeps are done in analysis.
Usage: python3 -I run_aqsol.py N_REPEATS OUT.npz"""
import sys, os, numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(f"{HERE}/aqsoldb_features.npz", allow_pickle=True)
X, logS, logP = d["X"], d["logS"], d["logP"]; y = (logS >= -4).astype(int)
R, OUT = int(sys.argv[1]), sys.argv[2]; res = {}
for r in range(R):
    rng = np.random.default_rng(7000 + r); tr, ca, te = [], [], []
    for c in (0, 1):                                   # stratified
        idx = rng.permutation(np.where(y == c)[0]); a, b = int(0.70 * len(idx)), int(0.85 * len(idx))
        tr += list(idx[:a]); ca += list(idx[a:b]); te += list(idx[b:])
    tr, ca, te = map(np.array, (tr, ca, te))
    clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0, thread_count=1, random_seed=7000 + r)
    clf.fit(X[tr], y[tr])
    for name, idx in (("cal", ca), ("test", te)):
        res[f"r{r}_{name}_u"] = clf.predict_proba(X[idx])[:, 1]; res[f"r{r}_{name}_y"] = y[idx]; res[f"r{r}_{name}_logP"] = logP[idx]
        res[f"r{r}_{name}_idx"] = idx
    lip = logP[te] > 3.5
    print(r, "AUROC all", round(roc_auc_score(y[te], res[f"r{r}_test_u"]), 3),
          "lipophilic", round(roc_auc_score(y[te][lip], res[f"r{r}_test_u"][lip]), 3),
          "n_cal_lip", int(np.sum(logP[ca] > 3.5)), "n_test_lip", int(lip.sum()),
          "soluble frac lip", round(float(y[te][lip].mean()), 3), flush=True)
np.savez_compressed(OUT, **res)

"""Can a pre-flight test tell whether calibration, audit and batch are exchangeable?

Exchangeability is required between the calibration set, the audit set and the batch, and
not between any of those and the training set, which only supplies a frozen score.  The
arrangement that satisfies it while still measuring a model on unseen chemistry is therefore
to split at the training boundary and to draw calibration, audit and batch from the
deployment chemistry at random.  Whether a given arrangement satisfies it is a question that
can be asked before the batch is made, because it concerns the structures rather than their
labels.

Two label-free statistics are computed here for each split design.  A domain classifier is
trained to tell calibration compounds from batch compounds using fingerprints alone; under
exchangeability the two are draws from one distribution, so no classifier can separate them
and its area under the curve sits at one half.  Anything appreciably above that is evidence
the intervals do not apply, and the test is one-sided: failing to separate them is weak
evidence of exchangeability, since a classifier that cannot see a difference may only lack
the features for it, while separating them is strong evidence against.  Alongside it, the
share of batch molecules whose building block at each position also occurs in the
calibration set measures the overlap directly, which is the quantity the design controls.

The two are compared against the coverage each design actually achieved, so the question is
whether a test run before the batch would have predicted the failure that followed.

Usage: python3 -I exch_check.py PARQUET RUNS.jsonl OUT.json [REPS]
"""
import sys, os, json, numpy as np, pandas as pd

PARQUET, RUNS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
NSUB = 200000
FRACS = (0.40, 0.20, 0.20)
COLS = ("bb1_id", "bb2_id", "bb3_id")
DESIGNS = ("random", "block3", "block2", "block1")
CATS = ("confident hit", "false lead", "missed binder", "deferred binder",
        "correct rejection", "deferred non-binder")


def fingerprints(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s.replace("[Dy]", "[H]"))
        if m is not None:
            X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


def split_random(rng, n):
    p = rng.permutation(n)
    a = int(FRACS[0] * n); b = a + int(FRACS[1] * n); c = b + int(FRACS[2] * n)
    return p[:a], p[a:b], p[b:c], p[c:]


def split_block(rng, bb, pos):
    u = np.unique(bb[:, pos]); rng.shuffle(u)
    a = int(FRACS[0] * len(u)); b = a + int(FRACS[1] * len(u)); c = b + int(FRACS[2] * len(u))
    r = np.empty(len(u), int)
    r[:a], r[a:b], r[b:c], r[c:] = 0, 1, 2, 3
    role = dict(zip(u.tolist(), r.tolist()))
    assign = np.array([role[v] for v in bb[:, pos]])
    return [np.where(assign == k)[0] for k in range(4)]


def domain_auc(X, ci, fi, rng, cap=12000):
    """how well a classifier separates calibration compounds from batch compounds"""
    c = rng.choice(ci, min(cap, len(ci)), replace=False)
    f = rng.choice(fi, min(cap, len(fi)), replace=False)
    idx = np.concatenate([c, f]); y = np.r_[np.zeros(len(c)), np.ones(len(f))]
    p = rng.permutation(len(idx)); idx, y = idx[p], y[p]
    half = len(idx) // 2
    clf = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, verbose=0,
                             thread_count=2, random_seed=0).fit(X[idx[:half]], y[:half])
    return float(roc_auc_score(y[half:], clf.predict_proba(X[idx[half:]])[:, 1]))


def main():
    b = pd.read_parquet(PARQUET)
    rng0 = np.random.default_rng(0)
    sub = rng0.choice(len(b), min(NSUB, len(b)), replace=False)
    b = b.iloc[sub].reset_index(drop=True)
    X = fingerprints(b.molecule_smiles.to_numpy())
    bb = b[list(COLS)].to_numpy()

    runs = [json.loads(l) for l in open(RUNS)]
    runs = [r for r in runs if not r.get("skipped") and r["label"] == "activity"
            and r["score"] == "model"]
    cov = {}
    for d in DESIGNS:
        v = [r for r in runs if r["design"] == d]
        ins = tot = 0
        for r in v:
            g = [r["audit_intervals"][c]["inside"] for c in CATS]
            ins += sum(g); tot += len(g)
        cov[d] = 100.0 * ins / tot if tot else None

    res = {"n_molecules": int(len(b)), "reps": REPS, "coverage": cov, "designs": {}}
    print(f"{'design':9} {'audit|batch':>11} {'cal|batch':>11} {'bb1':>7} {'bb2':>7} "
          f"{'bb3':>7} {'coverage':>9}")
    for design in DESIGNS:
        aucs, aucs_cal, shares = [], [], []
        for r in range(REPS):
            rng = np.random.default_rng(80000 + r)
            ti, ci, ai, fi = (split_random(rng, len(b)) if design == "random"
                              else split_block(rng, bb, int(design[-1]) - 1))
            aucs.append(domain_auc(X, ai, fi, np.random.default_rng(1000 + r)))      # audit vs batch
            aucs_cal.append(domain_auc(X, ci, fi, np.random.default_rng(2000 + r)))  # calibration vs batch
            sh = []
            for j in range(3):
                seen = set(np.unique(bb[ci, j]).tolist())
                sh.append(float(np.mean([v in seen for v in bb[fi, j]])))
            shares.append(sh)
        sh = np.array(shares).mean(0)
        res["designs"][design] = {"domain_auc_audit_vs_batch": float(np.mean(aucs)),
                                  "domain_auc_cal_vs_batch": float(np.mean(aucs_cal)),
                                  "domain_auc": float(np.mean(aucs)),
                                  "domain_auc_all": aucs,
                                  "block_share": sh.tolist(), "coverage": cov[design]}
        print(f"{design:9} {np.mean(aucs):>11.3f} {np.mean(aucs_cal):>11.3f} {sh[0]:>7.2f} "
              f"{sh[1]:>7.2f} {sh[2]:>7.2f} {cov[design]:>8.0f}%")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()

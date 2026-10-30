"""Does re-expressing the score on a potency-like scale change anything?

A threshold is an order statistic of the calibration scores of one class, and a compound's
prediction set is decided by where its score falls among those order statistics.  Both are
statements about ordering, so the outcome table is a function of ranks alone.  A project is
therefore free to replace a model's raw output by any strictly increasing function of it:
the logit, an affine map onto a potency-like scale, a saturating curve fitted to a handful
of measured compounds.  The thresholds are recomputed on the new scale and the resulting
table is identical, cell for cell.

This checks that claim on the screen rather than asserting it.  One model is trained, ties
are broken once on the original scale, and the whole analysis is then run three times: on
the model's probability, on its logit, and on a pseudo-potency scale built from the logit
by an affine map and a strictly increasing nonlinearity.  Every count in every outcome cell,
and every published interval, must agree exactly.

What this buys is reporting, not coverage.  A threshold quoted in potency units can be read
by a chemist and compared against an assay cutoff, where a threshold quoted as a probability
of a binary label cannot.  What it does not buy is any additional guarantee: the identity of
the three tables is the point, and it is also the reason the transformation is free.

Usage: python3 -I monotone_check.py PARQUET OUT.json
"""
import sys, os, json, time, numpy as np, pandas as pd

PARQUET = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
sys.argv = [sys.argv[0], os.path.dirname(PARQUET) or ".", ".", "1"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cadel
from cadel import analyse
from conformal import break_ties
from catboost import CatBoostClassifier
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
NSUB = 200000
TRAIN_BLOCK_FRAC = 0.50
DEPLOY_FRACS = (0.40, 0.30)
LABEL = "binds_sEH"


def fingerprints(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s.replace("[Dy]", "[H]"))
        if m is not None:
            X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


# three readings of the same ordering: the model's own output, its logit, and a
# potency-like scale on which larger means more potent
SCALES = {
    "probability of in-class": lambda p: p,
    "logit": logit,
    "pseudo-pKi": lambda p: 4.5 + 0.62 * np.arcsinh(0.8 * logit(p)),
}


def main():
    t0 = time.time()
    b = pd.read_parquet(PARQUET)
    rng0 = np.random.default_rng(0)
    b = b.iloc[rng0.choice(len(b), min(NSUB, len(b)), replace=False)].reset_index(drop=True)
    X = fingerprints(b.molecule_smiles.to_numpy())
    y = b[LABEL].to_numpy().astype(int)
    bb1 = b.bb1_id.to_numpy()

    rng = np.random.default_rng(99000)
    bl = np.unique(bb1).copy(); rng.shuffle(bl)
    train_blocks = set(bl[:int(TRAIN_BLOCK_FRAC * len(bl))].tolist())
    in_train = np.array([v in train_blocks for v in bb1])
    held = np.where(in_train)[0]; rng.shuffle(held)
    deploy = np.where(~in_train)[0]; rng.shuffle(deploy)

    ti = held[:int(0.70 * len(held))]
    clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                             thread_count=2, random_seed=0).fit(X[ti], y[ti])
    p = clf.predict_proba(X)[:, 1]
    print(f"model trained on {len(ti)} molecules  [{time.time()-t0:.0f}s]", flush=True)

    # the audit travels with the batch: all three sets come from the deployment blocks
    c = int(DEPLOY_FRACS[0] * len(deploy)); d = c + int(DEPLOY_FRACS[1] * len(deploy))
    ci, ai, fi = deploy[:c], deploy[c:d], deploy[d:]

    # ties are broken once, on the original scale, so the ordering is fixed before any
    # transformation is applied; a strictly increasing map then preserves it exactly
    pc, pa, pf = break_ties(np.random.default_rng(7), p[ci], p[ai], p[fi])
    yc, ya, yf = y[ci], y[ai], y[fi]

    def table(o):
        return {k: (v["audit_count"], v["future_count"], tuple(v["interval_future"]),
                    v["inside"]) for k, v in o["audit_intervals"].items()}

    res, tabs = {}, []
    n0, n1 = int((yc == 0).sum()), int((yc == 1).sum())
    m1 = int(round(cadel.M * n1 / (n0 + n1))); m0 = cadel.M - m1
    k0 = cadel.k_batch(n0, cadel.A, cadel.D, m0); k1 = cadel.k_batch(n1, cadel.A, cadel.D, m1)
    for name, f in SCALES.items():
        v = f(pc)
        o = analyse(v, yc, f(pa), ya, f(pf), yf)
        t0, t1 = cadel.thresholds(v, yc, k0, k1)
        # the two cut points as a chemist would read them: a compound joins the out-class
        # set at or below cut0 and the in-class set at or above cut1.  Both are order
        # statistics of the calibration scores of the corresponding class, so each should
        # equal the k-th smallest, respectively k-th largest, value on whatever scale is
        # in use -- which is the invariance being checked.
        cut0, cut1 = t0, 1.0 - t1
        s0 = np.sort(v[yc == 0]); s1 = np.sort(v[yc == 1])
        ok0 = np.isclose(cut0, s0[k0 - 1]); ok1 = np.isclose(cut1, s1[len(s1) - k1])
        res[name] = {"cut_out_class": float(cut0), "cut_in_class": float(cut1),
                     "is_order_statistic": [bool(ok0), bool(ok1)], "table": table(o)}
        tabs.append(table(o))
        print(f"  {name:24} out-class at or below {cut0:+9.4f}   in-class at or above "
              f"{cut1:+9.4f}   both order statistics: {bool(ok0 and ok1)}   "
              f"table identical to the first: {tabs[-1] == tabs[0]}", flush=True)

    ok = all(t == tabs[0] for t in tabs)
    print(f"\nall three outcome tables agree cell for cell: {ok}")
    print(f"covered {sum(v[3] for v in tabs[0].values())}/{len(tabs[0])} lines")
    if OUT:
        json.dump({"identical": bool(ok), "scales": list(SCALES),
                   "n": {"cal": len(ci), "audit": len(ai), "batch": len(fi)},
                   "k": [k0, k1],
                   "results": {k: {"cut_out_class": v["cut_out_class"],
                                   "cut_in_class": v["cut_in_class"],
                                   "is_order_statistic": v["is_order_statistic"],
                                   "table": {kk: list(vv) for kk, vv in v["table"].items()}}
                               for k, v in res.items()}}, open(OUT, "w"), indent=1)
        print(f"written to {OUT}")


if __name__ == "__main__":
    main()

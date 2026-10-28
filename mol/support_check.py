"""Label-free check that a proxy-trained model is being applied outside its training data.

Conformal prediction needs exchangeability *within* the screened pool — calibration, audit
and future compounds drawn alike — and that is all it needs.  Nothing in the construction
requires the model's training data to resemble the screened pool, and nothing in it reports
when they do not.  Validity therefore survives an arbitrarily bad model, which is the point
of a distribution-free guarantee and also its blind spot.

Usefulness is the part that needs the training pool to resemble the screened one.  This
script tests that second, untested relation, using features only: no physical labels, and
nothing from the fitted model.  Three statistics, in increasing order of what they tell you.

  support      For each screened compound, the greatest Tanimoto similarity to any training
               molecule, with the training set's own held-out similarity as the reference
               scale.  A screened pool whose support sits far below that reference is being
               extrapolated into.

  p-values     Conformal p-values of the screened compounds under the training distribution,
               taking the negative support as the nonconformity measure.  Under exchangeability
               of the two pools these are uniform, so a Kolmogorov-Smirnov test against the
               uniform is an exact, distribution-free test of the relation that conformal
               prediction assumes nothing about.  This uses conformal machinery to test an
               assumption conformal does not itself require.

  separability AUROC of a classifier trained to tell the two pools apart.  One near 1 means
               the pools occupy different regions of feature space altogether.

Reported alongside is the same support split by a chemotype flag, which shows whether the
mismatch is wholesale or confined to part of the screened pool: only in the second case can
a stratified audit localise it.

Usage: python3 -I support_check.py DATADIR OUT.json [N_TRAIN_SAMPLE]
"""
import sys, os, json, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import kstest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
DATA, OUT = sys.argv[1], sys.argv[2]
NSUB = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
PRIMARY_SULFONAMIDE = "[SX4](=[OX1])(=[OX1])[NX3;H2]"


def fingerprints(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


def flag(smiles, smarts):
    pat = Chem.MolFromSmarts(smarts)
    return np.array([(lambda m: m is not None and m.HasSubstructMatch(pat))(Chem.MolFromSmiles(s))
                     for s in smiles])


def max_similarity(A, B, chunk=400):
    """greatest Tanimoto similarity of each row of A to any row of B"""
    nb = B.sum(1); out = np.zeros(len(A))
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        inter = a @ B.T
        union = a.sum(1)[:, None] + nb[None, :] - inter
        out[i:i + chunk] = (inter / np.maximum(union, 1)).max(1)
    return out


def main():
    rng = np.random.default_rng(0)
    dl = pd.read_parquet(f"{DATA}/cadel_del_CAXII.parquet")
    sub = rng.choice(len(dl), min(NSUB, len(dl)), replace=False)
    Xd = fingerprints(dl.smiles.to_numpy()[sub])
    a = pd.read_parquet(f"{DATA}/cadel_chembl_CAXII.parquet")
    b = pd.read_parquet(f"{DATA}/cadel_chembl_CAII.parquet")
    a, b = a[~a.inconsistent], b[~b.inconsistent]
    ch = a.merge(b[["inchikey", "pKi_median"]], on="inchikey", suffixes=("_t", "_a")).reset_index(drop=True)
    Xc = fingerprints(ch.smiles.to_numpy())
    sulf = flag(ch.smiles.to_numpy(), PRIMARY_SULFONAMIDE)

    half = len(Xd) // 2
    s_ref = max_similarity(Xd[:half], Xd[half:])     # the training set's own scale
    s_scr = max_similarity(Xc, Xd)                   # screened pool against training

    p = np.array([(1 + (s_ref <= v).sum()) / (len(s_ref) + 1) for v in s_scr])
    X = np.vstack([Xd[:5000], Xc]); y = np.r_[np.zeros(5000), np.ones(len(Xc))]
    perm = rng.permutation(len(X)); k = int(0.7 * len(X))
    clf = LogisticRegression(max_iter=2000).fit(X[perm[:k]], y[perm[:k]])
    auc = roc_auc_score(y[perm[k:]], clf.predict_proba(X[perm[k:]])[:, 1])

    q = lambda v: {f"q{int(100 * t)}": float(np.quantile(v, t)) for t in (.1, .25, .5, .75, .9)}
    res = {"n_train_sampled": len(Xd), "n_screened": len(Xc),
           "support_training_reference": q(s_ref), "support_screened": q(s_scr),
           "support_screened_sulfonamide": q(s_scr[sulf]),
           "support_screened_other": q(s_scr[~sulf]),
           "n_sulfonamide": int(sulf.sum()),
           "conformal_p_median": float(np.median(p)),
           "conformal_p_frac_below_0.01": float((p < 0.01).mean()),
           "ks_uniform_pvalue": float(kstest(p, "uniform").pvalue),
           "domain_classifier_auroc": float(auc),
           "wholesale_mismatch": bool(np.median(s_scr) < np.quantile(s_ref, 0.1))}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

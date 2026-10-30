"""What a surrogate calibration label has to get right, and where.

A campaign that cannot measure what it cares about calibrates against something cheaper and
hopes the substitution is harmless.  It is harmless under a condition that is narrower than
"the surrogate ranks compounds about as the truth does" and wider than "the surrogate is an
increasing function of the truth".

The condition follows from an identity.  The out-class threshold is the k0-th order statistic
of the scores of the calibration compounds the surrogate calls out-class, so it sits at a
fixed quantile of the score distribution conditioned on the surrogate's label, while the
coverage the campaign achieves is the score distribution conditioned on the truth's label
evaluated at that same threshold.  The shortfall is therefore

    achieved - nominal  =  F(tau | truth out-class) - F(tau | surrogate out-class)

a difference between two class-conditional score distributions at one point.  Nothing in it
refers to how often the two labels disagree.  What matters is whether the disagreement is
structured with respect to the score: a surrogate that mislabels compounds without regard to
where they sit in the ranking moves neither distribution and costs nothing, while one whose
mistakes are concentrated at one end of the ranking moves the threshold and costs coverage.

This separates the two at a matched error rate.  The population, the frozen score and the
measured affinities are held fixed, and three families of surrogate are compared:

  monotone            a strictly increasing relabelling of the affinity at matched
                      prevalence.  It must return the truth's own labels exactly, and is
                      the control.
  unstructured        a fraction f of each class relabelled, chosen uniformly at random,
                      so the disagreement carries no information about the score.
  structured          the same fraction f of each class relabelled, chosen as the compounds
                      the score ranks most extremely in the wrong direction, so the
                      disagreement is as informative about the score as it can be.

The second and third disagree with the truth on the same number of compounds.  If the
identity above is the mechanism, only the third loses coverage, and the loss is predicted by
the measured CDF gap rather than by any global summary of agreement.

The audit-set intervals are reported throughout, because they are a different guarantee:
they price whatever rule they are handed, and ask only that the audit and the batch be
exchangeable, not that the rule be well calibrated.

Usage: python3 -I surrogate_structure.py DATADIR OUT.json [REPS]
"""
import sys, os, json, time, numpy as np, pandas as pd

DATA, OUT = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import lru_cache
import intervals, conformal
from conformal import thresholds, cells, break_ties
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
TH = lru_cache(maxsize=None)(intervals.thatcher)
k_batch = lru_cache(maxsize=None)(conformal.k_batch)

A = D_ = 0.10
G = 0.05
M = 1000
VAL_FRAC = 0.10
ISOFORMS = ("CAII", "CAIX", "CAXII")
CUTOFFS = (7, 8)
FLIPS = (0.0, 0.05, 0.10, 0.20)
G8 = {"correct rejection": ["0_s0"], "false lead": ["0_s1"],
      "deferred non-binder": ["0_both", "0_empty"], "confident hit": ["1_s1"],
      "missed binder": ["1_s0"], "deferred binder": ["1_both", "1_empty"]}


def fingerprints(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise SystemExit(f"unparsable SMILES: {smi}")
        X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


def grouped(c):
    return {k: sum(c[x] for x in v) for k, v in G8.items()}


def frozen_score(data, iso):
    """the DEL-trained score mapped to (0,1) from the DEL training molecules only, cached"""
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), f".score_{iso}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["u_ch"], z["pki"], json.loads(str(z["info"]))
    dl = pd.read_parquet(f"{data}/cadel_del_{iso}.parquet")
    ch = pd.read_parquet(f"{data}/cadel_chembl_{iso}.parquet")
    n_all = len(ch); ch = ch[~ch.inconsistent].reset_index(drop=True)
    y_del = np.log1p(dl.del_enrich.to_numpy(float))
    Xd = fingerprints(dl.smiles.to_numpy())
    tr = np.random.default_rng(0).permutation(len(dl))[int(VAL_FRAC * len(dl)):]
    model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                              thread_count=2, random_seed=0).fit(Xd[tr], y_del[tr])
    yhat_tr = model.predict(Xd[tr])
    mu, sd = float(yhat_tr.mean()), float(yhat_tr.std())
    u_ch = 1.0 / (1.0 + np.exp(-(model.predict(fingerprints(ch.smiles.to_numpy())) - mu) / sd))
    pki = ch.pKi_median.to_numpy(float)
    info = {"isoform": iso, "n_del": int(len(dl)), "n_chembl": int(len(ch)),
            "n_chembl_inconsistent_excluded": int(n_all - len(ch)),
            "spearman_score_vs_pki": float(spearmanr(u_ch, pki).statistic)}
    np.savez(cache, u_ch=u_ch, pki=pki, info=json.dumps(info))
    return u_ch, pki, info


def flip_unstructured(rng, y, u, f):
    """relabel a fraction f of each class, chosen uniformly: disagreement carries no
    information about where a compound sits in the score ranking"""
    ys = y.copy()
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        k = int(round(f * len(idx)))
        if k:
            ys[rng.choice(idx, k, replace=False)] = 1 - cls
    return ys


def flip_structured(rng, y, u, f):
    """relabel the same fraction of each class, chosen as the compounds the score ranks
    most extremely against their class: out-class compounds the score likes best become
    in-class, in-class compounds the score likes least become out-class.  The disagreement
    is then as informative about the score as a given error rate allows."""
    ys = y.copy()
    jit, = break_ties(rng, u)
    for cls, sign in ((0, -1), (1, 1)):
        idx = np.where(y == cls)[0]
        k = int(round(f * len(idx)))
        if k:
            order = np.argsort(sign * jit[idx])       # cls 0: highest score first
            ys[idx[order[:k]]] = 1 - cls
    return ys


def run(u_all, pki, c, maker, f, reps, tag):
    n = len(u_all)
    y_true = (pki >= c).astype(int)
    acc = {"cov0": [], "cov1": [], "v0": 0, "v1": 0, "g0": [], "g1": [],
           "disc": [], "sp": [], "auc": [], "aud": [0, 0], "ident": 0,
           "pshift": []}
    for r in range(reps):
        rng = np.random.default_rng(41000 + 97 * r + 7 * c + int(10000 * f) + len(tag))
        y_sur = maker(rng, y_true, u_all, f)
        acc["ident"] += int(np.array_equal(y_sur, y_true))
        acc["disc"].append(float(np.mean(y_sur != y_true)))
        acc["sp"].append(float(spearmanr(y_sur, pki).statistic))
        acc["auc"].append(float(roc_auc_score(y_true, y_sur))
                          if 0 < y_sur.sum() < n else float("nan"))
        acc["pshift"].append(float(y_sur.mean() - y_true.mean()))

        p = rng.permutation(n)
        a, b = int(0.30 * n), int(0.60 * n)
        ci, ai, fi = p[:a], p[a:b], p[b:]
        uc, ua, uf = break_ties(rng, u_all[ci], u_all[ai], u_all[fi])

        ys_c = y_sur[ci]
        n0, n1 = int((ys_c == 0).sum()), int((ys_c == 1).sum())
        if min(n0, n1) < 20:
            continue
        m1 = int(round(M * n1 / (n0 + n1))); m0 = M - m1
        t_0, t_1 = thresholds(uc, ys_c, k_batch(n0, A, D_, m0), k_batch(n1, A, D_, m1))

        ya, yf = y_true[ai], y_true[fi]
        cc = cells(ua, ya, t_0, t_1)
        ma0, ma1 = int((ya == 0).sum()), int((ya == 1).sum())
        cov0 = (cc["0_s0"] + cc["0_both"]) / max(1, ma0)
        cov1 = (cc["1_s1"] + cc["1_both"]) / max(1, ma1)
        acc["cov0"].append(cov0); acc["cov1"].append(cov1)
        acc["v0"] += int(cov0 < 1 - A); acc["v1"] += int(cov1 < 1 - A)

        uc0, uc1 = uc[ys_c == 0], uc[ys_c == 1]
        ua0, ua1 = ua[ya == 0], ua[ya == 1]
        acc["g0"].append(float((ua0 <= t_0).mean() - (uc0 <= t_0).mean()))
        acc["g1"].append(float((ua1 >= 1 - t_1).mean() - (uc1 >= 1 - t_1).mean()))

        ga, gf = grouped(cc), grouped(cells(uf, yf, t_0, t_1))
        for k in ga:
            lo, hi = TH(ga[k], len(ya), len(yf), G)
            acc["aud"][1] += 1; acc["aud"][0] += int(lo <= gf[k] <= hi)

    if not acc["cov0"]:
        return None
    return {"surrogate": tag, "flip": f, "cutoff": c, "reps": len(acc["cov0"]),
            "identical_to_truth": acc["ident"],
            "disagreement": float(np.mean(acc["disc"])),
            "prevalence_shift": float(np.mean(acc["pshift"])),
            "spearman_surrogate_vs_pki": float(np.mean(acc["sp"])),
            "auroc_surrogate_vs_truth": float(np.nanmean(acc["auc"])),
            "cov_out": float(np.mean(acc["cov0"])), "cov_in": float(np.mean(acc["cov1"])),
            "viol_out": [acc["v0"], len(acc["cov0"])],
            "viol_in": [acc["v1"], len(acc["cov1"])],
            "cdf_gap_out": float(np.mean(acc["g0"])),
            "cdf_gap_in": float(np.mean(acc["g1"])),
            "audit_lines_covered": acc["aud"]}


def main():
    t0 = time.time()
    res = {"alpha": A, "gamma": G, "reps": REPS, "cutoffs": list(CUTOFFS),
           "flips": list(FLIPS), "isoforms": {}, "rows": []}
    print(f"{'iso':6}{'cut':>4}{'surrogate':>14}{'f':>6}{'disagree':>9}"
          f"{'cov out':>9}{'cov in':>8}{'viol out':>10}{'viol in':>9}"
          f"{'gap out':>9}{'gap in':>8}{'rho':>7}{'audit':>9}", flush=True)
    for iso in ISOFORMS:
        u_all, pki, info = frozen_score(DATA, iso)
        res["isoforms"][iso] = info
        for c in CUTOFFS:
            jobs = [("monotone", lambda rng, y, u, f: y, 0.0)]
            for f in FLIPS:
                if f == 0.0:
                    continue
                jobs.append(("unstructured", flip_unstructured, f))
                jobs.append(("structured", flip_structured, f))
            for tag, maker, f in jobs:
                row = run(u_all, pki, c, maker, f, REPS, tag)
                if row is None:
                    continue
                row["isoform"] = iso
                res["rows"].append(row)
                print(f"{iso:6}{c:>4}{tag:>14}{f:>6.2f}{row['disagreement']:>9.3f}"
                      f"{row['cov_out']:>9.3f}{row['cov_in']:>8.3f}"
                      f"{row['viol_out'][0]:>6}/{row['viol_out'][1]:<3}"
                      f"{row['viol_in'][0]:>5}/{row['viol_in'][1]:<3}"
                      f"{row['cdf_gap_out']:>+9.3f}{row['cdf_gap_in']:>+8.3f}"
                      f"{row['spearman_surrogate_vs_pki']:>7.3f}"
                      f"{row['audit_lines_covered'][0]:>5}/{row['audit_lines_covered'][1]:<4}",
                      flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwritten to {OUT}  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

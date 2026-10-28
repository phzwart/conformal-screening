"""Positive control for the CA-DEL selectivity screen: the same compounds, the same label,
the same audit, trained on physical measurements instead of on the cheap DEL proxy.

Every arm of cadel_selectivity.py trains on DEL enrichment and transfers to ChEMBL, and
none of them carries information about measured selectivity.  That on its own cannot say
whether the audit is detecting a bad model or whether the screening question is simply
unanswerable from structure.  This script answers that by changing one thing: the training
labels are physical pKi values from a held-out part of the same pool.

The pool is the ChEMBL compounds measured against both isoforms and carrying a primary
sulfonamide, the zinc-binding group of essentially every carbonic anhydrase inhibitor.
Each repeat splits it four ways - training, calibration, audit and future - so nothing is
reused between fitting the model, setting the thresholds, measuring the consequences and
checking them.  Because all four come from one pool by random assignment, exchangeability
holds by construction and the audit intervals are valid whatever the model turns out to be.

Two in-class definitions are reported.

  gap        pKi(CAXII) >= 8 and pKi(CAXII) - pKi(CAII) >= 0.5, the definition the DEL arms
             use, kept for comparability.
  rectangle  pKi(CAXII) >= 8 and pKi(CAII) <= 7: potent on the target and genuinely weak on
             the off-target, each axis on its own terms.  A margin alone admits a compound
             at 6.0 against 4.5, which no programme would advance.

Designs
  P  random assignment of compounds to the four splits.
  S  whole Bemis-Murcko scaffolds assigned to the splits, so the future compounds come from
     series the model and the calibration never saw.

Usage: python3 -I cadel_chembl.py DATADIR OUTDIR REPS
"""
import sys, os, json, time, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cadel  # reads the same three command-line arguments at import
from cadel import fingerprints, scaffolds, analyse
from cadel_selectivity import paired_chembl, has_primary_sulfonamide
from conformal import break_ties
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact

DATA, OUT, REPS = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(OUT, exist_ok=True)
LABELS = {"gap": lambda t, a: (t >= 8.0) & ((t - a) >= 0.5),
          "rectangle": lambda t, a: (t >= 8.0) & (a <= 7.0)}
FRACS = (0.40, 0.20, 0.20)        # training, calibration, audit; the rest is the future batch
M, J, EPS = 1000, 144, 0.05


def split_random(rng, n):
    p = rng.permutation(n)
    a = int(FRACS[0] * n); b = a + int(FRACS[1] * n); c = b + int(FRACS[2] * n)
    return p[:a], p[a:b], p[b:c], p[c:]


def split_scaffold(rng, scaf):
    """whole scaffolds, so the future compounds come from unseen series"""
    us = np.unique(scaf); rng.shuffle(us)
    order = np.concatenate([np.where(scaf == s)[0] for s in us])
    n = len(scaf)
    a = int(FRACS[0] * n); b = a + int(FRACS[1] * n); c = b + int(FRACS[2] * n)
    return order[:a], order[a:b], order[b:c], order[c:]


def enrichment(ua, ya, uc, yc, n0, n1, m0, m1):
    """Fisher test that the advanced compounds are enriched, as in beyond_chance.py"""
    from cadel import k_batch
    from conformal import thresholds, cells
    t0, t1 = thresholds(uc, yc, k_batch(n0, 0.10, 0.10, m0), k_batch(n1, 0.10, 0.10, m1))
    c = cells(ua, ya, t0, t1)
    adv_in, adv_out = c["1_s1"], c["0_s1"]
    rest_in = int((ya == 1).sum()) - adv_in; rest_out = int((ya == 0).sum()) - adv_out
    if adv_in + adv_out == 0:
        return None
    odds, p = fisher_exact([[adv_in, adv_out], [rest_in, rest_out]], alternative="greater")
    prev = float(ya.mean())
    return {"advanced": adv_in + adv_out, "advanced_in_class": adv_in,
            "precision": adv_in / (adv_in + adv_out), "prevalence": prev,
            "fold": (adv_in / (adv_in + adv_out)) / prev if prev > 0 else None,
            "p_one_sided": float(p), "significant": bool(p < EPS / J)}


def main():
    t_all = time.time()
    ch = paired_chembl()
    n_all = len(ch)
    ch = ch[has_primary_sulfonamide(ch.smiles.to_numpy())].reset_index(drop=True)
    X = fingerprints(ch.smiles.to_numpy())
    scaf = scaffolds(ch.smiles.to_numpy())
    pki_t = ch.pKi_median_t.to_numpy(float); pki_a = ch.pKi_median_a.to_numpy(float)
    info = {"pool": "ChEMBL CAXII+CAII, primary sulfonamides", "n_paired": n_all,
            "n_sulfonamide": len(ch), "fracs": FRACS, "reps": REPS,
            "model": "Morgan r=2 2048 bits + CatBoostClassifier(500, depth 6, lr 0.1)",
            "prevalence": {k: float(f(pki_t, pki_a).mean()) for k, f in LABELS.items()}}
    print(json.dumps(info, indent=1), flush=True)
    res = {"info": info, "runs": []}
    for lname, lf in LABELS.items():
        y = lf(pki_t, pki_a).astype(int)
        for design, splitter in (("P", split_random), ("S", split_scaffold)):
            for r in range(REPS):
                rng = np.random.default_rng(50000 + r)
                ti, ci, ai, fi = (splitter(rng, len(ch)) if design == "P"
                                  else splitter(rng, scaf))
                o = {"label": lname, "design": design, "rep": r}
                if min(y[ti].sum(), (1 - y[ti]).sum()) < 5 or min((y[ci] == 0).sum(), (y[ci] == 1).sum()) < 5 \
                   or min((y[ai] == 0).sum(), (y[ai] == 1).sum()) < 1 or len(fi) < 50:
                    o["skipped"] = True; res["runs"].append(o); continue
                clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1,
                                         verbose=0, thread_count=8, random_seed=0).fit(X[ti], y[ti])
                u = clf.predict_proba(X)[:, 1]
                uc_, ua_, uf_ = break_ties(rng, u[ci], u[ai], u[fi])
                yc_, ya_, yf_ = y[ci], y[ai], y[fi]
                o["n"] = {"train": len(ti), "cal": len(ci), "audit": len(ai), "future": len(fi)}
                o["prevalence_audit"] = float(ya_.mean())
                o["auroc_future"] = float(roc_auc_score(yf_, uf_)) if 0 < yf_.sum() < len(yf_) else None
                o["auroc_audit"] = float(roc_auc_score(ya_, ua_)) if 0 < ya_.sum() < len(ya_) else None
                o.update(analyse(uc_, yc_, ua_, ya_, uf_, yf_))
                n0, n1 = int((yc_ == 0).sum()), int((yc_ == 1).sum())
                m1 = int(round(M * n1 / (n0 + n1))); m0 = M - m1
                o["enrichment"] = enrichment(ua_, ya_, uc_, yc_, n0, n1, m0, m1)
                if r == 0 and design == "P":
                    np.savez_compressed(f"{OUT}/scores_chembl_{lname}.npz", u_cal=uc_, y_cal=yc_,
                                        u_audit=ua_, y_audit=ya_)
                res["runs"].append(o)
                if r == 0:
                    e = o["enrichment"]
                    print(f"  {lname} {design} rep0 AUROC(future) {o['auroc_future']:.3f} "
                          f"enrichment {e and round(e['fold'], 2)} p {e and '%.1e' % e['p_one_sided']}", flush=True)
    json.dump(res, open(f"{OUT}/cadel_chembl_runs.json", "w"))
    print(f"total {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

"""Conformal screening round on BRD4, entirely within ChEMBL.

No DEL data is used.  The pool is ChEMBL BRD4 IC50 measurements; each repeat splits it four
ways - training, calibration, audit and future - so that fitting the model, setting the
thresholds, measuring the consequences and checking them never share a compound.  All four
come from one pool by random assignment, so exchangeability holds by construction and the
audit intervals are valid whatever the model turns out to be.

Curation.  Measurements are grouped by canonical SMILES.  A molecule's potency is the
median of its exact ('=') values in nM, converted to pIC50 = 9 - log10(IC50).  Molecules
whose exact values span more than one log unit are dropped as inconsistent.  Censored rows
are not thrown away: a '<' bound below the threshold puts a molecule in class whatever its
true value, and a '>' bound above the threshold puts it out of class, so molecules with no
exact measurement are still labelled when their censoring is decisive.  Those that remain
ambiguous are dropped, and all three counts are reported.

What the pool is.  These are published medicinal-chemistry compounds, not a screening
library: 'too potent to measure' outnumbers 'inactive at the top concentration' by about
six to one, and most of the pool is sub-micromolar.  The guarantees therefore describe a
future batch drawn from this same published-compound distribution - triage of proposed
compounds rather than a primary screen - and the in-class rates quoted here are not primary
screening hit rates.

Labels.  IC50 <= 10 nM (pIC50 >= 8) is primary, the rate at which a model has room to add
something within a pool already selected for interest; IC50 <= 100 nM is the looser check.

Each arm is run twice: with the fitted model, and with a uniform random score through the
identical pipeline.  The random arm is the null the enrichment test has to reject, and it
also shows that validity and interval width survive a worthless score untouched.

Usage: python3 -I brd4_screen.py CSV OUTDIR REPS
"""
import sys, os, json, time, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cadel  # reads the same three command-line arguments at import
from cadel import fingerprints, scaffolds, analyse, k_batch
from conformal import thresholds, cells, break_ties
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact
import intervals

CSV, OUT, REPS = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(OUT, exist_ok=True)
CUTS = {"10nM": 8.0, "100nM": 7.0}
PRIMARY = "10nM"
FRACS = (0.40, 0.20, 0.20)
M, J, EPS, A, D = 1000, 144, 0.05, 0.10, 0.10


def curate(csv, cut):
    """label molecules at a pIC50 cut, using censored rows where they are decisive"""
    d = pd.read_csv(csv)
    d = d[d.ic50_units == "nM"].copy()
    d["v"] = pd.to_numeric(d.ic50_value, errors="coerce")
    d = d[d.v > 0]
    d["p"] = 9 - np.log10(d.v)
    out = {}
    stats = {"exact": 0, "censored_decisive": 0, "ambiguous_dropped": 0, "inconsistent_dropped": 0}
    for smi, g in d.groupby("smiles"):
        ex = g[g.ic50_relation == "="]
        if len(ex):
            if np.log10(ex.v.max() / ex.v.min()) > 1.0:
                stats["inconsistent_dropped"] += 1; continue
            out[smi] = int(np.median(ex.p) >= cut); stats["exact"] += 1; continue
        lo = g[g.ic50_relation.isin(["<", "<="])]      # more potent than the bound
        hi = g[g.ic50_relation.isin([">", ">="])]      # weaker than the bound
        if len(lo) and lo.p.max() >= cut:
            out[smi] = 1; stats["censored_decisive"] += 1
        elif len(hi) and hi.p.max() < cut:
            out[smi] = 0; stats["censored_decisive"] += 1
        else:
            stats["ambiguous_dropped"] += 1
    return pd.Series(out), stats


def split(rng, n, scaf=None):
    if scaf is None:
        o = rng.permutation(n)
    else:
        us = np.unique(scaf); rng.shuffle(us)
        o = np.concatenate([np.where(scaf == s)[0] for s in us])
    a = int(FRACS[0] * n); b = a + int(FRACS[1] * n); c = b + int(FRACS[2] * n)
    return o[:a], o[a:b], o[b:c], o[c:]


def consequences(uc, yc, ua, ya):
    n0, n1 = int((yc == 0).sum()), int((yc == 1).sum())
    m1 = int(round(M * n1 / (n0 + n1))); m0 = M - m1; na = len(ya)
    t0, t1 = thresholds(uc, yc, k_batch(n0, A, D, m0), k_batch(n1, A, D, m1))
    c = cells(ua, ya, t0, t1)
    adv_in, adv_out = c["1_s1"], c["0_s1"]
    rest_in = int(ya.sum()) - adv_in; rest_out = int((1 - ya).sum()) - adv_out
    prev = float(ya.mean())
    o = {"per1000": {"confident hit": 1000 * adv_in / na, "false lead": 1000 * adv_out / na,
                     "missed": 1000 * c["1_s0"] / na,
                     "deferred": 1000 * sum(c[k] for k in ("0_both", "0_empty", "1_both", "1_empty")) / na},
         "prevalence": prev}
    if adv_in + adv_out > 0:
        prec = adv_in / (adv_in + adv_out)
        _, p = fisher_exact([[adv_in, adv_out], [rest_in, rest_out]], alternative="greater")
        o["precision"] = prec; o["fold"] = prec / prev if prev else None
        o["fisher_p"] = float(p); o["significant"] = bool(p < EPS / J)
    else:
        o["precision"] = None; o["fold"] = None; o["fisher_p"] = 1.0; o["significant"] = False
    o["width_factor"] = {}
    for lab, keys in (("missed", ["1_s0"]), ("false lead", ["0_s1"]), ("confident hit", ["1_s1"]),
                      ("deferred", ["0_both", "0_empty", "1_both", "1_empty"])):
        k = sum(c[x] for x in keys); lo, hi = intervals.thatcher(k, na, M, EPS)
        o["width_factor"][lab] = M / max(hi - lo, 1)
    return o


def main():
    t0 = time.time(); res = {"info": {"pool": "ChEMBL BRD4 IC50, no DEL data",
                                      "cuts": CUTS, "fracs": FRACS, "reps": REPS,
                                      "model": "Morgan r=2 2048 bits + CatBoostClassifier(500, depth 6, lr 0.1)"},
                             "runs": []}
    for cname, cut in CUTS.items():
        y_ser, stats = curate(CSV, cut)
        smis = np.array(y_ser.index); y = y_ser.to_numpy()
        X = fingerprints(smis); scaf = scaffolds(smis)
        res["info"][f"curation_{cname}"] = stats
        res["info"][f"n_{cname}"] = int(len(y)); res["info"][f"prevalence_{cname}"] = float(y.mean())
        print(f"{cname}: {len(y)} molecules, prevalence {y.mean():.4f}, {stats}", flush=True)
        for design in ("random", "scaffold"):
            for score in ("model", "random"):
                for r in range(REPS):
                    rng = np.random.default_rng(70000 + r)
                    ti, ci, ai, fi = split(rng, len(y), None if design == "random" else scaf)
                    o = {"cut": cname, "design": design, "score": score, "rep": r}
                    if min(y[ti].sum(), (1 - y[ti]).sum()) < 5 or min(y[ci].sum(), (1 - y[ci]).sum()) < 5:
                        o["skipped"] = True; res["runs"].append(o); continue
                    if score == "model":
                        clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1,
                                                 verbose=0, thread_count=8, random_seed=0).fit(X[ti], y[ti])
                        u = clf.predict_proba(X)[:, 1]
                    else:
                        u = rng.random(len(y))
                    uc, ua, uf = break_ties(rng, u[ci], u[ai], u[fi])
                    yc, ya, yf = y[ci], y[ai], y[fi]
                    o["n"] = {"train": len(ti), "cal": len(ci), "audit": len(ai), "future": len(fi)}
                    o["auroc_future"] = float(roc_auc_score(yf, uf)) if 0 < yf.sum() < len(yf) else None
                    o.update(consequences(uc, yc, ua, ya))
                    o.update(analyse(uc, yc, ua, ya, uf, yf))
                    if r == 0 and score == "model":
                        np.savez_compressed(f"{OUT}/scores_brd4_{cname}_{design}.npz",
                                            u_cal=uc, y_cal=yc, u_audit=ua, y_audit=ya)
                    res["runs"].append(o)
                    if r == 0:
                        print(f"  {cname} {design} {score} rep0 AUROC {o['auroc_future']:.3f} "
                              f"fold {o['fold'] and round(o['fold'], 2)} p {o['fisher_p']:.1e}", flush=True)
    json.dump(res, open(f"{OUT}/brd4_runs.json", "w"))
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

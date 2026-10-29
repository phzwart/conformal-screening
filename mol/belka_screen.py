"""Conformal screening round inside a DNA-encoded library: soluble epoxide hydrolase.

Everything here stays within one library and one readout, so the transfer problem that
defeats a DEL-trained model applied to published medicinal chemistry does not arise.
Training, calibration, audit and future compounds are drawn from the same million-molecule
sample of BELKA, and the labels are that library's own binding calls.

Two decisions are screened, differing by 807 molecules out of a million.

  activity     binds sEH.  7416 molecules, 0.74% of the library.
  specificity  binds sEH and does not bind human serum albumin.  6609 molecules, 0.66%.
               Albumin is a binding sink, and 10.9% of the sEH hits also bind it, an
               enrichment of 26 over the library baseline, so sparing albumin is a real
               liability screen rather than a relabelling.  The same question is empty for
               BRD4, where 99.9% of hits already spare albumin.

Splits.  A molecule's nearest neighbour in a combinatorial library differs by one building
block, so assigning molecules at random leaves a held-out compound one substitution from a
training compound and measures interpolation.  The honest unit is the building block.

  block     whole building blocks are assigned to the four roles, at every position at
            once, so a future molecule shares no block with any training molecule.  This is
            the headline design.
  random    molecules assigned at random, reported alongside to show how much the easier
            split inflates the result.

Each design runs with the fitted model and with a uniform random score through the same
pipeline.  The random arm is the null the enrichment test must reject, and it also shows
that coverage and interval width survive a worthless score untouched.

The in-class rates are below one percent, so counts are reported for a batch of m = 100000
candidates; at m = 1000 a setting confirming a handful of hits cannot be bounded away from
zero.

Usage: python3 -I belka_screen.py PARQUET OUTDIR REPS
"""
import sys, os, json, time, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cadel  # reads the same three command-line arguments at import
from cadel import analyse, k_batch
from conformal import thresholds, cells, break_ties
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
import intervals

RDLogger.DisableLog("rdApp.*")
PARQUET, OUT, REPS = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(OUT, exist_ok=True)
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
M, J, EPS, A, D = 100000, 144, 0.05, 0.10, 0.10
NSUB = 200000          # molecules drawn from the parquet, to keep fingerprinting tractable
FRACS = (0.40, 0.20, 0.20)
LABELS = ("activity", "specificity")
DESIGNS = ("block", "random")


def fingerprints(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s.replace("[Dy]", "[H]"))
        if m is not None:
            X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


def split_random(rng, n):
    p = rng.permutation(n)
    a = int(FRACS[0] * n); b = a + int(FRACS[1] * n); c = b + int(FRACS[2] * n)
    return p[:a], p[a:b], p[b:c], p[c:]


def split_block(rng, bb):
    """assign whole building blocks at every position; a molecule joins the role of its
    rarest-role block, and molecules whose blocks disagree are dropped so the roles stay
    disjoint in every position"""
    role = {}
    for pos in range(bb.shape[1]):
        u = np.unique(bb[:, pos]); rng.shuffle(u)
        a = int(FRACS[0] * len(u)); b = a + int(FRACS[1] * len(u)); c = b + int(FRACS[2] * len(u))
        r = np.empty(len(u), int)
        r[:a], r[a:b], r[b:c], r[c:] = 0, 1, 2, 3
        role[pos] = dict(zip(u.tolist(), r.tolist()))
    assign = np.stack([[role[pos][v] for v in bb[:, pos]] for pos in range(bb.shape[1])], 1)
    same = (assign == assign[:, :1]).all(1)
    out = []
    for k in range(4):
        out.append(np.where(same & (assign[:, 0] == k))[0])
    return out


def enrichment_and_width(uc, yc, ua, ya):
    n0, n1 = int((yc == 0).sum()), int((yc == 1).sum())
    m1 = int(round(M * n1 / (n0 + n1))); m0 = M - m1; na = len(ya)
    t0, t1 = thresholds(uc, yc, k_batch(n0, A, D, m0), k_batch(n1, A, D, m1))
    c = cells(ua, ya, t0, t1)
    adv_in, adv_out = c["1_s1"], c["0_s1"]
    rest_in = int(ya.sum()) - adv_in; rest_out = int((1 - ya).sum()) - adv_out
    prev = float(ya.mean())
    o = {"prevalence_audit": prev,
         "per100k": {"confident hit": M * adv_in / na, "false lead": M * adv_out / na,
                     "missed": M * c["1_s0"] / na,
                     "deferred": M * sum(c[k] for k in ("0_both", "0_empty", "1_both", "1_empty")) / na}}
    if adv_in + adv_out > 0:
        prec = adv_in / (adv_in + adv_out)
        _, p = fisher_exact([[adv_in, adv_out], [rest_in, rest_out]], alternative="greater")
        o.update({"precision": prec, "fold": prec / prev if prev else None,
                  "fisher_p": float(p), "significant": bool(p < EPS / J)})
    else:
        o.update({"precision": None, "fold": None, "fisher_p": 1.0, "significant": False})
    o["width_factor"] = {}
    for lab, keys in (("missed", ["1_s0"]), ("false lead", ["0_s1"]), ("confident hit", ["1_s1"]),
                      ("deferred", ["0_both", "0_empty", "1_both", "1_empty"])):
        k = sum(c[x] for x in keys); lo, hi = intervals.thatcher(k, na, M, EPS)
        o["width_factor"][lab] = M / max(hi - lo, 1)
    return o


def main():
    t_all = time.time()
    b = pd.read_parquet(PARQUET)
    rng0 = np.random.default_rng(0)
    sub = rng0.choice(len(b), min(NSUB, len(b)), replace=False)
    b = b.iloc[sub].reset_index(drop=True)
    X = fingerprints(b.molecule_smiles.to_numpy())
    bb = b[["bb1_id", "bb2_id", "bb3_id"]].to_numpy()
    Y = {"activity": b.binds_sEH.to_numpy().astype(int),
         "specificity": ((b.binds_sEH == 1) & (b.binds_HSA == 0)).to_numpy().astype(int)}
    info = {"library": "BELKA, soluble epoxide hydrolase", "n_sampled": len(b), "batch_m": M,
            "model": "Morgan r=2 2048 bits + CatBoostClassifier(500, depth 6, lr 0.1)",
            "prevalence": {k: float(v.mean()) for k, v in Y.items()},
            "blocks": {c: int(b[c].nunique()) for c in ("bb1_id", "bb2_id", "bb3_id")}}
    print(json.dumps(info, indent=1), flush=True)
    res = {"info": info, "runs": []}
    for lname in LABELS:
        y = Y[lname]
        for design in DESIGNS:
            for score in ("model", "random"):
                for r in range(REPS):
                    rng = np.random.default_rng(80000 + r)
                    ti, ci, ai, fi = (split_random(rng, len(y)) if design == "random"
                                      else split_block(rng, bb))
                    o = {"label": lname, "design": design, "score": score, "rep": r,
                         "n": {"train": len(ti), "cal": len(ci), "audit": len(ai), "future": len(fi)}}
                    if min(y[ti].sum(), (1 - y[ti]).sum()) < 20 or min(y[ci].sum(), (1 - y[ci]).sum()) < 20 \
                       or len(ai) < 100 or y[ai].sum() < 1:
                        o["skipped"] = True; res["runs"].append(o); continue
                    if score == "model":
                        clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1,
                                                 verbose=0, thread_count=8, random_seed=0).fit(X[ti], y[ti])
                        u = clf.predict_proba(X)[:, 1]
                    else:
                        u = rng.random(len(y))
                    uc, ua, uf = break_ties(rng, u[ci], u[ai], u[fi])
                    yc, ya, yf = y[ci], y[ai], y[fi]
                    o["auroc_future"] = float(roc_auc_score(yf, uf)) if 0 < yf.sum() < len(yf) else None
                    o.update(enrichment_and_width(uc, yc, ua, ya))
                    o.update(analyse(uc, yc, ua, ya, uf, yf))
                    if r == 0 and score == "model":
                        np.savez_compressed(f"{OUT}/scores_belka_{lname}_{design}.npz",
                                            u_cal=uc, y_cal=yc, u_audit=ua, y_audit=ya)
                    res["runs"].append(o)
                    if r == 0:
                        print(f"  {lname} {design} {score} rep0 n_future {len(fi)} "
                              f"AUROC {o['auroc_future']} fold {o['fold'] and round(o['fold'], 2)} "
                              f"p {o['fisher_p']:.1e}", flush=True)
    json.dump(res, open(f"{OUT}/belka_runs.json", "w"))
    print(f"total {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

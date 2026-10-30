"""The guarantee is a statement about ranks, so state the analysis in ranks.

Nothing in the construction uses the numerical value of a score.  A threshold is an order
statistic of the calibration scores of one class, a compound's prediction set is decided by
where its score falls among those order statistics, and every count in the outcome table is
therefore a function of ranks alone.  The guarantee is invariant to any strictly increasing
transformation of the score: a model whose outputs are squared, logged or passed through a
sigmoid gives the identical table.

That has two consequences this takes seriously.

First, the quantity that must be exchangeable between the audit and the batch is the rank,
not the structure and not the score.  For a compound with score u, its calibration rank is
the number of calibration scores at or below it, and under exchangeability with the
calibration set that rank is uniform on its range.  Comparing the rank distribution of the
audit against the batch is therefore a direct test of the assumption the intervals need.  It
uses no features, so it cannot inherit a representation's blind spots, and it uses no labels,
so it can be run before the batch is assayed.  A classifier on fingerprints answers a
different and weaker question, whether two sets of structures differ at all.

Second, the proper construction is the one in which that rank distribution is shared.  The
split that separates the model from the chemistry it has not seen belongs at the training
boundary; the calibration, audit and batch sets are then drawn together from the deployment
chemistry, which is the condition the method was always stated under.  The arrangement that
leaves the audit behind with the training chemistry is a violation of the premise rather
than a finding about it, and is reported here only as the contrast.

Usage: python3 -I rank_analysis.py PARQUET OUTDIR REPS      (REPS = n, or first:last)
"""
import sys, os, json, time, numpy as np, pandas as pd

PARQUET, OUT = sys.argv[1], sys.argv[2]
R0, R1 = ((0, int(sys.argv[3])) if ":" not in sys.argv[3]
          else tuple(int(x) for x in sys.argv[3].split(":")))
os.makedirs(OUT, exist_ok=True)
sys.argv = [sys.argv[0], os.path.dirname(PARQUET) or ".", OUT, str(R1)]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cadel
from cadel import analyse
from conformal import break_ties
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
NSUB = 200000
TRAIN_BLOCK_FRAC = 0.50
DEPLOY_FRACS = (0.40, 0.30)
HELD_FRACS = (0.55, 0.45)
LABEL = "binds_sEH"
CATS = ("confident hit", "false lead", "missed binder", "deferred binder",
        "correct rejection", "deferred non-binder")
ARMS = ("audit travels with the batch", "audit stays with the model")


def fingerprints(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s.replace("[Dy]", "[H]"))
        if m is not None:
            X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


def calib_rank(u_cal, u):
    """fraction of calibration scores at or below each score: the rank, on [0,1]"""
    s = np.sort(u_cal)
    return np.searchsorted(s, u, side="right") / (len(s) + 1.0)


def rank_report(uc, yc, ua, ya, uf, yf):
    """how far the batch's rank distribution sits from the audit's, pooled and per class"""
    out = {}
    ra, rf = calib_rank(uc, ua), calib_rank(uc, uf)
    d, p = ks_2samp(ra, rf)
    out["pooled"] = {"ks": float(d), "p": float(p), "n_audit": int(len(ra)),
                     "n_batch": int(len(rf)),
                     "mean_audit": float(ra.mean()), "mean_batch": float(rf.mean())}
    for lab, cls in (("in_class", 1), ("out_class", 0)):
        m_c, m_a, m_f = yc == cls, ya == cls, yf == cls
        if m_a.sum() < 5 or m_f.sum() < 5 or m_c.sum() < 5:
            out[lab] = None; continue
        rra = calib_rank(uc[m_c], ua[m_a]); rrf = calib_rank(uc[m_c], uf[m_f])
        d, p = ks_2samp(rra, rrf)
        out[lab] = {"ks": float(d), "p": float(p), "n_audit": int(m_a.sum()),
                    "n_batch": int(m_f.sum()),
                    "mean_audit": float(rra.mean()), "mean_batch": float(rrf.mean())}
    return out


def main():
    t_all = time.time()
    b = pd.read_parquet(PARQUET)
    rng0 = np.random.default_rng(0)
    sub = rng0.choice(len(b), min(NSUB, len(b)), replace=False)
    b = b.iloc[sub].reset_index(drop=True)
    X = fingerprints(b.molecule_smiles.to_numpy())
    y = b[LABEL].to_numpy().astype(int)
    bb1 = b.bb1_id.to_numpy()
    blocks = np.unique(bb1)

    info = {"library": "BELKA, soluble epoxide hydrolase", "label": LABEL,
            "n_molecules": int(len(b)), "prevalence": float(y.mean()),
            "n_blocks_position1": int(len(blocks)), "train_block_frac": TRAIN_BLOCK_FRAC,
            "statistic": "calibration rank, the share of calibration scores at or below"}
    print(json.dumps(info, indent=1), flush=True)

    log = f"{OUT}/rank_analysis.jsonl"
    done = set()
    if os.path.exists(log):
        for line in open(log):
            o = json.loads(line); done.add((o["arm"], o["rep"]))
        print(f"resuming, {len(done)} runs on disk", flush=True)
    fh = open(log, "a")

    for r in range(R0, R1):
        rng = np.random.default_rng(99000 + r)
        bl = blocks.copy(); rng.shuffle(bl)
        ntr = int(TRAIN_BLOCK_FRAC * len(bl))
        train_blocks = set(bl[:ntr].tolist())
        in_train = np.array([v in train_blocks for v in bb1])
        held = np.where(in_train)[0]; rng.shuffle(held)
        deploy = np.where(~in_train)[0]; rng.shuffle(deploy)

        ti = held[:int(0.70 * len(held))]
        rest_held = held[int(0.70 * len(held)):]
        if min(y[ti].sum(), (1 - y[ti]).sum()) < 20:
            print(f"  rep{r} skipped", flush=True); continue
        clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                                 thread_count=2, random_seed=0).fit(X[ti], y[ti])
        u = clf.predict_proba(X)[:, 1]

        c = int(DEPLOY_FRACS[0] * len(deploy)); d = c + int(DEPLOY_FRACS[1] * len(deploy))
        a = int(HELD_FRACS[0] * len(rest_held))
        arms = {"audit travels with the batch": (deploy[:c], deploy[c:d], deploy[d:]),
                "audit stays with the model": (rest_held[:a], rest_held[a:], deploy)}

        for arm in ARMS:
            if (arm, r) in done:
                continue
            ci, ai_, fi = arms[arm]
            o = {"arm": arm, "rep": r,
                 "n": {"train": len(ti), "cal": len(ci), "audit": len(ai_), "batch": len(fi)}}
            if min(y[ci].sum(), (1 - y[ci]).sum()) < 20 or y[ai_].sum() < 1 or y[fi].sum() < 1:
                o["skipped"] = True
                fh.write(json.dumps(o) + "\n"); fh.flush(); continue
            uc, ua, uf = break_ties(rng, u[ci], u[ai_], u[fi])
            yc, ya, yf = y[ci], y[ai_], y[fi]
            o["auroc_batch"] = float(roc_auc_score(yf, uf)) if 0 < yf.sum() < len(yf) else None
            o["ranks"] = rank_report(uc, yc, ua, ya, uf, yf)
            o.update(analyse(uc, yc, ua, ya, uf, yf))
            cov = [o["audit_intervals"][k]["inside"] for k in CATS if k in o["audit_intervals"]]
            o["lines_covered"] = [int(sum(cov)), len(cov)]
            fh.write(json.dumps(o) + "\n"); fh.flush()
            pk = o["ranks"]["pooled"]
            print(f"  {arm:30} rep{r} AUROC {o['auroc_batch'] and round(o['auroc_batch'],3)} "
                  f"rank KS {pk['ks']:.4f} p {pk['p']:.1e} covered {sum(cov)}/{len(cov)} "
                  f"[{time.time()-t_all:.0f}s]", flush=True)
    fh.close()
    runs = [json.loads(l) for l in open(log)]
    json.dump({"info": info, "runs": runs}, open(f"{OUT}/rank_analysis.json", "w"))
    print(f"total {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

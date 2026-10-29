"""Where should a campaign put its split: at the training boundary, or at the audit's?

Exchangeability is required between the calibration set, the audit set and the batch.  It is
not required between any of those and the training set, which supplies only a frozen score.
Two arrangements follow, and a project can choose either without changing its model.

  at the calibration boundary   training, calibration and audit all come from one set of
                                building blocks and the batch is built on others.  This is
                                what a block-disjoint split does when it is applied to the
                                whole pipeline, and it leaves the batch holding chemistry
                                the audit never saw.

  at the training boundary      training uses one set of blocks, and calibration, audit and
                                batch are drawn at random from the molecules built on the
                                others.  The model is still measured on chemistry it never
                                trained on, so the guard against memorisation is intact,
                                while the three sets the guarantee binds are exchangeable
                                with one another by construction.

Both arms share one model per repeat, trained on the same molecules, so the comparison
isolates where the split is made and nothing else.  If the second covers at its nominal
level while the first does not, then a project loses nothing by splitting honestly: it gains
an unmemorised model and keeps its intervals, provided the split is made in the right place.

Usage: python3 -I train_boundary.py PARQUET OUTDIR REPS      (REPS = n, or first:last)
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
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
NSUB = 200000
TRAIN_BLOCK_FRAC = 0.50          # share of first-position blocks reserved for training
DEPLOY_FRACS = (0.40, 0.30)      # of the deployment pool: calibration, audit; rest is batch
HELD_FRACS = (0.55, 0.45)        # of the training-block pool left over: calibration, audit
LABEL = "binds_sEH"
CATS = ("confident hit", "false lead", "missed binder", "deferred binder",
        "correct rejection", "deferred non-binder")


def fingerprints(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s.replace("[Dy]", "[H]"))
        if m is not None:
            X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


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
            "n_blocks_position1": int(len(blocks)),
            "train_block_frac": TRAIN_BLOCK_FRAC}
    print(json.dumps(info, indent=1), flush=True)

    log = f"{OUT}/train_boundary.jsonl"
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
        train_blocks, deploy_blocks = set(bl[:ntr].tolist()), set(bl[ntr:].tolist())
        in_train = np.array([v in train_blocks for v in bb1])

        held = np.where(in_train)[0]; rng.shuffle(held)
        deploy = np.where(~in_train)[0]; rng.shuffle(deploy)

        # the model: trained on part of the training-block pool, identical in both arms
        ntrain = int(0.70 * len(held))
        ti = held[:ntrain]
        rest_held = held[ntrain:]
        if min(y[ti].sum(), (1 - y[ti]).sum()) < 20:
            print(f"  rep{r} skipped, training pool has too few in-class", flush=True)
            continue
        clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                                 thread_count=2, random_seed=0).fit(X[ti], y[ti])
        u = clf.predict_proba(X)[:, 1]

        # arm A: calibration and audit from the training blocks, batch from the others
        a = int(HELD_FRACS[0] * len(rest_held))
        arms = {"at the calibration boundary":
                (rest_held[:a], rest_held[a:], deploy)}
        # arm B: calibration, audit and batch all drawn from the deployment blocks
        c = int(DEPLOY_FRACS[0] * len(deploy)); d = c + int(DEPLOY_FRACS[1] * len(deploy))
        arms["at the training boundary"] = (deploy[:c], deploy[c:d], deploy[d:])

        for arm, (ci, ai_, fi) in arms.items():
            if (arm, r) in done:
                continue
            o = {"arm": arm, "rep": r,
                 "n": {"train": len(ti), "cal": len(ci), "audit": len(ai_), "batch": len(fi)}}
            if min(y[ci].sum(), (1 - y[ci]).sum()) < 20 or y[ai_].sum() < 1 or y[fi].sum() < 1:
                o["skipped"] = True
                fh.write(json.dumps(o) + "\n"); fh.flush()
                print(f"  {arm:28} rep{r} skipped", flush=True); continue
            uc, ua, uf = break_ties(rng, u[ci], u[ai_], u[fi])
            yc, ya, yf = y[ci], y[ai_], y[fi]
            o["rate_audit"] = float(ya.mean()); o["rate_batch"] = float(yf.mean())
            o["auroc_batch"] = float(roc_auc_score(yf, uf)) if 0 < yf.sum() < len(yf) else None
            o.update(analyse(uc, yc, ua, ya, uf, yf))
            cov = [o["audit_intervals"][k]["inside"] for k in CATS if k in o["audit_intervals"]]
            o["lines_covered"] = [int(sum(cov)), len(cov)]
            fh.write(json.dumps(o) + "\n"); fh.flush()
            print(f"  {arm:28} rep{r} AUROC {o['auroc_batch'] and round(o['auroc_batch'],3)} "
                  f"rate aud/bat {o['rate_audit']:.4f}/{o['rate_batch']:.4f} "
                  f"covered {sum(cov)}/{len(cov)}  [{time.time()-t_all:.0f}s]", flush=True)
    fh.close()
    runs = [json.loads(l) for l in open(log)]
    json.dump({"info": info, "runs": runs}, open(f"{OUT}/train_boundary.json", "w"))
    print(f"total {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

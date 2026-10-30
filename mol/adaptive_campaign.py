"""Can an audit price a batch the model chose?

Every design in this work divides a pool at random or by a structural rule.  A real campaign
does neither: a planner reads the scores and commits the top of the ranking, so the batch is
not a random sample of anything.  The intervals are then not expected to hold, and the paper
says so without testing it.  This tests it, and tests the two repairs available.

A fixed model scores a screening pool.  In each round the planner commits the m highest-scored
candidates that remain.  One batch is committed per round and priced three ways, so the batch
is identical across arms and only the origin of the audit changes:

  library audit    a random sample of the candidates still in the pool.  This is what a team
                   does when it reserves an audit from the library, and the arrangement the
                   paper predicts will fail: the batch is the top of the ranking and the audit
                   is typical of what is left.
  batch slice      a random slice of the committed batch itself, measured and used to price
                   the remainder.  A random division of the committed set, so exchangeable
                   with the rest of it by construction --- which is the point: the lot being
                   priced is the lot being sampled, as in acceptance sampling.
  selected holdout the same selection rule applied to a pool reserved from the planner, so the
                   audit is drawn from the top of a ranking as the batch is.  This is the
                   "selection is part of the design" case.

After each round the committed candidates leave the pool, which depletes the top of the
ranking.  The in-class rate of what remains is reported, because that depletion is the reason
a round cannot be priced by the round before it even when nothing else moves.

Usage: python3 -I adaptive_campaign.py PARQUET OUTDIR ROUNDS
"""
import sys, os, json, time, numpy as np, pandas as pd

PARQUET, OUT = sys.argv[1], sys.argv[2]
ROUNDS = int(sys.argv[3])
os.makedirs(OUT, exist_ok=True)
sys.argv = [sys.argv[0], os.path.dirname(PARQUET) or ".", OUT, "1"]

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
M = 8000                 # candidates committed per round
AUDIT_FRAC = 0.25        # share of the committed batch reserved as the audit
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
    t0 = time.time()
    b = pd.read_parquet(PARQUET)
    rng0 = np.random.default_rng(0)
    b = b.iloc[rng0.choice(len(b), min(NSUB, len(b)), replace=False)].reset_index(drop=True)
    X = fingerprints(b.molecule_smiles.to_numpy())
    y = b.binds_sEH.to_numpy().astype(int)
    print(json.dumps({"n": int(len(b)), "prevalence": float(y.mean()),
                      "m_per_round": M, "audit_frac": AUDIT_FRAC}), flush=True)

    rng = np.random.default_rng(777)
    p = rng.permutation(len(b))
    n_tr, n_cal, n_res = int(0.40 * len(b)), int(0.15 * len(b)), int(0.10 * len(b))
    tr = p[:n_tr]
    cal = p[n_tr:n_tr + n_cal]
    res = p[n_tr + n_cal:n_tr + n_cal + n_res]          # reserved from the planner
    pool = p[n_tr + n_cal + n_res:]                      # the screening pool
    print(f"train {len(tr)}  cal {len(cal)}  reserved {len(res)}  pool {len(pool)}"
          f"  [{time.time()-t0:.0f}s]", flush=True)

    clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                             thread_count=2, random_seed=0).fit(X[tr], y[tr])
    u = clf.predict_proba(X)[:, 1]
    print(f"model fitted, pool AUROC {roc_auc_score(y[pool], u[pool]):.3f}"
          f"  [{time.time()-t0:.0f}s]", flush=True)

    log = open(os.path.join(OUT, "adaptive_campaign.jsonl"), "a")
    runs = []
    remaining = pool.copy()
    res_remaining = res.copy()

    for r in range(ROUNDS):
        if len(remaining) < M:
            break
        # the planner commits the top of the ranking
        order = np.argsort(-u[remaining])
        committed = remaining[order[:M]]
        g = np.random.default_rng(5000 + r)
        q = g.permutation(len(committed))
        n_a = int(AUDIT_FRAC * M)
        slice_audit = committed[q[:n_a]]
        batch = committed[q[n_a:]]                       # the same batch in every arm

        rest = np.setdiff1d(remaining, committed, assume_unique=False)
        lib_audit = g.choice(rest, min(n_a, len(rest)), replace=False)

        # the same selection rule applied to the reserved pool
        if len(res_remaining) >= n_a:
            ro = np.argsort(-u[res_remaining])
            sel_audit = res_remaining[ro[:n_a]]
            res_remaining = np.setdiff1d(res_remaining, sel_audit, assume_unique=False)
        else:
            sel_audit = None

        pool_rate = float(y[remaining].mean())
        arms = {"library audit": lib_audit, "batch slice": slice_audit,
                "selected holdout": sel_audit}
        for arm, aud in arms.items():
            if aud is None or y[aud].sum() < 1 or y[batch].sum() < 1:
                print(f"  round {r} {arm}: skipped (no in-class)", flush=True)
                continue
            g2 = np.random.default_rng(31337 + r)
            uc, ua, uf = break_ties(g2, u[cal], u[aud], u[batch])
            o = {"arm": arm, "round": r,
                 "n": {"cal": len(cal), "audit": len(aud), "batch": len(batch)},
                 "prevalence": {"cal": float(y[cal].mean()), "audit": float(y[aud].mean()),
                                "batch": float(y[batch].mean()), "pool_remaining": pool_rate},
                 "auroc_batch": (float(roc_auc_score(y[batch], uf))
                                 if 0 < y[batch].sum() < len(batch) else None)}
            o["prevalence_ratio"] = float(y[aud].mean() / max(1e-12, y[batch].mean()))
            o.update(analyse(uc, y[cal], ua, y[aud], uf, y[batch]))
            qq = [o["audit_intervals"][k]["inside"] for k in CATS if k in o["audit_intervals"]]
            o["lines_covered"] = [int(sum(qq)), len(qq)]
            log.write(json.dumps(o) + "\n"); log.flush(); runs.append(o)
            print(f"  round {r}  {arm:17} prev aud/bat {o['prevalence']['audit']:.4f}/"
                  f"{o['prevalence']['batch']:.4f}  ratio {o['prevalence_ratio']:.2f}"
                  f"  covered {sum(qq)}/{len(qq)}  [{time.time()-t0:.0f}s]", flush=True)

        remaining = rest
        print(f"  round {r}: pool rate was {pool_rate:.4f}, "
              f"{len(remaining)} left at {y[remaining].mean():.4f}", flush=True)

    log.close()
    json.dump({"runs": runs, "m_per_round": M, "audit_frac": AUDIT_FRAC},
              open(os.path.join(OUT, "adaptive_campaign.json"), "w"), indent=1)
    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

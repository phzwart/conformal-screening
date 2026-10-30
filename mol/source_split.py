"""An audit and a batch that are not two halves of one set.

Every real-data check of the audit intervals in this work divides one labelled pool at
random, which makes the two halves exchangeable by construction and so verifies the
construction rather than testing it.  The question a campaign asks is different: does an
audit measured on one body of work price a batch measured on another?

The electrolyte compilation answers it, because it records which paper each measurement came
from.  Formulations are assigned to the audit or to the batch by source, so no paper
contributes to both, and the model and the thresholds are fixed before either set is seen.
Nothing is halved.  Three divisions are compared, all with the same model and the same
calibration set:

  halved        the published check: the audit pool is split at random into an audit half
                and a stand-in batch.  Exchangeable by construction, and reported here as
                the reference the other two are read against.
  by source     papers are partitioned at random into two groups and the formulations
                follow their paper.  The chemistry is not deliberately shifted, but no
                formulation shares a source, a cell, an operator or a calibration with the
                set that prices it.
  one source out  a single paper's formulations are the batch and the rest are the audit,
                repeated over the papers large enough to serve, which is the sharpest
                version of the same question.

If the intervals hold under the second and third, the construction survives a real division
of provenance and the halving results stop carrying the whole claim.  If they do not, the
honest reading is that provenance alone is enough to void them, which bears directly on
pooled compilations.

Usage: python3 -I source_split.py CSV OUTDIR KSTAR REPS
"""
import sys, os, json, time, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calisol as C                      # parses argv at import
# cadel reads argv[3] as an integer repeat count when imported; calisol has already taken
# its own arguments, so the slot is rewritten before the import
_argv = list(sys.argv)
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], str(int(float(sys.argv[4])))]
import cadel
sys.argv = _argv
from cadel import analyse
from conformal import break_ties
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

CATS = ("confident hit", "false lead", "missed binder", "deferred binder",
        "correct rejection", "deferred non-binder")
MIN_SOURCE = 40                          # formulations a single paper needs to serve as a batch


def main():
    t0 = time.time()
    d, solv, counts = C.load()
    X = C.featurize(d, solv)
    t = C.rt_table(d)
    y = (t.k.to_numpy(float) >= C.KSTAR).astype(int)
    # t.row holds labels of d's index; the feature matrix is positional in d
    rows = d.index.get_indexer(t.row.to_numpy())
    assert (rows >= 0).all(), "a formulation's representative row is missing from the table"
    doi = t.doi.to_numpy()
    Xt = X[rows]

    info = {"formulations": int(len(t)), "prevalence": float(y.mean()),
            "papers": int(len(np.unique(doi))), "kstar": C.KSTAR}
    print(json.dumps(info), flush=True)

    sizes = pd.Series(doi).value_counts()
    big = [s for s in sizes.index if sizes[s] >= MIN_SOURCE]
    print(f"papers with at least {MIN_SOURCE} formulations: {len(big)}", flush=True)

    out = {"info": info, "min_source": MIN_SOURCE, "runs": []}
    log = os.path.join(C.OUT, "source_split.jsonl")
    fh = open(log, "a")

    def run(tag, rep, cal_i, aud_i, bat_i, note=""):
        if min(y[cal_i].sum(), (1 - y[cal_i]).sum()) < 20 or y[aud_i].sum() < 1 or y[bat_i].sum() < 1:
            return None
        rng = np.random.default_rng(31337 + rep)
        clf = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.08, verbose=0,
                                 thread_count=2, random_seed=0).fit(Xt[cal_i], y[cal_i])
        u = clf.predict_proba(Xt)[:, 1]
        uc, ua, uf = break_ties(rng, u[cal_i], u[aud_i], u[bat_i])
        o = {"design": tag, "rep": rep, "note": note,
             "n": {"cal": len(cal_i), "audit": len(aud_i), "batch": len(bat_i)},
             "prevalence": {"cal": float(y[cal_i].mean()), "audit": float(y[aud_i].mean()),
                            "batch": float(y[bat_i].mean())},
             "auroc_batch": (float(roc_auc_score(y[bat_i], uf))
                             if 0 < y[bat_i].sum() < len(bat_i) else None)}
        o.update(analyse(uc, y[cal_i], ua, y[aud_i], uf, y[bat_i]))
        g = [o["audit_intervals"][k]["inside"] for k in CATS if k in o["audit_intervals"]]
        o["lines_covered"] = [int(sum(g)), len(g)]
        fh.write(json.dumps(o) + "\n"); fh.flush()
        out["runs"].append(o)
        print(f"  {tag:16} rep{rep:<3} n cal/aud/bat {len(cal_i)}/{len(aud_i)}/{len(bat_i)}"
              f"  prev {o['prevalence']['audit']:.3f}/{o['prevalence']['batch']:.3f}"
              f"  AUROC {o['auroc_batch'] and round(o['auroc_batch'],3)}"
              f"  covered {sum(g)}/{len(g)}  [{time.time()-t0:.0f}s]", flush=True)
        return o

    n = len(t)
    for rep in range(C.REPS):
        rng = np.random.default_rng(777 + rep)
        # one calibration set for all three divisions of this repeat
        perm = rng.permutation(n)
        cal_i = perm[: int(0.40 * n)]
        rest = perm[int(0.40 * n):]

        # halved: the published check
        half = len(rest) // 2
        run("halved", rep, cal_i, rest[:half], rest[half:])

        # by source: papers partitioned at random, formulations follow their paper
        papers = np.unique(doi[rest]); rng.shuffle(papers)
        left = set(papers[: len(papers) // 2].tolist())
        m = np.array([doi[i] in left for i in rest])
        run("by source", rep, cal_i, rest[m], rest[~m],
            note=f"{len(left)} papers against {len(papers)-len(left)}")

        # one source out: each sufficiently large paper in turn becomes the batch
        for s in big:
            bat = rest[doi[rest] == s]
            aud = rest[doi[rest] != s]
            if len(bat) < MIN_SOURCE:
                continue
            run("one source out", rep, cal_i, aud, bat, note=s)

    fh.close()
    json.dump(out, open(os.path.join(C.OUT, "source_split.json"), "w"), indent=1)
    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

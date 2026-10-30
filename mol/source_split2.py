"""An audit and a batch that are not two halves of one set, under the main protocol.

A first version of this experiment fitted the model on its own calibration set, which voids
the coverage guarantee for the design and makes it incomparable with the rest of the work.
Here the training set is disjoint from the calibration set and the protocol is the one used
everywhere else: formulations split by role, the model fitted on every temperature row of the
training formulations, scored at 298.15 K, 500 trees, depth 6, learning rate 0.1.

Formulations are divided into roles 40/20/40, and the last 40% is divided into an audit and a
batch three ways:

  halved        at random, which is exchangeable by construction and is the reference.
  by source     papers are partitioned at random and formulations follow their paper, so no
                formulation shares a source, a cell, an operator or a calibration with the
                set that prices it.
  one source out  a single paper's formulations are the batch and the rest the audit.

Usage: python3 -I source_split2.py CSV OUTDIR KSTAR REPS
"""
import sys, os, json, time, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calisol as C                      # parses argv at import
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
MIN_SOURCE = 40


def main():
    t0 = time.time()
    d, solv, counts = C.load()
    X = C.featurize(d, solv)
    y_all = (d.k.to_numpy() >= C.KSTAR).astype(int)
    rt = C.rt_table(d)
    rt["y"] = (rt.k >= C.KSTAR).astype(int)
    pos = {r: i for i, r in enumerate(d.index)}
    rt["pos"] = rt.row.map(pos)
    Xrt = X[rt.pos.to_numpy()].copy()
    Xrt[:, -1] = 298.15                  # as elsewhere: the label is a room-temperature median
    y = rt.y.to_numpy()
    forms = rt.index.to_numpy()
    doi = rt.doi.to_numpy()
    form_of_row = d.form.to_numpy()

    info = {"formulations": int(len(rt)), "prevalence": float(y.mean()),
            "papers": int(len(np.unique(doi))), "kstar": C.KSTAR,
            "protocol": "train disjoint from calibration; 500 trees, depth 6, lr 0.1"}
    print(json.dumps(info), flush=True)

    sizes = pd.Series(doi).value_counts()
    big = [s for s in sizes.index if sizes[s] >= MIN_SOURCE]
    print(f"papers with at least {MIN_SOURCE} formulations: {len(big)}", flush=True)

    out = {"info": info, "min_source": MIN_SOURCE, "runs": []}
    log = os.path.join(C.OUT, "source_split2.jsonl")
    done = set()
    if os.path.exists(log):
        for line in open(log):
            o = json.loads(line)
            done.add((o["design"], o["rep"], o.get("note", "")))
        print(f"resuming, {len(done)} runs on disk", flush=True)
    fh = open(log, "a")

    n = len(rt)
    for rep in range(C.REPS):
        rng = np.random.default_rng(777 + rep)
        perm = rng.permutation(n)
        tr_i = perm[: int(0.40 * n)]
        cal_i = perm[int(0.40 * n): int(0.60 * n)]
        rest = perm[int(0.60 * n):]
        if min(y[cal_i].sum(), (1 - y[cal_i]).sum()) < 20:
            print(f"rep {rep}: calibration set too one-sided, skipped", flush=True)
            continue

        # one model per repeat, fitted on every temperature row of the training formulations
        held = set(forms[cal_i]) | set(forms[rest])
        mask = np.array([f not in held for f in form_of_row])
        clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                                 thread_count=2, random_seed=30000 + rep).fit(X[mask], y_all[mask])
        u = clf.predict_proba(Xrt)[:, 1]
        print(f"rep {rep}: trained on {int(mask.sum())} rows of {len(tr_i)} formulations"
              f"  [{time.time()-t0:.0f}s]", flush=True)

        def run(tag, aud_i, bat_i, note=""):
            if (tag, rep, note) in done:
                return
            if y[aud_i].sum() < 1 or y[bat_i].sum() < 1:
                return
            g2 = np.random.default_rng(31337 + rep)
            uc, ua, uf = break_ties(g2, u[cal_i], u[aud_i], u[bat_i])
            o = {"design": tag, "rep": rep, "note": note,
                 "n": {"train_forms": int(len(tr_i)), "train_rows": int(mask.sum()),
                       "cal": len(cal_i), "audit": len(aud_i), "batch": len(bat_i)},
                 "prevalence": {"cal": float(y[cal_i].mean()), "audit": float(y[aud_i].mean()),
                                "batch": float(y[bat_i].mean())},
                 "auroc_batch": (float(roc_auc_score(y[bat_i], uf))
                                 if 0 < y[bat_i].sum() < len(bat_i) else None)}
            o["prevalence_ratio"] = float(y[aud_i].mean() / max(1e-12, y[bat_i].mean()))
            o.update(analyse(uc, y[cal_i], ua, y[aud_i], uf, y[bat_i]))
            q = [o["audit_intervals"][k]["inside"] for k in CATS if k in o["audit_intervals"]]
            o["lines_covered"] = [int(sum(q)), len(q)]
            fh.write(json.dumps(o) + "\n"); fh.flush()
            out["runs"].append(o)
            print(f"  {tag:16} rep{rep:<3} n aud/bat {len(aud_i)}/{len(bat_i)}"
                  f"  prev {o['prevalence']['audit']:.3f}/{o['prevalence']['batch']:.3f}"
                  f"  ratio {o['prevalence_ratio']:.2f}"
                  f"  AUROC {o['auroc_batch'] and round(o['auroc_batch'],3)}"
                  f"  covered {sum(q)}/{len(q)}  [{time.time()-t0:.0f}s]", flush=True)

        half = len(rest) // 2
        run("halved", rest[:half], rest[half:])

        papers = np.unique(doi[rest]); rng.shuffle(papers)
        left = set(papers[: len(papers) // 2].tolist())
        mm = np.array([doi[i] in left for i in rest])
        run("by source", rest[mm], rest[~mm],
            note=f"{len(left)} papers against {len(papers)-len(left)}")

        for s in big:
            bat = rest[doi[rest] == s]
            aud = rest[doi[rest] != s]
            if len(bat) < MIN_SOURCE:
                continue
            run("one source out", aud, bat, note=s)

    fh.close()
    runs = [json.loads(l) for l in open(log)]
    json.dump({"info": info, "runs": runs},
              open(os.path.join(C.OUT, "source_split2.json"), "w"), indent=1)
    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

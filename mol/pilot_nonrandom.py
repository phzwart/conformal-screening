"""Does a pilot of the right population price a batch when the division is not random?

Every arrangement in this work that recovers the nominal level divides one pool at random,
which makes audit and batch exchangeable by construction and so verifies the construction
rather than testing the prescription drawn from it.  The prescription is: when a round reaches
into a population the audit does not hold, buy a pilot of *that* population and audit with it.
A random split of the new population cannot test that advice, because it cannot fail.

This asks the question properly.  The thresholds are set on the other sources throughout, so
the rule is calibrated off-population exactly as it would be in a campaign, and only the
origin of the audit changes.  One source is held out as the screened population, and its
formulations are divided into an audit and a batch five ways:

  off source      the audit is drawn from the other sources, not from the held-out one.  This
                  is the naive arrangement and the one expected to fail.
  random          the held-out source split at random.  Exchangeable by construction, so this
                  is the positive control, reported to show the ceiling.
  by concentration  the low-concentration half audits the high-concentration half.  A real
                  division of the same population: same laboratory, same cells, same
                  operator, different part of the design space.
  by system       the source's salt--solvent systems are partitioned, so no system is shared
                  between audit and batch.  The hardest within-source division.
  as reported     the first half of the source's formulations in file order audits the second.
                  Non-random but chemically arbitrary, which separates "not random" from
                  "shifted in chemistry".

If the intervals hold under the three genuine divisions and fail off source, the prescription
is a result rather than advice.  If they fail under the genuine divisions too, then sampling
the batch's own population is not sufficient either, and the paper should say so.

Usage: python3 -I pilot_nonrandom.py CSV OUTDIR KSTAR REPS
"""
import sys, os, json, time, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calisol as C
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
MIN_HALF = 20          # formulations each side of a within-source division needs
MIN_CAL = 20           # calibration compounds of each class


def halves(kind, idx, conc, system, rng):
    """return (audit, batch) index arrays for one within-source division, or None"""
    if kind == "random":
        p = rng.permutation(len(idx)); h = len(idx) // 2
        return idx[p[:h]], idx[p[h:]]
    if kind == "by concentration":
        o = np.argsort(conc[idx] + rng.random(len(idx)) * 1e-9)
        h = len(idx) // 2
        return idx[o[:h]], idx[o[h:]]
    if kind == "as reported":
        h = len(idx) // 2
        return idx[:h], idx[h:]
    if kind == "by system":
        us = np.unique(system[idx]); rng.shuffle(us)
        # greedily fill the audit to about half the formulations, whole systems only
        target, got, take = len(idx) // 2, 0, set()
        for s in us:
            n = int((system[idx] == s).sum())
            if got < target:
                take.add(s); got += n
        m = np.array([s in take for s in system[idx]])
        if m.all() or not m.any():
            return None
        return idx[m], idx[~m]
    raise ValueError(kind)


def main():
    t0 = time.time()
    d, solv, counts = C.load()
    X = C.featurize(d, solv)
    y_all = (d.k.to_numpy() >= C.KSTAR).astype(int)
    rt = C.rt_table(d); rt["y"] = (rt.k >= C.KSTAR).astype(int)
    pos = {r: i for i, r in enumerate(d.index)}; rt["pos"] = rt.row.map(pos)
    Xrt = X[rt.pos.to_numpy()].copy(); Xrt[:, -1] = 298.15
    y = rt.y.to_numpy(); forms = rt.index.to_numpy(); doi = rt.doi.to_numpy()
    system = rt.system.to_numpy()
    conc = d.loc[rt.row.to_numpy(), "c"].to_numpy(float)
    form_of_row = d.form.to_numpy()

    info = {"formulations": int(len(rt)), "prevalence": float(y.mean()),
            "papers": int(len(np.unique(doi))), "kstar": C.KSTAR,
            "protocol": "thresholds from the other sources throughout; only the audit moves"}
    print(json.dumps(info), flush=True)

    sizes = pd.Series(doi).value_counts()
    cand = [s for s in sizes.index if sizes[s] >= 2 * MIN_HALF]
    print(f"sources with at least {2*MIN_HALF} formulations: {len(cand)}", flush=True)

    log = os.path.join(C.OUT, "pilot_nonrandom.jsonl")
    done = set()
    if os.path.exists(log):
        for line in open(log):
            o = json.loads(line); done.add((o["arm"], o["source"], o["rep"]))
        print(f"resuming, {len(done)} runs on disk", flush=True)
    fh = open(log, "a")
    runs = []

    for src in cand:
        inx = np.where(doi == src)[0]
        out = np.where(doi != src)[0]
        for rep in range(C.REPS):
            rng = np.random.default_rng(4242 + rep)
            # calibration and training from the other sources only
            p = rng.permutation(out)
            a, b = int(0.50 * len(p)), int(0.75 * len(p))
            tr_i, cal_i, off_i = p[:a], p[a:b], p[b:]   # off_i is the off-source audit
            if min(y[cal_i].sum(), (1 - y[cal_i]).sum()) < MIN_CAL:
                continue
            held = set(forms[cal_i]) | set(forms[off_i]) | set(forms[inx])
            mask = np.array([f not in held for f in form_of_row])
            clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                                     thread_count=2, random_seed=30000 + rep).fit(X[mask], y_all[mask])
            u = clf.predict_proba(Xrt)[:, 1]

            def run(arm, aud, bat):
                if (arm, src, rep) in done:
                    return
                if aud is None or len(aud) < MIN_HALF or len(bat) < MIN_HALF:
                    return
                if y[aud].sum() < 1 or y[bat].sum() < 1:
                    return
                if (1 - y[aud]).sum() < 1 or (1 - y[bat]).sum() < 1:
                    return
                g2 = np.random.default_rng(31337 + rep)
                uc, ua, uf = break_ties(g2, u[cal_i], u[aud], u[bat])
                o = {"arm": arm, "source": src, "rep": rep,
                     "n": {"cal": len(cal_i), "audit": len(aud), "batch": len(bat)},
                     "prevalence": {"cal": float(y[cal_i].mean()),
                                    "audit": float(y[aud].mean()),
                                    "batch": float(y[bat].mean())},
                     "auroc_batch": (float(roc_auc_score(y[bat], uf))
                                     if 0 < y[bat].sum() < len(bat) else None)}
                o["prevalence_ratio"] = float(y[aud].mean() / max(1e-12, y[bat].mean()))
                o.update(analyse(uc, y[cal_i], ua, y[aud], uf, y[bat]))
                q = [o["audit_intervals"][k]["inside"] for k in CATS if k in o["audit_intervals"]]
                o["lines_covered"] = [int(sum(q)), len(q)]
                fh.write(json.dumps(o) + "\n"); fh.flush(); runs.append(o)
                print(f"  {src[:28]:30} rep{rep:<2} {arm:18} n {len(aud)}/{len(bat)}"
                      f"  prev {o['prevalence']['audit']:.3f}/{o['prevalence']['batch']:.3f}"
                      f"  AUROC {o['auroc_batch'] and round(o['auroc_batch'],3)}"
                      f"  covered {sum(q)}/{len(q)}  [{time.time()-t0:.0f}s]", flush=True)

            # the naive arrangement: audit from the other sources, batch the held-out source
            run("off source", off_i, inx)
            for kind in ("random", "by concentration", "by system", "as reported"):
                h = halves(kind, inx, conc, system, np.random.default_rng(909 + rep))
                if h is None:
                    continue
                run(kind, h[0], h[1])

    fh.close()
    allruns = [json.loads(l) for l in open(log)]
    json.dump({"info": info, "runs": allruns},
              open(os.path.join(C.OUT, "pilot_nonrandom.json"), "w"), indent=1)
    print(f"\ntotal {time.time()-t0:.1f}s, {len(allruns)} runs")


if __name__ == "__main__":
    main()

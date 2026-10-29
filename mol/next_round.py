"""Declare a next round, then design the calibration and audit to match it, and see whether
the guarantee survives.

The library screen showed that holding out whole building blocks at the load-bearing
position voids the audit interval: the batch was promised 578 to 723 confirmed hits and
returned 119, and coverage over ten repeats fell to 35 percent against a nominal 95.  The
interval machinery was not at fault, since the same construction covered every line on a
random split and on a position-3 holdout.  It failed because a block-disjoint split is not
exchangeable with the batch it is meant to describe.

That suggests designing the audit to mirror the declared batch rather than to be maximally
strict.  The question this asks is what "mirror" has to mean.  A round is declared as a
batch that is 30 percent new chemistry at position 1, and three ways of preparing for it are
run through the identical pipeline with the identical model:

  naive     calibration and audit drawn only from blocks the model was trained on, as a
            team would do by habit.  The batch then contains chemistry nothing in the audit
            resembles.

  mirror    calibration and audit carry the same 30 percent novel fraction as the batch, but
            their novel blocks are disjoint from the batch's novel blocks.  This is the
            honest position of a team that has measured some new chemistry and is about to
            synthesise different new chemistry.  It tests whether matching the composition
            is enough.

  pilot     calibration and audit carry the same 30 percent novel fraction and draw it from
            the same blocks as the batch, split at the molecule level.  This is a pilot
            plate: a sample of the actual new chemistry is measured before the batch is
            committed.  Exchangeability holds by construction and the interval should cover,
            though it may be wide.

The three arms share one model per repeat, trained only on known-block molecules, so any
difference between them comes from the design of the calibration and audit sets and from
nothing else.

Usage: python3 -I next_round.py PARQUET OUTDIR REPS      (REPS = n, or first:last)
"""
import sys, os, json, time, numpy as np, pandas as pd

PARQUET, OUT = sys.argv[1], sys.argv[2]
R0, R1 = ((0, int(sys.argv[3])) if ":" not in sys.argv[3]
          else tuple(int(x) for x in sys.argv[3].split(":")))
sys.argv[3] = str(R1)   # cadel parses this as a plain count when imported

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cadel
from cadel import analyse
from conformal import break_ties
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
os.makedirs(OUT, exist_ok=True)
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
NSUB = 200000
NEW_FRAC = 0.15        # share of position-1 blocks treated as new chemistry
PHI = 0.30             # share of the declared batch built on that new chemistry
LABEL = "activity"
ARMS = ("naive", "mirror", "pilot")
CATS = ("confident hit", "false lead", "missed binder", "deferred binder",
        "correct rejection", "deferred non-binder")


def fingerprints(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s.replace("[Dy]", "[H]"))
        if m is not None:
            X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


def take(rng, pool, n):
    """draw n indices from pool without replacement, or all of it if it is smaller"""
    if n >= len(pool):
        return pool.copy(), np.empty(0, int)
    sel = rng.choice(len(pool), n, replace=False)
    mask = np.ones(len(pool), bool); mask[sel] = False
    return pool[sel], pool[mask]


def design(rng, bb1):
    """one draw of the whole experiment: which blocks are new, and the six index sets"""
    blocks = np.unique(bb1); rng.shuffle(blocks)
    n_new = int(NEW_FRAC * len(blocks))
    new_blocks, known_blocks = blocks[:n_new], blocks[n_new:]
    half = n_new // 2
    novel_a, novel_f = set(new_blocks[:half].tolist()), set(new_blocks[half:].tolist())

    is_known = np.isin(bb1, known_blocks)
    known = np.where(is_known)[0]; rng.shuffle(known)
    a = int(0.60 * len(known)); b = a + int(0.20 * len(known)); c = b + int(0.10 * len(known))
    train, cal_k, audit_k, future_k = known[:a], known[a:b], known[b:c], known[c:]

    nov_a = np.where(np.isin(bb1, list(novel_a)))[0]
    nov_f = np.where(np.isin(bb1, list(novel_f)))[0]
    nov_all = np.where(np.isin(bb1, new_blocks))[0]
    return {"train": train, "cal_k": cal_k, "audit_k": audit_k, "future_k": future_k,
            "nov_a": nov_a, "nov_f": nov_f, "nov_all": nov_all,
            "n_new_blocks": int(n_new), "n_blocks": int(len(blocks))}


def arms(rng, d):
    """calibration, audit and batch index sets for each of the three preparations.

    The batch is the same size in every arm and always carries PHI novel molecules, so the
    arms differ only in what the calibration and audit sets are allowed to contain.
    """
    out = {}
    n_fk = len(d["future_k"])
    n_bnov = int(PHI / (1 - PHI) * n_fk)          # novel molecules added to the batch

    # naive: the batch's novel chemistry comes from one disjoint pool; nothing else sees it
    bnov, _ = take(rng, d["nov_f"], n_bnov)
    out["naive"] = {"cal": d["cal_k"], "audit": d["audit_k"],
                    "batch": np.concatenate([d["future_k"], bnov])}

    # mirror: calibration and audit carry PHI novel too, from blocks the batch does not use
    n_cnov = int(PHI / (1 - PHI) * len(d["cal_k"]))
    n_anov = int(PHI / (1 - PHI) * len(d["audit_k"]))
    pool = d["nov_a"].copy()
    cnov, pool = take(rng, pool, n_cnov)
    anov, pool = take(rng, pool, n_anov)
    out["mirror"] = {"cal": np.concatenate([d["cal_k"], cnov]),
                     "audit": np.concatenate([d["audit_k"], anov]),
                     "batch": np.concatenate([d["future_k"], bnov])}

    # pilot: all three draw their novel molecules from the same blocks, split by molecule
    pool = d["nov_all"].copy()
    p_c, pool = take(rng, pool, n_cnov)
    p_a, pool = take(rng, pool, n_anov)
    p_b, pool = take(rng, pool, n_bnov)
    out["pilot"] = {"cal": np.concatenate([d["cal_k"], p_c]),
                    "audit": np.concatenate([d["audit_k"], p_a]),
                    "batch": np.concatenate([d["future_k"], p_b])}
    return out


def main():
    t_all = time.time()
    b = pd.read_parquet(PARQUET)
    rng0 = np.random.default_rng(0)
    sub = rng0.choice(len(b), min(NSUB, len(b)), replace=False)
    b = b.iloc[sub].reset_index(drop=True)
    X = fingerprints(b.molecule_smiles.to_numpy())
    bb1 = b["bb1_id"].to_numpy()
    y = (b.binds_sEH.to_numpy().astype(int) if LABEL == "activity"
         else ((b.binds_sEH == 1) & (b.binds_HSA == 0)).to_numpy().astype(int))
    info = {"library": "BELKA, soluble epoxide hydrolase", "label": LABEL, "n_sampled": len(b),
            "new_block_frac": NEW_FRAC, "batch_novel_frac": PHI,
            "prevalence": float(y.mean()),
            "declared_round": "30% of the batch is built on position-1 blocks the model never saw"}
    print(json.dumps(info, indent=1), flush=True)

    log = f"{OUT}/next_round.jsonl"
    done = set()
    if os.path.exists(log):
        for line in open(log):
            o = json.loads(line); done.add((o["arm"], o["rep"]))
        print(f"resuming, {len(done)} runs already on disk", flush=True)
    fh = open(log, "a")

    for r in range(R0, R1):
        rng = np.random.default_rng(90000 + r)
        d = design(rng, bb1)
        ti = d["train"]
        clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1,
                                 verbose=0, thread_count=2, random_seed=0).fit(X[ti], y[ti])
        u = clf.predict_proba(X)[:, 1]
        setup = arms(rng, d)
        for arm in ARMS:
            if (arm, r) in done:
                continue
            ci, ai_, fi = setup[arm]["cal"], setup[arm]["audit"], setup[arm]["batch"]
            o = {"arm": arm, "rep": r, "label": LABEL,
                 "n": {"train": len(ti), "cal": len(ci), "audit": len(ai_), "batch": len(fi)},
                 "blocks": {"total": d["n_blocks"], "new": d["n_new_blocks"]}}
            if min(y[ci].sum(), (1 - y[ci]).sum()) < 20 or y[ai_].sum() < 1 or y[fi].sum() < 1:
                o["skipped"] = True
                fh.write(json.dumps(o) + "\n"); fh.flush(); continue
            uc, ua, uf = break_ties(rng, u[ci], u[ai_], u[fi])
            yc, ya, yf = y[ci], y[ai_], y[fi]
            o["auroc_batch"] = float(roc_auc_score(yf, uf)) if 0 < yf.sum() < len(yf) else None
            o.update(analyse(uc, yc, ua, ya, uf, yf))
            cov = [o["audit_intervals"][c]["inside"] for c in CATS if c in o["audit_intervals"]]
            o["lines_covered"] = [int(sum(cov)), len(cov)]
            ch = o["audit_intervals"].get("confident hit", {})
            fh.write(json.dumps(o) + "\n"); fh.flush()
            print(f"  {arm:7} rep{r} AUROC {o['auroc_batch'] and round(o['auroc_batch'], 3)} "
                  f"covered {sum(cov)}/{len(cov)}  hits promised {ch.get('interval_future')} "
                  f"got {ch.get('future_count')}  [{time.time() - t_all:.0f}s]", flush=True)
    fh.close()
    runs = [json.loads(l) for l in open(log)]
    json.dump({"info": info, "runs": runs}, open(f"{OUT}/next_round.json", "w"))
    print(f"total {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

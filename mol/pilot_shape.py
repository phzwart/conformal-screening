"""Given a fixed pilot budget, what shape of pilot certifies the full library?

The declared-round experiment showed that a pilot restores the audit interval only when it
shares building blocks with the batch, and that matching the batch's composition without
sharing its blocks is worth nothing.  That leaves the practical question: a combinatorial
library is far too large to pilot by brute force, so which part of it must the pilot cover?

Strict exchangeability certifies only molecules all of whose blocks the pilot contained.
Drawn as a block subset at every position, a pilot of size n certifies n molecules and
nothing else, because the certified set is the pilot.  For a library of a thousand blocks at
each of three positions, a hundred thousand compounds certify one ten-thousandth of it.

The library screen suggests an escape.  Holding out position-3 blocks cost nothing and the
audit intervals covered every line, while holding out position-1 blocks voided them, so
novelty appears to be dangerous only at the position that carries the binding.  If that is
right, a pilot need not cover whole molecules; it needs to be complete at the load-bearing
position and may be thin everywhere else, which is cheaper by orders of magnitude.

Three pilot shapes of the same size are compared, each supplying its own training,
calibration and audit sets, and each used to quote intervals for the same full-library
batch:

  cube     a subset of blocks at all three positions, the brute-force shrunken library.
  cover1   every position-1 block, with few partners at the other two.
  cover3   every position-3 block, with few partners at the other two.  The control: if
           position 1 is what matters, completeness at position 3 should not help.

Usage: python3 -I pilot_shape.py PARQUET OUTDIR REPS      (REPS = n, or first:last)
"""
import sys, os, json, time, numpy as np, pandas as pd

PARQUET, OUT = sys.argv[1], sys.argv[2]
R0, R1 = ((0, int(sys.argv[3])) if ":" not in sys.argv[3]
          else tuple(int(x) for x in sys.argv[3].split(":")))
sys.argv[3] = str(R1)

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
NSUB = 500000
PILOT = 60000         # molecules the pilot may measure, identical in every arm
BATCH = 40000          # molecules in the declared full-library batch
FRACS = (0.40, 0.30)   # of the pilot: training, calibration; the rest is audit
ARMS = ("cube", "cover1", "cover3")
COLS = ("bb1_id", "bb2_id", "bb3_id")
CATS = ("confident hit", "false lead", "missed binder", "deferred binder",
        "correct rejection", "deferred non-binder")


def fingerprints(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s.replace("[Dy]", "[H]"))
        if m is not None:
            X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


def subset_mask(rng, bb, fracs):
    """molecules whose block at each position falls in a random subset of that position's
    blocks, the subset sizes given as fractions"""
    keep = np.ones(len(bb), bool)
    chosen = []
    for j, f in enumerate(fracs):
        u = np.unique(bb[:, j])
        if f >= 1.0:
            chosen.append(len(u)); continue
        rng.shuffle(u)
        sel = set(u[:max(1, int(f * len(u)))].tolist())
        keep &= np.array([v in sel for v in bb[:, j]])
        chosen.append(len(sel))
    return keep, chosen


def calibrate_fracs(rng, bb, shape, target):
    """find the per-position subset fractions of the given shape that yield about `target`
    molecules; the free fraction is found by bisection so the arms are the same size"""
    lo, hi = 0.005, 1.0
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        fr = [mid if s == "f" else 1.0 for s in shape]
        keep, _ = subset_mask(np.random.default_rng(12345), bb, fr)
        if keep.sum() < target:
            lo = mid
        else:
            hi = mid
    return [hi if s == "f" else 1.0 for s in shape]


def main():
    t_all = time.time()
    b = pd.read_parquet(PARQUET)
    rng0 = np.random.default_rng(0)
    sub = rng0.choice(len(b), min(NSUB, len(b)), replace=False)
    b = b.iloc[sub].reset_index(drop=True)
    X = fingerprints(b.molecule_smiles.to_numpy())
    bb = b[list(COLS)].to_numpy()
    y = b.binds_sEH.to_numpy().astype(int)
    shapes = {"cube": ["f", "f", "f"], "cover1": [1.0, "f", "f"], "cover3": ["f", "f", 1.0]}
    fr = {a: calibrate_fracs(rng0, bb, shapes[a], PILOT) for a in ARMS}
    info = {"library": "BELKA, soluble epoxide hydrolase", "n_sampled": len(b),
            "blocks": {c: int(b[c].nunique()) for c in COLS},
            "design_space": int(np.prod([b[c].nunique() for c in COLS])),
            "pilot": PILOT, "batch": BATCH, "prevalence": float(y.mean()),
            "subset_fractions": fr}
    print(json.dumps(info, indent=1), flush=True)

    log = f"{OUT}/pilot_shape.jsonl"
    done = set()
    if os.path.exists(log):
        for line in open(log):
            o = json.loads(line); done.add((o["arm"], o["rep"]))
        print(f"resuming, {len(done)} runs on disk", flush=True)
    fh = open(log, "a")

    for r in range(R0, R1):
        rng = np.random.default_rng(95000 + r)
        for arm in ARMS:
            if (arm, r) in done:
                continue
            keep, nblocks = subset_mask(rng, bb, fr[arm])
            pool = np.where(keep)[0]; rng.shuffle(pool)
            pilot = pool[:PILOT]
            rest = np.where(~keep)[0]
            batch = rng.choice(rest, min(BATCH, len(rest)), replace=False)
            a = int(FRACS[0] * len(pilot)); c = a + int(FRACS[1] * len(pilot))
            ti, ci, ai_ = pilot[:a], pilot[a:c], pilot[c:]
            o = {"arm": arm, "rep": r, "fracs": fr[arm], "blocks_in_pilot": nblocks,
                 "n": {"pool": int(keep.sum()), "pilot": len(pilot), "train": len(ti),
                       "cal": len(ci), "audit": len(ai_), "batch": len(batch)}}
            if min(y[ci].sum(), (1 - y[ci]).sum()) < 20 or y[ai_].sum() < 1 or y[batch].sum() < 1:
                o["skipped"] = True
                fh.write(json.dumps(o) + "\n"); fh.flush()
                print(f"  {arm:7} rep{r} skipped (cal in-class {int(y[ci].sum())})", flush=True)
                continue
            clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1,
                                     verbose=0, thread_count=2, random_seed=0).fit(X[ti], y[ti])
            u = clf.predict_proba(X)[:, 1]
            uc, ua, uf = break_ties(rng, u[ci], u[ai_], u[batch])
            yc, ya, yf = y[ci], y[ai_], y[batch]
            o["auroc_batch"] = float(roc_auc_score(yf, uf)) if 0 < yf.sum() < len(yf) else None
            o.update(analyse(uc, yc, ua, ya, uf, yf))
            cov = [o["audit_intervals"][k]["inside"] for k in CATS if k in o["audit_intervals"]]
            o["lines_covered"] = [int(sum(cov)), len(cov)]
            ch = o["audit_intervals"].get("confident hit", {})
            fh.write(json.dumps(o) + "\n"); fh.flush()
            print(f"  {arm:7} rep{r} blocks {nblocks} AUROC "
                  f"{o['auroc_batch'] and round(o['auroc_batch'], 3)} covered {sum(cov)}/{len(cov)} "
                  f"hits promised {ch.get('interval_future')} got {ch.get('future_count')} "
                  f"[{time.time() - t_all:.0f}s]", flush=True)
    fh.close()
    runs = [json.loads(l) for l in open(log)]
    json.dump({"info": info, "runs": runs}, open(f"{OUT}/pilot_shape.json", "w"))
    print(f"total {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

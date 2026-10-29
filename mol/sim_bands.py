"""Does block similarity predict whether an audit interval will cover?

The smoothness measurement found that along the rough axis, activity transfer rises with
the similarity of the two swapped blocks, from 10 times the independence baseline at 0.4 to
0.5 similarity up to 78 times above 0.7.  If that is the mechanism behind the coverage
failure, then a batch built on blocks close to the pilot's blocks should be priced correctly
and a batch built on distant ones should not, with the error growing as similarity falls.
That is a prediction, and it is the prediction that would turn the measurement into a check
worth running before committing a batch.

The design isolates it.  One pilot supplies the training, calibration and audit sets, so one
model and one audit serve every arm; the arms differ only in which held-out blocks their
batch is drawn from, graded by each block's maximum Tanimoto similarity to the pilot's
blocks.  Nothing else varies, which is what the earlier pilot-shape attempt got wrong when
its per-arm training sets let the model collapse in some arms and not others.

One band is a control that can refute the whole story: blocks whose fingerprints are
identical to a pilot block.  The model cannot distinguish those from blocks it trained on,
so if coverage still fails there, structural similarity is not what governs exchangeability
and something else is, most likely per-block differences in rate that structure does not
carry.

The two warhead blocks are dropped.  They hold 58 percent of the library's hits, so whether
they land in the pilot or in a batch would swamp any similarity effect; removing them leaves
a pool whose hit rate is near uniform across blocks and lets the similarity question be asked
on its own.

Usage: python3 -I sim_bands.py PARQUET BBPARQUET OUTDIR REPS     (REPS = n, or first:last)
"""
import sys, os, json, time, numpy as np, pandas as pd

PARQUET, BBP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
R0, R1 = ((0, int(sys.argv[4])) if ":" not in sys.argv[4]
          else tuple(int(x) for x in sys.argv[4].split(":")))
os.makedirs(OUT, exist_ok=True)
sys.argv = [sys.argv[0], os.path.dirname(PARQUET) or ".", OUT, str(R1)]  # what cadel expects

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cadel
from cadel import analyse
from conformal import break_ties
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, DataStructs

RDLogger.DisableLog("rdApp.*")
os.makedirs(OUT, exist_ok=True)
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
NSUB = 300000
PILOT_BLOCK_FRAC = 0.60
FRACS = (0.60, 0.20)          # of pilot molecules: train, calibration; rest is audit
WARHEADS = (87, 88)
BANDS = (("identical", 0.999, 1.01), ("near", 0.85, 0.999),
         ("mid", 0.72, 0.85), ("low", 0.0, 0.72))
CATS = ("confident hit", "false lead", "missed binder", "deferred binder",
        "correct rejection", "deferred non-binder")
LABEL = "binds_sEH"


def mol_fps(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s.replace("[Dy]", "[H]"))
        if m is not None:
            X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


def main():
    t_all = time.time()
    b = pd.read_parquet(PARQUET)
    b = b[~b.bb1_id.isin(WARHEADS)].reset_index(drop=True)
    rng0 = np.random.default_rng(0)
    sub = rng0.choice(len(b), min(NSUB, len(b)), replace=False)
    b = b.iloc[sub].reset_index(drop=True)
    X = mol_fps(b.molecule_smiles.to_numpy())
    y = b[LABEL].to_numpy().astype(int)
    bb1 = b.bb1_id.to_numpy()

    bbt = pd.read_parquet(BBP)
    b1 = bbt[bbt.position == 1].reset_index(drop=True)
    bfp = {int(i): GEN.GetFingerprint(Chem.MolFromSmiles(s))
           for i, s in zip(b1.bb_id, b1.smiles) if Chem.MolFromSmiles(s) is not None}
    blocks = np.array(sorted(set(bb1.tolist()) & set(bfp)))

    info = {"label": LABEL, "n_molecules": int(len(b)), "hit_rate": float(y.mean()),
            "blocks": int(len(blocks)), "warheads_dropped": list(WARHEADS),
            "bands": [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in BANDS]}
    print(json.dumps(info, indent=1), flush=True)

    log = f"{OUT}/sim_bands.jsonl"
    done = set()
    if os.path.exists(log):
        for line in open(log):
            o = json.loads(line); done.add((o["band"], o["rep"]))
        print(f"resuming, {len(done)} runs on disk", flush=True)
    fh = open(log, "a")

    for r in range(R0, R1):
        rng = np.random.default_rng(97000 + r)
        perm = rng.permutation(len(blocks))
        npil = int(PILOT_BLOCK_FRAC * len(blocks))
        pil_blocks = blocks[perm[:npil]]; out_blocks = blocks[perm[npil:]]
        pil_fps = [bfp[int(v)] for v in pil_blocks]
        maxsim = {int(v): max(DataStructs.BulkTanimotoSimilarity(bfp[int(v)], pil_fps))
                  for v in out_blocks}

        in_pilot = np.isin(bb1, pil_blocks)
        pool = np.where(in_pilot)[0]; rng.shuffle(pool)
        a = int(FRACS[0] * len(pool)); c = a + int(FRACS[1] * len(pool))
        ti, ci, ai_ = pool[:a], pool[a:c], pool[c:]
        if min(y[ci].sum(), (1 - y[ci]).sum()) < 20 or y[ai_].sum() < 1:
            print(f"  rep{r} pilot unusable (cal in-class {int(y[ci].sum())})", flush=True)
            continue
        clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                                 thread_count=2, random_seed=0).fit(X[ti], y[ti])
        u = clf.predict_proba(X)[:, 1]

        for name, lo, hi in BANDS:
            if (name, r) in done:
                continue
            sel = [v for v, s in maxsim.items() if lo <= s < hi]
            fi = np.where(np.isin(bb1, sel))[0]
            o = {"band": name, "rep": r, "sim_range": [lo, hi], "n_blocks": len(sel),
                 "n": {"train": len(ti), "cal": len(ci), "audit": len(ai_), "batch": len(fi)},
                 "mean_maxsim": float(np.mean([maxsim[v] for v in sel])) if sel else None}
            if len(fi) < 200 or y[fi].sum() < 1:
                o["skipped"] = True
                fh.write(json.dumps(o) + "\n"); fh.flush()
                print(f"  {name:9} rep{r} skipped ({len(sel)} blocks, {len(fi)} molecules)",
                      flush=True)
                continue
            uc, ua, uf = break_ties(rng, u[ci], u[ai_], u[fi])
            yc, ya, yf = y[ci], y[ai_], y[fi]
            o["rate_audit"] = float(ya.mean()); o["rate_batch"] = float(yf.mean())
            o["auroc_batch"] = float(roc_auc_score(yf, uf)) if 0 < yf.sum() < len(yf) else None
            o.update(analyse(uc, yc, ua, ya, uf, yf))
            cov = [o["audit_intervals"][k]["inside"] for k in CATS if k in o["audit_intervals"]]
            o["lines_covered"] = [int(sum(cov)), len(cov)]
            fh.write(json.dumps(o) + "\n"); fh.flush()
            print(f"  {name:9} rep{r} blocks {len(sel):>3} sim {o['mean_maxsim']:.3f} "
                  f"batch {len(fi):>6} AUROC {o['auroc_batch'] and round(o['auroc_batch'],3)} "
                  f"rate aud/bat {o['rate_audit']:.4f}/{o['rate_batch']:.4f} "
                  f"covered {sum(cov)}/{len(cov)}  [{time.time()-t_all:.0f}s]", flush=True)
    fh.close()
    runs = [json.loads(l) for l in open(log)]
    json.dump({"info": info, "runs": runs}, open(f"{OUT}/sim_bands.json", "w"))
    print(f"total {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

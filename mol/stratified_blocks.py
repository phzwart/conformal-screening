"""Is the block-holdout failure a change in chemistry, or a change in prevalence?

Withholding whole building blocks breaks the audit intervals, and the paper reads that as
the mapping from score to rate having moved with the chemistry.  There is a second
explanation the design does not exclude.  In-class molecules are concentrated by block --- a
few first-position blocks carry most of the binders --- so dealing whole blocks to the four
roles moves the in-class rate of the audit and of the batch independently.  A prevalence
shift of that size would break the intervals on its own, and it is something the audit set
reports without any new experiment.

The two are separated here by dealing the blocks differently and changing nothing else.

  shuffled    blocks are shuffled and split by molecule count, which is the published
              design: block-disjoint, and the in-class rate of each role left to chance.
  stratified  blocks are ordered by how many in-class molecules they carry and dealt to the
              four roles in turn, so that each role receives a comparable share of the rich
              blocks.  The design is still block-disjoint --- a batch molecule still carries
              a block no training molecule carries --- but the audit and the batch now hold
              the same in-class rate by construction.

If the stratified arm still fails, the loss is attributable to the chemistry and the paper's
reading stands.  If it recovers, the published result is substantially a prevalence shift,
and the remedy is to compare the audit's in-class rate against the library's rather than to
buy a pilot.

Usage: python3 -I stratified_blocks.py PARQUET OUTDIR REPS      (REPS = n, or first:last)
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
POS = 0                                   # the position whose blocks are dealt
ROLES = ("train", "cal", "audit", "batch")
ARMS = ("shuffled", "stratified")
CATS = ("confident hit", "false lead", "missed binder", "deferred binder",
        "correct rejection", "deferred non-binder")


def fingerprints(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s.replace("[Dy]", "[H]"))
        if m is not None:
            X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


def deal_shuffled(rng, blocks, bbcol, y):
    """the published design: shuffle blocks, cut by cumulative molecule count 40/20/20/20"""
    bl = blocks.copy(); rng.shuffle(bl)
    sizes = np.array([int((bbcol == v).sum()) for v in bl])
    cuts = np.cumsum(sizes) / sizes.sum()
    role = {}
    for v, c in zip(bl, cuts):
        role[v] = ROLES[0] if c <= 0.40 else ROLES[1] if c <= 0.60 else ROLES[2] if c <= 0.80 else ROLES[3]
    return role


def deal_stratified(rng, blocks, bbcol, y):
    """order blocks by in-class count and deal them to the four roles in turn, so each role
    receives a comparable share of the binder-rich blocks and the roles end at a comparable
    in-class rate; the assignment is still by whole block"""
    cnt = np.array([int(y[bbcol == v].sum()) for v in blocks])
    size = np.array([int((bbcol == v).sum()) for v in blocks])
    order = np.argsort(-(cnt + rng.random(len(cnt)) * 1e-6))     # rich blocks first, ties broken
    # deal round robin, but weight the training role twice so molecule shares stay near 40/20/20/20
    seq = ("train", "train", "cal", "audit", "batch")
    role = {}
    for j, i in enumerate(order):
        role[blocks[i]] = seq[j % len(seq)]
    return role


def main():
    t0 = time.time()
    b = pd.read_parquet(PARQUET)
    rng0 = np.random.default_rng(0)
    b = b.iloc[rng0.choice(len(b), min(NSUB, len(b)), replace=False)].reset_index(drop=True)
    X = fingerprints(b.molecule_smiles.to_numpy())
    y = b.binds_sEH.to_numpy().astype(int)
    bbcol = b[["bb1_id", "bb2_id", "bb3_id"]].to_numpy()[:, POS]
    blocks = np.unique(bbcol)

    info = {"library": "BELKA, soluble epoxide hydrolase", "position_dealt": POS + 1,
            "n_molecules": int(len(b)), "prevalence": float(y.mean()),
            "n_blocks": int(len(blocks))}
    print(json.dumps(info), flush=True)

    log = os.path.join(OUT, "stratified_blocks.jsonl")
    done = set()
    if os.path.exists(log):
        for line in open(log):
            o = json.loads(line); done.add((o["arm"], o["rep"]))
        print(f"resuming, {len(done)} runs on disk", flush=True)
    fh = open(log, "a")

    print(f"{'arm':12}{'rep':>4}{'AUROC':>8}{'prev aud':>10}{'prev bat':>10}"
          f"{'ratio':>7}{'covered':>10}", flush=True)
    for r in range(R0, R1):
        for arm in ARMS:
            if (arm, r) in done:
                continue
            rng = np.random.default_rng(61000 + r)
            role = (deal_shuffled if arm == "shuffled" else deal_stratified)(rng, blocks, bbcol, y)
            who = np.array([role[v] for v in bbcol])
            idx = {k: np.where(who == k)[0] for k in ROLES}
            for k in ROLES:
                rng.shuffle(idx[k])
            ti, ci, ai, fi = idx["train"], idx["cal"], idx["audit"], idx["batch"]
            o = {"arm": arm, "rep": r,
                 "n": {k: int(len(idx[k])) for k in ROLES},
                 "prevalence": {k: float(y[idx[k]].mean()) for k in ROLES}}
            if min(y[ti].sum(), (1 - y[ti]).sum()) < 20 or min(y[ci].sum(), (1 - y[ci]).sum()) < 20 \
               or y[ai].sum() < 1 or y[fi].sum() < 1:
                o["skipped"] = True
                fh.write(json.dumps(o) + "\n"); fh.flush()
                print(f"{arm:12}{r:>4}  skipped", flush=True); continue
            clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                                     thread_count=2, random_seed=0).fit(X[ti], y[ti])
            u = clf.predict_proba(X)[:, 1]
            uc, ua, uf = break_ties(rng, u[ci], u[ai], u[fi])
            yc, ya, yf = y[ci], y[ai], y[fi]
            o["auroc_batch"] = float(roc_auc_score(yf, uf)) if 0 < yf.sum() < len(yf) else None
            o["prevalence_ratio"] = float(y[ai].mean() / max(1e-12, y[fi].mean()))
            o.update(analyse(uc, yc, ua, ya, uf, yf))
            g = [o["audit_intervals"][k]["inside"] for k in CATS if k in o["audit_intervals"]]
            o["lines_covered"] = [int(sum(g)), len(g)]
            fh.write(json.dumps(o) + "\n"); fh.flush()
            print(f"{arm:12}{r:>4}{o['auroc_batch']:>8.3f}"
                  f"{o['prevalence']['audit']:>10.4f}{o['prevalence']['batch']:>10.4f}"
                  f"{o['prevalence_ratio']:>7.2f}{sum(g):>6}/{len(g):<3}"
                  f"   [{time.time()-t0:.0f}s]", flush=True)
    fh.close()
    runs = [json.loads(l) for l in open(log)]
    json.dump({"info": info, "runs": runs}, open(os.path.join(OUT, "stratified_blocks.json"), "w"))
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

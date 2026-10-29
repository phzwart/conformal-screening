"""How smooth is activity in building-block coordinates?

A combinatorial library is not a point cloud.  A molecule is a triple of building blocks,
and its nearest neighbours in the library differ from it at one position.  That makes the
library a lattice, and asks whether the response is smooth along each of its three axes.
Smoothness is what licenses interpolation, so it is also what a series ought to mean: a
region of the lattice over which the response moves little enough that measuring a few
members says something about the rest.  A Tanimoto ball is a crude stand-in for that, with
an arbitrary radius and a linkage rule to pick; the lattice coordinates are given by the
synthesis plan.

The screen showed that holding out whole building blocks at position 1 voided every audit
interval while holding out position 3 cost nothing.  If the lattice reading is right, that
is a statement about smoothness: swapping the block at position 1 moves the response a lot
and swapping at position 3 moves it little.  This measures that directly rather than
inferring it.

For each position, molecule pairs that share the other two blocks are enumerated, so the
pair differs by exactly one substitution.  Among pairs where at least one member binds, the
fraction where both bind is recorded against the Tanimoto similarity of the two swapped
blocks.  That curve is the empirical modulus of continuity along that axis: flat and high
means activity survives substitution there, so the axis is safe to interpolate along and to
extrapolate to new blocks; collapsing to the independence baseline means it does not.

The baseline for comparison is what the curve would read if the two members of a pair were
independent draws at the library's own hit rate, which for a rate p is p/(2-p).

Usage: python3 -I lipschitz.py PARQUET BBPARQUET OUT.json
"""
import sys, os, json, itertools, numpy as np, pandas as pd

PARQUET, BBP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, DataStructs

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
LABEL = "binds_sEH"
COLS = ("bb1_id", "bb2_id", "bb3_id")
BINS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01)
MAXPAIRS = 4_000_000


def block_fps(bb):
    """one fingerprint per building block, keyed by (position, bb_id)"""
    fps = {}
    for pos, bid, smi in zip(bb.position, bb.bb_id, bb.smiles):
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            fps[(int(pos), int(bid))] = GEN.GetFingerprint(m)
    return fps


def pairs_for_position(df, pos, y):
    """every pair of molecules sharing the two blocks other than `pos`"""
    others = [c for i, c in enumerate(COLS) if i != pos]
    swapped = COLS[pos]
    out = []
    g = df.groupby(others, sort=False)
    for _, idx in g.indices.items():
        if len(idx) < 2:
            continue
        if len(idx) > 60:                      # cap pathological groups
            idx = idx[:60]
        for a, b in itertools.combinations(idx, 2):
            if y[a] or y[b]:                   # only informative pairs
                out.append((a, b))
            if len(out) >= MAXPAIRS:
                return out, swapped
    return out, swapped


def main():
    b = pd.read_parquet(PARQUET)
    bb = pd.read_parquet(BBP)
    y = b[LABEL].to_numpy().astype(bool)
    p = float(y.mean())
    base = p / (2 - p)
    fps = block_fps(bb)
    ids = {c: b[c].to_numpy() for c in COLS}
    res = {"label": LABEL, "n_molecules": int(len(b)), "hit_rate": p,
           "independence_baseline": base, "bins": list(BINS), "positions": {}}
    print(f"{len(b)} molecules, hit rate {p:.5f}, independence baseline {base:.5f}", flush=True)

    for pos in (0, 1, 2):
        prs, swapped = pairs_for_position(b, pos, y)
        if not prs:
            res["positions"][swapped] = None; continue
        A = np.array([x[0] for x in prs]); B = np.array([x[1] for x in prs])
        both = (y[A] & y[B])
        ia, ib = ids[swapped][A], ids[swapped][B]
        sim = np.empty(len(prs), dtype=np.float32)
        cache = {}
        for i, (u, v) in enumerate(zip(ia, ib)):
            key = (u, v) if u <= v else (v, u)
            s = cache.get(key)
            if s is None:
                fu, fv = fps.get((pos + 1, int(u))), fps.get((pos + 1, int(v)))
                s = DataStructs.TanimotoSimilarity(fu, fv) if (fu and fv) else np.nan
                cache[key] = s
            sim[i] = s
        ok = ~np.isnan(sim)
        rows = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = ok & (sim >= lo) & (sim < hi)
            n = int(m.sum())
            rows.append({"lo": lo, "hi": hi, "n_pairs": n,
                         "p_both": float(both[m].mean()) if n else None,
                         "lift": float(both[m].mean() / base) if n else None})
        overall = float(both[ok].mean())
        res["positions"][swapped] = {"n_pairs": int(ok.sum()), "p_both_overall": overall,
                                     "lift_overall": overall / base, "by_similarity": rows}
        print(f"\n{swapped}: {int(ok.sum())} informative pairs, "
              f"P(both bind | one binds) = {overall:.4f}  = {overall/base:.1f}x baseline", flush=True)
        for r in rows:
            if r["n_pairs"]:
                print(f"   block similarity {r['lo']:.1f}-{r['hi']:.1f}: "
                      f"{r['n_pairs']:>8} pairs  P(both) {r['p_both']:.4f}  "
                      f"{r['lift']:>6.1f}x", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()

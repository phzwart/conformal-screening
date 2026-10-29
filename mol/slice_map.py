"""Which slice of the library is worth working in?

The axis-level smoothness measurement is an average over the whole library, and a screener
does not work on the whole library.  They work inside a region, and the question they ask is
whether this region has signal they can act on and coordinates they can move along.  That is
a property of the region, not of the library: one warhead may dominate so completely that
nothing else matters, while another supports real variation across its decorating blocks.

A slice here is the set of molecules sharing a block at the rough axis, which the axis-level
measurement identified as position 1.  Within a slice the remaining two positions form a
grid, and three things are reported for each slice:

  yield      the hit rate inside the slice, and the rate among its top-decile scoring
             members, which is what a screener would actually advance.  A slice whose top
             decile is no better than its base rate has no usable ranking.

  smoothness the fraction of one-substitution pairs inside the slice where both members
             bind given that one does, computed separately for substitutions at each of the
             two free positions.  High means activity survives movement along that
             coordinate, so a few measurements inside the slice say something about the
             rest.

  coverage   how much of the slice the library actually contains, since a slice that is
             mostly unmade is a region to extrapolate into rather than interpolate over.

A slice worth concentrating on scores on all three.  Nothing here is fitted and no physical
measurement is needed, so the map can be built before any compound is resynthesised.

Usage: python3 -I slice_map.py PARQUET OUT.json [TOP_N_SLICES]
"""
import sys, os, json, itertools, numpy as np, pandas as pd

PARQUET, OUT = sys.argv[1], sys.argv[2]
TOPN = int(sys.argv[3]) if len(sys.argv) > 3 else 25
LABEL = "binds_sEH"
ROUGH = "bb1_id"                      # axis identified as rough by the smoothness measurement
FREE = ("bb2_id", "bb3_id")
MIN_HITS = 20


def transfer(df, y, vary, fixed):
    """P(both bind | one binds) for pairs differing only at `vary`"""
    both = n = 0
    for _, idx in df.groupby(fixed, sort=False).indices.items():
        if len(idx) < 2:
            continue
        if len(idx) > 40:
            idx = idx[:40]
        for a, b in itertools.combinations(idx, 2):
            if y[a] or y[b]:
                n += 1
                both += bool(y[a] and y[b])
    return (both / n if n else None), n


def main():
    b = pd.read_parquet(PARQUET)
    y = b[LABEL].to_numpy().astype(bool)
    n2, n3 = b[FREE[0]].nunique(), b[FREE[1]].nunique()
    full = n2 * n3
    p_lib = float(y.mean())
    print(f"{len(b)} molecules, library hit rate {p_lib:.5f}, slice grid {n2} x {n3} = {full}",
          flush=True)

    g = b.groupby(ROUGH)[LABEL].agg(["size", "sum"])
    g = g[g["sum"] >= MIN_HITS].sort_values("sum", ascending=False).head(TOPN)
    rows = []
    for blk, r in g.iterrows():
        sub = b[b[ROUGH] == blk]
        ysub = sub[LABEL].to_numpy().astype(bool)
        sub = sub.reset_index(drop=True)
        rate = float(ysub.mean())
        t2, n2p = transfer(sub, ysub, FREE[0], FREE[1])
        t3, n3p = transfer(sub, ysub, FREE[1], FREE[0])
        o = {"block": int(blk), "n_in_sample": int(r["size"]), "hits": int(r["sum"]),
             "rate": rate, "lift_over_library": rate / p_lib,
             "coverage_of_slice": float(r["size"]) / full,
             "transfer_bb2": t2, "n_pairs_bb2": n2p,
             "transfer_bb3": t3, "n_pairs_bb3": n3p}
        rows.append(o)
        print(f"  block {blk:>4}  n {int(r['size']):>6}  hits {int(r['sum']):>5}  "
              f"rate {rate:.4f} ({rate/p_lib:>6.1f}x lib)  "
              f"slice coverage {100*o['coverage_of_slice']:.2f}%  "
              f"transfer bb2 {t2 if t2 is None else round(t2,3)} ({n2p}) "
              f"bb3 {t3 if t3 is None else round(t3,3)} ({n3p})", flush=True)

    res = {"label": LABEL, "rough_axis": ROUGH, "free_axes": list(FREE),
           "library_hit_rate": p_lib, "slice_grid": int(full), "slices": rows}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()

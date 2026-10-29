"""Can a fingerprint-based check see the distinctions that decide activity in this library?

The similarity-band experiment included a control: batches built on blocks whose fingerprints
are identical to a block the pilot trained on.  The model cannot tell those apart from blocks
it has seen, so if structural similarity governed exchangeability, that band should have been
the safe one.  It was the worst.

This asks what those identical-fingerprint blocks actually are.  Blocks at the first position
are grouped by their exact fingerprint bitstring, each group is classified by comparing its
members' canonical SMILES with and without stereochemistry, and the library hit rate of each
member is reported.  If the groups are stereoisomers, then the feature set is blind to a
distinction the assay is not, and no check built on those features can certify a batch that
differs stereochemically from the audit, however the similarity threshold is set.

That is a statement about the representation rather than about the model or the library, and
it bounds what any amount of calibration data can buy.

Usage: python3 -I stereo_blocks.py PARQUET BBPARQUET OUT.json
"""
import sys, json, collections, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
PARQUET, BBP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
LABEL = "binds_sEH"
POSITION = 1
MIN_N = 200          # a block must appear this often for its rate to carry a ratio


def classify(smis):
    flat = {Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in smis}
    stereo = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smis}
    if len(stereo) == 1:
        return "duplicate"
    return "stereoisomers" if len(flat) == 1 else "other"


def main():
    bb = pd.read_parquet(BBP)
    b1 = bb[bb.position == POSITION].reset_index(drop=True)
    groups = collections.defaultdict(list)
    for bid, smi in zip(b1.bb_id, b1.smiles):
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            groups[GEN.GetFingerprint(m).ToBitString()].append((int(bid), smi))
    dup = [v for v in groups.values() if len(v) > 1]

    b = pd.read_parquet(PARQUET)
    g = b.groupby(f"bb{POSITION}_id")[LABEL].agg(["size", "sum"])
    g["rate"] = g["sum"] / g["size"]

    kinds = collections.Counter()
    rows = []
    for v in dup:
        ids = [i for i, _ in v]
        kind = classify([s for _, s in v]); kinds[kind] += 1
        present = [i for i in ids if i in g.index]
        rates = [float(g.loc[i, "rate"]) for i in present]
        sizes = [int(g.loc[i, "size"]) for i in present]
        hits = [int(g.loc[i, "sum"]) for i in present]
        # a block appearing a handful of times gives a rate that is mostly noise, and a
        # ratio against it is meaningless, so a spread is only quoted when every member of
        # the group is measured often enough to carry one
        ok = len(rates) > 1 and min(sizes) >= MIN_N and min(rates) > 0
        o = {"blocks": ids, "kind": kind, "rates": rates, "n": sizes, "hits": hits,
             "well_measured": bool(ok),
             "fold_spread": (max(rates) / min(rates)) if ok else None,
             "smiles": [s for _, s in v]}
        rows.append(o)
    rows.sort(key=lambda o: -(o["fold_spread"] or 0))

    n_in_groups = sum(len(v) for v in dup)
    res = {"label": LABEL, "position": POSITION, "n_blocks": int(len(b1)),
           "n_groups": len(dup), "n_blocks_in_groups": n_in_groups,
           "group_kinds": dict(kinds), "groups": rows}
    print(f"position-{POSITION} blocks: {len(b1)}")
    print(f"fingerprint-identical groups: {len(dup)}, covering {n_in_groups} blocks "
          f"({100*n_in_groups/len(b1):.0f}%)")
    print(f"group kinds: {dict(kinds)}")
    print()
    nw = sum(1 for o in rows if o["well_measured"])
    print(f"groups where every member is measured at least {MIN_N} times: {nw}")
    print("hit-rate spread inside those groups:")
    for o in rows[:8]:
        if o["fold_spread"]:
            print(f"   blocks {o['blocks']}  rates " +
                  ", ".join(f"{x:.4f}" for x in o["rates"]) +
                  f"   {o['fold_spread']:.2f}-fold  ({o['kind']})")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()

"""Does the label-free check see the arrangement that defeats the guarantee?

The pre-flight test of the main text was reported on four split designs: random assignment,
and the three holdouts in which every molecule carrying a given building block is withheld.
In all four the answer is known before the test is run, because a withheld block is a
substructure that the fingerprint represents directly, so the classifier separates the sets
at once and the block-share column gives the same verdict by inspection.

The arrangements the check has to adjudicate are the three preparations of a declared round.
All three carry the same share of novel chemistry in the batch; they differ in where the
calibration and audit sets draw theirs.  Two of them lose the guarantee and one keeps it,
and no fingerprint distinguishes the two that differ only in which novel blocks the audit
holds --- unless it does, which is what this measures.

For each preparation and repeat the same classifier is trained to separate audit molecules
from batch molecules on fingerprints alone, with the comparison between calibration and
batch reported alongside, a permutation null obtained by shuffling the audit/batch labels,
and the block shares the design controls.  The coverage each preparation achieved is read
from the runs already on disk, so the test and the outcome sit in one table.

If mirror reads near one half while covering 63%, the check passes an arrangement that
destroys the guarantee and the workflow step that recommends it is wrong.

Usage: python3 -I preflight_arms.py PARQUET OUTDIR REPS      (REPS = n, or first:last)
"""
import sys, os, json, time, numpy as np

PARQUET, OUT = sys.argv[1], sys.argv[2]
R0, R1 = ((0, int(sys.argv[3])) if ":" not in sys.argv[3]
          else tuple(int(x) for x in sys.argv[3].split(":")))
os.makedirs(OUT, exist_ok=True)
sys.argv = [sys.argv[0], PARQUET, OUT, str(R1)]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import next_round as nr                      # parses argv at import; set above
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

CAP = 12000                                  # molecules per side, as in the published check
NPERM = 20                                   # permutation replicates for the null
ARMS = ("naive", "mirror", "pilot")


def domain_auc(X, ia, ib, rng, cap=CAP):
    """held-out area under the curve for telling one set of molecules from another"""
    a = rng.choice(ia, min(cap, len(ia)), replace=False)
    b = rng.choice(ib, min(cap, len(ib)), replace=False)
    idx = np.concatenate([a, b]); lab = np.r_[np.zeros(len(a)), np.ones(len(b))]
    p = rng.permutation(len(idx)); idx, lab = idx[p], lab[p]
    half = len(idx) // 2
    clf = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, verbose=0,
                             thread_count=2, random_seed=0).fit(X[idx[:half]], lab[:half])
    return float(roc_auc_score(lab[half:], clf.predict_proba(X[idx[half:]])[:, 1]))


def perm_null(X, ia, ib, rng, nperm=NPERM, cap=CAP):
    """the same statistic with the audit/batch labels shuffled, which is the null it needs.

    The cap must match the one the observed statistic uses.  An earlier version drew the null
    at 3000 molecules per side against an observed statistic at 12000, which inflates the
    null's spread relative to the statistic's own sampling distribution, and inflates it in
    the direction that makes a reading near one half easier to obtain.
    """
    pool = np.concatenate([ia, ib])
    out = []
    for j in range(nperm):
        q = rng.permutation(len(pool))
        half = len(pool) // 2
        out.append(domain_auc(X, pool[q[:half]], pool[q[half:]],
                              np.random.default_rng(7000 + j), cap=cap))
    return out


def main():
    t0 = time.time()
    b = pd.read_parquet(PARQUET)
    rng0 = np.random.default_rng(0)
    b = b.iloc[rng0.choice(len(b), min(nr.NSUB, len(b)), replace=False)].reset_index(drop=True)
    X = nr.fingerprints(b.molecule_smiles.to_numpy())
    bb = b[["bb1_id", "bb2_id", "bb3_id"]].to_numpy()
    bb1 = b["bb1_id"].to_numpy()
    y = b.binds_sEH.to_numpy().astype(int)

    # coverage already measured for these preparations
    cov = {}
    log = os.path.join(OUT, "next_round.jsonl")
    if os.path.exists(log):
        for line in open(log):
            o = json.loads(line)
            if o.get("skipped"):
                continue
            c = o.get("lines_covered")
            if c:
                cov.setdefault(o["arm"], [0, 0])
                cov[o["arm"]][0] += c[0]; cov[o["arm"]][1] += c[1]

    out_path = os.path.join(OUT, "preflight_arms.json")
    prior = []
    if os.path.exists(out_path):
        prior = json.load(open(out_path)).get("rows", [])
    res = {"cap": CAP, "nperm": NPERM, "coverage_on_disk": cov, "rows": list(prior)}
    have = {(o["arm"], o["rep"]) for o in prior}
    print(f"{'arm':8}{'rep':>4}{'audit|batch':>13}{'cal|batch':>11}"
          f"{'null mean':>11}{'null max':>10}{'bb1':>7}{'bb2':>7}{'bb3':>7}"
          f"{'prev aud':>10}{'prev bat':>10}", flush=True)
    for r in range(R0, R1):
        rng = np.random.default_rng(90000 + r)
        d = nr.design(rng, bb1)
        A = nr.arms(rng, d)
        for arm in ARMS:
            if (arm, r) in have:
                continue
            s = A[arm]
            ai, fi, ci = s["audit"], s["batch"], s["cal"]
            g = np.random.default_rng(5000 + r)
            auc_ab = domain_auc(X, ai, fi, g)
            auc_cb = domain_auc(X, ci, fi, np.random.default_rng(6000 + r))
            null = perm_null(X, ai, fi, np.random.default_rng(8000 + r))
            shares = []
            for j in range(3):
                seen = set(np.unique(bb[ai, j]).tolist())
                shares.append(float(np.mean([v in seen for v in bb[fi, j]])))
            row = {"arm": arm, "rep": r,
                   "auc_audit_vs_batch": auc_ab, "auc_cal_vs_batch": auc_cb,
                   "null_mean": float(np.mean(null)), "null_max": float(np.max(null)),
                   "null": null, "block_share": shares,
                   "prevalence_audit": float(y[ai].mean()),
                   "prevalence_batch": float(y[fi].mean()),
                   "n_audit": int(len(ai)), "n_batch": int(len(fi))}
            res["rows"].append(row)
            print(f"{arm:8}{r:>4}{auc_ab:>13.3f}{auc_cb:>11.3f}"
                  f"{np.mean(null):>11.3f}{np.max(null):>10.3f}"
                  f"{shares[0]:>7.2f}{shares[1]:>7.2f}{shares[2]:>7.2f}"
                  f"{row['prevalence_audit']:>10.4f}{row['prevalence_batch']:>10.4f}"
                  f"   [{time.time()-t0:.0f}s]", flush=True)
            json.dump(res, open(out_path, "w"), indent=1)
    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

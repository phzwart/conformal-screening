"""Does a screen's apparent skill come from telling series apart, or from ranking within one?

A marginal AUROC is an average over all discordant pairs, and in a clumpy chemical space
almost every such pair joins two different series.  The statistic then measures series
recognition, which is real but is not the discrimination a project needs once it is working
inside a series.  Worse, a pooled figure can sit below every subgroup when the between-group
contrast runs against the within-group relation, so a marginal number is not even bounded by
what it summarises.

This splits the concordance of an audit set in two.  Compounds are clustered by sphere
exclusion at a Tanimoto radius; a pair is within-cluster if both members fall in the same
cluster and between-cluster otherwise.  Reported for each screen: the overall AUROC, the two
concordances, how many pairs support each, and how much of the audit sits in clusters that
contain both classes, since only those can contribute a within-cluster pair.

Nothing here is refitted.  The stored calibration and audit scores of the molecular runs are
read as they are, so this characterises exactly the screens the paper reports.

Usage: python3 -I within_between.py OUT.json [RADIUS]
"""
import sys, os, json, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score

OUT = sys.argv[1]
RADIUS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
HERE = os.path.dirname(os.path.abspath(__file__))
MAXN = 4000


def cluster(X, radius):
    """greedy sphere exclusion: densest fingerprints seed clusters"""
    order = np.argsort(-X.sum(1)); lab = -np.ones(len(X), int); c = 0
    n = X.sum(1)
    for i in order:
        if lab[i] >= 0:
            continue
        inter = X @ X[i]
        sim = inter / np.maximum(n + n[i] - inter, 1)
        lab[np.where((sim >= radius) & (lab < 0))[0]] = c; c += 1
    return lab


def decompose(u, y, lab):
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    win = []; btw = []
    for i in pos:
        same = neg[lab[neg] == lab[i]]; diff = neg[lab[neg] != lab[i]]
        win.extend(u[i] > u[same]); btw.extend(u[i] > u[diff])
    mixed = [k for k in np.unique(lab) if 0 < y[lab == k].sum() < (lab == k).sum()]
    in_mixed = int(sum((lab == k).sum() for k in mixed))
    return {"auroc": float(roc_auc_score(y, u)),
            "n_within": len(win), "within": float(np.mean(win)) if win else None,
            "n_between": len(btw), "between": float(np.mean(btw)) if btw else None,
            "clusters": int(len(np.unique(lab))), "mixed_clusters": len(mixed),
            "frac_audit_in_mixed": in_mixed / len(y)}


def report(name, res):
    if res is None:
        print(f"{name}: no usable pairs"); return
    w = "n/a" if res["within"] is None else "%.3f" % res["within"]
    print(f"{name}: AUROC {res['auroc']:.3f} | within {w} ({res['n_within']} pairs) | "
          f"between {res['between']:.3f} ({res['n_between']} pairs) | "
          f"{res['mixed_clusters']}/{res['clusters']} clusters mixed, "
          f"{100 * res['frac_audit_in_mixed']:.0f}% of the audit in them")


def main():
    out = {"radius": RADIUS, "screens": {}}

    # --- AqSolDB: lipophilic solubility screen, split 0
    F = np.load(f"{HERE}/aqsoldb_features.npz", allow_pickle=True)
    D = np.load(f"{HERE}/aqsol_runs.npz")
    idx = D["r0_test_idx"]; keep = D["r0_test_logP"] > 3.5
    idx = idx[keep]; u = D["r0_test_u"][keep]; y = D["r0_test_y"][keep]
    X = F["X"][idx].astype(np.float32)
    r = decompose(u, y, cluster(X, RADIUS)); out["screens"]["AqSolDB lipophilic"] = r
    report("AqSolDB lipophilic ", r)

    # --- Tox21: every endpoint, random split 0
    T = np.load(f"{HERE}/tox21_features.npz", allow_pickle=True)
    tasks = list(T["tasks"])
    for t in tasks:
        f = f"{HERE}/v2/scores/{t}_random_0.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        ti = z["test"]
        if len(ti) > MAXN:
            sel = np.random.default_rng(0).choice(len(ti), MAXN, replace=False)
            ti2, u, y = ti[sel], z["u_test"][sel], z["y_test"][sel]
        else:
            ti2, u, y = ti, z["u_test"], z["y_test"]
        X = T["X"][ti2].astype(np.float32)
        r = decompose(u, y, cluster(X, RADIUS)); out["screens"][f"Tox21 {t}"] = r
        report(f"Tox21 {t:<16}", r)

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()

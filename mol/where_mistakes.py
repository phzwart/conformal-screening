"""Where in the ranking a proxy's mistakes have to be absent.

A campaign chooses a quantile: the compounds it ranks above some cut are the ones it will
make.  The rule used here is of that shape, with one cut per class, each an order statistic
of the calibration scores of its own class.  At a coverage request of 1 - alpha, the
out-class cut sits near the (1 - alpha) quantile of the out-class scores and the in-class cut
near the alpha quantile of the in-class scores.

If the shortfall is the gap between two class-conditional score distributions at the cut,
then a proxy's mistakes matter only where they overlap that cut, and each class should be
damaged only by mistakes near its own.  That is a sharp prediction and this tests it: a fixed
fraction of each class is relabelled, but the compounds relabelled are chosen in a window
centred on a target position in the within-class ranking, and the position is swept from one
end to the other.  Coverage is then read as a function of where the mistakes were put.

The prediction is a dip in out-class coverage when the window reaches the upper tail of the
out-class scores, a dip in in-class coverage when it reaches the lower tail of the in-class
scores, and no appreciable loss anywhere else -- with the error rate, the confusion matrix
and the implied prevalence held constant throughout, so that nothing a label-only summary can
see distinguishes the conditions.

Usage: python3 -I where_mistakes.py DATADIR OUT.json [REPS]
"""
import sys, os, json, time, numpy as np

DATA, OUT = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import lru_cache
import intervals, conformal
from conformal import thresholds, cells, break_ties

TH = lru_cache(maxsize=None)(intervals.thatcher)
k_batch = lru_cache(maxsize=None)(conformal.k_batch)

A = D_ = 0.10
G = 0.05
M = 1000
ISOFORMS = ("CAII", "CAIX", "CAXII")
CUTOFFS = (7, 8)
FLIP = 0.10                                     # share of each class relabelled, held fixed
POSITIONS = (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95)
G8 = {"correct rejection": ["0_s0"], "false lead": ["0_s1"],
      "deferred non-binder": ["0_both", "0_empty"], "confident hit": ["1_s1"],
      "missed binder": ["1_s0"], "deferred binder": ["1_both", "1_empty"]}


def grouped(c):
    return {k: sum(c[x] for x in v) for k, v in G8.items()}


def load_score(iso):
    """the frozen score and the measured affinities, written by the proxy-structure run"""
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), f".score_{iso}.npz")
    if not os.path.exists(cache):
        raise SystemExit(f"no cached score for {iso}; run surrogate_structure.py first")
    z = np.load(cache)
    return z["u_ch"], z["pki"], json.loads(str(z["info"]))


def flip_at(rng, y, u, f, pos):
    """relabel a fraction f of each class, taken from a window centred at within-class
    ranking position `pos`.  The count per class does not depend on pos, so the confusion
    matrix against the truth is the same at every position; only the location moves."""
    ys = y.copy()
    jit, = break_ties(rng, u)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        k = int(round(f * len(idx)))
        if not k:
            continue
        order = idx[np.argsort(jit[idx])]           # ascending score within the class
        centre = int(round(pos * (len(order) - 1)))
        lo = min(max(0, centre - k // 2), len(order) - k)
        ys[order[lo:lo + k]] = 1 - cls
    return ys


def main():
    t0 = time.time()
    res = {"alpha": A, "gamma": G, "reps": REPS, "flip": FLIP,
           "positions": list(POSITIONS), "isoforms": {}, "rows": []}
    print(f"flipping {FLIP:.0%} of each class, moving only where in the ranking they sit")
    print(f"{'iso':6}{'cut':>4}{'position':>10}{'cov out':>9}{'cov in':>8}"
          f"{'viol out':>10}{'viol in':>9}{'gap out':>9}{'gap in':>8}"
          f"{'tau0 q':>8}{'tau1 q':>8}{'audit':>10}", flush=True)
    for iso in ISOFORMS:
        u_all, pki, info = load_score(iso)
        res["isoforms"][iso] = info
        n = len(u_all)
        for c in CUTOFFS:
            y_true = (pki >= c).astype(int)
            for pos in POSITIONS:
                acc = {"c0": [], "c1": [], "v0": 0, "v1": 0, "g0": [], "g1": [],
                       "q0": [], "q1": [], "disc": [], "aud": [0, 0]}
                for r in range(REPS):
                    rng = np.random.default_rng(57000 + 97 * r + 7 * c + int(1000 * pos))
                    y_sur = flip_at(rng, y_true, u_all, FLIP, pos)
                    acc["disc"].append(float(np.mean(y_sur != y_true)))

                    p = rng.permutation(n)
                    a, b = int(0.30 * n), int(0.60 * n)
                    ci, ai, fi = p[:a], p[a:b], p[b:]
                    uc, ua, uf = break_ties(rng, u_all[ci], u_all[ai], u_all[fi])
                    ys_c = y_sur[ci]
                    n0, n1 = int((ys_c == 0).sum()), int((ys_c == 1).sum())
                    if min(n0, n1) < 20:
                        continue
                    m1 = int(round(M * n1 / (n0 + n1))); m0 = M - m1
                    t_0, t_1 = thresholds(uc, ys_c, k_batch(n0, A, D_, m0),
                                          k_batch(n1, A, D_, m1))

                    ya, yf = y_true[ai], y_true[fi]
                    cc = cells(ua, ya, t_0, t_1)
                    ma0, ma1 = int((ya == 0).sum()), int((ya == 1).sum())
                    cov0 = (cc["0_s0"] + cc["0_both"]) / max(1, ma0)
                    cov1 = (cc["1_s1"] + cc["1_both"]) / max(1, ma1)
                    acc["c0"].append(cov0); acc["c1"].append(cov1)
                    acc["v0"] += int(cov0 < 1 - A); acc["v1"] += int(cov1 < 1 - A)

                    uc0, uc1 = uc[ys_c == 0], uc[ys_c == 1]
                    ua0, ua1 = ua[ya == 0], ua[ya == 1]
                    acc["g0"].append(float((ua0 <= t_0).mean() - (uc0 <= t_0).mean()))
                    acc["g1"].append(float((ua1 >= 1 - t_1).mean() - (uc1 >= 1 - t_1).mean()))
                    # where each cut sits in the within-class ranking of the truth's classes
                    acc["q0"].append(float((u_all[y_true == 0] <= t_0).mean()))
                    acc["q1"].append(float((u_all[y_true == 1] <= 1 - t_1).mean()))

                    ga, gf = grouped(cc), grouped(cells(uf, yf, t_0, t_1))
                    for k in ga:
                        lo, hi = TH(ga[k], len(ya), len(yf), G)
                        acc["aud"][1] += 1; acc["aud"][0] += int(lo <= gf[k] <= hi)

                if not acc["c0"]:
                    continue
                row = {"isoform": iso, "cutoff": c, "position": pos,
                       "reps": len(acc["c0"]), "disagreement": float(np.mean(acc["disc"])),
                       "cov_out": float(np.mean(acc["c0"])),
                       "cov_in": float(np.mean(acc["c1"])),
                       "viol_out": [acc["v0"], len(acc["c0"])],
                       "viol_in": [acc["v1"], len(acc["c1"])],
                       "cdf_gap_out": float(np.mean(acc["g0"])),
                       "cdf_gap_in": float(np.mean(acc["g1"])),
                       "tau_out_quantile": float(np.mean(acc["q0"])),
                       "tau_in_quantile": float(np.mean(acc["q1"])),
                       "audit_lines_covered": acc["aud"]}
                res["rows"].append(row)
                print(f"{iso:6}{c:>4}{pos:>10.2f}{row['cov_out']:>9.3f}{row['cov_in']:>8.3f}"
                      f"{acc['v0']:>7}/{len(acc['c0']):<3}{acc['v1']:>6}/{len(acc['c1']):<3}"
                      f"{row['cdf_gap_out']:>+9.3f}{row['cdf_gap_in']:>+8.3f}"
                      f"{row['tau_out_quantile']:>8.2f}{row['tau_in_quantile']:>8.2f}"
                      f"{acc['aud'][0]:>5}/{acc['aud'][1]:<4}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwritten to {OUT}  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

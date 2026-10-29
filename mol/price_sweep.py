"""Alpha sweep, Pareto front and priced consequences for a screen.

Takes the calibration and audit scores of one screen, sweeps the two coverage requests over
the 12 x 12 grid, and reports for every setting the audited consequences of a batch of
m = 1000 candidates: the expected counts and the one-sided bounds that hold jointly over
the whole grid, so a setting chosen by looking at the map keeps its guarantee.

Three things are then derived from that map.

  Pareto front   The settings not dominated on the four bounded outcomes (fewer
                 measurements, false leads and misses, more confirmed hits).  With monotone
                 costs an optimal setting always lies on it, so the front is the complete
                 menu and no price needs to be named to draw it.

  Price list     Costs are counted in units of one measurement, the assay that resolves a
                 deferred candidate.  A false lead costs LAM_FL measurements, being the
                 downstream work spent on a candidate that turns out not to qualify; a
                 missed in-class candidate costs LAM_MISS, the opportunity given up.  The
                 total for a setting is

                     measurements + LAM_FL * false leads + LAM_MISS * missed

                 reported both at the expected counts and at the joint bounds, the latter
                 being what a team can commit to.  Because the ratios are rarely known, the
                 cheapest setting is reported over a grid of them rather than at one point.

  Price curve    For a run of measurement budgets, the most confirmed hits that can be
                 guaranteed within the budget, and the marginal cost of the next hit.  This
                 is the shadow price of the assay budget, in measurements per hit.

A rare in-class rate makes the exercise degenerate at a small batch: when a setting is
expected to confirm a couple of dozen candidates per 1000, the joint lower bound on that
count falls to zero and every setting survives on the Pareto front, so nothing can be
priced.  The batch size is therefore an argument, and a screen for a rare class has to be
run at the scale such a campaign actually operates.

Usage: python3 -I price_sweep.py SCORES.npz OUT.json [BATCH]
       where SCORES.npz holds u_cal, y_cal, u_audit, y_audit.
"""
import sys, os, json, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenario_map as SM

if len(sys.argv) > 3:
    SM.M = int(sys.argv[3])
M = SM.M
LAM_FL = [5, 10, 20, 50]        # cost of a false lead, in measurements
LAM_MISS = [5, 20, 50, 200]     # cost of a missed in-class candidate, in measurements
BUDGETS = None  # set from the batch size: 10% .. 90%


def pareto(V):
    """indices of rows not dominated; V is minimised column-wise"""
    keep = []
    for i in range(len(V)):
        if not any(np.all(V[j] <= V[i]) and np.any(V[j] < V[i]) for j in range(len(V)) if j != i):
            keep.append(i)
    return np.array(keep)


def main():
    z = np.load(sys.argv[1])
    G, info = SM.grid(z["u_cal"], z["y_cal"], z["u_audit"], z["y_audit"])
    A = SM.ALPHAS
    flat = lambda k, q: G[k][q].ravel()
    # minimise measurements, false leads, misses; maximise confirmed hits
    V = np.stack([flat("deferred", "bound"), flat("false lead", "bound"),
                  flat("missed in-class", "bound"), -flat("confirmed in-class", "bound")], 1)
    front = pareto(V)
    res = {"info": info, "requests": SM.J, "per_bound_level": SM.LEV, "m": M,
           "n_pareto": int(len(front)), "n_settings": int(len(V))}
    print(f"audit {info['n_audit']}, calibration {info['n_cal']}, batch m={M}")
    print(f"{len(V)} settings, {len(front)} on the Pareto front of the four bounded outcomes\n")

    def total(q, lf, lm):
        return flat("deferred", q) + lf * flat("false lead", q) + lm * flat("missed in-class", q)

    print("cheapest setting over cost ratios (cost per 1000 candidates, in measurements)")
    print("  false lead  missed   alpha0 alpha1   expected   guaranteed   hits(exp/bound)")
    rows = []
    for lf in LAM_FL:
        for lm in LAM_MISS:
            c_pt = total("point", lf, lm); i = int(np.argmin(c_pt))
            c_bd = total("bound", lf, lm)
            r = {"lam_fl": lf, "lam_miss": lm, "alpha0": float(A[i // len(A)]),
                 "alpha1": float(A[i % len(A)]), "cost_expected": float(c_pt[i]),
                 "cost_bound": float(c_bd[i]),
                 "hits_expected": float(flat("confirmed in-class", "point")[i]),
                 "hits_bound": float(flat("confirmed in-class", "bound")[i]),
                 "on_pareto": bool(i in front)}
            rows.append(r)
            print("  %9d %7d   %5.3f %6.3f   %8.0f   %10.0f   %4.0f / %.0f%s" % (
                lf, lm, r["alpha0"], r["alpha1"], r["cost_expected"], r["cost_bound"],
                r["hits_expected"], r["hits_bound"], "" if r["on_pareto"] else "  (off front)"))
    res["cheapest"] = rows

    print("\nprice of the measurement budget (guaranteed hits within budget)")
    print("  budget   best setting   hits>=   measurements<=   marginal cost of a hit")
    dbound = flat("deferred", "bound"); hbound = flat("confirmed in-class", "bound")
    curve = []; prev = None
    for U in [int(f * M) for f in (.1,.2,.3,.4,.5,.6,.7,.8,.9)]:
        ok = dbound <= U
        if not ok.any():
            print("  %6d   infeasible" % U); curve.append({"budget": U, "feasible": 0}); continue
        j = int(np.where(ok, hbound, -np.inf).argmax())
        mc = None if prev is None or hbound[j] <= prev[1] else (U - prev[0]) / (hbound[j] - prev[1])
        curve.append({"budget": U, "feasible": int(ok.sum()), "alpha0": float(A[j // len(A)]),
                      "alpha1": float(A[j % len(A)]), "hits_bound": float(hbound[j]),
                      "measurements_bound": float(dbound[j]),
                      "marginal_measurements_per_hit": None if mc is None else float(mc)})
        print("  %6d   %5.3f %5.3f    %5.0f    %10.0f       %s" % (
            U, A[j // len(A)], A[j % len(A)], hbound[j], dbound[j],
            "-" if mc is None else "%.0f" % mc))
        prev = (U, hbound[j])
    res["price_curve"] = curve
    json.dump(res, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()

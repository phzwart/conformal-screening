"""Selection among candidate rules (replaces Part B of decision_sims.py).
Among J candidate threshold pairs (a 10 x 10 grid of class-wise miscoverage levels,
de-duplicated to distinct threshold pairs), choose the rule with the fewest test-set errors
(missed actives + false leads) whose upper prediction limit for deferrals is within assay
capacity. Three ways to set the level of the capacity check:
  unadjusted   each rule checked at gamma (no account of selection);
  bonferroni   each rule checked at gamma / J, J = number of distinct rules;
  fixed order  rules tested in an order fixed before the test set is seen (ascending
               deferral count on the calibration set, which is independent of the test set),
               each at gamma, stopping at the first rule that fails the check; family-wise
               valid without division by J.
For the chosen rule we record whether realized deferrals exceed capacity and whether the
two-sided error interval (at the same per-rule level) covers the realized errors.
Usage: python3 -I selection_sim.py REPS OUT.json"""
import numpy as np, json, sys, os
from functools import lru_cache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intervals import thatcher

@lru_cache(maxsize=None)
def TH(k, n, m, g): return thatcher(k, n, m, g)

def draw(rng, pi, a1, b1, n):
    y = (rng.random(n) < pi).astype(int)
    u = np.where(y == 1, rng.beta(a1, b1, n), rng.beta(2, 7, n)); return y, u

def stats(y, u, t0, t1):
    in0 = u <= t0; in1 = u >= 1 - t1
    err = np.sum((y == 0) & in1 & ~in0) + np.sum((y == 1) & in0 & ~in1)
    return int(np.sum(in0 == in1)), int(err)

def run(R=4000, n_cal=500, n_test=200, m=1000, cap=0.15, g=0.05, seed=1, n_alpha=10):
    rng = np.random.default_rng(seed)
    alphas = np.linspace(0.02, 0.30, n_alpha)
    grid = [(a0, a1) for a0 in alphas for a1 in alphas]
    modes = ("unadjusted", "bonferroni", "fixed order")
    acc = {k: {"found": 0, "cap_exceeded": 0, "err_covered": 0, "errors": 0} for k in modes}
    Js = []
    for r in range(R):
        pi, a1, b1 = rng.uniform(0.1, 0.4), rng.uniform(3, 9), rng.uniform(2, 4)
        yc, uc = draw(rng, pi, a1, b1, n_cal); yt, ut = draw(rng, pi, a1, b1, n_test); yf, uf = draw(rng, pi, a1, b1, m)
        s0 = np.sort(uc[yc == 0]); s1 = np.sort(1 - uc[yc == 1]); n0, n1 = len(s0), len(s1)
        cand, seen = [], set()
        for a0_, a1_ in grid:                       # duplicates (same threshold pair) dropped
            t0 = s0[min(n0, int(np.ceil((n0 + 1) * (1 - a0_)))) - 1]; t1 = s1[min(n1, int(np.ceil((n1 + 1) * (1 - a1_)))) - 1]
            if (t0, t1) in seen: continue
            seen.add((t0, t1)); cand.append((stats(yc, uc, t0, t1), t0, t1))
        cand.sort(key=lambda c: c[0])               # order fixed by calibration data only
        rules = [stats(yt, ut, t0, t1) + stats(yf, uf, t0, t1) for _, t0, t1 in cand]
        J = len(rules); Js.append(J)
        for mode in modes:
            gg = g / J if mode == "bonferroni" else g
            ok = []
            for i, (dt, et, df, ef) in enumerate(rules):
                passed = TH(dt, n_test, m, 2 * gg)[1] <= cap * m      # one-sided upper limit at level gg
                if mode == "fixed order" and not passed: break
                if passed: ok.append(i)
            if not ok: continue
            i = min(ok, key=lambda j: rules[j][1]); dt, et, df, ef = rules[i]
            lo, hi = TH(et, n_test, m, gg)
            a = acc[mode]; a["found"] += 1; a["cap_exceeded"] += df > cap * m; a["err_covered"] += lo <= ef <= hi; a["errors"] += ef
    out = {"reps": R, "grid_rules": n_alpha ** 2, "distinct_rules_median": int(np.median(Js)),
           "distinct_rules_range": [int(min(Js)), int(max(Js))], "capacity_fraction": cap, "gamma": g}
    for k, a in acc.items():
        f = max(1, a["found"])
        out[k] = {"no_rule_feasible_fraction": round(1 - a["found"] / R, 4), "capacity_exceeded_fraction": round(a["cap_exceeded"] / f, 4),
                  "error_interval_coverage": round(a["err_covered"] / f, 4), "future_errors_per_batch": round(a["errors"] / f, 1)}
    return out

if __name__ == "__main__":
    o = run(R=int(sys.argv[1])); print(json.dumps(o, indent=1)); json.dump(o, open(sys.argv[2], "w"), indent=1)

"""Scenario maps: certified bounds for every setting of the coverage requests.
For one split of a dataset, every request (alpha0, alpha1) on the 12 x 12 grid (delta = 0.10,
batch correction) sets thresholds on the calibration set; the audit set gives, for a future
batch of M = 1000, the point estimate of each outcome count and a one-sided bound: an upper
bound for resources and errors (deferrals = assays, false leads, missed in-class compounds),
a lower bound for yield (confirmed in-class compounds). Each bound is computed at level
EPS / (J * C), J = 144 requests and C = number of bounded outcomes, so that all bounds of all
settings hold together with probability at least 1 - EPS.
A scenario is a set of limits on these bounds; its feasible settings are those whose bounds
satisfy every limit, and among them the objective (fewest expected false leads) picks one.
Usage: python3 -I scenario_map.py OUT.json FIGDIR"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import lru_cache
import intervals
from conformal import k_batch, thresholds, cells, break_ties
TH = lru_cache(maxsize=None)(intervals.thatcher)
HERE = os.path.dirname(os.path.abspath(__file__))
ALPHAS = np.round(np.arange(0.025, 0.301, 0.025), 3); J = len(ALPHAS) ** 2; M = 1000; EPS = 0.05; D = 0.10
OUTCOMES = {"deferred": ("upper", ["0_both", "0_empty", "1_both", "1_empty"]), "false lead": ("upper", ["0_s1"]),
            "missed in-class": ("upper", ["1_s0"]), "confirmed in-class": ("lower", ["1_s1"])}
C = len(OUTCOMES); LEV = EPS / (J * C)

def grid(uc, yc, ua, ya):
    n0, n1 = int((yc == 0).sum()), int((yc == 1).sum()); m1 = int(round(M * n1 / (n0 + n1))); m0 = M - m1; na = len(ya)
    G = {k: {"point": np.zeros((len(ALPHAS), len(ALPHAS))), "bound": np.zeros((len(ALPHAS), len(ALPHAS)))} for k in OUTCOMES}
    for i, a0 in enumerate(ALPHAS):
        for j, a1 in enumerate(ALPHAS):
            t0, t1 = thresholds(uc, yc, k_batch(n0, a0, D, m0), k_batch(n1, a1, D, m1)); c = cells(ua, ya, t0, t1)
            for k, (side, keys) in OUTCOMES.items():
                cnt = sum(c[x] for x in keys); lo, hi = TH(cnt, na, M, 2 * LEV)       # one-sided at LEV
                G[k]["point"][i, j] = 1000 * cnt / na; G[k]["bound"][i, j] = hi if side == "upper" else lo
    return G, {"n_cal": [n0, n1], "n_audit": [int((ya == 0).sum()), int((ya == 1).sum())]}

def feasible(G, lim):
    ok = np.ones((len(ALPHAS), len(ALPHAS)), bool)
    for k, (side, v) in lim.items():
        ok &= G[k]["bound"] <= v if side == "max" else G[k]["bound"] >= v
    return ok

def solve(G, lim):
    ok = feasible(G, lim); obj = np.where(ok, G["false lead"]["point"], np.inf)
    if not ok.any(): return ok, None
    i, j = np.unravel_index(np.argmin(obj), obj.shape)
    return ok, {"alpha0": float(ALPHAS[i]), "alpha1": float(ALPHAS[j]), **{k: {"point": float(G[k]["point"][i, j]), "bound": float(G[k]["bound"][i, j])} for k in G}}

def load_aqsol(r=0):
    D_ = np.load(f"{HERE}/aqsol_runs.npz"); rng = np.random.default_rng(7000 + r)
    jc, jt = break_ties(rng, D_[f"r{r}_cal_u"], D_[f"r{r}_test_u"]); sc, st = D_[f"r{r}_cal_logP"] > 3.5, D_[f"r{r}_test_logP"] > 3.5
    return jc[sc], D_[f"r{r}_cal_y"][sc], jt[st], D_[f"r{r}_test_y"][st]

def load_calisol(design="random", r=0):
    z = np.load(f"{HERE}/v2/calisol_scores/scores_{design}_{r}.npz"); return z["u_cal"], z["y_cal"], z["u_audit"], z["y_audit"]

def load_cadel(isoform="CAXII", c=8):
    z = np.load(f"{HERE}/v2/cadel/scores_P_{isoform}_c{c}.npz"); return z["u_cal"], z["y_cal"], z["u_audit"], z["y_audit"]

SCEN = {"aqsol": {"loose": {"deferred": ("max", 450), "confirmed in-class": ("min", 60), "missed in-class": ("max", 90)},
                  "tight": {"deferred": ("max", 250), "confirmed in-class": ("min", 60), "missed in-class": ("max", 90)}},
        "calisol": {"loose": {"deferred": ("max", 200), "confirmed in-class": ("min", 60), "missed in-class": ("max", 40)},
                    "tight": {"deferred": ("max", 100), "confirmed in-class": ("min", 60), "missed in-class": ("max", 40)}},
        # the DEL-trained scores never defer fewer than 395 per 1000 at any request, so the
        # electrolyte limits (200 / 60 / 40) are infeasible over the whole grid; relaxed here
        "cadel": {"loose": {"deferred": ("max", 750), "confirmed in-class": ("min", 60), "missed in-class": ("max", 100)}}}

if __name__ == "__main__":
    OUT, FIG = sys.argv[1], sys.argv[2]; res = {"eps": EPS, "requests": J, "bounded_outcomes": C, "per_bound_level": LEV, "alphas": ALPHAS.tolist()}
    for name, loader in (("aqsol", load_aqsol), ("calisol", load_calisol), ("cadel", load_cadel)):
        G, info = grid(*loader()); res[name] = {"info": info, "grid": {k: {q: v.tolist() for q, v in G[k].items()} for k in G}, "scenarios": {}}
        for sname, lim in SCEN[name].items():
            ok, best = solve(G, lim); res[name]["scenarios"][sname] = {"limits": {k: list(v) for k, v in lim.items()}, "feasible": int(ok.sum()), "best": best,
                                                                       "feasible_mask": ok.astype(int).tolist()}
        print(name, info, {k: (round(float(G[k]["bound"].min())), round(float(G[k]["bound"].max()))) for k in G},
              {s: (v["feasible"], v["best"] and (v["best"]["alpha0"], v["best"]["alpha1"], round(v["best"]["false lead"]["point"]))) for s, v in res[name]["scenarios"].items()})
    json.dump(res, open(OUT, "w"), indent=1)

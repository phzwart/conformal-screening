"""Width reduction and enrichment beyond blind selection (main text Sections 3.1, 3.4, 3.6).

Two questions are kept apart.

1. Width. Validity alone is attained by the trivial interval [0, m] for a count in a batch
   of m, so an interval earns its place by its width. We report how many times narrower it
   is than that baseline, m / (U - L), pointwise at 95% and for the bounds that hold jointly
   over the J = 144 requests of the sweep and the four bounded outcomes.

2. Enrichment. Whether the advanced compounds (those given the single label in-class, with
   no assay spent) are enriched over the batch as a whole is a property of the rule, tested
   on the audit set with a one-sided Fisher exact test, Bonferroni-corrected over the J
   settings of the sweep. It is a statement about the deployed rule, not a prediction of a
   future count, and the two should not be conflated.

Usage: python3 -I beyond_chance.py OUT.json
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import lru_cache
from scipy.stats import fisher_exact
import intervals
from conformal import k_batch, thresholds, cells
import scenario_map as SM

TH = lru_cache(maxsize=None)(intervals.thatcher)
M, EPS, D = 1000, 0.05, 0.10
J = len(SM.ALPHAS) ** 2
C = 4
LEV = EPS / (J * C)
OUTCOMES = {"missed in-class": ["1_s0"], "false leads": ["0_s1"], "confirmed in-class": ["1_s1"],
            "measurements": ["0_both", "0_empty", "1_both", "1_empty"]}

def analyse(uc, yc, ua, ya, a0, a1):
    n0, n1 = int((yc == 0).sum()), int((yc == 1).sum())
    m1 = int(round(M * n1 / (n0 + n1))); m0 = M - m1; na = len(ya)
    t0, t1 = thresholds(uc, yc, k_batch(n0, a0, D, m0), k_batch(n1, a1, D, m1))
    c = cells(ua, ya, t0, t1)
    res = {"n_audit": na, "prevalence": round(int((ya == 1).sum()) / na, 4), "outcomes": {}}
    for lab, keys in OUTCOMES.items():
        k = sum(c[x] for x in keys)
        lo, hi = TH(k, na, M, EPS); blo, bhi = TH(k, na, M, 2 * LEV)
        res["outcomes"][lab] = {"point_per1000": round(1000 * k / na, 1),
                                "interval": [int(lo), int(hi)], "width": int(hi - lo),
                                "factor": round(M / (hi - lo), 1),
                                "joint": [int(blo), int(bhi)], "joint_width": int(bhi - blo),
                                "joint_factor": round(M / (bhi - blo), 1)}
    adv_in, adv_out = c["1_s1"], c["0_s1"]
    rest_in, rest_out = int((ya == 1).sum()) - adv_in, int((ya == 0).sum()) - adv_out
    odds, p = fisher_exact([[adv_in, adv_out], [rest_in, rest_out]], alternative="greater")
    res["enrichment"] = {"advanced_in_class": adv_in, "advanced_total": adv_in + adv_out,
                         "rest_in_class": rest_in, "rest_total": rest_in + rest_out,
                         "precision": round(adv_in / (adv_in + adv_out), 4),
                         "fold_over_prevalence": round((adv_in / (adv_in + adv_out)) / res["prevalence"], 2),
                         "odds_ratio": round(float(odds), 1), "p_one_sided": float(p),
                         "bonferroni_level": EPS / J, "significant": bool(p < EPS / J)}
    return res

if __name__ == "__main__":
    R = json.load(open("v2/scenario_map.json"))
    out = {"m": M, "eps": EPS, "requests": J, "bounded_outcomes": C, "per_bound_level": LEV,
           "trivial_width": M, "cases": {}}
    for name, loader in (("aqsol", SM.load_aqsol), ("calisol", SM.load_calisol)):
        uc, yc, ua, ya = loader()
        b = R[name]["scenarios"]["loose"]["best"]
        out["cases"][name] = {"alpha0": b["alpha0"], "alpha1": b["alpha1"],
                              **analyse(uc, yc, ua, ya, b["alpha0"], b["alpha1"])}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for k, v in out["cases"].items():
        e = v["enrichment"]
        print(f"{k}: narrower than trivial by " + ", ".join(f"{lab} {o['factor']}x/{o['joint_factor']}x"
              for lab, o in v["outcomes"].items()))
        print(f"   enrichment {e['precision']:.2f} vs {v['prevalence']:.2f} = {e['fold_over_prevalence']}x, "
              f"p={e['p_one_sided']:.1e}, significant={e['significant']}")

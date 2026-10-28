"""Per-plate and audit-size numbers quoted in the main text (Sections 3.4 and 3.6).
1. Electrolyte screen, new salt calibrated on a pilot: median audited counts of the default
   setting, and intervals for one 96-well plate (audit interval and m x Clopper-Pearson),
   with the one-sided measurement bound for one plate and for all of K = 10 plates.
2. Solubility scenario 1: joint bound on measurements for the chosen setting at other audit
   sizes, at the same audited rate.
Usage: python3 -I plate_bounds.py v2/calisol/calisol_runs.json v2/scenario_map.json OUT.json"""
import sys, json, numpy as np
from scipy.stats import beta
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__))); import intervals
runs = json.load(open(sys.argv[1]))["runs"]; SM = json.load(open(sys.argv[2])); out = {}
DEF = ["0_both", "0_empty", "1_both", "1_empty"]
for des in ("salt", "salt_pilot"):
    X = [z for z in runs if z["design"] == des]
    per = lambda keys: float(np.median([1000 * sum(z["default"]["cells"][c] for c in keys) / sum(z["n_audit"]) for z in X]))
    out[des] = {"missed_good_per1000": per(["1_s0"]), "confident_hits_per1000": per(["1_s1"]), "false_leads_per1000": per(["0_s1"]),
                "measurements_per1000": per(DEF), "n_cal_median": float(np.median([sum(z["n_cal"]) for z in X])),
                "n_audit_median": float(np.median([sum(z["n_audit"]) for z in X]))}
n = int(out["salt_pilot"]["n_audit_median"]); K = round(out["salt_pilot"]["measurements_per1000"] * n / 1000); m = 96
lo, hi = intervals.thatcher(K, n, m, 0.05)
cl, cu = beta.ppf(0.025, K, n - K + 1), beta.ppf(0.975, K + 1, n - K)
out["plate"] = {"n_audit": n, "K": K, "m": m, "expected": m * K / n, "audit_interval": [int(lo), int(hi)],
                "m_times_cp": [int(np.floor(m * cl)), int(np.ceil(m * cu))],
                "upper_one_plate": int(intervals.thatcher(K, n, m, 0.10)[1]), "upper_all_10_plates": int(intervals.thatcher(K, n, m, 0.01)[1])}
lev = SM["per_bound_level"]; na = sum(SM["aqsol"]["info"]["n_audit"]); p = SM["aqsol"]["scenarios"]["loose"]["best"]["deferred"]["point"] / 1000
out["audit_size"] = {str(k): int(intervals.thatcher(round(p * k), k, 1000, 2 * lev)[1]) for k in (na // 2, na, 2 * na, 4 * na)}
json.dump(out, open(sys.argv[3], "w"), indent=1); print(json.dumps(out, indent=1))

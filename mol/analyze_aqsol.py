"""AqSolDB lipophilic screen (log P > 3.5): sweep, Pareto front, cost map, capacity rule,
Chow baseline and restricted vs unrestricted calibration. Scores are tie-broken at random
(jitter below the smallest gap) before any threshold is computed. Pareto, cost-map and
supported-rule statistics are reported for every split; the figure uses split 0.
The sweep summaries (Pareto front, settings A and B, cost map, regions) use the audit-set
(test) rates of each rule; the calibration-only capacity rule and the Chow baselines use
calibration data (ESI). Choosing on the audit set and checking on fresh data: audit_check.py. Class 1 = soluble (wanted).
Cells: 1_s0 missed soluble, 0_s1 false lead (insoluble labelled soluble), 1_s1 soluble
confirmed, 0_s0 insoluble rejected, *_both / *_empty deferred.
Usage: python3 -I analyze_aqsol.py aqsol_runs.npz OUTDIR"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from conformal import k_batch, thresholds, cells, cross_counts, break_ties
import rowint
from functools import lru_cache
rowint.thatcher = lru_cache(maxsize=None)(rowint.thatcher); rowint.rank_law = lru_cache(maxsize=None)(rowint.rank_law)   # exact limits, memoized
from rowint import calonly_mixed
from intervals import thatcher
from scipy.stats import beta as betad
D = np.load(sys.argv[1]); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
R = len([k for k in D.files if k.endswith("_cal_u")]); M = 1000; G = 0.05; DELTA = 0.10
ALPHAS = np.round(np.arange(0.025, 0.301, 0.025), 3)
CELLS = ["0_s0", "0_s1", "0_both", "0_empty", "1_s0", "1_s1", "1_both", "1_empty"]

_J = {}
def jittered(r):
    """tie-broken scores for split r (calibration incl. compounds outside the subset, and test)"""
    if r not in _J:
        rng = np.random.default_rng(7000 + r); _J[r] = dict(zip(("cal", "test"), break_ties(rng, D[f"r{r}_cal_u"], D[f"r{r}_test_u"])))
    return _J[r]
def get(r, part, lip=True):
    u, y, lp = jittered(r)[part], D[f"r{r}_{part}_y"], D[f"r{r}_{part}_logP"]
    s = lp > 3.5 if lip else np.ones_like(lp, bool); return u[s], y[s]

def est_rates(rule, n0, n1):
    """calibration-only point estimates of the 8 cell rates (per compound, mixed batch)"""
    p1 = n1 / (n0 + n1); p0 = 1 - p1; a0 = rule["k0"] / (n0 + 1); a1 = rule["k1"] / (n1 + 1)
    b0 = rule["c0"] / n0; b1 = rule["c1"] / n1; e = dict.fromkeys(CELLS, 0.0)
    if rule["two_label"]:
        e["0_s1"], e["0_s0"] = p0 * (1 - a0), p0 * (1 - b0); e["0_both"] = p0 - e["0_s1"] - e["0_s0"]
        e["1_s0"], e["1_s1"] = p1 * (1 - a1), p1 * (1 - b1); e["1_both"] = p1 - e["1_s0"] - e["1_s1"]
    else:
        e["0_s0"], e["0_s1"] = p0 * a0, p0 * b0; e["0_empty"] = p0 - e["0_s0"] - e["0_s1"]
        e["1_s1"], e["1_s0"] = p1 * a1, p1 * b1; e["1_empty"] = p1 - e["1_s1"] - e["1_s0"]
    return {k: max(0.0, v) for k, v in e.items()}

def summ(c):  # screening quantities from cell values
    return {"missed": c["1_s0"], "false_lead": c["0_s1"], "deferred": c["0_both"] + c["0_empty"] + c["1_both"] + c["1_empty"],
            "confirmed": c["1_s1"], "rejected": c["0_s0"]}

def sweep(uc, yc, ut, yt):
    n0, n1 = int(np.sum(yc == 0)), int(np.sum(yc == 1)); p1 = n1 / (n0 + n1)
    m0, m1 = int(round(M * (1 - p1))), int(round(M * p1)); rules = {}
    for a0 in ALPHAS:
        for a1 in ALPHAS:
            k0, k1 = k_batch(n0, a0, DELTA, m0), k_batch(n1, a1, DELTA, m1)
            if (k0, k1) in rules: rules[(k0, k1)]["requests"].append((float(a0), float(a1))); continue
            t0, t1 = thresholds(uc, yc, k0, k1); c0, c1 = cross_counts(uc, yc, t0, t1)
            rule = {"k0": k0, "k1": k1, "t0": t0, "t1": t1, "two_label": bool(t0 + t1 >= 1), "c0": c0, "c1": c1,
                    "requests": [(float(a0), float(a1))], "test": cells(ut, yt, t0, t1)}
            rule["est"] = est_rates(rule, n0, n1); rule["aud"] = {k: v / len(yt) for k, v in rule["test"].items()}; rules[(k0, k1)] = rule
    return list(rules.values()), n0, n1

def pareto(rules):
    pts = [(r["aud"]["1_s0"], r["aud"]["0_s1"], summ(r["aud"])["deferred"]) for r in rules]; keep = []
    for i, p in enumerate(pts):
        dom = any(all(q[j] <= p[j] for j in range(3)) and any(q[j] < p[j] for j in range(3)) for q in pts)
        if not dom: keep.append(i)
    return keep

def cost(c, lam, rho):  # in units of a false lead, per compound
    s = summ(c); return lam * s["missed"] + s["false_lead"] + rho * s["deferred"]

results = {"reps": R, "alphas": ALPHAS.tolist(), "delta": DELTA}
# ---------- rep 0: Pareto, representative settings, cost map ----------
uc, yc = get(0, "cal"); ut, yt = get(0, "test")
rules, n0, n1 = sweep(uc, yc, ut, yt); P = pareto(rules); nt = len(yt)
results["rep0"] = {"n_cal": [n0, n1], "n_test": [int(np.sum(yt == 0)), int(np.sum(yt == 1))], "distinct_rules": len(rules), "pareto": len(P)}
def describe(rule):
    ci = calonly_mixed(rule, n0, n1, M, G); ti = {k: thatcher(v, nt, M, G) for k, v in rule["test"].items()}
    agg = lambda d, keys: (sum(d[k][0] for k in keys), sum(d[k][1] for k in keys))
    keys = {"missed": ["1_s0"], "false_lead": ["0_s1"], "confirmed": ["1_s1"], "rejected": ["0_s0"],
            "deferred_insoluble": ["0_both", "0_empty"], "deferred_soluble": ["1_both", "1_empty"]}
    out = {"requests": rule["requests"][:4], "n_requests": len(rule["requests"]), "two_label": rule["two_label"]}
    for q, ks in keys.items():
        out[q] = {"estimate_cal": round(M * sum(rule["est"][k] for k in ks)), "test_per1000": round(M * sum(rule["test"][k] for k in ks) / nt),
                  "calonly_interval": [int(x) for x in agg({k: ci.get(k, (0, 0)) for k in ks}, ks)],
                  "testset_interval": [int(v) for v in thatcher(sum(rule["test"][k] for k in ks), nt, M, G)],
                  "testset_interval_bonferroni": [int(x) for x in thatcher(sum(rule["test"][k] for k in ks), nt, M, G / (len(ALPHAS) ** 2))]}
    return out
front = [rules[i] for i in P]
A = min(front, key=lambda r: (r["aud"]["1_s0"], summ(r["aud"])["deferred"]))
B = min(front, key=lambda r: (summ(r["aud"])["deferred"], r["aud"]["1_s0"]))
results["setting_A_fewest_missed"] = describe(A); results["setting_B_fewest_deferred"] = describe(B)
results["pareto_points"] = [{"missed": 1000 * r["aud"]["1_s0"], "false_lead": 1000 * r["aud"]["0_s1"], "deferred": 1000 * summ(r["aud"])["deferred"],
                             "requests": r["requests"][:1]} for r in front]
results["all_points"] = [{"missed": 1000 * r["aud"]["1_s0"], "false_lead": 1000 * r["aud"]["0_s1"], "deferred": 1000 * summ(r["aud"])["deferred"]} for r in rules]
lams = np.geomspace(1, 100, 41); rhos = np.linspace(0.02, 1.0, 50)
def costmap(rules):
    E = np.array([[r["aud"]["1_s0"], r["aud"]["0_s1"], summ(r["aud"])["deferred"]] for r in rules])
    C = lams[None, :, None] * E[None, None, :, 0] + E[None, None, :, 1] + rhos[:, None, None] * E[None, None, :, 2]
    return np.argmin(C, axis=2)
best = costmap(rules)
uniq = sorted(set(best.ravel().tolist())); results["costmap"] = {"lambda": lams.tolist(), "rho": rhos.tolist(),
    "best_rule_index": [[uniq.index(v) for v in row] for row in best.tolist()],
    "rules": [{"requests": rules[v]["requests"][:1], "missed": 1000 * rules[v]["aud"]["1_s0"], "false_lead": 1000 * rules[v]["aud"]["0_s1"],
               "deferred": 1000 * summ(rules[v]["aud"])["deferred"]} for v in uniq]}
# Is acting on the single labels (and deferring the rest) the cheapest action within each set for the chosen
# rule?  NOTE: this check uses the TEST-SET fractions of actives within each prediction set (Clopper-Pearson,
# Bonferroni over the 3 sets), so it needs labelled test compounds; it is not a calibration-only statement.
def eta_bounds(rule, g=G / 3):
    t = rule["test"]; out = {}
    for s_, a, b in (("s1", "1_s1", "0_s1"), ("s0", "1_s0", "0_s0"), ("mid", ["1_both", "1_empty"], ["0_both", "0_empty"])):
        k = t[a] if isinstance(a, str) else sum(t[x] for x in a); n = k + (t[b] if isinstance(b, str) else sum(t[x] for x in b))
        if n == 0: out[s_] = None; continue
        lo = 0.0 if k == 0 else betad.ppf(g / 2, k, n - k + 1); hi = 1.0 if k == n else betad.ppf(1 - g / 2, k + 1, n - k)
        out[s_] = (lo, hi, k / n)
    return out
def rule_ok(eb, lam, rho, robust):
    ok = True
    if eb["s1"]: e = eb["s1"][0] if robust else eb["s1"][2]; ok &= e >= max(1 / (1 + lam), 1 - rho)
    if eb["s0"]: e = eb["s0"][1] if robust else eb["s0"][2]; ok &= e <= min(1 / (1 + lam), rho / lam)
    if eb["mid"]:
        lo, hi = (eb["mid"][0], eb["mid"][1]) if robust else (eb["mid"][2], eb["mid"][2]); ok &= (rho / lam <= lo) and (hi <= 1 - rho)
    return bool(ok)
def regions(rules, best):
    ok_pt = np.zeros_like(best); ok_rb = np.zeros_like(best)
    for i, rho in enumerate(rhos):
        for j, lam in enumerate(lams):
            eb = eta_bounds(rules[best[i, j]]); ok_pt[i, j] = rule_ok(eb, lam, rho, False); ok_rb[i, j] = rule_ok(eb, lam, rho, True)
    return ok_pt, ok_rb
ok_pt, ok_rb = regions(rules, best)
results["costmap"]["convention_optimal_point"] = ok_pt.tolist(); results["costmap"]["convention_optimal_robust"] = ok_rb.tolist()
results["costmap"]["region_fraction_point"] = float(ok_pt.mean()); results["costmap"]["region_fraction_robust"] = float(ok_rb.mean())
results["costmap"]["region_uses_test_set"] = True

# ---------- across reps ----------
COSTS = [(5, 0.3), (10, 0.5), (20, 0.8), (50, 0.5)]
DEF = ("0_both", "0_empty", "1_both", "1_empty")
CAP = 0.5                                     # capacity: at most half of the batch deferred
GRID_J = len(ALPHAS) ** 2                     # Bonferroni over every request in the grid (valid without de-duplication)
ORDER = sorted([(float(a0), float(a1)) for a0 in ALPHAS for a1 in ALPHAS], key=lambda p: (-(p[0] + p[1]), -p[0]))
def emp_cost(act, y, lam, rho):  # act: 0 label inactive, 1 label soluble, 2 defer
    return np.mean(np.where(act == 0, lam * y, np.where(act == 1, 1 - y, rho)))
def chow_actions(eta, lam, rho):
    return np.argmin(np.vstack([lam * eta, 1 - eta, np.full_like(eta, rho)]), axis=0)
def erm_thresholds(uc, yc, lam, rho):
    """two thresholds l <= h on u minimizing empirical calibration cost (label 0 below l, defer between, label 1 above h)"""
    q = np.unique(np.quantile(uc, np.linspace(0, 1, 201))); best = None
    for a, l in enumerate(q):
        for h in q[a:]:
            act = np.where(uc < l, 0, np.where(uc > h, 1, 2)); c = emp_cost(act, yc, lam, rho)
            if best is None or c < best[0]: best = (c, l, h)
    return best[1], best[2]
per_split = []; cov_hits = [0, 0]; cap_rows = []; chow_rows = []; restr = {"restricted": [], "unrestricted": []}
for r in range(R):
    uc, yc = get(r, "cal"); ut, yt = get(r, "test"); rules, n0, n1 = sweep(uc, yc, ut, yt); nt = len(yt); p1 = n1 / (n0 + n1)
    # per-split structure
    Pr = pareto(rules); br = costmap(rules); okp, okr = regions(rules, br)
    front = [rules[i] for i in Pr]
    Ar = min(front, key=lambda x: (x["aud"]["1_s0"], summ(x["aud"])["deferred"])); Br = min(front, key=lambda x: (summ(x["aud"])["deferred"], x["aud"]["1_s0"]))
    pick = lambda rule: {"requests": rule["requests"][:1], "deferred_test_per1000": round(1000 * sum(rule["test"][k] for k in DEF) / nt),
                         "missed_test_per1000": round(1000 * rule["test"]["1_s0"] / nt), "false_lead_test_per1000": round(1000 * rule["test"]["0_s1"] / nt)}
    per_split.append({"rep": r, "distinct_rules": len(rules), "pareto": len(Pr), "supported": len(set(br.ravel().tolist())),
                      "region_point": float(okp.mean()), "region_robust": float(okr.mean()), "setting_A": pick(Ar), "setting_B": pick(Br)})
    for rule in rules:                       # calibration-only interval for the realized test batch (m = n_test)
        for k, (lo, hi) in calonly_mixed(rule, n0, n1, nt, G).items():
            cov_hits[0] += lo <= rule["test"][k] <= hi; cov_hits[1] += 1
    # capacity rule: fewest estimated errors among rules whose calibration-only upper limit for deferrals is within capacity
    updef = lambda rule, g: sum(calonly_mixed(rule, n0, n1, M, g).get(k, (0, 0))[1] for k in DEF)
    by_req = {req: rule for rule in rules for req in rule["requests"]}
    row = {"rep": r, "rules": len(rules)}
    for mode in ("bonferroni",):
        ok = []
        if mode == "bonferroni":
            ok = [(rule, updef(rule, G / GRID_J)) for rule in rules]; ok = [(x, u) for x, u in ok if u <= CAP * M]
        else:                                  # fixed order: rules expected to defer least first, each at level G, stop at first failure
            seen = set()
            for req in ORDER:
                rule = by_req[req]; key_ = (rule["k0"], rule["k1"])
                if key_ in seen: continue
                seen.add(key_); u_ = updef(rule, G)
                if u_ > CAP * M: break
                ok.append((rule, u_))
        if ok:
            ch, ud = min(ok, key=lambda x: x[0]["est"]["1_s0"] + x[0]["est"]["0_s1"])
            row[mode] = {"admissible": len(ok), "chosen_requests": ch["requests"][:1], "upper_deferred_per1000": int(ud),
                         "test_deferred_per1000": round(1000 * sum(ch["test"][k] for k in DEF) / nt),
                         "test_errors_per1000": round(1000 * (ch["test"]["1_s0"] + ch["test"]["0_s1"]) / nt)}
        else: row[mode] = None
    cap_rows.append(row)
    # Chow baselines, all fitted on the full calibration set, evaluated on the test set
    z = lambda u: np.log(np.clip(u, 1e-6, 1 - 1e-6) / (1 - np.clip(u, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    platt = LogisticRegression(C=1e6).fit(z(uc), yc); eta_platt = platt.predict_proba(z(ut))[:, 1]
    eta_iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(uc, yc).predict(ut)
    for lam, rho in COSTS:
        tc = lambda rule: cost({k: v / nt for k, v in rule["test"].items()}, lam, rho)
        trivial = {"defer all": (rho, rho), "label all inactive": (lam * p1, lam * np.mean(yt)), "label all soluble": (1 - p1, 1 - np.mean(yt))}
        bset = min(rules, key=lambda rule: cost(rule["est"], lam, rho))
        cands = [(cost(bset["est"], lam, rho), tc(bset), "set rule")] + [(e, t, name) for name, (e, t) in trivial.items()]
        bcand = min(cands, key=lambda x: x[0])
        cons = [rule for rule in rules if any(a <= 0.10 + 1e-9 and b <= 0.10 + 1e-9 for a, b in rule["requests"])]
        bcons = min(cons, key=lambda rule: cost(rule["est"], lam, rho)) if cons else None
        l, h = erm_thresholds(uc, yc, lam, rho)
        chow_rows.append({"rep": r, "lambda": lam, "rho": rho,
                          "chow_platt": emp_cost(chow_actions(eta_platt, lam, rho), yt, lam, rho),
                          "chow_isotonic": emp_cost(chow_actions(eta_iso, lam, rho), yt, lam, rho),
                          "erm_two_thresholds": emp_cost(np.where(ut < l, 0, np.where(ut > h, 1, 2)), yt, lam, rho),
                          "best_set_rule": tc(bset), "best_candidate_incl_trivial": bcand[1], "best_candidate_kind": bcand[2],
                          "best_set_rule_coverage_0.9": tc(bcons) if bcons else None, "defer_all": rho})
    # restricted vs unrestricted calibration at alpha = delta = 0.10 (m_y from calibration prevalence)
    for name, (ucc, ycc) in (("restricted", (uc, yc)), ("unrestricted", get(r, "cal", lip=False))):
        a0n, a1n = int(np.sum(ycc == 0)), int(np.sum(ycc == 1)); m0, m1 = int(np.sum(yt == 0)), int(np.sum(yt == 1))
        q1 = int(round(nt * a1n / (a0n + a1n))); k0, k1 = k_batch(a0n, 0.10, DELTA, nt - q1), k_batch(a1n, 0.10, DELTA, q1); t0, t1 = thresholds(ucc, ycc, k0, k1)
        c = cells(ut, yt, t0, t1)
        restr[name].append({"cov0": (c["0_s0"] + c["0_both"]) / m0, "cov1": (c["1_s1"] + c["1_both"]) / m1,
                            "deferred_per1000": 1000 * sum(c[k] for k in DEF) / nt, "n_cal": a0n + a1n})
results["per_split"] = per_split
stat = lambda v: {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}
results["per_split_summary"] = {k: stat([x[k] for x in per_split]) for k in ("distinct_rules", "pareto", "supported", "region_point", "region_robust")}
for st in ("setting_A", "setting_B"):
    results["per_split_summary"][st] = {q: stat([x[st][q] for x in per_split]) for q in ("deferred_test_per1000", "missed_test_per1000", "false_lead_test_per1000")}
results["calonly_validation"] = {"cells_checked": cov_hits[1], "coverage": cov_hits[0] / cov_hits[1]}
results["capacity"] = cap_rows
agg = {}
for row in chow_rows: agg.setdefault(f"{row['lambda']}_{row['rho']}", []).append(row)
MS = ("chow_platt", "chow_isotonic", "erm_two_thresholds", "best_set_rule", "best_candidate_incl_trivial", "best_set_rule_coverage_0.9", "defer_all")
results["chow"] = {}
for k, v in agg.items():
    e = {m: float(np.mean([x[m] for x in v if x[m] is not None])) for m in MS}
    for base in ("chow_platt", "chow_isotonic", "erm_two_thresholds"):
        d = np.array([x["best_candidate_incl_trivial"] - x[base] for x in v])
        e[f"paired_diff_vs_{base}"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    e["trivial_chosen"] = int(sum(x["best_candidate_kind"] != "set rule" for x in v)); results["chow"][k] = e
results["restriction"] = {k: {"mean_cov0": float(np.mean([x["cov0"] for x in v])), "mean_cov1": float(np.mean([x["cov1"] for x in v])),
                              "frac_cov0_below_0.9": float(np.mean([x["cov0"] < 0.9 for x in v])), "frac_cov1_below_0.9": float(np.mean([x["cov1"] < 0.9 for x in v])),
                              "mean_deferred_per1000": float(np.mean([x["deferred_per1000"] for x in v])), "mean_n_cal": float(np.mean([x["n_cal"] for x in v]))} for k, v in restr.items()}
json.dump(results, open(f"{OUT}/aqsol_results.json", "w"), indent=1)
print(json.dumps({k: results[k] for k in ("rep0", "setting_A_fewest_missed", "setting_B_fewest_deferred", "per_split_summary", "calonly_validation", "capacity", "chow", "restriction")}, indent=1))

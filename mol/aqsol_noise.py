"""Label-noise sensitivity of the AqSolDB lipophilic screen (referee response).

The screen labels a compound soluble if log S >= -4 and restricts to log P > 3.5.  The
inter-source spread of AqSolDB log S is 0.5-0.7 log units, so the label of a compound near
the cut is itself uncertain; this script asks how much of the published analysis survives
when the label is perturbed at the conceded rate.  The model and its scores are NEVER
recomputed: the CatBoost probabilities of aqsol_runs.npz (with the same random tie-breaking,
rng 7000 + r) are held fixed and only y changes, so every difference reported here is a
label effect alone.

Perturbations (applied to the pooled screened calibration + audit compounds of each split,
then split back, so calibration and audit are perturbed consistently):
  base              y = 1[log S >= -4], as published
  cut_-3.5/-4.5     the cut moved by half a log unit
  random_flip       8% of the screened compounds flipped, drawn uniformly without
                    replacement from the near-cut band |log S + 4| <= 0.5 (five seeds)
  structured_flip   the same per-class budget, drawn from the same band, but taking in each
                    class the compounds whose score ranks most extremely against that class
                    (lowest u among soluble, highest u among insoluble).  This is the
                    mechanism the proxy-label section identifies as the damaging one: the
                    noise is correlated with the score, so it makes the labels agree with
                    the model and the measured performance is biased upwards.
  structured_adverse the mirror image, reported so that the direction of the structured bias
                    is visible: the same budget taken from the compounds whose score most
                    strongly supports their own label (highest u among soluble, lowest u
                    among insoluble), which biases the measured performance downwards.
  sd_flip           the same budget, taken from the screened compounds with a recorded
                    non-zero inter-source SD, largest SD first.

Reported per split and aggregated: distinguished settings and Pareto-front size of the
12 x 12 sweep, the scenario feasible count under the published limits (deferred <= 450,
confirmed in-class >= 60, missed in-class <= 90 per 1000) with the minimum certifiable
measurement budget, the enrichment of the advanced compounds with its one-sided Fisher
p-value, and the fraction of the base Pareto front that stays on the perturbed front.

Usage: python3 -I aqsol_noise.py OUTDIR
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import fisher_exact
from conformal import k_batch, thresholds, cells, cross_counts, break_ties
import scenario_map as SM
import intervals
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
TH = lru_cache(maxsize=None)(intervals.thatcher)
ALPHAS = SM.ALPHAS                      # 0.025 .. 0.300, 12 values
J = len(ALPHAS) ** 2                    # 144 requests
M, DELTA, EPS = 1000, 0.10, 0.05
CUT, LIP, BAND, RATE, SEEDS = -4.0, 3.5, 0.5, 0.08, 5
LIMITS = {"deferred": ("max", 450), "confirmed in-class": ("min", 60), "missed in-class": ("max", 90)}
DEF = ("0_both", "0_empty", "1_both", "1_empty")
CELLS = ["0_s0", "0_s1", "0_both", "0_empty", "1_s0", "1_s1", "1_both", "1_empty"]
BASE_SETTING = (0.15, 0.125)            # loose-scenario choice of the published scenario map

# ---------------------------------------------------------------- data

D = np.load(f"{HERE}/aqsol_runs.npz")
F = np.load(f"{HERE}/aqsoldb_features.npz", allow_pickle=True)
LOGS = F["logS"]
R = len([k for k in D.files if k.endswith("_cal_u")])


def sd_map(outdir):
    """Per-feature-row inter-source SD and Occurrences, recovered from the curated CSV.

    The _idx arrays of aqsol_runs.npz index the de-duplicated feature table, not the CSV, so
    the CSV rows are standardized the same way as in featurization and grouped by canonical
    SMILES; a merged group takes the largest SD of its rows.  Cached, because it needs RDKit.
    """
    cache = f"{outdir}/aqsol_sd_map.npz"
    if os.path.exists(cache):
        z = np.load(cache); return z["SD"], z["OCC"]
    import pandas as pd
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    chooser = rdMolStandardize.LargestFragmentChooser(); unch = rdMolStandardize.Uncharger()

    def std(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None: return None
        try:
            m = unch.uncharge(chooser.choose(m)); Chem.SanitizeMol(m)
        except Exception:
            return None
        return None if m.GetNumHeavyAtoms() < 2 else m

    aq = pd.read_csv(f"{HERE}/../data/aqsoldb_curated.csv")
    pos = {s: i for i, s in enumerate(F["smiles"])}
    sd = [[] for _ in pos]; occ = [[] for _ in pos]
    for smi, s_, o_ in zip(aq.SMILES, aq.SD, aq.Occurrences):
        m = std(smi)
        if m is None: continue
        i = pos.get(Chem.MolToSmiles(m))
        if i is None: continue
        sd[i].append(float(s_)); occ[i].append(float(o_))
    SD = np.array([max(x) if x else np.nan for x in sd])
    OCC = np.array([max(x) if x else np.nan for x in occ])
    np.savez_compressed(cache, SD=SD, OCC=OCC)
    return SD, OCC


_J = {}
def jittered(r):
    """the published tie-broken scores for split r (identical rng stream to analyze_aqsol.py)"""
    if r not in _J:
        rng = np.random.default_rng(7000 + r)
        _J[r] = dict(zip(("cal", "test"), break_ties(rng, D[f"r{r}_cal_u"], D[f"r{r}_test_u"])))
    return _J[r]


def screened(r):
    """pooled screened (log P > 3.5) calibration + audit compounds of split r"""
    u, idx, part = [], [], []
    for p in ("cal", "test"):
        s = D[f"r{r}_{p}_logP"] > LIP
        u.append(jittered(r)[p][s]); idx.append(D[f"r{r}_{p}_idx"][s]); part.append(np.full(int(s.sum()), p == "test"))
    return np.concatenate(u), np.concatenate(idx), np.concatenate(part).astype(bool)


# ---------------------------------------------------------------- perturbations

def perturb(u, logs, y0, sd, kind, rng=None):
    """return perturbed labels and a record of what was changed"""
    n = len(y0); budget = int(round(RATE * n)); band = np.abs(logs - CUT) <= BAND
    info = {"kind": kind, "n_screened": n, "budget": budget, "band": int(band.sum())}
    if kind == "base":
        return y0.copy(), {**info, "budget": 0, "flipped": 0}
    if kind.startswith("cut_"):
        cut = float(kind.split("_")[1]); y = (logs >= cut).astype(int)
        return y, {**info, "cut": cut, "budget": None, "flipped": int((y != y0).sum())}
    pick = np.zeros(n, bool)
    if kind == "sd_flip":
        cand = np.where(np.nan_to_num(sd, nan=0.0) > 0)[0]
        order = cand[np.argsort(-sd[cand], kind="stable")][:budget]
        pick[order] = True
        info["candidates_sd_positive"] = int(len(cand))
        info["min_sd_flipped"] = float(sd[order].min()) if len(order) else None
        info["median_sd_flipped"] = float(np.median(sd[order])) if len(order) else None
    else:
        # same eligible pool (the near-cut band) and the same per-class budget for the random
        # and the structured case, so the only difference is the choice within the band
        share = {c: int(((y0 == c) & band).sum()) for c in (0, 1)}
        tot = share[0] + share[1]
        nc = {c: min(share[c], int(round(budget * share[c] / tot))) for c in (0, 1)}
        for c in (0, 1):
            pool = np.where(band & (y0 == c))[0]
            if kind == "random_flip":
                take = rng.choice(pool, size=nc[c], replace=False)
            elif kind == "structured_flip":        # most extreme score AGAINST the class
                take = pool[np.argsort(u[pool] if c == 1 else -u[pool], kind="stable")][:nc[c]]
            else:                                  # structured_adverse: score most supports the class
                take = pool[np.argsort(-u[pool] if c == 1 else u[pool], kind="stable")][:nc[c]]
            pick[take] = True
        info["per_class_budget"] = {str(c): nc[c] for c in (0, 1)}
    y = y0.copy(); y[pick] = 1 - y[pick]
    info["flipped"] = int(pick.sum()); info["flip_rate"] = round(float(pick.mean()), 4)
    info["flipped_1to0"] = int((pick & (y0 == 1)).sum()); info["flipped_0to1"] = int((pick & (y0 == 0)).sum())
    return y, info


# ---------------------------------------------------------------- sweep, Pareto (as analyze_aqsol.py)

def summ(c):
    return {"missed": c["1_s0"], "false_lead": c["0_s1"], "confirmed": c["1_s1"], "rejected": c["0_s0"],
            "deferred": sum(c[k] for k in DEF)}


def sweep(uc, yc, ut, yt):
    """the published 12 x 12 sweep: distinct (k0, k1) rules, each with its audit-set cells"""
    n0, n1 = int((yc == 0).sum()), int((yc == 1).sum()); p1 = n1 / (n0 + n1)
    m1 = int(round(M * p1)); m0 = M - m1; rules = {}
    for a0 in ALPHAS:
        for a1 in ALPHAS:
            k0, k1 = k_batch(n0, a0, DELTA, m0), k_batch(n1, a1, DELTA, m1)
            if (k0, k1) in rules:
                rules[(k0, k1)]["requests"].append((float(a0), float(a1))); continue
            t0, t1 = thresholds(uc, yc, k0, k1); c0, c1 = cross_counts(uc, yc, t0, t1)
            rules[(k0, k1)] = {"k0": k0, "k1": k1, "t0": t0, "t1": t1, "c0": c0, "c1": c1,
                               "requests": [(float(a0), float(a1))], "test": cells(ut, yt, t0, t1)}
    out = list(rules.values())
    for rule in out:
        rule["aud"] = {k: v / len(yt) for k, v in rule["test"].items()}
    return out, n0, n1


def pareto(rules):
    pts = [(r["aud"]["1_s0"], r["aud"]["0_s1"], summ(r["aud"])["deferred"]) for r in rules]
    keep = []
    for i, p in enumerate(pts):
        dom = any(all(q[j] <= p[j] for j in range(3)) and any(q[j] < p[j] for j in range(3)) for q in pts)
        if not dom: keep.append(i)
    return keep


def front_requests(rules, P):
    """the grid settings (alpha0, alpha1) carried by the rules on the front"""
    return {req for i in P for req in rules[i]["requests"]}


# ---------------------------------------------------------------- scenario and enrichment

def scenario(uc, yc, ut, yt):
    """published scenario map on the perturbed labels: feasible count and cheapest certified budget"""
    G, info = SM.grid(uc, yc, ut, yt)
    ok, best = SM.solve(G, LIMITS)
    out = {"feasible": int(ok.sum()), "n_requests": J, "best": None, "min_certified_budget": None}
    if best is not None:
        out["best"] = {"alpha0": best["alpha0"], "alpha1": best["alpha1"],
                       "false_lead_point": round(best["false lead"]["point"], 1),
                       "deferred_bound": round(best["deferred"]["bound"], 1)}
        out["min_certified_budget"] = float(np.min(G["deferred"]["bound"][ok]))
    out["grid_min_deferred_bound"] = float(G["deferred"]["bound"].min())
    out["info"] = info
    return out, G


def enrichment(uc, yc, ut, yt, a0, a1):
    """Fisher enrichment of the advanced compounds (single label in-class) over the base rate"""
    n0, n1 = int((yc == 0).sum()), int((yc == 1).sum())
    m1 = int(round(M * n1 / (n0 + n1))); m0 = M - m1
    t0, t1 = thresholds(uc, yc, k_batch(n0, a0, DELTA, m0), k_batch(n1, a1, DELTA, m1))
    c = cells(ut, yt, t0, t1); na = len(yt); prev = float((yt == 1).mean())
    adv_in, adv_out = c["1_s1"], c["0_s1"]
    rest_in, rest_out = int((yt == 1).sum()) - adv_in, int((yt == 0).sum()) - adv_out
    o = {"alpha0": a0, "alpha1": a1, "advanced_in_class": adv_in, "advanced_total": adv_in + adv_out,
         "base_rate": round(prev, 4), "n_audit": na}
    if adv_in + adv_out == 0:
        o.update({"precision": None, "fold": None, "fisher_p": 1.0, "significant": False}); return o
    odds, p = fisher_exact([[adv_in, adv_out], [rest_in, rest_out]], alternative="greater")
    o.update({"precision": round(adv_in / (adv_in + adv_out), 4),
              "fold": round((adv_in / (adv_in + adv_out)) / prev, 3) if prev > 0 else None,
              "odds_ratio": round(float(odds), 2), "fisher_p": float(p),
              "bonferroni_level": EPS / J, "significant": bool(p < EPS / J)})
    return o


def evaluate(r, u, part, y, base_front, label, return_front=False):
    """all reported quantities for one split under one labelling"""
    uc, yc, ut, yt = u[~part], y[~part], u[part], y[part]
    if min((yc == 0).sum(), (yc == 1).sum(), (yt == 0).sum(), (yt == 1).sum()) < 2:
        err = {"rep": r, "perturbation": label, "error": "a class has fewer than 2 compounds"}
        return (err, set()) if return_front else err
    rules, n0, n1 = sweep(uc, yc, ut, yt); P = pareto(rules); fr = front_requests(rules, P)
    scen, _ = scenario(uc, yc, ut, yt)
    row = {"rep": r, "perturbation": label, "n_cal": [n0, n1],
           "n_audit": [int((yt == 0).sum()), int((yt == 1).sum())],
           "distinguished": len(rules), "pareto": len(P), "pareto_settings": len(fr),
           "scenario": scen,
           "enrichment_base_setting": enrichment(uc, yc, ut, yt, *BASE_SETTING),
           "enrichment_own_setting": (enrichment(uc, yc, ut, yt, scen["best"]["alpha0"], scen["best"]["alpha1"])
                                      if scen["best"] else None)}
    if base_front is not None:
        keep = fr & base_front
        row["front_overlap"] = {"base_settings": len(base_front), "perturbed_settings": len(fr),
                                "retained": len(keep),
                                "retained_frac": round(len(keep) / len(base_front), 4) if base_front else None,
                                "jaccard": round(len(keep) / len(fr | base_front), 4) if (fr | base_front) else None}
    else:
        row["front_overlap"] = {"base_settings": len(fr), "perturbed_settings": len(fr),
                                "retained": len(fr), "retained_frac": 1.0, "jaccard": 1.0}
    return (row, fr) if return_front else row


# ---------------------------------------------------------------- run

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/v2"
    os.makedirs(outdir, exist_ok=True)
    SD, OCC = sd_map(outdir)
    rows = []; perts = []
    for r in range(R):
        u, idx, part = screened(r); logs = LOGS[idx]; sd = SD[idx]
        y0 = (logs >= CUT).astype(int)
        base, base_front = evaluate(r, u, part, y0, None, "base", return_front=True)
        rows.append(base)
        bi = perturb(u, logs, y0, sd, "base")[1]
        bi.update({"rep": r, "label": "base", "prevalence": round(float(y0.mean()), 4),
                   "prevalence_base": round(float(y0.mean()), 4)})
        perts.append(bi)
        print(f"  rep {r} {'base':<20s} flipped   0  distinguished {base['distinguished']}  "
              f"pareto {base['pareto']}  feasible {base['scenario']['feasible']}", flush=True)
        jobs = [("cut_-3.5", None), ("cut_-4.5", None), ("structured_flip", None),
                ("structured_adverse", None), ("sd_flip", None)]
        jobs += [("random_flip", 9000 + 100 * r + s) for s in range(SEEDS)]
        for kind, seed in jobs:
            rng = np.random.default_rng(seed) if seed is not None else None
            y, info = perturb(u, logs, y0, sd, kind, rng)
            label = kind if seed is None else f"{kind}_s{seed - 9000 - 100 * r}"
            info["rep"] = r; info["label"] = label; info["prevalence"] = round(float(y.mean()), 4)
            info["prevalence_base"] = round(float(y0.mean()), 4); perts.append(info)
            rows.append(evaluate(r, u, part, y, base_front, label))
            print(f"  rep {r} {label:<20s} flipped {info['flipped']:>3d}  "
                  f"distinguished {rows[-1].get('distinguished')}  pareto {rows[-1].get('pareto')}  "
                  f"feasible {rows[-1]['scenario']['feasible']}", flush=True)

    # ------------- aggregate over splits (random seeds pooled into one family)
    fam = lambda lab: "random_flip" if lab.startswith("random_flip") else lab
    FAMS = ["base", "cut_-3.5", "cut_-4.5", "random_flip", "structured_flip",
            "structured_adverse", "sd_flip"]
    agg = {}
    stat = lambda v: ({"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                       "mean": round(float(np.mean(v)), 3)} if len(v) else None)
    for f in FAMS:
        g = [x for x in rows if fam(x["perturbation"]) == f and "error" not in x]
        if not g: continue
        feas = [x["scenario"]["feasible"] for x in g]
        bud = [x["scenario"]["min_certified_budget"] for x in g if x["scenario"]["min_certified_budget"] is not None]
        e_own = [x["enrichment_own_setting"] for x in g if x["enrichment_own_setting"]]
        e_base = [x["enrichment_base_setting"] for x in g]
        pick = lambda lst, k: [x[k] for x in lst if x.get(k) is not None]
        agg[f] = {"n_evaluations": len(g), "n_splits": len(set(x["rep"] for x in g)),
                  "distinguished": stat([x["distinguished"] for x in g]),
                  "pareto_rules": stat([x["pareto"] for x in g]),
                  "pareto_settings": stat([x["pareto_settings"] for x in g]),
                  "scenario_feasible": stat(feas),
                  "scenario_infeasible_splits": int(sum(v == 0 for v in feas)),
                  "min_certified_budget": stat(bud),
                  "front_overlap_frac": stat([x["front_overlap"]["retained_frac"] for x in g]),
                  "enrichment_own_fold": stat(pick(e_own, "fold")),
                  "enrichment_own_p_max": (max(pick(e_own, "fisher_p")) if pick(e_own, "fisher_p") else None),
                  "enrichment_own_significant": int(sum(x["significant"] for x in e_own)),
                  "enrichment_base_fold": stat(pick(e_base, "fold")),
                  "enrichment_base_p_max": (max(pick(e_base, "fisher_p")) if pick(e_base, "fisher_p") else None),
                  "enrichment_base_significant": int(sum(x["significant"] for x in e_base)),
                  "flipped": stat([x["flipped"] for x in perts if fam(x.get("label", x["kind"])) == f]),
                  "prevalence": stat([x["prevalence"] for x in perts if "prevalence" in x and fam(x.get("label", x["kind"])) == f])}
    res = {"description": __doc__.strip().splitlines()[0], "reps": R, "alphas": ALPHAS.tolist(),
           "requests": J, "delta": DELTA, "eps": EPS, "m": M, "limits": {k: list(v) for k, v in LIMITS.items()},
           "cut": CUT, "logP_min": LIP, "band": BAND, "flip_rate": RATE, "random_seeds": SEEDS,
           "base_setting": list(BASE_SETTING), "families": FAMS,
           "perturbations": perts, "per_split": rows, "aggregate": agg}
    json.dump(res, open(f"{outdir}/aqsol_noise.json", "w"), indent=1)

    # ------------- compact summary table
    W = ["perturbation", "flips", "distinguished", "pareto settings", "feasible/144", "min budget",
         "front overlap", "fold (own)", "max p", "sig"]
    fmt = "{:<19s}{:>7s}{:>17s}{:>19s}{:>16s}{:>18s}{:>20s}{:>20s}{:>10s}{:>8s}"
    print("\nAqSolDB lipophilic screen, label-noise sensitivity (10 splits, model and scores fixed)")
    print("values are median [min, max] over the evaluations of each family\n")
    print(fmt.format(*W)); print("-" * 154)
    def rng_s(s, d=0):
        if s is None: return "-"
        f = (lambda v: f"{v:.0f}") if d == 0 else (lambda v: f"{v:.2f}")
        return f"{f(s['median'])} [{f(s['min'])},{f(s['max'])}]"
    for f in FAMS:
        if f not in agg: continue
        a = agg[f]
        print(fmt.format(f, f"{a['flipped']['median']:.0f}", rng_s(a["distinguished"]),
                         rng_s(a["pareto_settings"]), rng_s(a["scenario_feasible"]),
                         rng_s(a["min_certified_budget"]), rng_s(a["front_overlap_frac"], 2),
                         rng_s(a["enrichment_own_fold"], 2),
                         (f"{a['enrichment_own_p_max']:.1e}" if a["enrichment_own_p_max"] is not None else "-"),
                         f"{a['enrichment_own_significant']}/{a['n_evaluations']}"))
    print("\nenrichment held at the published setting alpha = (0.15, 0.125):")
    for f in FAMS:
        if f not in agg: continue
        a = agg[f]
        print(f"  {f:<19s} fold {rng_s(a['enrichment_base_fold'], 2):<20s} max p "
              f"{a['enrichment_base_p_max']:.1e}  significant {a['enrichment_base_significant']}/{a['n_evaluations']}")
    print(f"\nwritten: {outdir}/aqsol_noise.json")


if __name__ == "__main__":
    main()

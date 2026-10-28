"""CALiSol-23 electrolyte screen: will a formulation (Li salt, concentration, solvent mixture)
reach a target ionic conductivity at room temperature?  Same protocol as the other molecular
examples: one fixed model (CatBoost), class-conditional conformal thresholds with the batch
correction, audit-set intervals, a sweep of coverage requests, and three ways of drawing the
calibration, audit and screened compounds.

Data handling
  - rows with missing conductivity are dropped; one source (DOI 10.1007/s10008-002-0300-9)
    reports values about 100 times below every other source for the same formulations and is
    excluded as a probable unit error;
  - a formulation is (salt, concentration and its unit, solvent fractions and their basis);
  - the screened quantity is the median conductivity of a formulation over measurements at
    20-30 C; label 1 ("good") if it is at least KSTAR mS/cm;
  - the model is trained on all temperatures of the training formulations (temperature is a
    feature) and scores the room-temperature formulations at 298.15 K.
Features: salt one-hot, concentration, unit and basis flags, the 38 solvent fractions
(normalized), fraction-weighted RDKit descriptors of the solvents, temperature.
Designs (REPS repeats each)
  random    formulations split 50/25/25 into training / calibration / audit at random;
  system    the same, but whole salt-solvent systems are assigned (all concentrations and
            fractions of a system go to one set);
  salt      screening a new salt: training and calibration on formulations of all other salts,
            audit and screened batch from LiPF6 formulations (the shift case);
  salt_pilot  the same model, calibrated instead on a random 40% of the LiPF6 formulations
            (a pilot sample of the screened library), audited on the remaining 60%.
  (Non-carbonate systems hold too few room-temperature formulations, 114, for a solvent-family
  shift with a pilot sample.)
Usage: python3 -I calisol.py CSV OUTDIR KSTAR REPS"""
import sys, os, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import lru_cache
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
import intervals, rowint
TH = lru_cache(maxsize=None)(intervals.thatcher)
rowint.thatcher = lru_cache(maxsize=None)(rowint.thatcher); rowint.rank_law = lru_cache(maxsize=None)(rowint.rank_law)
from rowint import calonly_class, calonly_mixed
from conformal import k_batch, thresholds, cells, cross_counts, break_ties

CSV, OUT, KSTAR, REPS = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]); os.makedirs(OUT, exist_ok=True)
A = D = 0.10; G = 0.05; M = 1000; ALPHAS = np.round(np.arange(0.025, 0.301, 0.025), 3); J = len(ALPHAS) ** 2
SMILES = {"EC": "C1COC(=O)O1", "PC": "CC1COC(=O)O1", "DMC": "COC(=O)OC", "EMC": "CCOC(=O)OC", "DEC": "CCOC(=O)OCC", "DME": "COCCOC",
          "DMSO": "CS(C)=O", "AN": "CC#N", "MOEMC": "COCCOC(=O)OC", "TFP": "O=P(OCC(F)(F)F)(OCC(F)(F)F)OCC(F)(F)F", "EA": "CCOC(C)=O",
          "MA": "COC(C)=O", "FEC": "O=C1OCC(F)O1", "DOL": "C1COCO1", "2-MeTHF": "CC1CCCO1", "DMM": "COCOC", "Freon 11": "FC(Cl)(Cl)Cl",
          "Methylene chloride": "ClCCl", "THF": "C1CCOC1", "Toluene": "Cc1ccccc1", "Sulfolane": "O=S1(=O)CCCC1", "2-Glyme": "COCCOCCOC",
          "3-Glyme": "COCCOCCOCCOC", "4-Glyme": "COCCOCCOCCOCCOC", "3-Me-2-Oxazolidinone": "CN1CCOC1=O", "3-MeSulfolane": "CC1CCS(=O)(=O)C1",
          "Ethyldiglyme": "CCOCCOCCOCC", "DMF": "CN(C)C=O", "Ethylbenzene": "CCc1ccccc1", "Ethylmonoglyme": "CCOCCOCC", "Benzene": "c1ccccc1",
          "g-Butyrolactone": "O=C1CCCO1", "Cumene": "CC(C)c1ccccc1", "Propylsulfone": "CCCS(=O)(=O)CCC", "Pseudocumeme": "Cc1ccc(C)c(C)c1",
          "TEOS": "CCO[Si](OCC)(OCC)OCC", "m-Xylene": "Cc1cccc(C)c1", "o-Xylene": "Cc1ccccc1C"}
DESIGNS = ["random", "system", "salt", "salt_pilot"]; NEWSALT = "LiPF6"
CARBONATES = {"EC", "PC", "DMC", "EMC", "DEC", "FEC", "MOEMC"}

def load():
    d = pd.read_csv(CSV, index_col=0); n0 = len(d)
    d = d[d.k.notna()]; n1 = len(d)
    d = d[d.doi != "10.1007/s10008-002-0300-9"]; n2 = len(d)
    d["salt"] = d.salt.str.strip(); solv = [c for c in d.columns[7:] if c in SMILES]
    assert len(solv) == 38, solv
    F = d[solv].to_numpy(float); F = F / F.sum(1, keepdims=True); d[solv] = F
    d["system"] = d.salt + "|" + d[solv].apply(lambda r: "+".join(s for s in solv if r[s] > 0), axis=1)
    d["form"] = d.system + "|" + d[solv].round(3).astype(str).agg(",".join, axis=1) + "|" + d.c.round(3).astype(str) + d["c units"] + d["solvent ratio type"]
    d["carbonate_only"] = d[solv].apply(lambda r: all(s in CARBONATES for s in solv if r[s] > 0), axis=1)
    d["has_carbonate"] = d[solv].apply(lambda r: any(s in CARBONATES for s in solv if r[s] > 0), axis=1)
    return d, solv, {"rows_total": n0, "rows_with_k": n1, "rows_after_exclusion": n2}

def featurize(d, solv):
    desc = {s: [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBA(m),
                rdMolDescriptors.CalcFractionCSP3(m), rdMolDescriptors.CalcNumRotatableBonds(m), Crippen.MolMR(m)]
            for s, m in ((s, Chem.MolFromSmiles(SMILES[s])) for s in solv)}
    Dm = np.array([desc[s] for s in solv]); W = d[solv].to_numpy(float)
    salts = sorted(d.salt.unique()); S = np.stack([(d.salt == s).to_numpy(float) for s in salts], 1)
    X = np.hstack([S, d[["c"]].to_numpy(float), (d["c units"] == "mol/l").to_numpy(float)[:, None],
                   np.stack([(d["solvent ratio type"] == t).to_numpy(float) for t in ("w", "v", "mol")], 1),
                   W, W @ Dm, d[["T"]].to_numpy(float)])
    return X

def rt_table(d):
    rt = d[(d["T"] >= 293.15) & (d["T"] <= 303.15)]
    g = rt.groupby("form"); idx = g.apply(lambda x: x.index[0])
    t = pd.DataFrame({"k": g.k.median(), "row": idx, "system": g.system.first(), "doi": g.doi.first(),
                      "carbonate_only": g.carbonate_only.first(), "has_carbonate": g.has_carbonate.first()})
    return t

def grouped(c):
    G8 = {"correct rejection": ["0_s0"], "false lead": ["0_s1"], "deferred inactive": ["0_both", "0_empty"], "confident hit": ["1_s1"],
          "missed active": ["1_s0"], "deferred active": ["1_both", "1_empty"], "single label": ["0_s0", "0_s1", "1_s0", "1_s1"],
          "deferred": ["0_both", "0_empty", "1_both", "1_empty"]}
    return {k: sum(c[x] for x in v) for k, v in G8.items()}

def exact_violation(n, k, m):
    from scipy.stats import betabinom
    if m <= 0: return 0.0
    x = int(np.ceil(round((1 - A) * m, 9))); return float(betabinom.cdf(x - 1, m, k, n + 1 - k))

def analyse(uc, yc, ua, ya, rng):
    """thresholds on calibration; audit checks, sweep, capacity, calibration-only, per-class coverage"""
    n0, n1 = int((yc == 0).sum()), int((yc == 1).sum()); p1 = n1 / (n0 + n1); m1 = int(round(M * p1)); m0 = M - m1
    out = {"n_cal": [n0, n1], "n_audit": [int((ya == 0).sum()), int((ya == 1).sum())]}
    k0, k1 = k_batch(n0, A, D, int(round(len(ya) * (1 - p1)))), k_batch(n1, A, D, int(round(len(ya) * p1)))
    t0, t1 = thresholds(uc, yc, k0, k1); c = cells(ua, ya, t0, t1); ma0, ma1 = out["n_audit"]
    cov0 = (c["0_s0"] + c["0_both"]) / max(1, ma0); cov1 = (c["1_s1"] + c["1_both"]) / max(1, ma1)
    out["default"] = {"k": [k0, k1], "cells": c, "cov": [cov0, cov1], "viol": [bool(cov0 < 0.9), bool(cov1 < 0.9)],
                      "p_viol": [exact_violation(n0, k0, ma0), exact_violation(n1, k1, ma1)],
                      "feasible": [bool(exact_violation(n0, n0, ma0) <= D), bool(exact_violation(n1, n1, ma1) <= D)]}
    c0, c1 = cross_counts(uc, yc, t0, t1); rec = {"k0": k0, "k1": k1, "t0": t0, "t1": t1, "two_label": bool(t0 + t1 >= 1), "c0": c0, "c1": c1}
    hit = tot = 0
    for y, n, m in ((0, n0, ma0), (1, n1, ma1)):
        for kk, (lo, hi) in calonly_class(rec, y, n, m).items(): hit += lo <= c[kk] <= hi; tot += 1
    out["calonly"] = [hit, tot]
    # audit halvings: default thresholds and the whole sweep
    rules = {}
    for a0 in ALPHAS:
        for a1 in ALPHAS:
            key = (k_batch(n0, a0, D, m0), k_batch(n1, a1, D, m1))
            if key not in rules: rules[key] = thresholds(uc, yc, *key)
    out["distinct_rules"] = len(rules)
    ah = {k: 0 for k in grouped(c)}; nh = 0; sw_hit = sw_tot = 0; bonf_all = 0; cap = []
    for h in range(20):
        p = rng.permutation(len(ya)); X, Y = p[: len(p) // 2], p[len(p) // 2:]; nA, nF = len(X), len(Y)
        ga, gf = grouped(cells(ua[X], ya[X], t0, t1)), grouped(cells(ua[Y], ya[Y], t0, t1))
        for k in ga: lo, hi = TH(ga[k], nA, nF, G); ah[k] += lo <= gf[k] <= hi
        nh += 1; allok = True; cand = []
        for (r0, r1) in rules.values():
            ga, gf = grouped(cells(ua[X], ya[X], r0, r1)), grouped(cells(ua[Y], ya[Y], r0, r1))
            for k in ga:
                lo, hi = TH(ga[k], nA, nF, G); sw_hit += lo <= gf[k] <= hi; sw_tot += 1
                lo, hi = TH(ga[k], nA, nF, G / J); allok &= lo <= gf[k] <= hi
            up = TH(ga["deferred"], nA, nF, 2 * G / J)[1]
            if up <= 0.5 * nF: cand.append((ga["false lead"] + ga["missed active"], up, gf["deferred"], nF))
        bonf_all += allok
        if cand: e, up, dF, nF_ = min(cand); cap.append([up / nF_, dF / nF_, bool(dF > 0.5 * nF_)])
    out["audit_default"] = {k: v / nh for k, v in ah.items()}; out["audit_sweep"] = [sw_hit, sw_tot]; out["bonf_all"] = bonf_all / nh; out["capacity"] = cap
    # sweep summary on the full audit set
    pts = []
    for (r0, r1) in rules.values():
        g = grouped(cells(ua, ya, r0, r1)); pts.append([1000 * g["missed active"] / len(ya), 1000 * g["false lead"] / len(ya), 1000 * g["deferred"] / len(ya)])
    pts = np.array(pts); par = [i for i, p in enumerate(pts) if not any(np.all(q <= p) and np.any(q < p) for q in pts)]
    out["sweep_points"] = pts.round(1).tolist(); out["pareto"] = len(par)
    return out

def main():
    d, solv, counts = load(); X = featurize(d, solv); y_all = (d.k.to_numpy() >= KSTAR).astype(int)
    rt = rt_table(d); rt["y"] = (rt.k >= KSTAR).astype(int)
    pos = {r: i for i, r in enumerate(d.index)}; rt["pos"] = rt.row.map(pos)
    Xrt = X[rt.pos.to_numpy()].copy(); Xrt[:, -1] = 298.15; yrt = rt.y.to_numpy()
    form_of_row = d.form.to_numpy()
    info = {**counts, "formulations_all": int(d.form.nunique()), "systems": int(d.system.nunique()), "papers": int(d.doi.nunique()),
            "rt_formulations": len(rt), "rt_good": int(yrt.sum()), "kstar": KSTAR,
            "rt_newsalt": int((rt.system.str.split("|").str[0] == NEWSALT).sum()), "rt_newsalt_good": int(rt.y[rt.system.str.split("|").str[0] == NEWSALT].sum()),
            "rt_carbonate_only": int(rt.carbonate_only.sum()), "rt_carbonate_only_good": int(rt.y[rt.carbonate_only].sum())}
    print(json.dumps(info), flush=True); res = {"info": info, "runs": []}
    forms = rt.index.to_numpy(); systems = rt.system.to_numpy()
    for design in DESIGNS:
        for r in range(REPS):
            seed = 30000 + 1000 * DESIGNS.index(design) + r; rng = np.random.default_rng(seed); t = time.time()
            if design == "random":
                p = rng.permutation(len(rt)); tr, ca, au = p[: len(p) // 2], p[len(p) // 2: 3 * len(p) // 4], p[3 * len(p) // 4:]
            elif design == "system":
                us = np.unique(systems); rng.shuffle(us); target = np.array([0.5, 0.25, 0.25]) * len(rt); parts = [[], [], []]
                for s in us:
                    i = int(np.argmax(target - np.array([len(q) for q in parts]))); parts[i] += list(np.where(systems == s)[0])
                tr, ca, au = (np.array(q) for q in parts)
            else:
                is_new = (rt.system.str.split("|").str[0] == NEWSALT).to_numpy()
                po = rng.permutation(np.where(~is_new)[0]); pn = rng.permutation(np.where(is_new)[0])
                if design == "salt":
                    tr, ca, au = po[: 2 * len(po) // 3], po[2 * len(po) // 3:], pn
                else:
                    npil = int(0.4 * len(pn)); tr, ca, au = po, pn[:npil], pn[npil:]
            train_forms = set(forms[tr]); held = set(forms[ca]) | set(forms[au])
            if design.startswith("salt"):
                mask = (d.salt != NEWSALT).to_numpy() & np.array([f not in held for f in form_of_row])   # every other-salt row not held out
            elif design == "system":
                held_sys = set(systems[ca]) | set(systems[au]); mask = ~d.system.isin(held_sys).to_numpy()  # no row of a held system
            else:
                mask = np.array([f not in held for f in form_of_row])        # all temperatures of every non-held formulation
            clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0, thread_count=2, random_seed=seed).fit(X[mask], y_all[mask])
            u = clf.predict_proba(Xrt)[:, 1]; uc, ua = break_ties(rng, u[ca], u[au]); yc, ya = yrt[ca], yrt[au]
            if os.environ.get("CALISOL_SAVE_SCORES"):                # stored scores for the scenario maps (same seeds as the full run)
                np.savez_compressed(f"{OUT}/scores_{design}_{r}.npz", u_cal=uc, y_cal=yc, u_audit=ua, y_audit=ya)
            o = {"design": design, "rep": r, "seed": seed, "n_train_rows": int(mask.sum()), "auroc_audit": float(roc_auc_score(ya, ua)) if 0 < ya.sum() < len(ya) else None,
                 "prevalence_cal": float(yc.mean()), "prevalence_audit": float(ya.mean())}
            if min((yc == 0).sum(), (yc == 1).sum()) < 5: o["skipped"] = True
            else: o.update(analyse(uc, yc, ua, ya, rng))
            res["runs"].append(o); print(design, r, f"{time.time() - t:.1f}s", o.get("auroc_audit"), o.get("default", {}).get("viol"), flush=True)
    json.dump(res, open(f"{OUT}/calisol_runs.json", "w"))

if __name__ == "__main__":
    main()

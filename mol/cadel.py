"""CA-DEL screen: train on cheap DEL enrichment, screen ChEMBL compounds whose labels are
physical affinities (Ki).  Same protocol as the other molecular examples: one fixed model
(Morgan r=2, 2048 bits + CatBoost), class-conditional conformal thresholds with the batch
correction, audit-set intervals, a sweep of coverage requests, and several ways of drawing
the calibration, audit and screened compounds.

Data handling
  - training uses DEL molecules only; the target is y = log(1 + del_enrich), a monotone
    transform defined at zero (del_enrich is a read count for CAII and CAXII and a
    normalized fold-change for CAIX; the same transform is used for all three);
  - a random 10% of the DEL molecules is held out as a DEL validation set and never trained
    on; it supplies the transfer check and the calibration set of design D;
  - the model is frozen after training: it is never fitted or recalibrated on ChEMBL;
  - scores are mapped to u in [0, 1] by u = sigmoid((y_hat - mu) / sd) with mu, sd from the
    DEL *training* molecules, so no ChEMBL label enters the map; only the ordering matters;
  - the screened pool is the ChEMBL table; in-class (binder) means pKi_median >= c for
    c in CUTOFFS; compounds flagged `inconsistent` (Ki spread > 1 log) are excluded.
Designs (REPS repeats each, seed 50000 + r)
  P  physical pilot: ChEMBL pool split at random 30/30/40 into calibration / audit / future;
     thresholds come from the ChEMBL calibration set with Ki labels.
  D  DEL calibration: thresholds from the DEL validation molecules with a binary DEL label,
     applied to the ChEMBL scores; audit and future are the ChEMBL pool split 50/50.
     CAII uses the CA-DEL `label` column (the release defines one; its positive rate is
     about 2%, far below the ChEMBL in-class rate, and that mismatch is part of the result).
     CAIX and CAXII have no released label, so the top q of del_enrich is labelled positive
     with q equal to the ChEMBL in-class rate at that cut-off (ranked with random
     tie-breaking, so the rate is exact despite the many tied counts).
  S  series split: as P, but whole Bemis-Murcko scaffolds are assigned; calibration and audit
     come from 60% of the scaffolds, the future batch from the other 40%.  The release
     carries no ChEMBL document id (`chembl_ids` is empty), so the scaffold is the stand-in
     for a publication series.
  N  calibration size: as P, with the calibration set fixed at n_cal compounds (audit 30%,
     future the rest).
The DEL model is trained ONCE per isoform (seed 0) and reused across every repeat, design
and cut-off.
Usage: python3 -I cadel.py DATADIR OUTDIR REPS"""
import sys, os, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import lru_cache
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
import intervals, conformal
TH = lru_cache(maxsize=None)(intervals.thatcher)
# memoized only: the DEL calibration sets are large and every repeat of a design reuses the
# same (n, alpha, delta, m), so the beta-binomial search would otherwise be repeated 144 times
k_batch = lru_cache(maxsize=None)(conformal.k_batch)
from conformal import thresholds, cells, break_ties

DATA, OUT, REPS = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(OUT, exist_ok=True)
A = D = 0.10; G = 0.05; M = 1000; PLATE = 96
ALPHAS = np.round(np.arange(0.025, 0.301, 0.025), 3); J = len(ALPHAS) ** 2
ISOFORMS = ["CAII", "CAIX", "CAXII"]
CUTOFFS = [7, 8, 9]
DESIGNS = ["P", "D", "S"]
NCALS = [50, 100, 200, 400, 800]
VAL_FRAC = 0.10
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprints(smiles):
    """Morgan r=2, 2048 bits.  SMILES in data/cadel are already standardized by cadel_fetch.py."""
    X = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise SystemExit(f"unparsable standardized SMILES: {smi}")
        X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


def scaffolds(smiles):
    out = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        try:
            out.append(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)))
        except Exception:
            out.append("")
    return np.array(out)


def grouped(c):
    G8 = {"correct rejection": ["0_s0"], "false lead": ["0_s1"], "deferred non-binder": ["0_both", "0_empty"],
          "confident hit": ["1_s1"], "missed binder": ["1_s0"], "deferred binder": ["1_both", "1_empty"],
          "single label": ["0_s0", "0_s1", "1_s0", "1_s1"], "deferred": ["0_both", "0_empty", "1_both", "1_empty"]}
    return {k: sum(c[x] for x in v) for k, v in G8.items()}


def exact_violation(n, k, m):
    from scipy.stats import betabinom
    if m <= 0: return 0.0
    x = int(np.ceil(round((1 - A) * m, 9))); return float(betabinom.cdf(x - 1, m, k, n + 1 - k))


def analyse(uc, yc, ua, ya, uf, yf):
    """thresholds on calibration; audit checks against a held-out future batch; sweep; capacity"""
    n0, n1 = int((yc == 0).sum()), int((yc == 1).sum()); p1 = n1 / (n0 + n1)
    m1 = int(round(M * p1)); m0 = M - m1
    na, nf = len(ya), len(yf)
    out = {"n_cal": [n0, n1], "n_audit": [int((ya == 0).sum()), int((ya == 1).sum())],
           "n_future": [int((yf == 0).sum()), int((yf == 1).sum())]}
    ma0, ma1 = out["n_audit"]
    k0, k1 = k_batch(n0, A, D, m0), k_batch(n1, A, D, m1)
    t0, t1 = thresholds(uc, yc, k0, k1)
    c = cells(ua, ya, t0, t1)
    cov0 = (c["0_s0"] + c["0_both"]) / max(1, ma0); cov1 = (c["1_s1"] + c["1_both"]) / max(1, ma1)
    out["default"] = {"k": [k0, k1], "planned_m": [m0, m1], "cells": c, "cov": [cov0, cov1],
                      "viol": [bool(cov0 < 0.9), bool(cov1 < 0.9)],
                      "p_viol": [exact_violation(n0, k0, ma0), exact_violation(n1, k1, ma1)],
                      "feasible": [bool(exact_violation(n0, n0, ma0) <= D), bool(exact_violation(n1, n1, ma1) <= D)]}
    # audit intervals from the audit set, checked against the realized future batch,
    # and reported for a planned batch of M and for one 96-well plate
    ga, gf = grouped(c), grouped(cells(uf, yf, t0, t1))
    ai = {}
    for k in ga:
        lo, hi = TH(ga[k], na, nf, G)
        ai[k] = {"audit_count": ga[k], "future_count": gf[k], "interval_future": [lo, hi],
                 "inside": bool(lo <= gf[k] <= hi),
                 "per1000": [TH(ga[k], na, M, G)[0], TH(ga[k], na, M, G)[1]],
                 "per_plate": list(TH(ga[k], na, PLATE, G))}
    out["audit_intervals"] = ai
    out["audit_default"] = {k: float(ai[k]["inside"]) for k in ai}
    # sweep over the 12 x 12 grid of coverage requests
    rules = {}
    for a0 in ALPHAS:
        for a1 in ALPHAS:
            key = (k_batch(n0, a0, D, m0), k_batch(n1, a1, D, m1))
            if key not in rules: rules[key] = thresholds(uc, yc, *key)
    out["distinct_rules"] = len(rules)
    sw_hit = sw_tot = 0; allok = True; cand = []; pts = []
    for (r0, r1) in rules.values():
        ca_ = grouped(cells(ua, ya, r0, r1)); cf_ = grouped(cells(uf, yf, r0, r1))
        for k in ca_:
            lo, hi = TH(ca_[k], na, nf, G); sw_hit += lo <= cf_[k] <= hi; sw_tot += 1
            lo, hi = TH(ca_[k], na, nf, G / J); allok &= lo <= cf_[k] <= hi
        up = TH(ca_["deferred"], na, nf, 2 * G / J)[1]
        if up <= 0.5 * nf: cand.append((ca_["false lead"] + ca_["missed binder"], up, cf_["deferred"], nf))
        pts.append([1000 * ca_["missed binder"] / na, 1000 * ca_["false lead"] / na, 1000 * ca_["deferred"] / na])
    out["audit_sweep"] = [sw_hit, sw_tot]; out["bonf_all"] = float(allok)
    if cand:
        e, up, dF, nF_ = min(cand); out["capacity"] = [[up / nF_, dF / nF_, bool(dF > 0.5 * nF_)]]
    else:
        out["capacity"] = []
    pts = np.array(pts)
    par = [i for i, p in enumerate(pts) if not any(np.all(q <= p) and np.any(q < p) for q in pts)]
    out["sweep_points"] = pts.round(1).tolist(); out["pareto"] = len(par)
    return out


def del_binary(iso, enrich, label, rate, rng):
    """binary DEL label for design D: the released column where there is one, else the top
    `rate` fraction of del_enrich (ranked with random tie-breaking, so the rate is exact)"""
    if label is not None:
        return label, "released label column"
    jit, = break_ties(rng, enrich.astype(float))
    n_pos = int(round(rate * len(jit)))
    thr_idx = np.argsort(-jit)[:max(1, n_pos)]
    y = np.zeros(len(jit), dtype=int); y[thr_idx] = 1
    return y, f"top {rate:.4f} of del_enrich (tie-broken ranks)"


def splits_P(rng, n):
    p = rng.permutation(n); a, b = int(0.3 * n), int(0.6 * n)
    return p[:a], p[a:b], p[b:]


def splits_N(rng, n, n_cal):
    p = rng.permutation(n); a = n_cal; b = a + int(0.3 * n)
    return p[:a], p[a:b], p[b:]


def splits_S(rng, scaf):
    """whole scaffolds: 60% of scaffolds give calibration and audit, 40% the future batch"""
    us = np.unique(scaf); rng.shuffle(us)
    n = len(scaf); cut = 0.6 * n; held = []; tot = 0
    for s in us:
        idx = np.where(scaf == s)[0]
        if tot < cut: held.append(idx); tot += len(idx)
    held = np.concatenate(held); fut = np.setdiff1d(np.arange(n), held)
    rng.shuffle(held)
    half = len(held) // 2
    return held[:half], held[half:], fut


def main():
    t_all = time.time()
    res = {"info": {"alpha": A, "delta": D, "gamma": G, "batch": M, "plate": PLATE, "reps": REPS,
                    "cutoffs": CUTOFFS, "designs": DESIGNS + [f"N{n}" for n in NCALS],
                    "model": "Morgan r=2 2048 bits + CatBoostRegressor(500, depth 6, lr 0.1)",
                    "target": "y = log(1 + del_enrich)", "val_frac": VAL_FRAC,
                    "model_trained_once_per_isoform_seed": 0,
                    "score_map": "u = sigmoid((y_hat - mu) / sd), mu and sd from DEL training molecules"},
           "isoforms": {}, "runs": []}
    for iso in ISOFORMS:
        t_iso = time.time()
        dl = pd.read_parquet(f"{DATA}/cadel_del_{iso}.parquet")
        ch = pd.read_parquet(f"{DATA}/cadel_chembl_{iso}.parquet")
        n_chembl_all = len(ch); ch = ch[~ch.inconsistent].reset_index(drop=True)
        y_del = np.log1p(dl.del_enrich.to_numpy(float))
        Xd = fingerprints(dl.smiles.to_numpy())
        rng0 = np.random.default_rng(0)
        perm = rng0.permutation(len(dl)); nval = int(VAL_FRAC * len(dl))
        val, tr = perm[:nval], perm[nval:]
        model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                                  thread_count=8, random_seed=0).fit(Xd[tr], y_del[tr])
        yhat_tr = model.predict(Xd[tr]); yhat_val = model.predict(Xd[val])
        mu, sd = float(yhat_tr.mean()), float(yhat_tr.std())
        Xc = fingerprints(ch.smiles.to_numpy()); yhat_ch = model.predict(Xc)
        u_ch = 1.0 / (1.0 + np.exp(-(yhat_ch - mu) / sd))
        u_val = 1.0 / (1.0 + np.exp(-(yhat_val - mu) / sd))
        pki = ch.pKi_median.to_numpy(float)
        info = {"n_del": len(dl), "n_del_train": len(tr), "n_del_val": len(val),
                "n_chembl_all": n_chembl_all, "n_chembl": len(ch),
                "n_chembl_inconsistent_excluded": n_chembl_all - len(ch),
                "del_enrich_zero_fraction": float((dl.del_enrich.to_numpy(float) == 0).mean()),
                "del_val_spearman": float(spearmanr(yhat_val, y_del[val]).statistic),
                "del_val_rmse": float(np.sqrt(np.mean((yhat_val - y_del[val]) ** 2))),
                "del_val_sd": float(y_del[val].std()),
                "chembl_spearman_pki": float(spearmanr(yhat_ch, pki).statistic),
                "mu": mu, "sd": sd,
                "has_released_del_label": bool("label" in dl.columns)}
        for c in CUTOFFS:
            yb = (pki >= c).astype(int)
            info[f"prevalence_c{c}"] = float(yb.mean())
            info[f"auroc_c{c}"] = float(roc_auc_score(yb, u_ch)) if 0 < yb.sum() < len(yb) else None
        res["isoforms"][iso] = info
        print(iso, json.dumps(info), flush=True)
        scaf = scaffolds(ch.smiles.to_numpy())
        label_col = dl.label.to_numpy()[val] if "label" in dl.columns else None
        n = len(ch)
        jobs = [(d, None) for d in DESIGNS] + [("N", k) for k in NCALS]
        for c in CUTOFFS:
            yb = (pki >= c).astype(int)
            for design, n_cal in jobs:
                name = design if n_cal is None else f"N{n_cal}"
                for r in range(REPS):
                    seed = 50000 + r; rng = np.random.default_rng(seed + 1000 * (CUTOFFS.index(c) + 1) + 7 * (jobs.index((design, n_cal)) + 1))
                    t = time.time()
                    o = {"isoform": iso, "cutoff": c, "design": name, "rep": r, "seed": seed}
                    if design == "D":
                        ydel_bin, rule = del_binary(iso, dl.del_enrich.to_numpy()[val], label_col, float(yb.mean()), rng)
                        p = rng.permutation(n); half = n // 2
                        ai, fi = p[:half], p[half:]
                        uc_, ua_, uf_ = break_ties(rng, u_val, u_ch[ai], u_ch[fi])
                        yc_, ya_, yf_ = ydel_bin, yb[ai], yb[fi]
                        o["del_label_rule"] = rule
                        o["prevalence_cal"] = float(yc_.mean())
                    else:
                        if design == "P":
                            ci, ai, fi = splits_P(rng, n)
                        elif design == "S":
                            ci, ai, fi = splits_S(rng, scaf)
                        else:
                            ci, ai, fi = splits_N(rng, n, n_cal)
                        if len(fi) < 50 or len(ci) < 10:
                            o["skipped"] = True; res["runs"].append(o); continue
                        uc_, ua_, uf_ = break_ties(rng, u_ch[ci], u_ch[ai], u_ch[fi])
                        yc_, ya_, yf_ = yb[ci], yb[ai], yb[fi]
                        o["prevalence_cal"] = float(yc_.mean())
                    o["prevalence_audit"] = float(ya_.mean()); o["prevalence_future"] = float(yf_.mean())
                    o["auroc_audit"] = float(roc_auc_score(ya_, ua_)) if 0 < ya_.sum() < len(ya_) else None
                    if min((yc_ == 0).sum(), (yc_ == 1).sum()) < 5 or min((ya_ == 0).sum(), (ya_ == 1).sum()) < 1:
                        o["skipped"] = True
                    else:
                        o.update(analyse(uc_, yc_, ua_, ya_, uf_, yf_))
                        if design == "P" and r == 0:
                            np.savez_compressed(f"{OUT}/scores_P_{iso}_c{c}.npz", u_cal=uc_, y_cal=yc_,
                                                u_audit=ua_, y_audit=ya_)
                    res["runs"].append(o)
                    if r == 0:
                        print(f"  {iso} c{c} {name} rep0 {time.time() - t:.1f}s auroc {o.get('auroc_audit')} "
                              f"viol {o.get('default', {}).get('viol')} skipped {o.get('skipped', False)}", flush=True)
        res["isoforms"][iso]["runtime_s"] = time.time() - t_iso
        print(f"{iso} done {time.time() - t_iso:.1f}s", flush=True)
    json.dump(res, open(f"{OUT}/cadel_runs.json", "w"))
    print(f"total {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

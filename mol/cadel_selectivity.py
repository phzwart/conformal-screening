"""CA-DEL selectivity screen: find CAXII inhibitors that spare CAII.

Potency alone is not the decision a carbonic-anhydrase programme makes.  CAXII is
tumour-associated; CAII is the ubiquitous off-target whose inhibition carries the side
effects of the classical sulfonamides.  The screening question is therefore potency on
CAXII *together with* a margin over CAII, and that is the in-class label used here:

    selective  <=>  pKi(CAXII) >= POT  and  pKi(CAXII) - pKi(CAII) >= GAP

The CA-DEL release screened the identical 122,500-molecule library against both isoforms
(the two DEL tables share every InChIKey), so the cheap proxy for this decision is the DEL
*differential* enrichment of each molecule between the two screens.  The model is trained on

    y = log(1 + del_enrich[CAXII]) - log(1 + del_enrich[CAII])

and frozen; nothing is fitted or recalibrated on ChEMBL.  Scores map to u in [0, 1] by
u = sigmoid((y_hat - mu) / sd) with mu, sd from the DEL training molecules, so no physical
label enters the map and only the ordering matters.

The screened pool is the ChEMBL compounds measured on both isoforms, excluding those
flagged `inconsistent` in either.  Designs, conformal settings, intervals and sweep are
exactly those of cadel.py, whose helpers this script reuses; design D calibrates on the DEL
validation molecules with a binary DEL selectivity label (the top q of the differential).

Reported as a diagnostic only: the AUROC of the single-isoform CAXII potency score against
the same selective label, which says whether the differential score is needed at all.

CAIX cannot be paired this way: its DEL library is disjoint from the other two.

A fourth argument restricts both the DEL library and the screened pool to molecules
carrying a primary sulfonamide, the zinc-binding group of essentially every carbonic
anhydrase inhibitor.  Only 2.0% of the DEL library carries one, against 78% of the paired
ChEMBL compounds, and the two screens enrich sulfonamides very differently (median counts
464 on CAII against 48 on CAXII), so on the whole library the differential largely encodes
the presence of that group rather than any per-molecule isoform preference.  Restricting to
sulfonamides holds the zinc-binding chemotype fixed and asks the question the tail of the
molecule actually decides.

Usage: python3 -I cadel_selectivity.py DATADIR OUTDIR REPS [all|sulfonamide]
"""
import sys, os, json, time, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cadel  # reads the same three command-line arguments at import
from cadel import (fingerprints, scaffolds, analyse, del_binary, splits_P, splits_N,
                   splits_S, VAL_FRAC)
from conformal import break_ties
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score

DATA, OUT, REPS = sys.argv[1], sys.argv[2], int(sys.argv[3])
SUBSET = sys.argv[4] if len(sys.argv) > 4 else "all"
if SUBSET not in ("all", "sulfonamide"):
    raise SystemExit("subset must be 'all' or 'sulfonamide'")
os.makedirs(OUT, exist_ok=True)
SUF = "" if SUBSET == "all" else "_sulf"
PRIMARY_SULFONAMIDE = "[SX4](=[OX1])(=[OX1])[NX3;H2]"


def has_primary_sulfonamide(smiles):
    from rdkit import Chem
    pat = Chem.MolFromSmarts(PRIMARY_SULFONAMIDE)
    out = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        out[i] = m is not None and m.HasSubstructMatch(pat)
    return out
TARGET, ANTI = "CAXII", "CAII"
LABELS = [(8, 0.5), (8, 1.0), (7, 0.5), (7, 1.0)]   # (potency pKi, margin over CAII)
PRIMARY = (8, 0.5)
DESIGNS = ["P", "D", "S"]
NCALS = [50, 100, 200, 400, 800]


def paired_del():
    """the two DEL tables share one library; return it keyed by InChIKey, with both readouts"""
    a = pd.read_parquet(f"{DATA}/cadel_del_{TARGET}.parquet")[["inchikey", "smiles", "del_enrich"]]
    b = pd.read_parquet(f"{DATA}/cadel_del_{ANTI}.parquet")[["inchikey", "del_enrich"]]
    m = a.merge(b, on="inchikey", suffixes=("_t", "_a"))
    if len(m) != len(a) or len(m) != len(b):
        raise SystemExit(f"DEL libraries differ: {len(a)} / {len(b)} / {len(m)} shared")
    return m


def paired_chembl():
    """ChEMBL compounds measured on both isoforms, consistent in both"""
    a = pd.read_parquet(f"{DATA}/cadel_chembl_{TARGET}.parquet")
    b = pd.read_parquet(f"{DATA}/cadel_chembl_{ANTI}.parquet")
    a = a[~a.inconsistent]; b = b[~b.inconsistent]
    m = a.merge(b[["inchikey", "pKi_median"]], on="inchikey", suffixes=("_t", "_a"))
    return m.reset_index(drop=True)


def main():
    t_all = time.time()
    dl, ch = paired_del(), paired_chembl()
    n_dl_all, n_ch_all = len(dl), len(ch)
    if SUBSET == "sulfonamide":
        dl = dl[has_primary_sulfonamide(dl.smiles.to_numpy())].reset_index(drop=True)
        ch = ch[has_primary_sulfonamide(ch.smiles.to_numpy())].reset_index(drop=True)
    y_del = np.log1p(dl.del_enrich_t.to_numpy(float)) - np.log1p(dl.del_enrich_a.to_numpy(float))
    X = fingerprints(dl.smiles.to_numpy())
    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(len(dl)); nval = int(VAL_FRAC * len(dl))
    val, tr = perm[:nval], perm[nval:]
    model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                              thread_count=8, random_seed=0).fit(X[tr], y_del[tr])
    yhat_tr, yhat_val = model.predict(X[tr]), model.predict(X[val])
    mu, sd = float(yhat_tr.mean()), float(yhat_tr.std())
    sig = lambda z: 1.0 / (1.0 + np.exp(-(z - mu) / sd))
    Xc = fingerprints(ch.smiles.to_numpy())
    u_ch, u_val = sig(model.predict(Xc)), sig(yhat_val)
    delta_del = dl.del_enrich_t.to_numpy(float) - dl.del_enrich_a.to_numpy(float)

    # diagnostic: the potency-only score, from the single-isoform model of cadel.py
    pot_model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                                  thread_count=8, random_seed=0).fit(
        X[tr], np.log1p(dl.del_enrich_t.to_numpy(float))[tr])
    u_pot = pot_model.predict(Xc)

    pki_t, pki_a = ch.pKi_median_t.to_numpy(float), ch.pKi_median_a.to_numpy(float)
    gap = pki_t - pki_a
    from scipy.stats import spearmanr
    info = {"pair": f"{TARGET} over {ANTI}", "subset": SUBSET,
            "n_del_before_subset": n_dl_all, "n_chembl_before_subset": n_ch_all,
            "n_del": len(dl), "n_del_train": len(tr),
            "n_del_val": len(val), "n_chembl_paired": len(ch),
            "target": "y = log1p(del_enrich[CAXII]) - log1p(del_enrich[CAII])",
            "del_val_spearman": float(spearmanr(yhat_val, y_del[val]).statistic),
            "del_val_rmse": float(np.sqrt(np.mean((yhat_val - y_del[val]) ** 2))),
            "del_val_sd": float(y_del[val].std()),
            "chembl_spearman_gap": float(spearmanr(u_ch, gap).statistic),
            "chembl_spearman_potency": float(spearmanr(u_ch, pki_t).statistic),
            "gap_quantiles": [float(q) for q in np.quantile(gap, [.05, .25, .5, .75, .95])],
            "mu": mu, "sd": sd}
    for pot, g in LABELS:
        yb = ((pki_t >= pot) & (gap >= g)).astype(int)
        key = f"p{pot}_g{g}"
        info[f"prevalence_{key}"] = float(yb.mean())
        info[f"auroc_{key}"] = float(roc_auc_score(yb, u_ch))
        info[f"auroc_potency_score_{key}"] = float(roc_auc_score(yb, u_pot))
    res = {"info": info, "runs": []}
    print(json.dumps(info, indent=1), flush=True)

    scaf = scaffolds(ch.smiles.to_numpy()); n = len(ch)
    jobs = [(d, None) for d in DESIGNS] + [("N", k) for k in NCALS]
    for li, (pot, g) in enumerate(LABELS):
        yb = ((pki_t >= pot) & (gap >= g)).astype(int)
        for ji, (design, n_cal) in enumerate(jobs):
            name = design if n_cal is None else f"N{n_cal}"
            for r in range(REPS):
                seed = 50000 + r
                rng = np.random.default_rng(seed + 1000 * (li + 1) + 7 * (ji + 1))
                o = {"potency": pot, "gap": g, "design": name, "rep": r, "seed": seed}
                if design == "D":
                    ydel_bin, rule = del_binary(TARGET, delta_del[val], None, float(yb.mean()), rng)
                    p = rng.permutation(n); half = n // 2
                    ai, fi = p[:half], p[half:]
                    uc_, ua_, uf_ = break_ties(rng, u_val, u_ch[ai], u_ch[fi])
                    yc_, ya_, yf_ = ydel_bin, yb[ai], yb[fi]
                    o["del_label_rule"] = rule.replace("del_enrich", "differential del_enrich")
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
                o["prevalence_cal"] = float(np.mean(yc_))
                o["prevalence_audit"] = float(ya_.mean()); o["prevalence_future"] = float(yf_.mean())
                o["auroc_audit"] = float(roc_auc_score(ya_, ua_)) if 0 < ya_.sum() < len(ya_) else None
                if min((yc_ == 0).sum(), (yc_ == 1).sum()) < 5 or min((ya_ == 0).sum(), (ya_ == 1).sum()) < 1:
                    o["skipped"] = True
                else:
                    o.update(analyse(uc_, yc_, ua_, ya_, uf_, yf_))
                    if design == "P" and r == 0 and (pot, g) == PRIMARY:
                        np.savez_compressed(f"{OUT}/scores_sel_P_{TARGET}{SUF}.npz", u_cal=uc_,
                                            y_cal=yc_, u_audit=ua_, y_audit=ya_)
                res["runs"].append(o)
                if r == 0:
                    print(f"  p{pot} g{g} {name} rep0 auroc {o.get('auroc_audit')} "
                          f"viol {o.get('default', {}).get('viol')} skipped {o.get('skipped', False)}", flush=True)
    json.dump(res, open(f"{OUT}/cadel_selectivity_runs{SUF}.json", "w"))
    print(f"total {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

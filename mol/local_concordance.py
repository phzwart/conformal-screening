"""Calibrating against a surrogate: what has to agree with the truth, and where.

A campaign often cannot calibrate against the quantity it cares about.  The thresholds are
set using something cheaper -- an enrichment count, a single-shot readout, a computed proxy
-- while the statement the campaign wants is about affinity.  The question is what that
substitution costs.

A strictly increasing transformation of the quantity costs nothing at all, and the reason is
bookkeeping: if the surrogate is g(potency) for increasing g, then the event {potency >= c}
and the event {surrogate >= g(c)} are the same event, so the calibration labels are the same
labels and every threshold, count and interval is unchanged.  Exact monotonicity is not,
however, what a real surrogate offers.  A real surrogate ranks compounds nearly as potency
does, and the useful question is how near, and near where.

The mechanism is local, and an identity says so.  The out-class threshold is the k0-th order
statistic of the scores of the calibration compounds the surrogate calls out-class, so it
sits at a fixed quantile of the score distribution conditioned on the surrogate's label.
The coverage the campaign actually achieves is the score distribution conditioned on the
*truth's* label, evaluated at that same threshold.  The shortfall is therefore the vertical
gap between two class-conditional score CDFs at one point:

    achieved - nominal  =  F(tau | truth out-class) - F(tau | surrogate out-class)

Nothing in that expression refers to the agreement of the two labels away from tau.  A
surrogate may rank compounds perfectly over most of the range and still lose coverage if it
misorders them in the narrow band where the threshold falls; it may rank them poorly overall
and lose nothing if it happens to agree there.  Global summaries -- a Spearman correlation,
an area under the curve -- average over the whole range and so cannot see this.

This takes the ChEMBL pool of one isoform with its measured affinities, holds the population
and the frozen score fixed, and changes only the label used for calibration: the surrogate
is the affinity plus noise, thresholded at a matched prevalence.  At zero noise the surrogate
is an exact monotone relabelling and must reproduce the truth exactly, which is the control.
As the noise grows, the per-class coverage, the CDF gap, a local discordance measured in a
band around the threshold, and the two global summaries are reported together, so it can be
seen which of them tracks the damage.

The audit-set intervals are reported alongside, because they are a different guarantee: they
price whatever rule they are handed and need only the audit and the batch to be exchangeable,
not the rule to be well calibrated.

Usage: python3 -I local_concordance.py DATADIR OUT.json [ISOFORM] [REPS]
"""
import sys, os, json, time, numpy as np, pandas as pd

DATA, OUT = sys.argv[1], sys.argv[2]
ISO = sys.argv[3] if len(sys.argv) > 3 else "CAII"
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import lru_cache
import intervals, conformal
from conformal import thresholds, cells, break_ties
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
TH = lru_cache(maxsize=None)(intervals.thatcher)
k_batch = lru_cache(maxsize=None)(conformal.k_batch)

A = D_ = 0.10               # per-class coverage request, and its tolerance
G = 0.05                    # audit interval level
M = 1000
VAL_FRAC = 0.10
CUTOFFS = (7, 8)
SIGMAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
BAND = 0.10                 # half-width of the quantile band used for local discordance
CATS = ("confident hit", "false lead", "missed binder", "deferred binder",
        "correct rejection", "deferred non-binder")
G8 = {"correct rejection": ["0_s0"], "false lead": ["0_s1"],
      "deferred non-binder": ["0_both", "0_empty"], "confident hit": ["1_s1"],
      "missed binder": ["1_s0"], "deferred binder": ["1_both", "1_empty"]}


def fingerprints(smiles):
    X = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise SystemExit(f"unparsable SMILES: {smi}")
        X[i] = GEN.GetFingerprintAsNumPy(m)
    return X


def grouped(c):
    return {k: sum(c[x] for x in v) for k, v in G8.items()}


def frozen_score(data, iso, cache):
    """the DEL-trained score, mapped to (0,1) using the DEL training molecules only.

    Identical in construction to the screen's own score and cached, because it is frozen:
    no ChEMBL label enters it and it does not change with the surrogate under test.
    """
    if os.path.exists(cache):
        z = np.load(cache)
        return z["u_ch"], z["pki"], json.loads(str(z["info"]))
    dl = pd.read_parquet(f"{data}/cadel_del_{iso}.parquet")
    ch = pd.read_parquet(f"{data}/cadel_chembl_{iso}.parquet")
    n_all = len(ch); ch = ch[~ch.inconsistent].reset_index(drop=True)
    y_del = np.log1p(dl.del_enrich.to_numpy(float))
    Xd = fingerprints(dl.smiles.to_numpy())
    perm = np.random.default_rng(0).permutation(len(dl))
    tr = perm[int(VAL_FRAC * len(dl)):]
    model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0,
                              thread_count=2, random_seed=0).fit(Xd[tr], y_del[tr])
    yhat_tr = model.predict(Xd[tr])
    mu, sd = float(yhat_tr.mean()), float(yhat_tr.std())
    u_ch = 1.0 / (1.0 + np.exp(-(model.predict(fingerprints(ch.smiles.to_numpy())) - mu) / sd))
    pki = ch.pKi_median.to_numpy(float)
    info = {"isoform": iso, "n_del": int(len(dl)), "n_chembl": int(len(ch)),
            "n_chembl_inconsistent_excluded": int(n_all - len(ch)),
            "spearman_score_vs_pki": float(spearmanr(u_ch, pki).statistic)}
    np.savez(cache, u_ch=u_ch, pki=pki, info=json.dumps(info))
    return u_ch, pki, info


def surrogate_label(rng, pki, sigma, rate):
    """affinity plus noise, thresholded so its positive rate matches the truth's.

    At sigma = 0 this is a strictly increasing relabelling of the affinity at a matched
    prevalence, so it returns exactly the truth's label; that is the control.
    """
    z = pki + sigma * rng.standard_normal(len(pki))
    jit, = break_ties(rng, z)
    n_pos = int(round(rate * len(jit)))
    y = np.zeros(len(jit), dtype=int)
    y[np.argsort(-jit)[:max(1, n_pos)]] = 1
    return y, z


def band_discordance(u, y_true, y_sur, t_0, t_1):
    """disagreement between the two labels inside the region the rule leaves indecisive.

    The two thresholds bound that region: a compound below both gets the single label 0, one
    above both gets the single label 1, and one between them gets either both labels or
    neither, depending on which way round they fall.  In the interior of a single-label
    region a compound's own order can move without changing its set and without moving a
    threshold, so a surrogate is free to misorder it there.  The compounds whose order
    decides anything are the ones in the band, because the band's edges are the thresholds.
    The rule therefore marks out the region in which the surrogate has to agree with the
    truth, and no window has to be chosen by hand.
    """
    lo, hi = min(t_0, 1 - t_1), max(t_0, 1 - t_1)
    m = (u >= lo) & (u <= hi)
    out = {"band": [float(lo), float(hi)], "mass": float(m.mean()),
           "n": int(m.sum()), "overlapping": bool(t_0 >= 1 - t_1)}
    out["discordance"] = (float(np.mean(y_true[m] != y_sur[m])) if m.sum() >= 20 else None)
    ms = ~m
    out["discordance_outside"] = (float(np.mean(y_true[ms] != y_sur[ms]))
                                 if ms.sum() >= 20 else None)
    return out


def main():
    t0 = time.time()
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f".score_{ISO}.npz")
    u_all, pki, info = frozen_score(DATA, ISO, cache)
    print(json.dumps(info), flush=True)

    rows = []
    n = len(u_all)
    for c in CUTOFFS:
        y_true_all = (pki >= c).astype(int)
        rate = float(y_true_all.mean())
        auroc_global = float(roc_auc_score(y_true_all, u_all))
        for sigma in SIGMAS:
            acc = {"cov0": [], "cov1": [], "viol0": 0, "viol1": 0, "gap0": [], "gap1": [],
                   "dband": [], "dout": [], "mass": [], "ovl": 0,
                   "sp": [], "auc": [], "aud": [0, 0], "ident": 0}
            for r in range(REPS):
                rng = np.random.default_rng(31000 + 97 * r + 7 * c + int(1000 * sigma))
                y_sur_all, z = surrogate_label(rng, pki, sigma, rate)
                acc["ident"] += int(np.array_equal(y_sur_all, y_true_all))
                acc["sp"].append(float(spearmanr(z, pki).statistic))
                acc["auc"].append(float(roc_auc_score(y_true_all, y_sur_all)))

                p = rng.permutation(n)
                a, b = int(0.30 * n), int(0.60 * n)
                ci, ai, fi = p[:a], p[a:b], p[b:]
                uc, ua, uf = break_ties(rng, u_all[ci], u_all[ai], u_all[fi])

                # thresholds from the SURROGATE's labels; everything checked against the TRUTH
                ys_c = y_sur_all[ci]
                n0, n1 = int((ys_c == 0).sum()), int((ys_c == 1).sum())
                if min(n0, n1) < 20:
                    continue
                m1 = int(round(M * n1 / (n0 + n1))); m0 = M - m1
                k0, k1 = k_batch(n0, A, D_, m0), k_batch(n1, A, D_, m1)
                t_0, t_1 = thresholds(uc, ys_c, k0, k1)

                ya, yf = y_true_all[ai], y_true_all[fi]
                cc = cells(ua, ya, t_0, t_1)
                ma0, ma1 = int((ya == 0).sum()), int((ya == 1).sum())
                cov0 = (cc["0_s0"] + cc["0_both"]) / max(1, ma0)
                cov1 = (cc["1_s1"] + cc["1_both"]) / max(1, ma1)
                acc["cov0"].append(cov0); acc["cov1"].append(cov1)
                acc["viol0"] += int(cov0 < 1 - A); acc["viol1"] += int(cov1 < 1 - A)

                # the identity: achieved minus nominal is a gap between two CDFs at tau
                uc0, uc1 = uc[ys_c == 0], uc[ys_c == 1]
                ua0, ua1 = ua[ya == 0], ua[ya == 1]
                acc["gap0"].append(float((ua0 <= t_0).mean() - (uc0 <= t_0).mean()))
                acc["gap1"].append(float((ua1 >= 1 - t_1).mean() - (uc1 >= 1 - t_1).mean()))

                bd = band_discordance(u_all, y_true_all, y_sur_all, t_0, t_1)
                acc["mass"].append(bd["mass"]); acc["ovl"] += int(bd["overlapping"])
                if bd["discordance"] is not None: acc["dband"].append(bd["discordance"])
                if bd["discordance_outside"] is not None:
                    acc["dout"].append(bd["discordance_outside"])

                ga, gf = grouped(cc), grouped(cells(uf, yf, t_0, t_1))
                for k in ga:
                    lo, hi = TH(ga[k], len(ya), len(yf), G)
                    acc["aud"][1] += 1; acc["aud"][0] += int(lo <= gf[k] <= hi)

            if not acc["cov0"]:
                continue
            row = {"cutoff": c, "sigma": sigma, "reps": len(acc["cov0"]),
                   "identical_to_truth": acc["ident"],
                   "prevalence": rate, "auroc_score_global": auroc_global,
                   "spearman_surrogate_vs_pki": float(np.mean(acc["sp"])),
                   "auroc_surrogate_vs_truth": float(np.mean(acc["auc"])),
                   "cov_out": float(np.mean(acc["cov0"])),
                   "cov_in": float(np.mean(acc["cov1"])),
                   "viol_out": [acc["viol0"], len(acc["cov0"])],
                   "viol_in": [acc["viol1"], len(acc["cov1"])],
                   "cdf_gap_out": float(np.mean(acc["gap0"])),
                   "cdf_gap_in": float(np.mean(acc["gap1"])),
                   "band_mass": float(np.mean(acc["mass"])),
                   "band_overlapping": [acc["ovl"], len(acc["cov0"])],
                   "discordance_in_band": float(np.mean(acc["dband"])) if acc["dband"] else None,
                   "discordance_outside_band": float(np.mean(acc["dout"])) if acc["dout"] else None,
                   "audit_lines_covered": acc["aud"]}
            rows.append(row)
            db = row["discordance_in_band"]; do = row["discordance_outside_band"]
            print(f"  c={c} sigma={sigma:<4} ident {row['identical_to_truth']:>2}/{REPS}  "
                  f"cov out {row['cov_out']:.3f} in {row['cov_in']:.3f}  "
                  f"gap out {row['cdf_gap_out']:+.3f} in {row['cdf_gap_in']:+.3f}  "
                  f"disc band {db if db is None else round(db,3)} "
                  f"outside {do if do is None else round(do,3)}  "
                  f"band mass {row['band_mass']:.2f}  "
                  f"rho {row['spearman_surrogate_vs_pki']:.3f}  "
                  f"audit {acc['aud'][0]}/{acc['aud'][1]}  [{time.time()-t0:.0f}s]", flush=True)

    json.dump({"info": info, "alpha": A, "gamma": G, "band": BAND, "reps": REPS,
               "rows": rows}, open(OUT, "w"), indent=1)
    print(f"\nwritten to {OUT}  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

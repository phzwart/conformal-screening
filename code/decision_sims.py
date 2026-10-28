"""Decision-layer and dependence simulations.

Part A (population, exact): set-based rules vs Chow's rule on the true posterior.
  With binary probability scores u = p_hat(active|x), every conformal set rule
  ({0}->label 0, {1}->label 1, otherwise defer) is a two-threshold rule on u:
  label 0 if u < l, label 1 if u > h, defer otherwise. When the posterior eta(u)
  is increasing in u, Chow's optimal rule is also of this form. We compute expected
  cost per compound (units of a false-lead cost) for (i) Chow's rule on the true
  posterior, (ii) the best rule subject to class-conditional coverage >= 0.90 for
  both classes, (iii) the default conformal rule (coverage exactly 0.90 per class).
Part B (selection; superseded by selection_sim.py, which the paper reports): among J candidate rules, choose the one with the fewest test-set
  errors (missed actives + false leads) whose upper prediction limit for deferrals is
  within assay capacity, with and without a
  Bonferroni adjustment over the J rules; record how often the realized deferrals
  exceed capacity.
Part C (scaffold dependence): compounds belong to clusters with a shared shift of the
  model score; calibration and test sets come from one set of clusters; the future
  batch comes either from the same clusters or from new clusters."""
import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import beta
from scipy.special import expit, logit
from intervals import thatcher

# ---------------- Part A ----------------
def part_a():
    geoms = {"well separated": (0.15, 9, 3), "overlapping": (0.15, 4, 3), "balanced, overlapping": (0.5, 4, 3)}
    a0, b0 = 2, 7
    lams = [2, 5, 10, 20, 50]; rhos = [0.1, 0.3, 0.5, 0.8]
    grid = np.linspace(0, 1, 801)
    out = []
    for gname, (pi, a1, b1) in geoms.items():
        F0 = beta.cdf(grid, a0, b0); F1 = beta.cdf(grid, a1, b1)
        L, H = np.meshgrid(np.arange(len(grid)), np.arange(len(grid)), indexing="ij")
        ok = H >= L
        # P(Y=1,u<l)=pi F1(l); P(Y=0,u>h)=(1-pi)(1-F0(h)); deferred = P(l<=u<=h)
        miss = pi * F1[L]; lead = (1 - pi) * (1 - F0[H])
        defer = pi * (F1[H] - F1[L]) + (1 - pi) * (F0[H] - F0[L])
        cov0 = F0[H]; cov1 = 1 - F1[L]           # two-label convention: 0 kept iff u<=h, 1 kept iff u>=l
        feas = ok & (cov0 >= 0.9) & (cov1 >= 0.9)
        t0 = beta.ppf(0.9, a0, b0); t1l = beta.ppf(0.1, a1, b1)   # default: coverage exactly 0.9
        lo_d, hi_d = min(t0, t1l), max(t0, t1l)
        def cost_at(l, h, lam, rho):
            return lam * pi * beta.cdf(l, a1, b1) + (1 - pi) * beta.sf(h, a0, b0) + rho * (
                pi * (beta.cdf(h, a1, b1) - beta.cdf(l, a1, b1)) + (1 - pi) * (beta.cdf(h, a0, b0) - beta.cdf(l, a0, b0)))
        for lam in lams:
            for rho in rhos:
                C = lam * miss + lead + rho * defer
                chow = np.min(np.where(ok, C, np.inf)); cons = np.min(np.where(feas, C, np.inf))
                dflt = cost_at(lo_d, hi_d, lam, rho)
                out.append({"geometry": gname, "lambda": lam, "rho": rho, "chow": round(float(chow), 4),
                            "constrained": round(float(cons), 4), "default_conformal": round(float(dflt), 4),
                            "excess_constrained_pct": round(100 * (cons / chow - 1), 1),
                            "excess_default_pct": round(100 * (dflt / chow - 1), 1)})
    return out

# ---------------- Part B ----------------
def draw(rng, pi, a1, b1, n):
    y = (rng.random(n) < pi).astype(int)
    u = np.where(y == 1, rng.beta(a1, b1, n), rng.beta(2, 7, n)); return y, u

def part_b(R=4000, n_cal=500, n_test=200, m=1000, cap=0.15, g=0.05, seed=1, n_alpha=10):
    rng = np.random.default_rng(seed)
    alphas = list(np.linspace(0.02, 0.30, n_alpha)); J = len(alphas) ** 2
    viol = {"naive": 0, "bonferroni": 0}; errs = {"naive": 0, "bonferroni": 0}; chosen = {"naive": 0, "bonferroni": 0}; reps = 0
    for r in range(R):
        pi, a1, b1 = rng.uniform(0.1, 0.4), rng.uniform(3, 9), rng.uniform(2, 4)
        yc, uc = draw(rng, pi, a1, b1, n_cal); yt, ut = draw(rng, pi, a1, b1, n_test); yf, uf = draw(rng, pi, a1, b1, m)
        s0 = np.sort(uc[yc == 0]); s1 = np.sort(1 - uc[yc == 1]); n0, n1 = len(s0), len(s1)
        rules = []
        for a0_ in alphas:
            for a1_ in alphas:
                t0 = s0[min(n0, int(np.ceil((n0 + 1) * (1 - a0_)))) - 1]; t1 = s1[min(n1, int(np.ceil((n1 + 1) * (1 - a1_)))) - 1]
                def stats(y, u):
                    in0 = u <= t0; in1 = u >= 1 - t1
                    err = np.sum((y == 0) & in1 & ~in0) + np.sum((y == 1) & in0 & ~in1)
                    return int(np.sum(in0 == in1)), int(err)
                (dt, et), (df, ef) = stats(yt, ut), stats(yf, uf)
                rules.append((dt, df, et, ef))
        reps += 1
        for mode, gg in (("naive", g), ("bonferroni", g / J)):
            best = None
            for kt, kf, et, ef in rules:
                hi = thatcher(kt, n_test, m, 2 * gg)[1]       # one-sided upper limit at level gg
                if hi <= cap * m and (best is None or et < best[2]): best = (kt, kf, et, ef)
            if best is not None:
                chosen[mode] += 1; viol[mode] += best[1] > cap * m; errs[mode] += best[3]
    return {"reps": reps, "candidate_rules": J, "capacity_fraction": cap,
            "naive": {"rule_found": chosen["naive"], "capacity_exceeded_fraction": round(viol["naive"] / max(1, chosen["naive"]), 4), "future_errors_per_batch": round(errs["naive"] / max(1, chosen["naive"]), 1)},
            "bonferroni": {"rule_found": chosen["bonferroni"], "capacity_exceeded_fraction": round(viol["bonferroni"] / max(1, chosen["bonferroni"]), 4), "future_errors_per_batch": round(errs["bonferroni"] / max(1, chosen["bonferroni"]), 1)}}

# ---------------- Part C ----------------
def clustered(rng, n, clusters, sig, pi=0.2, a1=6, b1=3):
    c = rng.integers(0, len(clusters), n); e = clusters[c]
    y = (rng.random(n) < expit(logit(pi) + e[:, 0])).astype(int)
    u = np.where(y == 1, rng.beta(a1, b1, n), rng.beta(2, 7, n))
    return y, expit(logit(np.clip(u, 1e-9, 1 - 1e-9)) + e[:, 1])

def part_c(R=3000, n_clusters=60, n_cal=1000, n_test=500, m=1000, g=0.05, seed=2):
    rng = np.random.default_rng(seed); out = []
    for sig in (0.0, 0.5, 1.0, 1.5):
        hit = {"same clusters": np.zeros(3), "new clusters": np.zeros(3)}
        for r in range(R):
            cl = rng.normal(0, sig, (n_clusters, 2)); new = rng.normal(0, sig, (n_clusters, 2))
            yc, uc = clustered(rng, n_cal, cl, sig); yt, ut = clustered(rng, n_test, cl, sig)
            s0 = np.sort(uc[yc == 0]); s1 = np.sort(1 - uc[yc == 1])
            t0 = s0[int(np.ceil((len(s0) + 1) * 0.9)) - 1]; t1 = s1[int(np.ceil((len(s1) + 1) * 0.9)) - 1]
            def counts(y, u):
                in0 = u <= t0; in1 = u >= 1 - t1
                return np.array([np.sum((y == 0) & in1 & ~in0), np.sum((y == 1) & in0 & ~in1), np.sum(in0 == in1)])
            kt = counts(yt, ut)
            for name, clu in (("same clusters", cl), ("new clusters", new)):
                kf = counts(*clustered(rng, m, clu, sig))
                for i in range(3):
                    lo, hi = thatcher(int(kt[i]), n_test, m, g); hit[name][i] += lo <= kf[i] <= hi
        out.append({"cluster_sd": sig, **{k: dict(zip(["false leads", "missed actives", "deferred"], (v / R).round(3).tolist())) for k, v in hit.items()}})
    return out

if __name__ == "__main__":
    res = {"A_cost": part_a()}
    print("Part A done"); sys.stdout.flush()
    res["B_selection"] = part_b(); print(res["B_selection"]); sys.stdout.flush()
    res["C_scaffold"] = part_c(); print(json.dumps(res["C_scaffold"], indent=1))
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    for row in res["A_cost"]:
        print(f"{row['geometry']:22s} lam={row['lambda']:3d} rho={row['rho']:.1f}  chow={row['chow']:.4f}  constrained +{row['excess_constrained_pct']:5.1f}%  default +{row['excess_default_pct']:6.1f}%")

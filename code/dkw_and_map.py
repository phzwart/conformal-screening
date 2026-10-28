"""Usage: python3 -I dkw_and_map.py [dkw]   (the DKW rows, now superseded by threshold_sim.py, only with "dkw")
(1) One-sided DKW rows for ESI Table S1: selected index, deployed level, batch violation
(exact Beta-Binomial, m = 100) and Monte Carlo check (1e5 calibration draws).
(2) ESI Fig. S1: deployed miscoverage level of the batch-free correction over (alpha, delta)."""
import numpy as np, sys, os, json, matplotlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mol"))
from conformal import k_dkw
from scipy.stats import betabinom, beta
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(3); m = 100; x = 90; rows = []
for n in ((50, 75, 100, 150, 200, 250, 300, 500) if "dkw" in sys.argv else ()):
    k = k_dkw(n, 0.10, 0.10); th = float(betabinom.cdf(x - 1, m, k, n + 1 - k))
    U = np.sort(rng.random((100000, n)), axis=1)[:, k - 1]; cnt = rng.binomial(m, U)
    rows.append({"n": n, "u": n + 1 - k, "level": round((n + 1 - k) / (n + 1), 3), "observed": round(float(np.mean(cnt < x)), 4), "theory": round(th, 4)})
if rows: print(json.dumps(rows)); json.dump(rows, open("dkw_onesided.json", "w"), indent=1)
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
al = np.linspace(0.01, 0.30, 120); de = np.linspace(0.01, 0.30, 120)
for ax, n in zip(axes, (50, 100, 300)):
    Z = np.full((len(de), len(al)), np.nan); us = np.arange(1, n + 1)
    for i, d in enumerate(de):
        for j, a in enumerate(al):
            ok = beta.sf(1 - a, n + 1 - us, us) >= 1 - d
            if ok.any(): Z[i, j] = us[ok].max() / (n + 1)
    im = ax.pcolormesh(al, de, Z, cmap="Blues", shading="nearest", vmin=0, vmax=0.3)
    ax.contour(al, de, np.nan_to_num(Z, nan=-1), levels=np.arange(0.025, 0.30, 0.025), colors="#52514e", linewidths=0.3)
    ax.set_title(f"{n} calibration compounds", fontsize=8); ax.set_xlabel("requested miscoverage $\\alpha$")
axes[0].set_ylabel("requested failure probability $\\delta$")
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02); cb.set_label("miscoverage level actually used, $u/(n+1)$", fontsize=7)
fig.savefig("../figures/figS1_correction_map.pdf", bbox_inches="tight"); fig.savefig("../figures/figS1_correction_map.png", dpi=250, bbox_inches="tight")

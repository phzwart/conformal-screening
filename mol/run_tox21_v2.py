"""Tox21, revision: fresh seeds, randomized tie-breaking, batch correction with class counts
planned from the calibration prevalence, and stored scores, labels and split indices for every
run (so all intervals can be recomputed without refitting). One fixed model: CatBoost on Morgan
fingerprints (r=2, 2048 bits). Usage: python3 -I run_tox21_v2.py N_RANDOM N_SCAFFOLD OUTDIR"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from conformal import record_planned, break_ties
HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(f"{HERE}/tox21_features.npz", allow_pickle=True)
X, Y, tasks, scaf = d["X"], d["Y"], list(d["tasks"]), d["scaffold"]
NR, NS, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; os.makedirs(f"{OUT}/scores", exist_ok=True)
SEED0 = 100000

def random_split(n, rng):
    p = rng.permutation(n); a, b = int(0.5 * n), int(0.75 * n); return p[:a], p[a:b], p[b:]

def scaffold_split(sc, rng):
    groups = {}
    for i, s in enumerate(sc): groups.setdefault(s if s else f"acyclic_{i}", []).append(i)
    keys = list(groups); rng.shuffle(keys); n = len(sc); target = np.array([0.50, 0.25, 0.25]) * n; parts = [[], [], []]
    for k in keys:
        deficit = target - np.array([len(p) for p in parts]); parts[int(np.argmax(deficit))] += groups[k]
    return tuple(np.array(p, dtype=int) for p in parts)

done = set(); jl = f"{OUT}/tox21_runs_v2.jsonl"
if os.path.exists(jl):
    for l in open(jl): r = json.loads(l); done.add((r["task"], r["split"], r["rep"]))
with open(jl, "a") as fh:
    for j, task in enumerate(tasks):
        mask = ~np.isnan(Y[:, j]); idx_all = np.where(mask)[0]; Xj, yj, sj = X[mask], Y[mask, j].astype(int), scaf[mask]
        for kind, reps in (("random", NR), ("scaffold", NS)):
            for r in range(reps):
                if (task, kind, r) in done: continue
                seed = SEED0 + 1000 * j + r + (0 if kind == "random" else 500); rng = np.random.default_rng(seed)
                tr, ca, te = random_split(len(yj), rng) if kind == "random" else scaffold_split(sj, rng)
                t = time.time()
                clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0, thread_count=2, random_seed=seed).fit(Xj[tr], yj[tr])
                uc_raw, ut_raw = clf.predict_proba(Xj[ca])[:, 1], clf.predict_proba(Xj[te])[:, 1]
                ties = float(1 - len(np.unique(np.concatenate([uc_raw, ut_raw]))) / (len(ca) + len(te)))
                uc, ut = break_ties(rng, uc_raw, ut_raw)
                out = {"task": task, "split": kind, "rep": r, "seed": seed, "n_train": len(tr), "tied_fraction": ties,
                       "auroc_test": float(roc_auc_score(yj[te], ut_raw)), "record": record_planned(uc, yj[ca], ut, yj[te])}
                np.savez_compressed(f"{OUT}/scores/{task}_{kind}_{r}.npz", train=idx_all[tr], cal=idx_all[ca], test=idx_all[te],
                                    u_cal=uc_raw.astype(np.float64), u_test=ut_raw.astype(np.float64), y_cal=yj[ca], y_test=yj[te])
                if kind == "scaffold":
                    out["pilot"] = {}
                    for npil in (200, 500, 1000):
                        if npil >= len(te) - 200: continue
                        p = rng.permutation(len(te)); pil, fut = p[:npil], p[npil:]
                        if np.sum(yj[te][pil] == 1) < 5: out["pilot"][str(npil)] = None; continue
                        out["pilot"][str(npil)] = record_planned(ut[pil], yj[te][pil], ut[fut], yj[te][fut])
                fh.write(json.dumps(out) + "\n"); fh.flush()
                print(task, kind, r, f"{time.time() - t:.1f}s", flush=True)

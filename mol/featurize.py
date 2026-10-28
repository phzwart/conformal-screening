"""Standardize molecules, de-duplicate and compute Morgan fingerprints.

Standardization: RDKit parse -> largest organic fragment (salt/solvent removal) ->
neutralize charges where possible -> canonical isomeric SMILES.
De-duplication: compounds with the same canonical SMILES are merged.
  Tox21: per endpoint, labels of duplicates are merged; conflicting labels -> missing.
  AqSolDB: log S of duplicates averaged if their range is <= 0.5 log units, else dropped.
Features: Morgan fingerprint, radius 2, 2048 bits (bit vector).
Bemis-Murcko scaffold (generic=False) is stored for scaffold splits."""
import gzip, json, sys, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Crippen
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
DATA, OUT = sys.argv[1], sys.argv[2]
chooser = rdMolStandardize.LargestFragmentChooser(); uncharger = rdMolStandardize.Uncharger()
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def standardize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        m = uncharger.uncharge(chooser.choose(m)); Chem.SanitizeMol(m)
    except Exception:
        return None
    if m.GetNumHeavyAtoms() < 2: return None
    return m

def featurize(mols):
    X = np.zeros((len(mols), 2048), dtype=np.uint8)
    for i, m in enumerate(mols): X[i] = gen.GetFingerprintAsNumPy(m)
    return X

def scaffold(m):
    try: return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception: return ""

stats = {}
# ---------------- Tox21 ----------------
df = pd.read_csv(f"{DATA}/tox21.csv.gz"); tasks = [c for c in df.columns if c not in ("mol_id", "smiles")]
stats["tox21_raw"] = len(df)
rows = {}
for _, r in df.iterrows():
    m = standardize(r["smiles"])
    if m is None: continue
    can = Chem.MolToSmiles(m)
    rows.setdefault(can, {"mol": m, "labels": []})["labels"].append([r[t] for t in tasks])
stats["tox21_parsed_unique"] = len(rows)
cans = list(rows); mols = [rows[c]["mol"] for c in cans]
Y = np.full((len(cans), len(tasks)), np.nan)
conflicts = 0
for i, c in enumerate(cans):
    L = np.array(rows[c]["labels"], dtype=float)
    for j in range(len(tasks)):
        v = L[:, j][~np.isnan(L[:, j])]
        if len(v) == 0: continue
        if np.all(v == v[0]): Y[i, j] = v[0]
        else: conflicts += 1
stats["tox21_label_conflicts_set_missing"] = conflicts
X = featurize(mols); scaf = np.array([scaffold(m) for m in mols])
np.savez_compressed(f"{OUT}/tox21_features.npz", X=X, Y=Y, tasks=np.array(tasks), smiles=np.array(cans), scaffold=scaf)
stats["tox21_per_task"] = {t: {"n": int(np.sum(~np.isnan(Y[:, j]))), "active": int(np.nansum(Y[:, j]))} for j, t in enumerate(tasks)}

# ---------------- AqSolDB ----------------
aq = pd.read_csv(f"{DATA}/aqsoldb_curated.csv"); stats["aqsoldb_raw"] = len(aq)
rows = {}
for _, r in aq.iterrows():
    m = standardize(r["SMILES"])
    if m is None: continue
    can = Chem.MolToSmiles(m)
    rows.setdefault(can, {"mol": m, "logS": []})["logS"].append(r["Solubility"])
keep = [(c, d) for c, d in rows.items() if np.ptp(d["logS"]) <= 0.5]
stats["aqsoldb_parsed_unique"] = len(rows); stats["aqsoldb_after_dedup"] = len(keep)
cans = [c for c, _ in keep]; mols = [d["mol"] for _, d in keep]
logS = np.array([np.mean(d["logS"]) for _, d in keep]); logP = np.array([Crippen.MolLogP(m) for m in mols])
X = featurize(mols)
np.savez_compressed(f"{OUT}/aqsoldb_features.npz", X=X, logS=logS, logP=logP, smiles=np.array(cans))
stats["aqsoldb_logP_gt_3.5"] = int(np.sum(logP > 3.5)); stats["aqsoldb_insoluble_logS_lt_-4"] = int(np.sum(logS < -4))
json.dump(stats, open(f"{OUT}/featurize_stats.json", "w"), indent=1); print(json.dumps(stats, indent=1))

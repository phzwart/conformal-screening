"""CA-DEL subset: DEL readouts and ChEMBL Ki for CAII, CAIX and CAXII.

Standardization matches mol/featurize.py. That module runs on import, so the
function is copied: RDKit parse, largest fragment, uncharge, sanitize, drop
molecules with fewer than 2 heavy atoms, then canonical isomeric SMILES and
an InChIKey of that same molecule.

del_enrich is the readout stored by the CA-DEL release, not a ratio invented here.
  CAIX:  ca9_Fn, the authors' normalized fold-change for the CAIX screen.
  CAII:  column P, the sum of the three purified-CAII replicates (Exp-P01/02/03).
         The release has no fold-change column. `label` is their binary hit column.
  CAXII: column A, the sum of the three endogenous-CAXII replicates (Exp-A01/02/03).
         The overexpression screen is kept as OA and Exp-OA01/02/03, not as del_enrich.
         The shared `label` column tracks CAII counts and is not written on CAXII.

Identical CAS-DEL rows that differ only by row_count are dropped before
aggregation so those copies are not summed. Remaining rows that share an
InChIKey have their counts summed. Float scores take the median. n_dup is the
number of rows that were aggregated.
"""
import hashlib, math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")
CHOOSER = rdMolStandardize.LargestFragmentChooser()
UNCHARGER = rdMolStandardize.Uncharger()
MAX_BYTES = 90 * 1024 * 1024

CAS_COUNTS_II = ["Exp-B01", "Exp-P01", "Exp-P02", "Exp-P03", "P"]
CAS_COUNTS_XII = ["Exp-B01", "Exp-A01", "Exp-A02", "Exp-A03", "A",
                  "Exp-OA01", "Exp-OA02", "Exp-OA03", "OA"]
CAS_IDS = ["Code_A", "Code_B", "Code_C"]

DOS_IDS = ["cpd_id", "scaffold", "BB1", "BB2"]
DOS_COUNTS = [
    "ap1_baseline",
    "hrp_beads_r1", "hrp_beads_r2", "hrp_beads_r3", "hrp_beads_r4",
    "hrp_exp_r1", "hrp_exp_r2",
    "ca9_beads_r1", "ca9_beads_r2",
    "ca9_exp_r1", "ca9_exp_r2", "ca9_exp_r3", "ca9_exp_r4",
    "cycle0", "cycle1", "cycle2",
]
DOS_FLOATS = [
    "hrp_B", "hrp_A", "hrp_Bp", "hrp_Ap", "hrp_Fn",
    "ca9_B", "ca9_A", "ca9_Bp", "ca9_Ap", "ca9_Fn",
    "cycle01", "cycle02", "cycle12",
]


def standardize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        m = UNCHARGER.uncharge(CHOOSER.choose(m))
        Chem.SanitizeMol(m)
    except Exception:
        return None
    if m.GetNumHeavyAtoms() < 2:
        return None
    return m


def parse_table(smiles):
    """Return canonical SMILES and InChIKey, with None where parsing fails."""
    cache = {}
    can = []
    inchikey = []
    failed = 0
    for raw in smiles.astype(str):
        if raw not in cache:
            mol = standardize(raw)
            cache[raw] = None if mol is None else (Chem.MolToSmiles(mol), MolToInchiKey(mol))
        rec = cache[raw]
        if rec is None:
            can.append(None)
            inchikey.append(None)
            failed += 1
        else:
            can.append(rec[0])
            inchikey.append(rec[1])
    return can, inchikey, failed


def join_ids(values):
    return ";".join(sorted(set(values.astype(str))))


def aggregate(df, count_cols, float_cols, id_cols, label_col=None):
    grouped = df.groupby("inchikey", sort=False)
    if label_col is not None and int((grouped[label_col].nunique() > 1).sum()):
        raise SystemExit(f"{label_col} disagrees inside an InChIKey group")
    out = pd.DataFrame({
        "smiles": grouped["smiles"].min(),
        "n_dup": grouped.size().astype("int64"),
    })
    for col in count_cols:
        out[col] = grouped[col].sum().astype("int64")
    for col in float_cols:
        out[col] = grouped[col].median().astype("float64")
    for col in id_cols:
        out[col] = grouped[col].agg(join_ids)
    if label_col is not None:
        out[label_col] = grouped[label_col].first().astype("int8")
    return out.reset_index()


def write_parquet(df, out_dir, stem):
    df = df.sort_values("inchikey").reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{stem}*.parquet"):
        old.unlink()
    path = out_dir / f"{stem}.parquet"
    df.to_parquet(path, compression="zstd", index=False)
    if path.stat().st_size < MAX_BYTES:
        return [path]
    path.unlink()
    n = len(df)
    n_shards = 2
    while n_shards <= max(n, 2):
        rows = math.ceil(n / n_shards)
        paths = []
        too_big = False
        for i in range(n_shards):
            part = df.iloc[i * rows:(i + 1) * rows]
            if part.empty:
                continue
            part_path = out_dir / f"{stem}_{i:03d}.parquet"
            part.to_parquet(part_path, compression="zstd", index=False)
            paths.append(part_path)
            if part_path.stat().st_size >= MAX_BYTES:
                too_big = True
        if paths and not too_big:
            return paths
        for part_path in paths:
            part_path.unlink()
        n_shards += 1
    raise SystemExit(f"could not split {stem} under {MAX_BYTES} bytes")


def quantiles(series):
    q = series.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return " ".join(f"{name}={value:.4g}" for name, value in q.items())


def count_summary(df, cols):
    lines = []
    for col in cols:
        values = df[col]
        lines.append(f"    {col}: min={int(values.min())} median={float(values.median()):.4g} max={int(values.max())}")
    return "\n".join(lines)


def require_columns(df, cols, path):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing {missing}")


def file_md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cas(csv_dir):
    primary = csv_dir / "P00918_CAH2" / "data_cah12and2.csv"
    other = csv_dir / "O43570_CAH12" / "data_cah12and2.csv"
    if file_md5(primary) != file_md5(other):
        raise SystemExit("P00918 and O43570 data_cah12and2.csv files differ")
    usecols = ["Product_smiles", "label", "row_count", *CAS_IDS, *CAS_COUNTS_II, *CAS_COUNTS_XII]
    # Exp-B01 is in both count lists.
    usecols = list(dict.fromkeys(usecols))
    raw = pd.read_csv(primary, usecols=usecols)
    measure = [col for col in raw.columns if col != "row_count"]
    before = len(raw)
    raw = raw.drop_duplicates(subset=measure)
    print(f"CAS-DEL rows {before} exact copies dropped {before - len(raw)}")
    raw["smiles"], raw["inchikey"], failed = parse_table(raw["Product_smiles"])
    print(f"CAS-DEL failed parses {failed}")
    kept = raw.dropna(subset=["inchikey"]).copy()
    print(f"CAS-DEL parsed rows {len(kept)}")
    return kept, failed


def build_cas_target(kept, count_cols, enrich_col, with_label):
    work = kept[["smiles", "inchikey", *CAS_IDS, *count_cols] + (["label"] if with_label else [])].copy()
    out = aggregate(work, count_cols, [], CAS_IDS, "label" if with_label else None)
    out["del_enrich"] = out[enrich_col].astype("float64")
    front = ["smiles", "inchikey", *CAS_IDS, *count_cols]
    if with_label:
        front.append("label")
    front += ["del_enrich", "n_dup"]
    return out[front]


def load_dos(csv_dir):
    path = csv_dir / "Q16790_CAH9" / "data_cah9.csv"
    usecols = ["smiles", *DOS_IDS, *DOS_COUNTS, *DOS_FLOATS]
    raw = pd.read_csv(path, usecols=usecols)
    raw["smiles_std"], raw["inchikey"], failed = parse_table(raw["smiles"])
    print(f"CAIX DEL rows {len(raw)} failed parses {failed}")
    kept = raw.dropna(subset=["inchikey"]).copy()
    kept["smiles"] = kept["smiles_std"]
    out = aggregate(kept, DOS_COUNTS, DOS_FLOATS, DOS_IDS)
    out["del_enrich"] = out["ca9_Fn"].astype("float64")
    front = ["smiles", "inchikey", *DOS_IDS, *DOS_COUNTS, *DOS_FLOATS, "del_enrich", "n_dup"]
    ligand = csv_dir / "Q16790_CAH9" / "data_cah9_ligand.csv"
    if ligand.exists():
        side = pd.read_csv(ligand, usecols=["cpd_id", "smiles"])
        main = pd.read_csv(path, usecols=["cpd_id", "smiles"])
        merged = main.merge(side, on="cpd_id", suffixes=("_main", "_ligand"))
        agree = float((merged["smiles_main"] == merged["smiles_ligand"]).mean()) if len(merged) else 0.0
        print(f"data_cah9_ligand.csv rows {len(side)} cpd_id matches {len(merged)} smiles agree {agree:.4f}")
    return out[front], failed


def censored_count(series):
    if pd.api.types.is_numeric_dtype(series):
        return 0
    text = series.astype(str)
    return int(text.str.contains(r"[<>≤≥]", regex=True, na=False).sum())


def build_chembl(csv_dir, target, filename, del_keys):
    path = csv_dir / "valid" / filename
    raw = pd.read_csv(path)
    require_columns(raw, ["smiles", "Ki (nM)", "Kd (nM)", "IC50 (nM)"], path)
    n_censored = censored_count(raw["Ki (nM)"]) + censored_count(raw["Kd (nM)"])
    ki = pd.to_numeric(raw["Ki (nM)"], errors="coerce")
    kd = pd.to_numeric(raw["Kd (nM)"], errors="coerce")
    ic50 = pd.to_numeric(raw["IC50 (nM)"], errors="coerce")
    smiles, inchikey, failed = parse_table(raw["smiles"])
    work = pd.DataFrame({
        "smiles": smiles,
        "inchikey": inchikey,
        "ki": ki,
        "kd": kd,
        "ic50": ic50,
    }).dropna(subset=["inchikey"])
    use_ki = work["ki"].notna()
    use_kd = ~use_ki & work["kd"].notna()
    n_ic50_only = int((~use_ki & ~use_kd & work["ic50"].notna()).sum())
    n_neither = int((~use_ki & ~use_kd & work["ic50"].isna()).sum())
    n_nonpos = int(((use_ki & (work["ki"] <= 0)) | (use_kd & (work["kd"] <= 0))).sum())
    keep = (use_ki & (work["ki"] > 0)) | (use_kd & (work["kd"] > 0))
    kept = work.loc[keep].copy()
    kept["from_kd"] = ~use_ki.loc[keep].to_numpy()
    kept["value_nM"] = np.where(kept["from_kd"], kept["kd"], kept["ki"])
    kept["pki"] = 9.0 - np.log10(kept["value_nM"].to_numpy(dtype=float))
    kept["saw_kd"] = kept["kd"].notna()
    ki_rows = kept.loc[~kept["from_kd"]]
    kd_rows = kept.loc[kept["from_kd"]]
    ki_keys = set(ki_rows["inchikey"])
    kd_only = kd_rows.loc[~kd_rows["inchikey"].isin(ki_keys)]

    def pack(part):
        grouped = part.groupby("inchikey", sort=False)
        frame = pd.DataFrame({
            "smiles": grouped["smiles"].min(),
            "pKi_median": grouped["pki"].median().astype("float64"),
            "pKi_min": grouped["pki"].min().astype("float64"),
            "pKi_max": grouped["pki"].max().astype("float64"),
            "n_meas": grouped.size().astype("int64"),
        })
        return frame.reset_index()

    ki_mol = pack(ki_rows) if len(ki_rows) else pd.DataFrame()
    if len(ki_mol):
        saw = ki_rows.groupby("inchikey")["saw_kd"].any()
        ki_mol["has_kd"] = ki_mol["inchikey"].map(saw).astype(bool)
        ki_mol["from_kd"] = False
    kd_mol = pack(kd_only) if len(kd_only) else pd.DataFrame()
    if len(kd_mol):
        kd_mol["has_kd"] = True
        kd_mol["from_kd"] = True
    out = pd.concat([ki_mol, kd_mol], ignore_index=True)
    out["chembl_ids"] = ""
    out["inconsistent"] = (out["pKi_max"] - out["pKi_min"]) > 1.0
    out["in_del"] = out["inchikey"].isin(del_keys)
    out["has_kd"] = out["has_kd"].astype(bool)
    out["from_kd"] = out["from_kd"].astype(bool)
    out["inconsistent"] = out["inconsistent"].astype(bool)
    out["in_del"] = out["in_del"].astype(bool)
    cols = ["smiles", "inchikey", "pKi_median", "pKi_min", "pKi_max", "n_meas",
            "chembl_ids", "inconsistent", "in_del", "has_kd", "from_kd"]
    stats = {
        "rows": len(raw),
        "failed": failed,
        "censored": n_censored,
        "ic50_only": n_ic50_only,
        "neither": n_neither,
        "nonpositive": n_nonpos,
        "measurements": int(out["n_meas"].sum()) if len(out) else 0,
    }
    return out[cols], stats


def connectivity(keys):
    return {key.split("-")[0] for key in keys}


def print_del(name, df, count_cols, label_col):
    print(f"DEL {name}: n_molecules {len(df)}")
    print(count_summary(df, count_cols))
    print(f"    del_enrich quantiles {quantiles(df['del_enrich'])}")
    if label_col and label_col in df.columns:
        print(f"    fraction labelled positive {float((df[label_col] == 1).mean()):.6f} n_positive {int((df[label_col] == 1).sum())}")
    else:
        print("    fraction labelled positive: no binary label for this target")
    print(f"    n_dup>1 {(df['n_dup'] > 1).sum()}")


def print_chembl(name, df, stats, del_n):
    ki = df.loc[~df["from_kd"], "pKi_median"]
    def frac(series, limit_pki):
        if len(series) == 0:
            return float("nan")
        return float((series >= limit_pki).mean())
    print(f"ChEMBL {name}: n_molecules {len(df)} n_measurements {stats['measurements']}")
    print(f"    pKi quantiles {quantiles(df['pKi_median'])}")
    print(f"    fraction Ki<=100 nM {frac(ki, 7):.4f} Ki<=1 uM {frac(ki, 6):.4f} (Ki rows only, n={len(ki)})")
    print(f"    n_inconsistent {int(df['inconsistent'].sum())} n_has_kd {int(df['has_kd'].sum())} n_from_kd {int(df['from_kd'].sum())}")
    print(f"    n_dropped_censored_relation {stats['censored']} n_dropped_nonpositive {stats['nonpositive']} "
          f"n_dropped_ic50_only {stats['ic50_only']} n_dropped_no_ki_or_kd {stats['neither']} failed_smiles {stats['failed']}")
    print(f"    DEL/ChEMBL InChIKey overlap {int(df['in_del'].sum())} of DEL {del_n}")
    print(f"    connectivity-block overlap {stats['connectivity_overlap']}")


def null_check(df, cols, name):
    for col in cols:
        if df[col].isna().any():
            raise SystemExit(f"{name} has nulls in {col}")


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: cadel_fetch.py CSV_DIR OUT_DIR")
    t0 = time.perf_counter()
    csv_dir = Path(argv[1])
    out_dir = Path(argv[2])
    cas, _cas_failed = load_cas(csv_dir)
    caii = build_cas_target(cas, CAS_COUNTS_II, "P", True)
    caxii = build_cas_target(cas, CAS_COUNTS_XII, "A", False)
    caix, _dos_failed = load_dos(csv_dir)
    del_tables = {"CAII": caii, "CAIX": caix, "CAXII": caxii}
    del_counts = {
        "CAII": CAS_COUNTS_II,
        "CAIX": DOS_COUNTS,
        "CAXII": CAS_COUNTS_XII,
    }
    del_label = {"CAII": "label", "CAIX": None, "CAXII": None}
    chembl_files = {
        "CAII": "df_eval_data_P00918.csv",
        "CAIX": "df_eval_data_Q16790.csv",
        "CAXII": "df_eval_data_O43570.csv",
    }
    written = []
    for name in ("CAII", "CAIX", "CAXII"):
        table = del_tables[name]
        null_check(table, ["smiles", "inchikey", "del_enrich"], f"DEL {name}")
        print_del(name, table, del_counts[name], del_label[name])
        written += write_parquet(table, out_dir, f"cadel_del_{name}")
        keys = set(table["inchikey"])
        chem, stats = build_chembl(csv_dir, name, chembl_files[name], keys)
        stats["connectivity_overlap"] = len(connectivity(keys) & connectivity(chem["inchikey"]))
        null_check(chem, ["smiles", "inchikey", "pKi_median"], f"ChEMBL {name}")
        print_chembl(name, chem, stats, len(table))
        written += write_parquet(chem, out_dir, f"cadel_chembl_{name}")
    elapsed = time.perf_counter() - t0
    print(f"runtime_s {elapsed:.1f}")
    for path in written:
        print(f"wrote {path} {path.stat().st_size}")


if __name__ == "__main__":
    main(sys.argv)

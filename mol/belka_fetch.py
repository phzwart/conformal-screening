"""Uniform subsample of the Leash Bio BELKA screen.

Reads either the Polaris dataset leash-bio/belka-v1 (wide zarr, one row per
molecule) or the Kaggle leash-BELKA train.parquet (one row per molecule and
protein). Molecules are ranked by the MD5 digest of

    UTF-8(molecule_smiles) || 0x1F || decimal(seed)

and the N smallest digests are kept. Binders are not enriched, and train/test
membership is not used to choose molecules.

Polaris stores an unlabeled kinase0 block (no binary labels). Those rows cannot
be written as int8 labels, so the Polaris sample is the uniform draw among
molecules that have a 0/1 label for every protein. The excluded count is printed.
The Kaggle training file is expected to be fully labeled; incomplete molecules
abort the run instead of being dropped.
"""
import argparse, hashlib, heapq, math, sys, time
from pathlib import Path

import duckdb

MAX_BYTES = 90 * 1024 * 1024
POLARIS_ZARR = "https://data.polarishub.io/dataset/leash-bio/belka-v1/data.zarr"
POLARIS_ID = "leash-bio/belka-v1"
POLARIS_CREATED = "2024-10-30T22:37:44.090Z"
POLARIS_VERSION = "artifactVersion=1 polarisVersion=0.8.7.dev4+gb6092bc.d20241030"
POLARIS_MANIFEST_MD5 = "368e2b0da67d49c46e8823cec84f866a"
STAGING = Path("/tmp/belka/staging")


class Candidate:
    __slots__ = ("digest", "smiles", "bb1", "bb2", "bb3", "b4", "hsa", "seh", "split_tag")

    def __init__(self, digest, smiles, bb1, bb2, bb3, b4, hsa, seh, split_tag):
        self.digest = digest
        self.smiles = smiles
        self.bb1 = bb1
        self.bb2 = bb2
        self.bb3 = bb3
        self.b4 = b4
        self.hsa = hsa
        self.seh = seh
        self.split_tag = split_tag

    def __lt__(self, other):
        # Max-heap of digest size: the largest digest sits at position 0.
        return self.digest > other.digest


def digest_of(smiles, seed_bytes):
    return hashlib.md5(smiles.encode("utf-8") + b"\x1f" + seed_bytes).digest()


def label_ok(value):
    if value is None:
        return False
    try:
        if value != value:
            return False
    except Exception:
        return False
    return value == 0 or value == 1


def as_text(value):
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text if text != "" else None


def copy_parquet(con, query, path):
    dest = str(path).replace("'", "''")
    con.execute(f"COPY ({query}) TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def export_table(con, table, columns, out_dir, stem, order_by, max_bytes=MAX_BYTES):
    """Write a ZSTD parquet, splitting into stem_000, stem_001, ... if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{stem}*.parquet"):
        old.unlink()
    STAGING.mkdir(parents=True, exist_ok=True)
    col_sql = ", ".join(columns)
    full = STAGING / f"{stem}_full.parquet"
    if full.exists():
        full.unlink()
    copy_parquet(con, f"SELECT {col_sql} FROM {table} ORDER BY {order_by}", full)
    size = full.stat().st_size
    if size <= max_bytes:
        dest = out_dir / f"{stem}.parquet"
        full.replace(dest)
        return [dest]

    n_rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    ranked = f"{table}_ranked"
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE {ranked} AS "
        f"SELECT {col_sql}, row_number() OVER (ORDER BY {order_by}) - 1 AS rn FROM {table}"
    )
    target = max(1, int(max_bytes * 0.85))
    n_shards = max(2, math.ceil(size / target))
    while True:
        paths = []
        rows_per = math.ceil(n_rows / n_shards)
        for i in range(n_shards):
            start, end = i * rows_per, min(n_rows, (i + 1) * rows_per)
            if start >= n_rows:
                break
            path = out_dir / f"{stem}_{i:03d}.parquet"
            copy_parquet(
                con,
                f"SELECT {col_sql} FROM {ranked} WHERE rn >= {start} AND rn < {end} ORDER BY {order_by}",
                path,
            )
            paths.append(path)
        if paths and all(p.stat().st_size < max_bytes for p in paths):
            full.unlink(missing_ok=True)
            return paths
        for path in paths:
            path.unlink()
        n_shards += 1
        if n_shards > max(n_rows, 1) or n_shards > 500:
            raise SystemExit(f"could not split {stem} under {max_bytes} bytes")


def connect():
    Path("/tmp/belka/duckdb_tmp").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET temp_directory = '/tmp/belka/duckdb_tmp'")
    con.execute("SET memory_limit = '80GB'")
    return con


def build_outputs(con, out_dir, with_split):
    con.execute("""
        CREATE TEMP TABLE bb AS
        SELECT row_number() OVER (ORDER BY position, smiles)::INTEGER AS bb_id,
               position::INTEGER AS position,
               smiles
        FROM (
            SELECT 1 AS position, bb1 AS smiles FROM mol
            UNION
            SELECT 2, bb2 FROM mol
            UNION
            SELECT 3, bb3 FROM mol
        )
    """)
    split_sql = ", m.split_tag" if with_split else ""
    con.execute(f"""
        CREATE TEMP TABLE mol_out AS
        SELECT m.molecule_smiles,
               b1.bb_id::INTEGER AS bb1_id,
               b2.bb_id::INTEGER AS bb2_id,
               b3.bb_id::INTEGER AS bb3_id,
               m.binds_BRD4::TINYINT AS binds_BRD4,
               m.binds_HSA::TINYINT AS binds_HSA,
               m.binds_sEH::TINYINT AS binds_sEH
               {split_sql}
        FROM mol m
        JOIN bb b1 ON b1.position = 1 AND b1.smiles = m.bb1
        JOIN bb b2 ON b2.position = 2 AND b2.smiles = m.bb2
        JOIN bb b3 ON b3.position = 3 AND b3.smiles = m.bb3
    """)
    bad = con.execute("""
        SELECT count(*) FROM mol_out
        WHERE molecule_smiles IS NULL
           OR bb1_id IS NULL OR bb2_id IS NULL OR bb3_id IS NULL
           OR binds_BRD4 IS NULL OR binds_HSA IS NULL OR binds_sEH IS NULL
           OR binds_BRD4 NOT IN (0, 1)
           OR binds_HSA NOT IN (0, 1)
           OR binds_sEH NOT IN (0, 1)
    """).fetchone()[0]
    if bad:
        raise SystemExit(f"{bad} output rows have a null or non-binary label")
    n_in = con.execute("SELECT count(*) FROM mol").fetchone()[0]
    n_out = con.execute("SELECT count(*) FROM mol_out").fetchone()[0]
    if n_out != n_in:
        raise SystemExit(f"building-block join dropped rows: {n_in} -> {n_out}")

    mol_cols = [
        "molecule_smiles", "bb1_id", "bb2_id", "bb3_id",
        "binds_BRD4", "binds_HSA", "binds_sEH",
    ]
    if with_split:
        mol_cols.append("split_tag")
    mol_paths = export_table(con, "mol_out", mol_cols, out_dir, "belka_sub", "molecule_smiles")
    bb_paths = export_table(con, "bb", ["bb_id", "position", "smiles"], out_dir, "belka_bb", "bb_id")
    return mol_paths + bb_paths


def print_summary(con, paths, elapsed):
    n, n1, n2, n3, k4, kh, ks, any_hit = con.execute("""
        SELECT count(*)::BIGINT,
               count(DISTINCT bb1_id)::BIGINT,
               count(DISTINCT bb2_id)::BIGINT,
               count(DISTINCT bb3_id)::BIGINT,
               sum(binds_BRD4)::BIGINT,
               sum(binds_HSA)::BIGINT,
               sum(binds_sEH)::BIGINT,
               count(*) FILTER (
                   WHERE binds_BRD4 = 1 OR binds_HSA = 1 OR binds_sEH = 1
               )::BIGINT
        FROM mol_out
    """).fetchone()
    print(f"molecules {n}")
    print(f"bb1 {n1}")
    print(f"bb2 {n2}")
    print(f"bb3 {n3}")
    for name, k in (("BRD4", k4), ("HSA", kh), ("sEH", ks)):
        print(f"binds_{name} {k}")
        print(f"hit_rate_{name} {k / n:.8f}")
    print(f"any_hit {any_hit}")
    print(f"fraction_any_hit {any_hit / n:.8f}")
    if con.execute(
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_name = 'mol_out' AND column_name = 'split_tag'"
    ).fetchone()[0]:
        for tag, c in con.execute(
            "SELECT split_tag, count(*) FROM mol_out GROUP BY 1 ORDER BY 1"
        ).fetchall():
            print(f"split_tag {tag} {c}")
    for path in paths:
        print(f"file {path.name} {path.stat().st_size}")
    print(f"seconds {elapsed:.1f}")


def load_polaris(con, input_path, n, seed):
    import zarr

    local = input_path / "data.zarr"
    if (local / ".zgroup").exists():
        root = zarr.open(str(local), mode="r")
        where = str(local)
    elif (input_path / ".zgroup").exists():
        root = zarr.open(str(input_path), mode="r")
        where = str(input_path)
    else:
        root = zarr.open(POLARIS_ZARR, mode="r")
        where = POLARIS_ZARR
    print(f"source polaris {POLARIS_ID}")
    print(f"polaris_created {POLARIS_CREATED}")
    print(f"polaris_version {POLARIS_VERSION}")
    print(f"polaris_manifest_md5 {POLARIS_MANIFEST_MD5}")
    print(f"polaris_location {where}")

    smiles_arr = root["molecule_smiles"]
    n_rows = int(smiles_arr.shape[0])
    chunk = int(smiles_arr.chunks[0])
    seed_bytes = str(seed).encode("ascii")
    heap = []
    members = {}
    n_dup = 0
    n_bad_label = 0
    n_bad_structure = 0
    t0 = time.perf_counter()
    n_chunks = math.ceil(n_rows / chunk)
    for chunk_i, start in enumerate(range(0, n_rows, chunk)):
        end = min(n_rows, start + chunk)
        smi = smiles_arr[start:end]
        bb1 = root["buildingblock1_smiles"][start:end]
        bb2 = root["buildingblock2_smiles"][start:end]
        bb3 = root["buildingblock3_smiles"][start:end]
        b4 = root["binds_BRD4"][start:end]
        hsa = root["binds_HSA"][start:end]
        seh = root["binds_sEH"][start:end]
        grp = root["split_group"][start:end]
        for s, a, b, c, p, q, r, g in zip(smi, bb1, bb2, bb3, b4, hsa, seh, grp):
            smiles, a, b, c = as_text(s), as_text(a), as_text(b), as_text(c)
            tag = as_text(g)
            if smiles is None or a is None or b is None or c is None:
                n_bad_structure += 1
                continue
            if not (label_ok(p) and label_ok(q) and label_ok(r)):
                n_bad_label += 1
                continue
            if smiles in members:
                n_dup += 1
                continue
            cand = Candidate(
                digest_of(smiles, seed_bytes), smiles, a, b, c, int(p), int(q), int(r), tag
            )
            if len(heap) < n:
                heapq.heappush(heap, cand)
                members[smiles] = cand
                continue
            if cand.digest >= heap[0].digest:
                continue
            old = heapq.heapreplace(heap, cand)
            del members[old.smiles]
            members[smiles] = cand
        print(
            f"chunk {chunk_i + 1}/{n_chunks} rows {end} "
            f"kept {len(heap)} excluded_label {n_bad_label} "
            f"seconds {time.perf_counter() - t0:.0f}",
            flush=True,
        )
    print(f"rows_seen {n_rows}")
    print(f"excluded_label {n_bad_label}")
    print(f"excluded_structure {n_bad_structure}")
    print(f"duplicate_smiles_in_sample {n_dup}")
    if len(heap) != n:
        raise SystemExit(f"expected {n} labeled molecules, kept {len(heap)}")
    _register_candidates(con, heap, with_split=True)
    return True


def _register_candidates(con, heap, with_split):
    import pyarrow as pa

    smiles, bb1, bb2, bb3, b4, hsa, seh, tags = [], [], [], [], [], [], [], []
    for cand in heap:
        smiles.append(cand.smiles)
        bb1.append(cand.bb1)
        bb2.append(cand.bb2)
        bb3.append(cand.bb3)
        b4.append(cand.b4)
        hsa.append(cand.hsa)
        seh.append(cand.seh)
        tags.append(cand.split_tag)
    arrays = {
        "molecule_smiles": pa.array(smiles, pa.string()),
        "bb1": pa.array(bb1, pa.string()),
        "bb2": pa.array(bb2, pa.string()),
        "bb3": pa.array(bb3, pa.string()),
        "binds_BRD4": pa.array(b4, pa.int16()),
        "binds_HSA": pa.array(hsa, pa.int16()),
        "binds_sEH": pa.array(seh, pa.int16()),
    }
    if with_split:
        arrays["split_tag"] = pa.array(tags, pa.string())
    con.register("mol_arrow", pa.table(arrays))
    con.execute("CREATE TEMP TABLE mol AS SELECT * FROM mol_arrow")


def load_kaggle(con, path, n, seed):
    import hashlib as hl

    print(f"source kaggle leash-BELKA")
    print(f"kaggle_sha256 {sha256_file(path)}")
    names = [r[0] for r in con.execute(
        "SELECT DISTINCT protein_name FROM read_parquet(?) ORDER BY 1", [str(path)]
    ).fetchall()]
    print(f"protein_names {' '.join(names)}")
    if set(names) != {"BRD4", "HSA", "sEH"}:
        raise SystemExit(f"unexpected protein names: {names}")
    seed_s = str(seed)
    # Same bytes Python hashes: smiles, 0x1F, decimal seed.
    probe = con.execute("SELECT md5(concat('C', chr(31), ?))", [seed_s]).fetchone()[0]
    expect = hl.md5(b"C" + b"\x1f" + seed_s.encode("ascii")).hexdigest()
    if probe != expect:
        raise SystemExit("duckdb md5 does not match hashlib")
    con.execute("""
        CREATE TEMP TABLE grouped AS
        SELECT molecule_smiles,
               any_value(buildingblock1_smiles) AS bb1,
               any_value(buildingblock2_smiles) AS bb2,
               any_value(buildingblock3_smiles) AS bb3,
               count(*)::BIGINT AS n_rows,
               count(DISTINCT protein_name)::INTEGER AS n_prot,
               count(DISTINCT buildingblock1_smiles)::INTEGER AS n_bb1,
               count(DISTINCT buildingblock2_smiles)::INTEGER AS n_bb2,
               count(DISTINCT buildingblock3_smiles)::INTEGER AS n_bb3,
               count(*) FILTER (WHERE binds IS NULL)::INTEGER AS n_null_bind,
               count(*) FILTER (WHERE protein_name NOT IN ('BRD4', 'HSA', 'sEH'))::INTEGER AS n_bad_prot,
               min(CASE WHEN protein_name = 'BRD4' THEN binds END)::INTEGER AS min_brd4,
               max(CASE WHEN protein_name = 'BRD4' THEN binds END)::INTEGER AS max_brd4,
               min(CASE WHEN protein_name = 'HSA' THEN binds END)::INTEGER AS min_hsa,
               max(CASE WHEN protein_name = 'HSA' THEN binds END)::INTEGER AS max_hsa,
               min(CASE WHEN protein_name = 'sEH' THEN binds END)::INTEGER AS min_seh,
               max(CASE WHEN protein_name = 'sEH' THEN binds END)::INTEGER AS max_seh
        FROM read_parquet(?)
        GROUP BY molecule_smiles
    """, [str(path)])
    n_bad = con.execute("""
        SELECT count(*) FROM grouped
        WHERE molecule_smiles IS NULL OR bb1 IS NULL OR bb2 IS NULL OR bb3 IS NULL
           OR n_rows <> 3 OR n_prot <> 3
           OR n_bb1 <> 1 OR n_bb2 <> 1 OR n_bb3 <> 1
           OR n_null_bind <> 0 OR n_bad_prot <> 0
           OR min_brd4 IS NULL OR min_hsa IS NULL OR min_seh IS NULL
           OR min_brd4 <> max_brd4 OR min_hsa <> max_hsa OR min_seh <> max_seh
           OR min_brd4 NOT IN (0, 1) OR min_hsa NOT IN (0, 1) OR min_seh NOT IN (0, 1)
    """).fetchone()[0]
    print(f"excluded_label {n_bad}")
    if n_bad:
        raise SystemExit(
            f"{n_bad} Kaggle molecules lack one complete 0/1 label per protein; "
            "not dropping them, because that would change the uniform sample"
        )
    con.execute("""
        CREATE TEMP TABLE mol AS
        SELECT molecule_smiles, bb1, bb2, bb3,
               min_brd4::SMALLINT AS binds_BRD4,
               min_hsa::SMALLINT AS binds_HSA,
               min_seh::SMALLINT AS binds_sEH
        FROM grouped
        ORDER BY md5(concat(molecule_smiles, chr(31), ?::VARCHAR))
        LIMIT ?
    """, [seed_s, n])
    con.execute("DROP TABLE grouped")
    got = con.execute("SELECT count(*) FROM mol").fetchone()[0]
    if got != n:
        raise SystemExit(f"expected {n} molecules, kept {got}")
    return False


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args(argv):
    p = argparse.ArgumentParser(description="Uniform BELKA molecule subsample")
    p.add_argument("--source", choices=("polaris", "kaggle"), required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=20261028)
    args = p.parse_args(argv)
    if args.n <= 0:
        p.error("--n must be positive")
    return args


def main(argv):
    args = parse_args(argv)
    t0 = time.perf_counter()
    print(f"n {args.n}")
    print(f"seed {args.seed}")
    print("hash md5(molecule_smiles || 0x1F || decimal seed), lowest N")
    con = connect()
    if args.source == "polaris":
        with_split = load_polaris(con, args.input, args.n, args.seed)
    else:
        if not args.input.is_file():
            raise SystemExit(f"missing Kaggle parquet: {args.input}")
        with_split = load_kaggle(con, args.input, args.n, args.seed)
    paths = build_outputs(con, args.out, with_split)
    print_summary(con, paths, time.perf_counter() - t0)


if __name__ == "__main__":
    main(sys.argv[1:])

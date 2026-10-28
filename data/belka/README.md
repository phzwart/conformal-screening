# BELKA subsample

Source: Polaris dataset `leash-bio/belka-v1` (Leash Biosciences), created 2024-10-30T22:37:44.090Z. Artifact version 1, Polaris library version `0.8.7.dev4+gb6092bc.d20241030`. The table is the zarr at `https://data.polarishub.io/dataset/leash-bio/belka-v1/data.zarr` (99,293,632 rows, one per molecule). Zarr manifest md5 `368e2b0da67d49c46e8823cec84f866a`. Licence on the Polaris record: CC BY-NC-SA 4.0 (non-commercial, share-alike).

The raw sequencing counts, replicates and controls are not columns of this table. They live in a separate bucket, `https://storage.polarishub.io/belka-v1-raw/` (about 600 GB), which was not downloaded. Counts per billion were therefore not aggregated, and the subsample has no `cpb_*` or `n_rep_*` columns. The source also has `molecule_score_BRD4`, `molecule_score_HSA` and `molecule_score_sEH`; those proprietary scores were not copied.

## Sample

1,000,000 distinct `molecule_smiles`, seed `20261028`. Rank is the MD5 digest of the UTF-8 bytes of `molecule_smiles`, the byte `0x1F`, and the decimal seed. The lowest 1,000,000 digests are kept. Binders are not enriched, and `split` / `split_group` are not used to choose molecules.

486,260 rows have no binary label (the unlabeled kinase0 block; `split` is empty). They cannot be stored as int8 labels, so the ranking is over the remaining 98,807,372 molecules that have a 0/1 label for BRD4, HSA and sEH. No other rows were dropped (`excluded_structure 0`, `duplicate_smiles_in_sample 0`).

Command that wrote these files (26 min 18 s):

```
python mol/belka_fetch.py --source polaris --input /tmp/belka --out data/belka --n 1000000
```

The zarr was read remotely. Nothing under `/tmp/belka` is part of this repository.

## Columns

`belka_sub.parquet` (12,455,334 bytes), one row per molecule:

| Column | Definition |
|---|---|
| `molecule_smiles` | Full molecule, including the triazine core when present. `[Dy]` stands in for the DNA linker. |
| `bb1_id`, `bb2_id`, `bb3_id` | Integer ids into `belka_bb.parquet`. |
| `binds_BRD4`, `binds_HSA`, `binds_sEH` | int8, 0 or 1, no nulls. Polaris stores these as float16; values were exactly 0 or 1. |
| `split_tag` | Polaris `split_group`: `train`, `share`, `non_share`, or `kin0`. |

`belka_bb.parquet` (14,320 bytes): `bb_id`, `position` (1, 2 or 3), `smiles`. Ids are `row_number()` ordered by `(position, smiles)`, so each position is a contiguous id block.

## Summary

| Quantity | Value |
|---|---|
| Molecules | 1,000,000 |
| Distinct bb1 / bb2 / bb3 | 306 / 754 / 935 |
| Hits BRD4 | 4,663 (0.4663%) |
| Hits HSA | 4,172 (0.4172%) |
| Hits sEH | 7,416 (0.7416%) |
| Hit on at least one protein | 15,434 (1.5434%) |
| `split_tag` train / share / non_share / kin0 | 996,039 / 3,750 / 209 / 2 |

Every molecule has one non-null 0/1 label per protein. Checksums of the files in this directory are in `SHA256SUMS`.

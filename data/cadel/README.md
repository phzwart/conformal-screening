# CA-DEL subset

DEL selection counts for three carbonic anhydrase isoforms, and ChEMBL Ki values for the same targets. One parquet file per target and per source. No molecule in this release has both a DEL readout and a Ki, so there is no joined table.

## Source

Zenodo record 17656024, *CA_DEL_database*, published 2025-11-20, DOI [10.5281/zenodo.17656024](https://doi.org/10.5281/zenodo.17656024). The record is CC BY 4.0. A later upload of the same concept (20322413, 2026-05-20) has the same archive MD5 checksums and adds a field-description note. The bytes used here are from 17656024.

He, Cao, Tan, Gao, Yao, Gu and Heng, *CA-DEL: An Open Multi-Target, Multi-Modal Benchmark for Learning from DNA-Encoded Library Screens*, arXiv:2605.07439.

ChEMBL-derived Ki and Kd values are under the ChEMBL licence, CC BY-SA 3.0.

Paper Table 1 counts rows, not unique molecules: DEL 127,500 / 108,528 / 127,500 and ChEMBL Ki 6,396 / 3,323 / 2,689 for CAII / CAIX / CAXII.

## Files used

| Archive | MD5 | Member used |
|---|---|---|
| `P00918_CAH2.7z` | `fdf913f1fa0336216c090c276fb68083` | `P00918_CAH2/data_cah12and2.csv` |
| `O43570_CAH12.7z` | `8e63be0cef8a9befe0dea82570c1ecfc` | `O43570_CAH12/data_cah12and2.csv` |
| `Q16790_CAH9.7z` | `db73c7cd1e424dd0d237e35d30e33d86` | `Q16790_CAH9/data_cah9.csv`, `Q16790_CAH9/data_cah9_ligand.csv` |
| `valid.zip` | `698b83be30a52a669e4688a6d9cf2b78` | `valid/df_eval_data_{P00918,Q16790,O43570}.csv` |
| `readme.pdf` | `a1176b8f7f33ecb765dc5562d5352405` | path list only |

The two `data_cah12and2.csv` files are identical (`md5 3205be8f5fadf64cd8c822c1377167c2`). That one table holds both CAII and CAXII. The target is the column group, not a row field:

- CAII (purified protein): `Exp-P01`, `Exp-P02`, `Exp-P03`, and `P` = their sum. Blank beads: `Exp-B01`.
- CAXII endogenous (A549): `Exp-A01`, `Exp-A02`, `Exp-A03`, and `A` = their sum.
- CAXII overexpressed (hypoxic A549): `Exp-OA01`, `Exp-OA02`, `Exp-OA03`, and `OA` = their sum.

`data_cah9_ligand.csv` repeats `cpd_id` and `smiles` from `data_cah9.csv` (SMILES agree on every matched id). It is not a second library and is not written out.

Skipped: every docking `*.tar.gz`, every CNN feature `*.pt`, `ecfp6`, the TPM-normalized count copies, building-block SMILES, `ase_system`, and the CAIX `affinity` float. That float does not track pKi.

The ChEMBL files have `smiles`, `Ki (nM)`, `Kd (nM)` and `IC50 (nM)`. They do not contain ChEMBL compound ids, assay ids, units other than nM, or a relation column. `chembl_ids` is therefore an empty string. Values are plain floats, so no row was dropped for a censored relation.

## Columns

Standardization is the same as `mol/featurize.py`: RDKit parse, largest fragment, uncharge, sanitize, drop molecules with fewer than two heavy atoms, canonical isomeric SMILES, InChIKey of that molecule. Rows that share an InChIKey are one output row. `n_dup` is how many source rows were merged. Counts are summed. Float scores take the median. Building-block ids that disagree are joined with `;`, sorted.

CAS-DEL rows that differ only by `row_count` are exact copies (5,000 of them). They are dropped before the sum so the counts are not doubled.

`del_enrich` is the readout the release actually stores. No fold-change was computed for the CAS-DEL table, because that table has no enrichment column.

| Target | `del_enrich` | Binary label |
|---|---|---|
| CAII | `P`, the sum of `Exp-P01/02/03` | `label`, the authors' hit column: 2,450 of 122,500 molecules are 1 (2,500 of 127,500 source rows). It tracks high `P` (median 468 against 1) but it is not a single threshold: positive `P` runs from 6 to 5,294 and some negatives reach 22. The release does not state the rule. |
| CAIX | `ca9_Fn`, the authors' normalized fold-change for the on-target screen. `hrp_Fn` is the HRP counter-screen and is kept separately. | none |
| CAXII | `A`, the sum of the endogenous replicates `Exp-A01/02/03`. The overexpression totals stay in `OA` and `Exp-OA*`. | none. The shared `label` column is the CAII hit list and is not copied onto CAXII. |

ChEMBL rows keep Ki. A row with no Ki and a Kd is kept: `from_kd` is true and `pKi_*` is then pKd, from the same formula. `has_kd` is true if any source row for that molecule had a Kd, including molecules that also have a Ki. IC50-only rows are dropped. Rows with neither Ki nor Kd are dropped. Non-positive values are dropped (none were). `pKi = 9 - log10(value_nM)`. Per molecule: median, min, max and `n_meas`. `inconsistent` is true when max − min is greater than 1. `in_del` is true when the InChIKey is in that target's DEL table.

## Summary

Runtime of `cadel_fetch.py`: 122 s.

| | CAII | CAIX | CAXII |
|---|---:|---:|---:|
| DEL molecules | 122,500 | 107,616 | 122,500 |
| DEL source rows | 127,500 | 108,528 | 127,500 |
| ChEMBL molecules | 3,965 | 3,103 | 2,176 |
| ChEMBL measurements kept | 6,370 | 3,106 | 2,690 |
| ChEMBL source rows | 6,395 | 3,324 | 2,690 |
| InChIKey overlap | 0 | 0 | 0 |

DEL molecule counts sit below the paper's row counts because of duplicate rows: 5,000 exact CAS-DEL copies, and 912 CAIX SMILES that share an InChIKey (counts summed, `ca9_Fn` median). ChEMBL CAII and CAXII are lower still because many rows repeat one SMILES. CAII also drops 25 SMILES that do not parse. CAIX drops 62 IC50-only rows and 156 rows with neither Ki nor Kd. The connectivity block of the InChIKey (stereochemistry ignored) also overlaps by 0.

`del_enrich` quantiles (0.05, 0.25, 0.50, 0.75, 0.95): CAII `0, 0, 2, 3, 7`; CAIX `0, 0.051, 0.240, 0.592, 1.562`; CAXII `0, 3, 7, 13, 28`. CAII positive fraction 0.020 (2,450 molecules).

pKi quantiles (same probabilities): CAII `5.06, 6.48, 7.30, 8.10, 9.00`; CAIX `5.09, 6.48, 7.19, 7.82, 8.54`; CAXII `5.13, 6.83, 7.50, 8.13, 8.70`.

Fraction of Ki molecules with Ki ≤ 100 nM / ≤ 1 µM: CAII 0.622 / 0.865; CAIX 0.598 / 0.848 (3,038 Ki molecules; 65 further molecules are Kd-only); CAXII 0.727 / 0.889. Inconsistent molecules: CAII 121, CAIX 0, CAXII 16. Censored relations dropped: 0.

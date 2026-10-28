# Scripts for "Predicting the consequences of conformal screening decisions: finite-sample audit intervals for batch outcomes"

Every script is seeded and writes its results to JSON, NPZ or a log, so every table and figure can be regenerated. Tables in the paper are written by scripts, not typed by hand.

The paper's methods are implemented here as plain scripts; they do not import the `ssbc` package.

## Environment

Python 3.13 with the versions in `requirements.txt` (`pip install -r requirements.txt`). Run every script with `python3 -I` from the directory it lives in. `code/` holds the simulations; `../mol/` holds the molecular pipeline.

## Simulations (`code/`)

| Output | Command | Runtime (1 core) |
|---|---|---|
| Identity check (ESI S3) | `python3 -I thatcher.py` | < 1 min |
| Threshold correction (Table 2, ESI Table S1) | `python3 -I threshold_sim.py 1000000 threshold_sim.json ../tables` | ~6 min |
| Test-set interval validation (Fig. 3, ESI) | `python3 -I validate_intervals.py 10000 validation.json` | ~10 min |
| Finite-library check (ESI) | `python3 -I finite_library.py 20 1000 finite_library.json` | ~12 min |
| Two-stage check (ESI) | `python3 -I cover2.py` | < 1 min |
| Calibration-only intervals (Fig. 4, ESI) | `python3 -I calib_only_rowsim.py 3000 calib_only.json` | ~60 min |
| Cost and scaffold dependence (Table 3, Fig. 5, ESI) | `python3 -I decision_sims.py decision_sims.json` | ~6 min |
| Selection among settings (ESI; the paper reports the unadjusted and Bonferroni columns) | `python3 -I selection_sim.py 4000 selection_sim.json` | ~2 min |
| Tied scores (ESI) | `python3 -I ties_sim.py 2000 ties_sim.json` | ~20 min |
| Figures 1, 3 | `python3 -I make_figures.py ../figures validation.json` | < 1 min |
| Figures 4, 5 | `python3 -I make_figures_rev.py ../figures .` | < 1 min |
| Fig. S1 | `python3 -I dkw_and_map.py` | < 1 min |
| Tables (LaTeX) | `python3 -I make_tables_rev.py . ../tables` and `python3 -I make_tables_v2.py ..` | < 1 min |

`decision_sims.py` Part B is kept for the record; the selection results in the paper come from `selection_sim.py`.

`intervals.py` holds the prediction limits used by the newer scripts: the lower limit is the smallest *j* with CDF(*j*) > γ/2, the upper limit the (1−γ/2) quantile. `validate_intervals.py` and `finite_library.py` use `scipy.stats.betabinom.ppf` for the lower limit, which admits *j* when CDF(*j*) = γ/2 exactly; the two conventions differ only at exact ties.

## Molecular experiments (`mol/`)

Raw data are in `../data/` with `SHA256SUMS`: the Tox21 MoleculeNet file, the curated AqSolDB set, CALiSol-23 (`data/calisol/`, DOI 10.11583/DTU.c.6929599) and CA-DEL (`data/cadel/`, DOI 10.5281/zenodo.17656024).

| Step | Command | Runtime |
|---|---|---|
| Standardize and featurize (Morgan r=2, 2048 bits) | `python3 -I featurize.py ../data .` | ~2 min |
| Tox21 runs: 12 endpoints × (50 random + 20 scaffold splits), CatBoost, tie-broken scores, pilots | `python3 -I run_tox21_v2.py 50 20 v2` | ~2.5 h |
| Tox21 analysis (Fig. 6, Table 5, ESI) | `python3 -I analyze_tox21_v2.py v2 v2` | ~15 min |
| Batch-law check on fixed models (ESI) | `python3 -I check_batch_law.py 5000` | ~25 min |
| AqSolDB runs: 10 splits | `python3 -I run_aqsol.py 10 aqsol_runs.npz` | ~5 min |
| AqSolDB analysis (Fig. 7, Table 6, ESI) | `python3 -I analyze_aqsol.py aqsol_runs.npz v2` | ~3 min |
| Real-data audit check (Table 4, ESI) | `python3 -I audit_check.py v2 aqsol_runs.npz v2/audit_check.json 20` | ~15 min |
| Scenario maps (Fig. 7, ESI Fig. S3) | `CALISOL_SAVE_SCORES=1 python3 -I calisol.py ../data/calisol/calisol23.csv v2/calisol_scores 8 1` then `python3 -I scenario_map.py v2/scenario_map.json ../figures` | ~1 min |
| Per-plate intervals and audit-size bounds (Sections 3.4, 3.6) | `python3 -I plate_bounds.py v2/calisol/calisol_runs.json v2/scenario_map.json v2/plate_bounds.json` | < 1 min |
| Figures 6, 7, ESI S2, S3 | `python3 -I make_figures_mol.py v2 ../figures` | < 1 min |
| CALiSol-23 electrolyte screen (Table 7, Fig. 8, ESI) | `python3 -I calisol.py ../data/calisol/calisol23.csv v2/calisol 8 50` then `python3 -I calisol_summary.py v2/calisol/calisol_runs.json v2/calisol/calisol_summary.json ../figures ../tables` | ~20 min |
| CA-DEL (DEL enrichment + ChEMBL Ki, CAII/CAIX/CAXII) | `python3 -I cadel_fetch.py /tmp/cadel/csv ../data/cadel` | ~2 min |

`v2/scores/` stores, for every Tox21 run, the training, calibration and test indices, the raw scores and the labels, so every interval can be recomputed without refitting.

## License

To be added by the author.

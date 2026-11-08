# Public-data provenance and frozen prediction protocol

All sources are public and require no login. Raw archives are obtained from their official sources and are not redistributed here. Download times are actual UTC times. A supplied archive is verified against the recorded SHA256 before preprocessing. These data support observational descriptions and covariate-based semi-synthetic experiments; the full observed sample is never a population ground truth.

## Raw sources

| File | Downloaded UTC | Bytes | SHA256 | Source |
|---|---|---:|---|---|
| adult.zip | 2026-10-22T10:42:54.960306+00:00 | 620237 | `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb` | [official archive](https://archive.ics.uci.edu/static/public/2/adult.zip) |
| acs_2018_1y_al_person.zip | 2026-10-22T10:43:02.455850+00:00 | 8463683 | `b3579bea88751f554e1e40d2f314f9aa08801a33b4886d962c1ec0b51b0f1d03` | [official archive](https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pal.zip) |
| diabetes_130.zip | 2026-10-22T10:44:35.800372+00:00 | 3170254 | `f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a` | [official archive](https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip) |

UCI documentation: [Adult](https://archive.ics.uci.edu/dataset/2/adult), DOI 10.24432/C5XW20; [Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008), DOI 10.24432/C5230J. UCI pages state CC BY 4.0. ACS source: [Census 2018 1-year PUMS](https://www.census.gov/programs-surveys/acs/microdata/access/2018.html). Task definitions follow [Folktables](https://github.com/socialfoundations/folktables), package version 0.0.12 (MIT code). Raw ACS archive also contains ACS2018_PUMS_README.pdf.

## Protocol fixed before audit

- Random seed 20261022. Approximate 60% train, 20% calibration, 20% audit. Adult/Diabetes split stratified on y; ACS split by household SERIALNO. Disjoint row IDs and unit IDs are checked in code. Adult provides no original person identifier; its synthetic IDs identify records only. Sex/race/age groups are never used to train the predictor or choose its threshold.
- Intersectional groups use sex × race × age (<40, 40–69, 70+). All source race categories are retained; missing values become Unknown. Full Cartesian group universes retain empty cells. Diabetes decade intervals align exactly with these boundaries. ACS code RAC1P=5 is retained with the label AIAN_combined. Source categories differ between datasets and are not asserted to be exchangeable.
- `n_g` counts audit positives y=1. A cell without positives has undefined FNR, preserved as missing in CSV; it is not assigned an FNR of zero.
- Both fixed CPU predictors exclude sex/race/age and their code equivalents. Other proxies remain: omission of protected features does not guarantee fairness. Adult fnlwgt is excluded. Preprocessing fits train only; no audit-dependent vocabulary, imputation, feature selection, or tuning.
- LR: median-impute/standardize numerical features, train-fit one-hot categorical features with min_frequency=10; C=1, lbfgs, max_iter=400. HGB: train-fit median and ordinal categorical preprocessing, min_frequency=10, max_categories=32, native categorical handling, learning rate .08, 100 iterations, 15 leaves, depth 5, minimum leaf 30, L2=1, early_stopping=False. Random states fixed.
- Each predictor chooses a single threshold from calibration positive scores, the floor(0.2*n) zero-based order statistic; scores tied at threshold predict positive. This gives empirical calibration FNR ≤0.2, but is no guarantee on unseen groups. All audit probabilities and decisions are frozen.
- Diabetes keeps the minimum encounter_id record for each patient, then uniformly subsamples 50,000 people without using labels. This is a deterministic deduplication rule, not verified temporal ordering. Features represent information available by discharge, including disposition and treatments. No prospective clinical usefulness is claimed.
- ACS uses official Folktables tasks. Employment includes all ages with no prefilter. Income uses the package's adult_filter. The analysis is unweighted, and makes no claim to represent US-population rates. ACS audit records may remain dependent within households: group-disjoint train splits address train leakage, not within-audit sampling dependence. Exact binomial calibration for these observational records is not claimed; independent semi-synthetic losses can be generated conditional on covariates.

## Prepared datasets

| Dataset | Raw | Eligible | Used | Train | Calibration | Audit | Audit positives | Groups | Positive audit groups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acs_employment | 47777 | 47777 | 47777 | 28650 | 9644 | 9483 | 3922 | 54 | 35 |
| acs_income | 47777 | 22268 | 22268 | 13383 | 4436 | 4449 | 1347 | 54 | 23 |
| adult | 48842 | 48842 | 48842 | 29305 | 9768 | 9769 | 2338 | 30 | 22 |
| diabetes | 101766 | 71518 | 50000 | 30000 | 10000 | 10000 | 876 | 54 | 29 |

## Locally regenerated cache contract

Individual-level NPZ caches and feature matrices described below are deliberately not distributed. Group-count CSVs and dataset/model metadata are included; regenerate individual caches locally only when reproducing classifier training.

`data/processed/{dataset}.npz`: `y` integer labels; `a` N×3 sensitive-category integer codes in sex/race/age order; `group_id` scalar cell ID per row; `groupcodes` K×3 code tuples; `groupkeys` K readable strings; `id` unique record ID; `unit_id` patient/person or household ID; `split` 0=train, 1=calibration, 2=audit; `feature_names`; `lr_probability`, `hgb_probability`, `lr_prediction`, `hgb_prediction`. No object arrays or pickle are required. `*_X.npz` is the train-fitted transformed CSR feature matrix in matching row order. `*_group_counts.csv` preserves every group, both row counts and positive counts, and model-specific false-negative counts for all splits. `*_metadata.json` records target, selection, predictor inputs, group levels, parameters, times and metrics. `audit_group_counts.csv` combines all datasets.

Reproduce: `python src/prepare_data.py` from the project directory (offline once raw archives exist). Download only: `python scripts/download_data.py`. Environment: {"numpy": "2.2.4", "pandas": "3.0.6", "scipy": "1.18.1", "scikit-learn": "1.9.1", "folktables": "0.0.12", "requests": "2.32.3"}.

# Intersectional fairness auditing

Code and retained numerical results for **Model mismatch and missed harms in
intersectional fairness auditing: an estimation–detection analysis**.

Authors: Cenfa Lu and Shuda Wu, School of Mathematics and Computing Science,
Guilin University of Electronic Technology, Guilin, 541004, China.
Correspondence: [Cenfa Lu](mailto:joffin@163.com).

The study separates average estimation loss, uncalibrated subgroup screens,
and error-controlled evidence for fairness violations. A lower average squared
error does not by itself establish better detection of a particular harmful
subgroup. See the manuscript, including the complete proofs and supplementary
tables in its appendices, in `paper/`.

## Repository contents

- `src/`: inference, estimators, experiment generation, numerical analysis and plotting.
- `results/`: original per-repetition synthetic and semi-synthetic count arrays,
  scenario definitions, complete summaries, and run metadata.
- `analysis/`: numerical summaries and the exploratory interval diagnostic.
- `figure_data/`: the plotted values for the five manuscript figures.
- `data/processed/`: aggregate group counts and dataset/model metadata only.
- `data/raw/download_manifest.json`: official raw-data locations and recorded hashes.
- `protocols/`: support-sensitivity and exploratory interval specifications.
- `vendor/SureMap/`: the MIT-licensed upstream estimator files used by the adapter.

All released simulation arrays are generated group-level counts, estimates,
decisions, intervals and weights. The real observational datasets are not
population ground truth. Individual-level records, IDs, feature matrices and
per-record prediction caches are not redistributed.

## Environment

The recorded numerical experiments used Python 3.13 on CPU. Install the pinned
packages in a separate environment:

```bash
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r requirements_figures.txt
```

`requirements_figures.txt` includes the numerical dependencies, PyMuPDF and
Windows timezone data. Figure scripts use Arial when available; a font fallback
can change the layout but does not change the numerical results.

Run the inference checks with:

```bash
python -m unittest discover -s src -p "test_*.py" -v
```

## Build the manuscript

The manuscript targets **The European Journal on Artificial Intelligence**
(formerly AI Communications). It uses the official, unmodified Sage LaTeX
class, Sage Harvard author-year references, and the template's `doublespace`
submission option. The abstract contains 140 words.

With TeX Live or MiKTeX providing `pdflatex` and `bibtex`, run:

```bash
python paper/build.py
```

This produces an attributed manuscript, an anonymous review copy, and a separate
title page in `build/`. The attributed PDF is also supplied as `paper/main.pdf`.
The source files preserve all five figures, nine tables, and theoretical proofs.
The official Sage class and bibliography style retain their own distribution
and usage terms; see `paper/readme.txt`.

## Rebuild numerical summaries and figures from retained results

This route uses the released simulation arrays and aggregate group metadata;
it does not require downloading individual-level records or refitting classifiers.

```bash
python src/analyze_experiments.py
python src/summarize_point_controls.py
python src/make_paper_assets.py
python src/plot_publication_figures.py
python src/plot_fig3_readable.py
```

Run the plotting scripts in the order shown: `plot_fig3_readable.py` provides
the final six-panel Figure 3. The drawing scripts export PDF, editable SVG and
PNG files. Historical styling and numerical-analysis helpers may create
intermediate figures before the final figure scripts replace them.

The delivered LaTeX tables include publication edits. Regeneration helpers
recover their numerical contents; preserve or reapply the journal formatting
when replacing a table in the manuscript. `src/make_tables.py` is a lower-level
helper and is not required as a standalone command.

The run metadata records the seeds, repetitions, parameters, software versions,
relative source/input hashes and execution times of the original experiments.
The scientific arrays and result CSVs are retained without numerical editing.
A few public helper paths and replication output options were adjusted during
packaging; original-run code hashes identify the executed historical sources,
not the subsequent helper-path edits. SureMap file hashes remain exact and are
protected from automatic line-ending conversion by `.gitattributes`.

## Regenerate synthetic experiments

Use fresh names to preserve the delivered numerical evidence:

```bash
python src/run_simulation.py --name reproduced_main --repetitions 2000
python src/run_extended.py --mode ablation --name reproduced_ablation --repetitions 1000
python src/run_extended.py --mode newer_baselines --name reproduced_baselines --repetitions 1000 --baseline-repetitions 120
python src/interval_diagnostics.py --output-name reproduced_intervals
```

The interval diagnostic reuses the first 120 retained repetitions in each of
seven baseline scenarios. It creates no new audit outcomes or classifier fits.
Its pointwise Wald and SAFT-style intervals are compared with a simultaneous
exact-binomial rectangle; these are different error targets and must not be
ranked as matched-FWER power comparisons.

Analysis helpers default to the delivered run names. To analyse a fresh named
replication instead, change the directory-name mapping in
`src/analyze_experiments.py`; keep the original results untouched.

## Reproduce public-data preprocessing and semi-synthetic generation

Download each original source from its official host and verify the hashes:

```bash
python scripts/download_data.py
python src/prepare_data.py
python src/run_extended.py --mode semisynthetic --name reproduced_semi --repetitions 1000 --baseline-repetitions 80
python src/run_support_sensitivity.py --name reproduced_support --prepare
python src/run_support_sensitivity.py --name reproduced_support --run
```

Classifier training creates individual-level NPZ caches locally. They are
excluded by `.gitignore`. Preprocessing follows the recorded splits,
patient/household grouping, positive-label denominators and threshold rules.
See `DATA_PROVENANCE.md` for the full data definitions, source attribution and
sampling boundaries. Hash verification deliberately fails if an upstream
archive changes; investigate rather than silently replacing the frozen input.

## Interpretation

- Fixed-prior/oracle screens, fitted point-estimator screens and calibrated
  flags are distinct outputs. A failure to flag is inconclusive.
- Finite-sample claims apply to the stated registered atomic groups and
  independent-error generating models. The observational ACS survey design
  and within-household dependence are not covered by those claims.
- SR is a local point-estimation adapter. SureMap uses the documented upstream
  numerical functions. Original SR rBLPR and Cherian–Candès bootstrap inference
  are not reproduced.
- The SAFT-style conditional-FNR interval diagnostic implements published
  ingredients with explicit adaptations; it is not the complete original
  SAFT study and carries no automatic familywise frequentist guarantee.

## Attribution and licensing

See `THIRD_PARTY_NOTICES.md` and `vendor/SureMap/LICENSE`. No additional project
license is granted for the authors' original code or manuscript.

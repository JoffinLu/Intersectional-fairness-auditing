# Manuscript figures

All five figures use a 174 mm white canvas, Nunito text, STIX mathematical
symbols, consistent method colors, pastel markers with darker edges, and
lightweight white panels. The exports include vector PDF, SVG with editable
text, and 600-dpi PNG. Font subsets are embedded in the PDFs.

Rebuild from the repository root with:

```bash
python src/plot_warm_ml_figures.py
```

This reads the retained source-data CSVs without modifying them or running new
experiments. `provenance.json` records source-data and export SHA256 hashes.
The optional `--qa-helper` argument accepts a directory containing an external
Matplotlib panel-alignment checker; it is not required to regenerate figures.

| Figure | Content | Source rows | Uncertainty |
|---|---|---:|---|
| 1 | Exact macro MSE and harmed-cell screening | 151 | Exact calculation; no Monte Carlo bars |
| 2 | Fitted point estimators: macro MSE and screening | 16 | MSE ±1.96 MCSE; retained pointwise Wilson screening bounds |
| 3 | Absolute confirmation rates and paired split-minus-full contrasts | 60 | Pointwise ±1.96 MCSE; paired for contrasts |
| 4 | Confirmation fraction and floor ablation | 24 | Signed means; paired MCSE remains in source data |
| 5 | Absolute rates and paired gains on real-support templates | 16 | Pointwise ±1.96 MCSE; paired for contrasts |

Figure 3 uses a logarithmic nominal-count axis and the original small method
offsets for visibility. Figure 4 has a common symmetric −40 to +40 percentage
point scale centered on white at zero. No values or error bars are clipped.
These are the manuscript's existing Monte Carlo uncertainty summaries, not
additional simultaneous confidence statements.

# Data distribution

The repository contains aggregate group counts and dataset/model metadata.
It does not contain individual-level public-data records, original patient or
household identifiers, transformed feature matrices, or per-record predictions.

Run `python scripts/download_data.py` from the repository root to obtain the
three official source archives and verify their recorded hashes. Then run
`python src/prepare_data.py` to regenerate local caches and frozen classifier
predictions. These local archives and NPZ caches are ignored by Git.

The arrays under `results/` are different: they contain independently generated
synthetic/semi-synthetic group counts and the resulting numerical decisions.
These are released to support direct inspection and reproduction of the
reported experiments without redistributing observational records.

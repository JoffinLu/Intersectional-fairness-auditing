# Third-party materials

## SureMap

`vendor/SureMap/data.py` and `vendor/SureMap/suremap.py` are unmodified files from
[Mikhail Khodak's SureMap repository](https://github.com/mkhodak/SureMap),
revision `ccd80ae74e20a044fd415a65f9946536c3b5fe4a`.
The original MIT license is retained in `vendor/SureMap/LICENSE`.
The local adapter in `src/suremap_adapter.py` imports the author's numerical
function bodies and verifies the exact vendored file hashes. It omits unused
heavyweight imports and an optional JIT decorator; this is an import adaptation,
not a reproduction of the entire upstream software environment.

SAFT and fairaudit source trees are not redistributed. The conditional-FNR
interval diagnostic is a local implementation of published ingredients, with
its adaptation and error target stated explicitly in the manuscript.

## Sage manuscript template

`paper/sagej.cls`, `paper/SageH.bst` and `paper/readme.txt` are distributed
unchanged from the official Sage LaTeX template. The template's own notice
restricts its use to preparation of articles for Sage journals. Those terms
remain applicable; the files are included for this journal submission.

## Nunito figure font

`vendor/Nunito/Nunito-Variable.ttf` is the Nunito variable font from
[Google Fonts](https://github.com/google/fonts/tree/main/ofl/nunito).
The SIL Open Font License 1.1 is retained in `vendor/Nunito/OFL.txt`.
Local static weight instances are generated into an ignored cache for plotting;
the authoritative variable font is distributed unchanged.

## Data sources

Public source datasets are downloaded from UCI and the US Census Bureau.
Their attribution, source URLs and recorded SHA256 values are documented in
`DATA_PROVENANCE.md` and `data/raw/download_manifest.json`.
No individual-level records, patient identifiers, household identifiers,
feature matrices or per-record prediction caches are included in this repository.

## Original project material

No additional license is granted for the original project code or manuscript.
Copyright remains with the respective rights holders. Third-party licenses
apply only to the materials they identify.

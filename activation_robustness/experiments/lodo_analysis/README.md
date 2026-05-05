# LODO analysis

Reproducible pipeline for the LODO numbers reported in App G of
`activation_robustness/paper/main.tex`.

## Source of truth

- Tarball: `./lodo_data/raw`
  (the older non-`_latest` tarball is corrupted; do not use it).
- README inside the tarball provides the canonical dataset class
  distribution and is treated as ground truth for validation.

## Pipeline

```
bash 00_extract.sh
python3 01_validate_integrity.py
python3 02_compute_clean_lodo.py
python3 03_compute_perturbation_lodo.py
```

Each step is idempotent; later steps depend on earlier outputs.

## Outputs

- `validation_report.json`         — per-fold integrity decisions
- `clean_lodo_results.json`        — feeds Tables G.1, G.2
- `perturbation_lodo_results.json` — feeds Tables G.3, G.4
- `./lodo_data/extracted/`         — raw tarball contents
- `./lodo_data/validated_perteval/` — chosen perteval scores per fold

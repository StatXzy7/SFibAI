# Artifacts

This directory stores representative public outputs and is also the default location for newly generated run artifacts.

## Included public sample outputs

- `sample_results/`: preserved example outputs used for repository inspection and figure reference.

Current sample subdirectories:
- `raw/`
- `random/`
- `A-results/`
- `B-results/`
- `A2-res-sing/`
- `A2-res-ours/`

These folders are kept to preserve the original analysis outputs referenced during manuscript preparation.

## Default generated output locations

New runs should write to:
- `runs/` for training outputs
- `eval_results/` for evaluation outputs
- `feature_cache/` for cached radiomics features
- `figures/` for generated figure outputs
- `statistics/` for generated dataset summary tables

Main evaluation now writes nine-scenario clinical metrics such as:
- `binary_metrics_<model>.csv`
- `clinical_roc_auc_<model>.pdf`

Feature-embedding scripts write:
- `feature_cache/*.npz` from `scripts/figure_generation/extract_features.py`
- `figures/tsne/` outputs from `scripts/figure_generation/plot_tsne.py`

These generated directories are intentionally excluded by `.gitignore`.

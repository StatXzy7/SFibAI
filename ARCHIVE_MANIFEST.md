# SFibAI Archive Manifest

Version: `2.0.1-ncomms`

This manifest records the public reproducibility contents expected for the Nature Communications archive.

## Core training and evaluation

- `src/sfibai/train.py`: main 36-class SFibAI training entry point.
- `src/sfibai/eval.py`: main evaluation entry point.
- `src/sfibai/utils/binary_evaluation.py`: shared nine-scenario binary/composite clinical evaluation with bootstrap AUC confidence intervals.
- `scripts/run/train.sh`: example training command for the bundled sample dataset.
- `scripts/run/eval.sh`: example evaluation command requiring a user-supplied trained checkpoint.

## Figure reproduction

- `scripts/figure_generation/dataset_size_distri.py`: dataset distribution summaries.
- `scripts/figure_generation/feature_heatmap.py`: Grad-CAM++ feature heatmap generation, defaulting to ResNet Stage 1 (`layer1`).
- `scripts/figure_generation/extract_features.py`: exports Stage 4/`avgpool` features, probabilities, scores and clinical grades to a compressed NPZ file.
- `scripts/figure_generation/plot_tsne.py`: plots t-SNE panels from the exported feature archive and records the embedding parameters.

## Public sample data and outputs

- `data/seg_samples_500/`: representative pre-segmented sample images for smoke tests and demonstration runs.
- `artifacts/sample_results/`: preserved representative outputs from the analysis workflow.

## Non-redistributed materials

The full study dataset, expert lesion annotations and trained manuscript checkpoint are not redistributed in this public repository. Local runs that reproduce manuscript-scale metrics require user-supplied data and checkpoint files placed outside version control or under `checkpoints/`.

## Zenodo release checklist

1. Confirm `VERSION` and `CITATION.cff` contain the intended release version.
2. Run smoke checks for `train.py --help`, `eval.py --help`, `feature_heatmap.py --help`, `extract_features.py --help` and `plot_tsne.py --help`.
3. Create an immutable git tag, for example `v2.0.1-ncomms`.
4. Push the tag to GitHub.
5. Let Zenodo create a new version DOI from the tag, rather than modifying an older tag.

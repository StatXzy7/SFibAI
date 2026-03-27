# Baseline methods used for manuscript comparison

This document describes the two comparison baselines released with the SFibAI repository.

## Reviewer-facing disclosure

The original source code for the radiomics-based SVM framework proposed by Guo et al. and the VGG-based model described by Lee et al. was not publicly accessible when this study was conducted. For that reason, both methods were independently re-implemented based on the published method descriptions and re-trained on our dataset for fair comparison.

The public code for these re-implemented baselines is included in this repository at:
- `src/baselines/svm_guo/train.py`
- `src/baselines/vgg_lee/train.py`
- `src/baselines/vgg_lee/test.py`

These implementations are intended to document the comparison workflow used in the manuscript. They are not official code releases from the original authors.

## Guo et al. radiomics + SVM baseline

Reference summary:
- radiomics feature extraction from ultrasound regions of interest;
- Mann-Whitney U test for preliminary filtering;
- LASSO for sparse feature selection;
- SVM with RBF kernel;
- SMOTE for class-imbalance handling.

Repository entry point:
- `src/baselines/svm_guo/train.py`

Data requirements:
- 4-class image folders for `train/` and `val/`;
- ROI annotation files in `train_label/` and `val_label/`.

Notes:
- this baseline depends on radiomics-related packages listed in `src/baselines/requirements.txt`;
- feature extraction is CPU-oriented and may be substantially slower than the deep-learning baselines.

## Lee et al. VGG baseline

Reference summary:
- VGG-16 architecture;
- 4-class fibrosis classification setup in this repository;
- resize to 256 × 256 followed by crop to 224 × 224;
- SGD with Nesterov momentum.

Repository entry points:
- `src/baselines/vgg_lee/train.py`
- `src/baselines/vgg_lee/test.py`

Data requirements:
- 4-class image folders for `train/` and `val/`.

Notes:
- the implementation in this repository is adapted to the comparison setting used in the manuscript;
- pre-trained VGG weights are user-supplied unless placed under `checkpoints/`.

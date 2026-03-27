# Reproduced baselines

This directory contains the two comparison baselines used in the SFibAI manuscript.

## Disclosure

The original source code for the Guo SVM baseline and the Lee VGG baseline was not publicly accessible when this study was conducted. The code in this directory is therefore an **independent re-implementation** based on the published method descriptions and was written for fair comparison in this study.

These implementations are provided for transparency and reproducibility. They are not official releases from the original authors.

## Included entry points

### Guo radiomics + SVM baseline

- `svm_guo/train.py`

Expected input layout:

```text
<baseline_data_root>/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   └── 3/
├── val/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   └── 3/
├── train_label/
└── val_label/
```

Default public target location:
- `data/baseline_data/cls_raw4`

Example:

```bash
python src/baselines/svm_guo/train.py \
  --data_root /path/to/cls_raw4 \
  --runs_dir artifacts/runs
```

## Lee VGG baseline

- `vgg_lee/train.py`
- `vgg_lee/test.py`

Expected input layout:

```text
<baseline_data_root>/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   └── 3/
└── val/
    ├── 0/
    ├── 1/
    ├── 2/
    └── 3/
```

Default public target location:
- `data/baseline_data/cls_raw4`

Example:

```bash
python src/baselines/vgg_lee/train.py \
  --data_root /path/to/cls_raw4 \
  --save_root artifacts/runs
```

```bash
python src/baselines/vgg_lee/test.py \
  --data_root /path/to/cls_raw4 \
  --model_path /path/to/lee_vgg_checkpoint.pth \
  --output_dir artifacts/eval_results
```

## Output layout

Generated outputs are written under:
- `artifacts/runs/`
- `artifacts/eval_results/`
- `artifacts/feature_cache/`

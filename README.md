# SFibAI

SFibAI is the official code repository for the manuscript **"Deep Learning for Precision Grading of *Schistosoma japonicum*-induced Liver Fibrosis in Ultrasound Images"**.

## Features

- **Fine-grained grading** — 36-class ordinal output (0.0–3.5, step 0.1), compatible with standard clinical grades (F0–F3)
- **Hybrid loss** — Local KL Divergence + MSE + Boundary Penalty, with configurable weights
- **backbone** — ResNet-50 (default)
- **Mixed crop augmentation** — optional ROI-crop strategy with YOLO-format annotation support
- **Mixed-precision & DDP** — built-in AMP and DistributedDataParallel support

## Repository layout

```text
SFibAI/
├── README.md
├── CITATION.cff
├── environment.yml
├── requirements.txt
├── .gitignore
├── artifacts/
│   ├── README.md
│   ├── figures/
│   ├── sample_results/
│   └── statistics/
├── checkpoints/
│   └── README.md
├── data/
│   ├── README.md
│   └── seg_samples_500/
│       ├── train/
│       └── val/
├── scripts/
│   ├── figure_generation/
│   ├── preprocessing/
│   └── run/
└── src/
    ├── baselines/
    │   └── README.md
    └── sfibai/
        ├── config.py
        ├── train.py
        ├── eval.py
        ├── data/
        │   ├── dataset.py
        │   └── transforms.py
        └── utils/
            ├── __init__.py
            ├── metrics.py
            ├── models.py
            └── visualization.py
```

## Environment setup

Create a clean environment with conda:

```bash
conda env create -f environment.yml
conda activate sfibai
```

Or install the main dependencies with pip:

```bash
pip install -r requirements.txt
```

## Data layout

The training code expects each dataset root to contain at minimum:

```text
<data_root>/
├── train/
│   ├── 0.0/            # Grade 0, fibrosis score 0.0
│   │   ├── img001.jpg
│   │   └── ...
│   ├── 0.1/            # Grade 0, fibrosis score 0.1
│   ├── 0.2/
│   ├── ...
│   ├── 1.0/            # Grade 1, fibrosis score 1.0
│   ├── 1.1/
│   ├── ...
│   ├── 2.0/            # Grade 2, fibrosis score 2.0
│   ├── ...
│   ├── 3.0/            # Grade 3, fibrosis score 3.0
│   ├── ...
│   └── 3.5/            # Grade 3, fibrosis score 3.5
└── val/
    ├── 0.0/
    ├── 0.1/
    ├── ...
    └── 3.5/
```

Optionally, YOLO-format ROI annotations can be provided to enable crop-based augmentation:

```text
<data_root>/
├── train/
│   └── (same structure as above)
├── val/
│   └── (same structure as above)
├── train_label/        # YOLO-format .txt annotation files
│   ├── img001.txt      # each line: class x_center y_center width height
│   └── ...
└── val_label/
    └── ...
```

When annotation directories are absent, the dataset loader automatically disables cropping and loads images directly. The bundled sample data (`data/seg_samples_500/`) contains pre-segmented images and does not require annotation files.

## Included sample data

- `data/seg_samples_500/` — 500 representative pre-segmented ultrasound images organized by fibrosis grade (0.0–3.4), sufficient for a demonstration run of the training and evaluation pipeline.
- `artifacts/sample_results/` — representative outputs preserved from the analysis workflow.

The bundled sample data does not include ROI annotation files. The training and evaluation scripts default to `--crop_mode none`, which loads images directly without cropping.

The full study dataset and all training artifacts are **not** redistributed in this public repository.

## Preprocessing (optional)

If your raw ultrasound images contain device-UI backgrounds (text overlays, colored borders, thumbnails, etc.), you can use the bundled preprocessing script to extract the ultrasound ROI and generate YOLO-format annotation files:

```bash
python scripts/preprocessing/extract_roi.py \
    --input_dir /path/to/raw_images/train \
    --output_dir /path/to/processed/train \
    --label_dir /path/to/processed/train_label
```

The script uses contour detection (mean-threshold binarisation) to locate the largest connected region in each image. Run `python scripts/preprocessing/extract_roi.py --help` for all available options. This step is **not required** if your images are already pre-segmented — in that case, simply train with `--crop_mode none`.

## Quick start

### Training

```bash
python src/sfibai/train.py \
  --root_dirs /path/to/data_root \
  --checkpoint_path checkpoints/SFibAI.pth \
  --backbone resnet50 \
  --loss hybrid \
  --alpha 1.0 --beta 0.02 --gamma 0.02 \
  --epochs 120 \
  --bs 32 \
  --save_root artifacts/runs
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--backbone` | `resnet50` | Backbone architecture (see table above) |
| `--loss` | `hybrid` | Loss function: `kl`, `mse`, `boundary`, `kl_mse`, `kl_boundary`, `mse_boundary`, or `hybrid` |
| `--alpha` | 1.0 | Weight for Local KL Divergence loss |
| `--beta` | 0.02 | Weight for Expectation MSE loss |
| `--gamma` | 0.02 | Weight for Boundary Penalty loss |
| `--crop_mode` | `mixed` | Crop strategy: `none`, `fixed`, `random`, `mixed` |
| `--scheduler` | `step` | LR scheduler: `step` or `cos` |
| `--checkpoint_path` | — | Path to pretrained backbone weights |

Run `python src/sfibai/train.py --help` for the full list.

### Evaluation

```bash
python src/sfibai/eval.py \
  --model_paths checkpoints/SFibAI.pth \
  --root_dirs /path/to/data_root \
  --save_dir artifacts/eval_results
```

The evaluation script generates:
- Per-image prediction CSV
- 36-class and 4-class (F0–F3) confusion matrices (PDF)
- Clinical-threshold ROC curves with AUC scores (PDF)
- Detailed metrics log file

### Helper scripts

```bash
bash scripts/run/train.sh
bash scripts/run/eval.sh
```

## Figure-generation scripts

Utilities under `scripts/figure_generation/`:
- `dataset_size_distri.py` — dataset distribution summaries
- `feature_heatmap.py` — Grad-CAM-style feature visualisation

Outputs are saved to `artifacts/figures/` and `artifacts/statistics/`.

## Reproduced baselines

This repository also includes independent re-implementations of two comparison baselines described in the manuscript (Guo et al. radiomics + SVM, Lee et al. VGG-based). The original source code was not publicly accessible when this study was conducted.

For baseline documentation, entry points and usage instructions, see **[`src/baselines/README.md`](src/baselines/README.md)**.

## Additional documentation

- [`data/README.md`](data/README.md) — data availability and expected layout
- [`artifacts/README.md`](artifacts/README.md) — sample outputs and generated artifacts
- [`checkpoints/README.md`](checkpoints/README.md) — checkpoint placement and expectations

## Citation

If you use this repository, please cite the associated manuscript and software metadata in [`CITATION.cff`](CITATION.cff).

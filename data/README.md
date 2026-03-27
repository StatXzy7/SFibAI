# Data notes

This public repository does **not** include the full study dataset.

## Included in the repository

- `seg_samples_500/`: 500 representative pre-segmented ultrasound image samples organized by fibrosis grade, for repository inspection and demonstration runs.

## Not included in the repository

- the full multi-center dataset used for the main SFibAI experiments;
- YOLO-format ROI annotation files (optional; only needed for ROI-based cropping — see `scripts/preprocessing/extract_roi.py` for generating these from raw images);
- the full 4-class dataset used for baseline re-training;
- complete private or institution-restricted metadata.

## Expected data locations

### Main SFibAI pipeline

The default data root is `data/seg_samples_500/`.

Minimum required structure (images only, sufficient for demo runs):

```text
seg_samples_500/
├── train/              # Training images organized by fibrosis grade (0.0 to 3.4)
│   ├── 0.0/
│   ├── 0.1/
│   └── ...
└── val/                # Validation images organized by fibrosis grade
    ├── 0.0/
    ├── 0.2/
    └── ...
```

Full structure with ROI annotations (enables crop-based augmentation):

```text
seg_samples_500/
├── train/
├── val/
├── train_label/        # (optional) YOLO-format .txt annotation files for training images
└── val_label/          # (optional) YOLO-format .txt annotation files for validation images
```

Each `.txt` annotation file corresponds to an image file with the same stem name, in the format:

```text
class_id x_center y_center width height
```

where coordinates are normalized to [0, 1].

When annotation files are absent, the dataset loader automatically disables cropping and loads images directly.

> **Tip:** If you have raw ultrasound images with device-UI backgrounds, you can generate both segmented ROI images and the corresponding YOLO annotation files automatically:
>
> ```bash
> python scripts/preprocessing/extract_roi.py --help
> ```

### Reproduced baselines

The repository defaults point to:

```text
data/baseline_data/cls_raw4/
```

Expected structure:

```text
cls_raw4/
├── train/
├── val/
├── train_label/
└── val_label/
```

The Lee VGG baseline uses only the image folders. The Guo SVM baseline also requires the ROI annotation files.

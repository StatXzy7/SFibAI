"""
Optional preprocessing utility: extract the ultrasound ROI from raw images
that contain device-UI backgrounds and generate YOLO-format annotation files.

This script is useful when working with raw ultrasound images that contain
surrounding device UI, text overlays, or colored borders.  It uses contour
detection to locate and extract the actual ultrasound region, and optionally
produces YOLO-format bounding-box annotations compatible with the SFibAI
dataset loader's crop-based augmentation.

Usage examples
--------------
Extract ROI and generate labels for training images:

    python scripts/preprocessing/extract_roi.py \
        --input_dir /path/to/raw_images/train \
        --output_dir /path/to/processed/train \
        --label_dir /path/to/processed/train_label

Process validation images:

    python scripts/preprocessing/extract_roi.py \
        --input_dir /path/to/raw_images/val \
        --output_dir /path/to/processed/val \
        --label_dir /path/to/processed/val_label

Only generate label files (without saving segmented images):

    python scripts/preprocessing/extract_roi.py \
        --input_dir /path/to/images/train \
        --label_dir /path/to/images/train_label
"""

import argparse
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core image processing helpers
# ---------------------------------------------------------------------------

def _mask_edges(gray, edge_percent=0.05):
    """Zero out a border strip to suppress device-UI artifacts."""
    h, w = gray.shape
    mask = np.ones((h, w), dtype=np.uint8) * 255
    edge_h = int(h * edge_percent)
    edge_w = int(w * edge_percent * 1.5)
    mask[:edge_h, :] = 0
    mask[:, :edge_w] = 0
    mask[-edge_h:, :] = 0
    mask[:, -edge_w:] = 0
    return cv2.bitwise_and(gray, mask)


def _clean_border_rows(thresholded, lines=5, cols=5, threshold_value=220):
    """Suppress bright border rows/columns that survive thresholding."""
    h, w = thresholded.shape
    for i in range(lines):
        if np.mean(thresholded[i, :]) > threshold_value:
            thresholded[i, :] = 0
    for i in range(h - lines, h):
        if np.mean(thresholded[i, :]) > threshold_value:
            thresholded[i, :] = 0
    for i in range(cols):
        if np.mean(thresholded[:, i]) > threshold_value:
            thresholded[:, i] = 0
    for i in range(w - cols, w):
        if np.mean(thresholded[:, i]) > threshold_value:
            thresholded[:, i] = 0
    return thresholded


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_roi(image_path, edge_percent=0.05):
    """Locate the ultrasound ROI via mean-threshold contour detection.

    Parameters
    ----------
    image_path : str
        Path to a raw ultrasound image (with device-UI background).
    edge_percent : float
        Fraction of the image border to mask before detection.

    Returns
    -------
    segmented : ndarray or None
        Cropped ROI image (BGR).  *None* when no valid contour is found.
    label : list[float] or None
        YOLO-format bounding box ``[x_center, y_center, width, height]``
        normalised to [0, 1].  *None* when no valid contour is found.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_masked = _mask_edges(gray, edge_percent)

    mean_val = cv2.mean(gray_masked)[0]
    _, thresholded = cv2.threshold(gray_masked, mean_val, 255, cv2.THRESH_BINARY)

    line_frac = int(0.1 * gray.shape[0])
    col_frac = int(0.1 * gray.shape[1])
    thresholded = _clean_border_rows(thresholded, lines=line_frac, cols=col_frac)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    img_h, img_w = gray.shape
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    rw = w / img_w
    rh = h / img_h

    segmented = image[y:y + h, x:x + w]
    return segmented, [cx, cy, rw, rh]


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def process_directory(input_dir, output_dir=None, label_dir=None,
                      edge_percent=0.05, class_id=0):
    """Walk *input_dir* and process every image file found.

    The directory may be flat or contain grade sub-folders (e.g. ``0.0/``,
    ``0.1/``).  Sub-folder structure is preserved in *output_dir*; label
    files are written flat into *label_dir* (matching the SFibAI dataset
    loader convention).
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if label_dir:
        os.makedirs(label_dir, exist_ok=True)

    processed = 0
    skipped = 0

    for root, _dirs, files in os.walk(input_dir):
        rel = os.path.relpath(root, input_dir)

        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() not in _IMAGE_EXTS:
                continue

            src_path = os.path.join(root, fname)
            segmented, label = extract_roi(src_path, edge_percent)

            if segmented is None:
                print(f"  [skip] no contour: {src_path}")
                skipped += 1
                continue

            stem = os.path.splitext(fname)[0]

            if output_dir:
                dst_dir = os.path.join(output_dir, rel) if rel != '.' else output_dir
                os.makedirs(dst_dir, exist_ok=True)
                cv2.imwrite(os.path.join(dst_dir, fname), segmented)

            if label_dir:
                label_line = (f"{class_id} {label[0]:.6f} {label[1]:.6f} "
                              f"{label[2]:.6f} {label[3]:.6f}\n")
                with open(os.path.join(label_dir, f"{stem}.txt"), 'w') as f:
                    f.write(label_line)

            processed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract ultrasound ROI from raw images and generate "
                    "YOLO-format annotation files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--input_dir', required=True,
                        help='Directory of raw ultrasound images '
                             '(may contain grade sub-folders)')
    parser.add_argument('--output_dir', default=None,
                        help='Directory to save cropped ROI images '
                             '(omit to skip saving images)')
    parser.add_argument('--label_dir', default=None,
                        help='Directory to save YOLO-format .txt label files '
                             '(omit to skip label generation)')
    parser.add_argument('--edge_percent', type=float, default=0.05,
                        help='Fraction of the border to mask before detection '
                             '(default: 0.05)')
    parser.add_argument('--class_id', type=int, default=0,
                        help='Class ID written into label files (default: 0)')
    args = parser.parse_args()

    if not args.output_dir and not args.label_dir:
        parser.error("At least one of --output_dir or --label_dir is required.")

    print(f"Input:  {args.input_dir}")
    if args.output_dir:
        print(f"Output: {args.output_dir}")
    if args.label_dir:
        print(f"Labels: {args.label_dir}")
    print()

    process_directory(args.input_dir, args.output_dir, args.label_dir,
                      args.edge_percent, args.class_id)


if __name__ == '__main__':
    main()

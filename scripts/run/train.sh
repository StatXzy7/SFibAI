#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${REPO_ROOT}/src/sfibai/train.py" \
    --epochs 120 \
    --bs 32 \
    --num_workers 8 \
    --device_id 0 \
    --lr0 1e-4 \
    --lr1 1e-5 \
    --scheduler step \
    --scheduler_step_size 15 \
    --num_classes 36 \
    --root_dirs "${REPO_ROOT}/data/seg_samples_500" \
    --backbone resnet50 \
    --loss hybrid \
    --save_root "${REPO_ROOT}/artifacts/runs" \
    --save_best_name "best_model.pth" \
    --save_time_format "%Y-%m-%d_%H-%M-%S" \
    --crop_mode none

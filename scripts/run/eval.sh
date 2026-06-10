#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${REPO_ROOT}/src/sfibai/eval.py" \
  --model_paths "${REPO_ROOT}/checkpoints/SFibAI.pth" \
  --backbone resnet50 \
  --num_classes 36 \
  --device cuda:0 \
  --root_dirs "${REPO_ROOT}/data/seg_samples_500" \
  --save_dir "${REPO_ROOT}/artifacts/eval_results" \
  --shape 512 512 \
  --batch_size 8 \
  --num_workers 8 \
  --mode val \
  --crop_mode none \
  --bootstrap 2000 \
  --seed 42

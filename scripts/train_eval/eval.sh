#!/bin/bash

# Evaluate models with random crop mode
python eval.py \
  --model_paths \
    ../../models/SFibAI.pth \
  --backbone resnet50 \
  --num_classes 36 \
  --device cuda:0 \
  --root_dirs ../../dataset/seg_samples_500 \
  --save_dir ../../eval_results \
  --shape 512 512 \
  --batch_size 8 \
  --num_workers 8 \
  --mode val \
  --crop_mode random
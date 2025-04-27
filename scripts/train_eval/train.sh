#!/bin/bash
#SBATCH --job-name=liver_fibrosis    # Job name
#SBATCH --output=train_%j.log        # Output log file name, %j will be replaced with job ID
#SBATCH --error=train_%j.err         # Error log file name
#SBATCH --partition=gpu              # Use GPU partition
#SBATCH --nodes=1                    # Use 1 node
#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --cpus-per-task=4           # Use 4 CPU cores per task
#SBATCH --mem=64G                    # Request 64GB memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=1000:00:00           # Maximum running time 1000 hours

# Load required modules
module load cuda12.2
module load gcc12

# Activate conda environment
source ../../env/bin/activate

# Run training script
python train.py \
    --epochs 120 \
    --bs 32 \
    --num_workers 8 \
    --device_id 0 \
    --lr0 1e-4 \
    --lr1 1e-5 \
    --scheduler "step" \
    --tmax 20 \
    --num_classes 36 \
    --root_dirs '../../dataset/cls_raw' \
    --checkpoint_path "../../pretrained_models/resnet50-19c8e357.pth" \
    --backbone "resnet50" \
    --loss hybrid \
    --save_root "../../runs" \
    --save_best_name "best_model.pth" \
    --save_time_format "%Y-%m-%d_%H-%M-%S"
# SFibAI: Deep Learning for Precision Grading of Schistosomiasis Liver Fibrosis

SFibAI is a deep learning framework for precise grading of liver fibrosis in ultrasound images. This project implements a multi-class classification system that can accurately assess the degree of liver fibrosis from ultrasound images.

## Project Structure

```
project_root/
├── dataset/                    # Dataset directory
│   └── seg_samples_500/       # 500 labelled image samples
│       ├── train/             # Training set
│       │   ├── 0.0/          # Images with fibrosis grade 0.0
│       │   ├── 0.1/          # Images with fibrosis grade 0.1
│       │   └── ...           # Other grades (0.2 to 3.4)
│       └── val/              # Validation set
│           ├── 0.0/          # Images with fibrosis grade 0.0
│           ├── 0.1/          # Images with fibrosis grade 0.1
│           └── ...           # Other grades (0.2 to 3.2)
├── models/                     # Model weights directory
│   └── SFibAI.pth             # Pre-trained model weights
├── runs/                      # Training logs and results
├── scripts/                   # Source code
│   ├── figure_generate/       # Visualization scripts
│   │   ├── dataset_size_distri.py    # Dataset distribution analysis
│   │   └── feature_heatmap.py        # Feature visualization
│   └── train_eval/            # Training and evaluation scripts
│       ├── data/              # Data processing modules
│       │   ├── dataset.py     # Dataset class implementation
│       │   └── transforms.py  # Image transformations
│       ├── utils/             # Utility functions
│       │   ├── metrics.py     # Evaluation metrics
│       │   ├── models.py      # Model architectures
│       │   └── visualization.py # Training visualization
│       ├── config.py          # Configuration parameters
│       ├── train.py           # Training script
│       ├── eval.py            # Evaluation script
│       ├── train.sh           # Training shell script
│       └── eval.sh            # Evaluation shell script
└── requirements.txt           # Python dependencies
```

## Environment Setup

1. Create a new conda environment:

```bash
conda create -n sfibai python=3.8
conda activate sfibai
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The dataset consists of 500 labelled ultrasound images of liver fibrosis, organized by fibrosis grade. The dataset is split into training and validation sets:

1. Training set (`dataset/seg_samples_500/train/`):
   - Contains images with fibrosis grades from 0.0 to 3.5
   - Each grade has its own subdirectory (e.g., 0.0/, 0.1/, etc.)
   - Images are organized by their fibrosis grade

2. Validation set (`dataset/seg_samples_500/val/`):
   - Contains images with fibrosis grades from 0.0 to 3.5
   - Similar structure to the training set
   - Used for model validation during training

Each image is labeled with its corresponding fibrosis grade, which is used as the target for the classification task.

## Training

1. Configure training parameters in `scripts/train_eval/config.py`:

   - Set dataset paths
   - Adjust model parameters
   - Configure training hyperparameters
2. Start training:

```bash
cd scripts/train_eval
bash train.sh
```

Training logs and model checkpoints will be saved in the `runs` directory.

## Evaluation

1. Evaluate model performance:

```bash
cd scripts/train_eval
bash eval.sh
```

Evaluation results will be saved in the `eval_results` directory.

## Visualization

1. Analyze dataset distribution:

```bash
cd scripts/figure_generate
python dataset_size_distri.py
```

2. Generate feature heatmaps:

```bash
cd scripts/figure_generate
python feature_heatmap.py
```

## File Descriptions

### Core Training Files

- `train.py`: Main training script implementing the training loop and model optimization
- `eval.py`: Evaluation script for model performance assessment
- `config.py`: Configuration file containing all hyperparameters and settings

### Data Processing

- `dataset.py`: Custom dataset class for loading and preprocessing ultrasound images
- `transforms.py`: Image transformation pipeline for data augmentation

### Model and Utilities

- `models.py`: Model architecture definitions and weight loading
- `metrics.py`: Implementation of evaluation metrics
- `visualization.py`: Functions for plotting training progress and results

### Visualization Scripts

- `dataset_size_distri.py`: Analyzes and visualizes dataset distribution
- `feature_heatmap.py`: Generates feature activation heatmaps for model interpretation

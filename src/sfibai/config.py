from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class Config:
    # Dataset configuration
    ROOT_DIRS = [
        str(REPO_ROOT / 'data' / 'seg_samples_500')
    ]
    IMAGE_SIZE = (512, 512)
    NUM_CLASSES = 36

    # Training configuration
    BATCH_SIZE = 32
    EPOCHS = 120
    LEARNING_RATE = 1e-4
    MIN_LEARNING_RATE = 1e-6
    SCHEDULER_STEP_SIZE = 15
    T_MAX = SCHEDULER_STEP_SIZE
    SCHEDULER = 'step'

    # Model configuration
    INIT_CHECKPOINT_PATH = None
    CHECKPOINT_PATH = INIT_CHECKPOINT_PATH

    # Other configurations
    DEVICE_ID = 0
    NUM_WORKERS = 4

    # DDP configuration
    DDP_ENABLED = False

    # Model configuration
    BACKBONE = 'resnet50'

    SAVE_ROOT = str(REPO_ROOT / 'artifacts' / 'runs')
    SAVE_TIME_FORMAT = '%Y-%m-%d_%H-%M-%S'
    SAVE_BEST_NAME = 'best_model.pth'

class Config:
    # Dataset configuration
    ROOT_DIRS = [
        '../../dataset/seg_samples_500'
    ]
    IMAGE_SIZE = (512, 512)  # Image dimensions 800*600 512*512
    NUM_CLASSES = 36
    
    # Training configuration
    BATCH_SIZE = 32
    EPOCHS = 120
    LEARNING_RATE = 1e-4
    MIN_LEARNING_RATE = 1e-6
    T_MAX = 15
    SCHEDULER = 'step'
    
    # Model configuration
    CHECKPOINT_PATH = '../../models/SFibAI.pth'
    
    # Other configurations
    DEVICE_ID = 0
    NUM_WORKERS = 4  # 8
    
    # DDP configuration
    DDP_ENABLED = False
    
    # Model configuration
    BACKBONE = "resnet50"
    
    SAVE_ROOT = "../../runs"
    SAVE_TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
    SAVE_BEST_NAME = "best_model.pth"

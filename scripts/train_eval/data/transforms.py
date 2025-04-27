import cv2
import numpy as np
import torch

class ImageTransforms:
    def __init__(self, shape, training=True):
        self.shape = shape
        self.training = training
        
    def __call__(self, img):
        # Resize image
        # print("self.shape", self.shape)
        img = cv2.resize(img, self.shape)
        
        if self.training:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            
            # Random color jitter
            if np.random.rand() > 0.5:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                s = hsv[:, :, 1].astype(np.float32) * (0.2 + 0.8 * np.random.rand())
                s = np.clip(s, 0, 255).astype(np.uint8)
                hsv[:, :, 1] = s
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
        # Normalize image
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to CHW format
        img = img.transpose(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32)

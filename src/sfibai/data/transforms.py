import cv2
import numpy as np
import torch

class ImageTransforms:
    def __init__(self, shape, training=True):
        self.shape = shape
        self.training = training
        
    def _augment(self, img, seglabel=None):
        img = img.astype(np.float32)
        seglabel = None if seglabel is None else list(seglabel)

        if self.training:
            gamma = np.random.uniform(1.0 / 2.2, 2.2)
            img = np.power(np.clip(img / 255.0, 0.0, 1.0), gamma) * 255.0

            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                if seglabel is not None:
                    seglabel[0] = 1.0 - seglabel[0]

            if np.random.rand() > 0.5:
                hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                hsv = hsv.astype(np.float32)
                hsv[:, :, 1] *= np.random.uniform(0.2, 1.0)
                hsv[:, :, 2] *= np.random.uniform(0.2, 1.0)
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.randint(-30, 30)
                img = alpha * img + beta

            if np.random.rand() > 0.5:
                img = img + np.random.normal(0.0, 10.0, img.shape).astype(np.float32)

        return img, seglabel

    def augment(self, img):
        img, _ = self._augment(img)
        return img

    def augment_with_seglabel(self, img, seglabel):
        return self._augment(img, seglabel)

    def finalize(self, img):
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.resize(img, self.shape)

        # Normalize image
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to CHW format
        img = img.transpose(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32)

    def __call__(self, img):
        return self.finalize(self.augment(img))

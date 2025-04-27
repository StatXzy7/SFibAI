import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class SchistosomiasisDataset(Dataset):
    def __init__(self, root_dirs, mode='train', transform=None, debug=False, crop_mode='mixed'):
        """
        Args:
            root_dirs: Dataset root directory
            mode: 'train', 'val' or 'test'
            transform: Image transformations
            debug: Whether to enable debug mode
            crop_mode: Crop mode, options:
                - 'none': No cropping
                - 'fixed': Only use fixed cropping
                - 'random': Only use random cropping
                - 'mixed': Mix fixed and random cropping (original random method)
        """
        self.images = []
        self.labels = []
        self.seglabels = []
        self.transform = transform
        self.mode = mode
        self.debug = debug
        self.crop_mode = crop_mode.lower()
        
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        print(f"\nLoading {mode} dataset...")
        
        total_valid = 0
        total_invalid = 0
        
        # Iterate through all root directories
        for root_dir in root_dirs:
            image_dir = os.path.join(root_dir, mode)
            label_dir = os.path.join(root_dir, f'{mode}_label')
            
            # Check path existence
            if not os.path.exists(image_dir):
                print(f"Warning: Directory does not exist {image_dir}")
                continue
            if not os.path.exists(label_dir):
                print(f"Warning: Directory does not exist {label_dir}")
                continue
            
            # Iterate through label folders (e.g., 0.0, 0.1, 1.5, etc.)
            for label_folder in os.listdir(image_dir):
                label_folder_path = os.path.join(image_dir, label_folder)
                if not os.path.isdir(label_folder_path):
                    continue
                    
                # Iterate through all image files in the folder
                for image_file in os.listdir(label_folder_path):
                    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    image_path = os.path.join(label_folder_path, image_file)
                    seg_file = image_file.rsplit('.', 1)[0] + '.txt'
                    seg_file_path = os.path.join(label_dir, seg_file)
                    
                    # Check if image and corresponding label file exist
                    if not os.path.exists(image_path) or not os.path.exists(seg_file_path):
                        total_invalid += 1
                        if self.debug:
                            print(f"Skipping invalid file: {image_path}, label: {seg_file_path}")
                        continue
                        
                    try:
                        # Validate segmentation label format
                        with open(seg_file_path, 'r') as f:
                            line = f.readline().strip()
                            parts = line.split()
                            if len(parts) != 5:
                                print(f"Warning: Invalid segmentation label format {seg_file_path}")
                                total_invalid += 1
                                continue
                            _, x, y, w, h = map(float, parts)
                        
                        # Add to dataset
                        self.images.append(image_path)
                        self.labels.append(int(float(label_folder) * 10))
                        self.seglabels.append([x, y, w, h])
                        total_valid += 1
                        
                    except Exception as e:
                        print(f"Error: Failed to process file {image_path}: {str(e)}")
                        total_invalid += 1
                        continue
        
        # ------------------------
        # Print dataset loading information
        # ------------------------
        print(f"\n{mode} dataset loading completed, statistics:")
        print(f"  - Valid files: {total_valid}")
        print(f"  - Invalid files: {total_invalid}")
        
        # Print label distribution (only once)
        label_dist = {}
        for label in self.labels:
            label_dist[label] = label_dist.get(label, 0) + 1
        
        # Concatenate into one line output, e.g., "0.0:748 0.1:166 0.2:1030 ..."
        distribution_str = " ".join(
            f"{label/10:.1f}:{count}"
            for label, count in sorted(label_dist.items())
        )
        print(f"  - Label distribution: {distribution_str}")
        
        print(f"\nTotal {len(self.images)} samples\n")

    def __len__(self):
        return len(self.images)

    def fixed_crop(self, img, seglabel):
        h, w, _ = img.shape
        x_center, y_center, bbox_width, bbox_height = seglabel
        x_center *= w
        y_center *= h
        bbox_width *= w
        bbox_height *= h
        top = int(max(0, y_center - bbox_height / 2))
        left = int(max(0, x_center - bbox_width / 2))
        bottom = int(min(h, y_center + bbox_height / 2))
        right = int(min(w, x_center + bbox_width / 2))
        img = img[top:bottom, left:right]
        return img

    def random_crop(self, img, seglabel):
        h, w, _ = img.shape
        x_center, y_center, bbox_width, bbox_height = seglabel
        x_center *= w
        y_center *= h
        bbox_width *= w
        bbox_height *= h

        top_limit = max(0, y_center - bbox_height / 2 * 1.2)
        left_limit = max(0, x_center - bbox_width / 2 * 1.2)
        bottom_limit = min(h, y_center + bbox_height / 2 * 1.2)
        right_limit = min(w, x_center + bbox_width / 2 * 1.2)

        top = np.random.randint(
            int(top_limit), int(y_center - bbox_height / 2 + 1))
        left = np.random.randint(
            int(left_limit), int(x_center - bbox_width / 2 + 1))
        bottom = np.random.randint(
            int(y_center + bbox_height / 2), int(bottom_limit) + 1)
        right = np.random.randint(
            int(x_center + bbox_width / 2), int(right_limit) + 1)

        img = img[top:bottom, left:right]
        return img

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.images[idx])
            if img is None:
                print(f"Error: Unable to load image: {self.images[idx]}")
                return self.__getitem__((idx + 1) % len(self))
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = self.labels[idx]
            seglabel = self.seglabels[idx]
            
            # According to crop mode, select cropping method
            if self.crop_mode == 'none':
                pass  # No cropping
            elif self.crop_mode == 'fixed':
                img = self.fixed_crop(img, seglabel)
            elif self.crop_mode == 'random':
                img = self.random_crop(img, seglabel)
            elif self.crop_mode == 'mixed':
                # Original random cropping logic
                prob = np.random.rand()
                if prob < 0.2:
                    pass  # No cropping
                elif prob < 0.4:
                    img = self.fixed_crop(img, seglabel)
                else:
                    img = self.random_crop(img, seglabel)
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
            
        except Exception as e:
            print(f"\nWarning: Error loading index {idx}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

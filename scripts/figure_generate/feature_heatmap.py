import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision import transforms

##########################
# 1. Dataset: Using cv2 for preprocessing
##########################

class SchistosomiasisDataset(Dataset):
    def __init__(self,
                 root_dir,
                 transform=None,
                 debug=False):
        """
        Args:
            root_dir: Image directory
            transform: Image transforms
            debug: Debug mode flag
        """
        self.images = []
        self.debug = debug
        self.transform = transform
        
        print("\nLoading dataset...")
        
        total_valid = 0
        total_invalid = 0
        
        # Directly iterate through all image files in the directory
        for image_file in os.listdir(root_dir):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            image_path = os.path.join(root_dir, image_file)
            
            if not os.path.exists(image_path):
                total_invalid += 1
                if self.debug:
                    print(f"Skipping invalid file: {image_path}")
                continue
                
            self.images.append(image_path)
            total_valid += 1
        
        print(f"\nDataset loading completed:")
        print(f"  - Valid files: {total_valid}")
        print(f"  - Invalid files: {total_invalid}")
        print(f"\nTotal {len(self.images)} samples\n")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image_path = self.images[idx]
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Cannot load image: {image_path}")
                return self.__getitem__((idx + 1) % len(self))
                
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Save a copy of original image for visualization
            orig_img = img.copy()
            
            # Apply transformations
            if self.transform:
                img = self.transform(img)
            
            return img, orig_img
            
        except Exception as e:
            print(f"\nWarning: Error loading index {idx}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))


##########################
# 2. Model & Feature Visualization
##########################

def load_resnet50_model(checkpoint_path, num_classes=36, device='cuda'):
    """
    Load a ResNet50 model and load the specified weights file.
    """
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    
    # Set requires_grad=True
    for param in model.parameters():
        param.requires_grad = True
        
    model.to(device)
    model.eval()
    return model


class FeatureExtractor(nn.Module):
    """
    Truncate ResNet50 to specified layers and return intermediate feature maps.
    Supports feature extraction from multiple layers.
    """
    def __init__(self, model, target_layers=["layer1", "layer2", "layer3", "layer4"]):
        super().__init__()
        self.target_layers = target_layers
        
        # Define base layers
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        
        # Define individual layers
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        features = {}
        
        # Base layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Pass through layers and save features
        if "layer1" in self.target_layers:
            x = self.layer1(x)
            features["layer1"] = x
        else:
            x = self.layer1(x)
            
        if "layer2" in self.target_layers:
            x = self.layer2(x)
            features["layer2"] = x
        else:
            x = self.layer2(x)
            
        if "layer3" in self.target_layers:
            x = self.layer3(x)
            features["layer3"] = x
        else:
            x = self.layer3(x)
            
        if "layer4" in self.target_layers:
            x = self.layer4(x)
            features["layer4"] = x
        else:
            x = self.layer4(x)
            
        return features

class MultiGradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {layer: None for layer in target_layers}
        self.activations = {layer: None for layer in target_layers}
        
        # Register hook functions
        self._register_hooks()

    def _register_hooks(self):
        def get_activation_hook(layer_name):
            def hook(module, input, output):
                self.activations[layer_name] = output
            return hook
            
        def get_gradient_hook(layer_name):
            def hook(module, grad_input, grad_output):
                self.gradients[layer_name] = grad_output[0]
            return hook
            
        # Register hooks for each layer
        for layer_name in self.target_layers:
            target_module = dict(self.model.named_modules())[layer_name]
            target_module.register_forward_hook(get_activation_hook(layer_name))
            target_module.register_backward_hook(get_gradient_hook(layer_name))

    def generate_cams(self, input_tensor, target_class):
        # Ensure input tensor requires gradient
        input_tensor.requires_grad = True
        
        # Perform forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Calculate gradient for target class
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Generate CAM for each layer
        cams = {}
        for layer_name in self.target_layers:
            gradients = self.gradients[layer_name].cpu().data.numpy()
            activations = self.activations[layer_name].cpu().data.numpy()
            
            # Weight all channels
            weights = np.mean(gradients, axis=(2, 3))[0, :]
            cam = np.zeros(activations.shape[2:], dtype=np.float32)
            
            for i, w in enumerate(weights):
                cam += w * activations[0, i, :, :]
            
            # Normalize CAM
            cam = np.maximum(cam, 0)
            cam -= np.min(cam)
            cam /= np.max(cam) + 1e-8
            
            cams[layer_name] = cam
            
        return cams


def generate_heatmap(feature_map):
    """
    Convert feature map to single-channel heatmap. Simply average channels and normalize to [0,1].
    feature_map: shape = [1, C, H, W]
    """
    fm = feature_map[0]           # Take first item from batch
    fm = torch.mean(fm, dim=0)    # Average across channels => [H, W]
    fm = fm.detach().cpu().numpy()
    fm -= fm.min()
    fm /= (fm.max() + 1e-8)
    return fm  # [H, W], 0~1


def overlay_heatmap_on_image(heatmap, image_rgb, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay single-channel heatmap on RGB image, image_rgb shape=(H, W, 3), RGB format
    """
    # Ensure image_rgb is numpy array
    if torch.is_tensor(image_rgb):
        image_rgb = image_rgb.cpu().numpy()

    # Resize heatmap to original image size, not resize original image to heatmap size
    orig_h, orig_w = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
    
    # Convert heatmap to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)  # BGR
    
    # Convert original image from RGB to BGR for addWeighted
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    overlaid = cv2.addWeighted(heatmap_color, alpha, image_bgr, 1 - alpha, 0)
    return overlaid


def plot_probability_histogram(probabilities, save_path):
    """
    Plot probability distribution histogram
    probabilities: shape=(36,) probability distribution
    save_path: save path
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(probabilities)), probabilities)
    plt.title('Category Probability Distribution')
    plt.xlabel('Category Index')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


##########################
# 3. Main Process
##########################

def visualize_feature_maps(
    dataset,
    model_paths,
    device="cuda",
    target_layers=["layer1", "layer2", "layer3", "layer4"],
    num_classes=36,
    save_dir="./vis_results"
):
    os.makedirs(save_dir, exist_ok=True)
    
    # Load models and Grad-CAM
    models_and_gradcams = []
    for ckpt_path in model_paths:
        model_full = load_resnet50_model(ckpt_path, num_classes=num_classes, device=device)
        grad_cam = MultiGradCAM(model_full, target_layers=target_layers)
        models_and_gradcams.append((ckpt_path, model_full, grad_cam))
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for idx, (img_tensor, orig_img) in enumerate(loader):
        print(f"\nProcessing image {idx}:")
        
        img_torch = img_tensor.to(device)
        base_name = f"sample_{idx:04d}"
        
        for ckpt_path, model_full, grad_cam in models_and_gradcams:
            model_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            
            # Get model output and predicted class
            logits = model_full(img_torch)
            probs = torch.softmax(logits, dim=1)
            
            # Save probability histogram
            prob_save_path = os.path.join(save_dir, f"{base_name}_{model_name}_probs.png")
            plot_probability_histogram(probs[0].cpu().detach().numpy(), prob_save_path)
            print(f"Probability histogram saved: {prob_save_path}")
            
            pred_value = torch.sum(probs * torch.arange(num_classes, device=device), dim=1)
            pred_value = torch.round(pred_value).long()
            pred_idx = pred_value.cpu().item()
            print(f"Model: {model_name}, Predicted Class Index = {pred_idx}")
            
            # Generate heatmaps for all layers using Grad-CAM
            cams = grad_cam.generate_cams(img_torch, target_class=pred_idx)
            
            # Save heatmap for each layer
            for layer_name, cam in cams.items():
                overlaid = overlay_heatmap_on_image(cam, orig_img[0], alpha=0.5)
                out_name = f"{base_name}_{model_name}_{layer_name}.png"
                out_path = os.path.join(save_dir, out_name)
                cv2.imwrite(out_path, overlaid)
                print(f"Feature heatmap saved for {layer_name}: {out_path}")


class ImageTransforms:
    def __init__(self, shape, training=True):
        self.shape = shape
        self.training = training
        
    def __call__(self, img):
        # Resize image
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

if __name__ == "__main__":
    from torchvision import transforms
    
    # Define transform same as training
    transform = ImageTransforms(shape=(512, 512), training=False)
    
    # =============================
    # (1) Prepare dataset
    # =============================
    root_dir = "../../cli_figure_test/images"
    
    dataset = SchistosomiasisDataset(
        root_dir=root_dir,
        transform=transform,
        debug=False
    )
    
    # =============================
    # (2) Model paths list
    # =============================
    model_paths = [
        "../../models/Single-scale.pth"
    ]
    
    # =============================
    # (3) Run visualization
    # =============================
    visualize_feature_maps(
        dataset=dataset,
        model_paths=model_paths,
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_layers=["layer1", "layer2", "layer3", "layer4"],
        save_dir="../../cli_figure_test/vis_results"
    )

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models


class SchistosomiasisDataset(Dataset):
    def __init__(self, root_dir, transform=None, debug=False):
        """Load all images directly from a flat directory."""
        self.images = []
        self.debug = debug
        self.transform = transform

        print("\nLoading dataset...")
        total_valid = 0
        total_invalid = 0

        for image_file in sorted(Path(root_dir).iterdir()):
            if image_file.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            if not image_file.exists():
                total_invalid += 1
                if self.debug:
                    print(f"Skipping invalid file: {image_file}")
                continue
            self.images.append(str(image_file))
            total_valid += 1

        print("\nDataset loading completed:")
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

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_img = img.copy()

            if self.transform:
                img = self.transform(img)

            return img, orig_img
        except Exception as exc:
            print(f"\nWarning: Error loading index {idx}: {exc}")
            return self.__getitem__((idx + 1) % len(self))


def load_resnet50_model(checkpoint_path, num_classes=36, device='cuda'):
    """Load a ResNet50 model and restore the supplied checkpoint."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)

    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict, strict=False)
    for param in model.parameters():
        param.requires_grad = True

    model.to(device)
    model.eval()
    return model


class MultiGradCAMPP:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {layer: None for layer in target_layers}
        self.activations = {layer: None for layer in target_layers}
        self._register_hooks()

    def _register_hooks(self):
        def get_activation_hook(layer_name):
            def hook(module, input_value, output):
                self.activations[layer_name] = output
            return hook

        def get_gradient_hook(layer_name):
            def hook(module, grad_input, grad_output):
                self.gradients[layer_name] = grad_output[0]
            return hook

        for layer_name in self.target_layers:
            target_module = dict(self.model.named_modules())[layer_name]
            target_module.register_forward_hook(get_activation_hook(layer_name))
            target_module.register_full_backward_hook(get_gradient_hook(layer_name))

    def generate_cams(self, input_tensor, target_class):
        input_tensor.requires_grad = True
        self.model.zero_grad()
        output = self.model(input_tensor)

        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

        cams = {}
        for layer_name in self.target_layers:
            gradients = self.gradients[layer_name].detach()
            activations = self.activations[layer_name].detach()
            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)
            spatial_sum = torch.sum(activations * grad_3, dim=(2, 3), keepdim=True)
            alpha = grad_2 / (2.0 * grad_2 + spatial_sum + 1e-8)
            weights = torch.sum(alpha * torch.relu(gradients), dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1)[0]

            cam = torch.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cams[layer_name] = cam.cpu().numpy().astype(np.float32)

        return cams


MultiGradCAM = MultiGradCAMPP


def overlay_heatmap_on_image(heatmap, image_rgb, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay a single-channel heatmap on an RGB image."""
    if torch.is_tensor(image_rgb):
        image_rgb = image_rgb.cpu().numpy()

    orig_h, orig_w = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(heatmap_color, alpha, image_bgr, 1 - alpha, 0)


def plot_probability_histogram(probabilities, save_path):
    """Plot a probability distribution histogram."""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(probabilities)), probabilities)
    plt.title('Category probability distribution')
    plt.xlabel('Category index')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


class ImageTransforms:
    def __init__(self, shape, training=True):
        self.shape = shape
        self.training = training

    def __call__(self, img):
        img = cv2.resize(img, self.shape)
        if self.training and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        if self.training and np.random.rand() > 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1].astype(np.float32) * (0.2 + 0.8 * np.random.rand())
            hsv[:, :, 1] = np.clip(saturation, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.transpose(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32)


def visualize_feature_maps(dataset, model_paths, device='cuda', target_layers=None, num_classes=36, save_dir='./vis_results'):
    """Generate Grad-CAM++ visualizations for all supplied checkpoints."""
    if target_layers is None:
        target_layers = ['layer1']

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    models_and_gradcams = []
    for ckpt_path in model_paths:
        model_full = load_resnet50_model(ckpt_path, num_classes=num_classes, device=device)
        grad_cam = MultiGradCAMPP(model_full, target_layers=target_layers)
        models_and_gradcams.append((ckpt_path, model_full, grad_cam))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for idx, (img_tensor, orig_img) in enumerate(loader):
        print(f"\nProcessing image {idx}:")
        img_torch = img_tensor.to(device)
        base_name = f"sample_{idx:04d}"

        for ckpt_path, model_full, grad_cam in models_and_gradcams:
            model_name = Path(ckpt_path).stem
            logits = model_full(img_torch)
            probs = torch.softmax(logits, dim=1)

            prob_save_path = save_dir / f"{base_name}_{model_name}_probs.png"
            plot_probability_histogram(probs[0].cpu().detach().numpy(), prob_save_path)
            print(f"Probability histogram saved: {prob_save_path}")

            pred_value = torch.sum(probs * torch.arange(num_classes, device=device), dim=1)
            pred_idx = torch.round(pred_value).long().cpu().item()
            print(f"Model: {model_name}, predicted class index = {pred_idx}")

            cams = grad_cam.generate_cams(img_torch, target_class=pred_idx)
            for layer_name, cam in cams.items():
                overlaid = overlay_heatmap_on_image(cam, orig_img[0], alpha=0.5)
                out_path = save_dir / f"{base_name}_{model_name}_{layer_name}.png"
                cv2.imwrite(str(out_path), overlaid)
                print(f"Feature heatmap saved for {layer_name}: {out_path}")


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Generate Grad-CAM++ visualizations for SFibAI checkpoints.')
    parser.add_argument('--image_dir', type=str, default=str(repo_root / 'data' / 'seg_samples_500' / 'val' / '0.0'))
    parser.add_argument('--model_paths', nargs='+', default=[str(repo_root / 'checkpoints' / 'SFibAI.pth')])
    parser.add_argument('--save_dir', type=str, default=str(repo_root / 'artifacts' / 'figures' / 'heatmaps'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--shape', nargs=2, type=int, default=[512, 512])
    parser.add_argument('--target_layers', nargs='+', default=['layer1'],
                        help='Target ResNet layers for Grad-CAM++; default is layer1 (Stage 1).')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    transform = ImageTransforms(shape=tuple(args.shape), training=False)
    dataset = SchistosomiasisDataset(root_dir=args.image_dir, transform=transform, debug=args.debug)
    visualize_feature_maps(
        dataset=dataset,
        model_paths=args.model_paths,
        device=args.device,
        target_layers=args.target_layers,
        save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()

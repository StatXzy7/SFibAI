import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
SFIBAI_SRC = REPO_ROOT / "src" / "sfibai"
sys.path.insert(0, str(SFIBAI_SRC))

from config import Config  # noqa: E402
from data.dataset import SchistosomiasisDataset  # noqa: E402
from data.transforms import ImageTransforms  # noqa: E402
from utils.models import create_model  # noqa: E402
from utils.scoring import clinical_grade_from_indices, label_indices_to_scores  # noqa: E402


def load_full_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    return model


class ActivationCollector:
    def __init__(self, model, layer_name):
        modules = dict(model.named_modules())
        if layer_name not in modules:
            available = ", ".join(sorted(modules.keys()))
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available}")
        self.activation = None
        self.handle = modules[layer_name].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.activation = output.detach()

    def pop(self):
        if self.activation is None:
            raise RuntimeError("No activation was captured during the forward pass.")
        activation = self.activation
        self.activation = None
        return activation

    def close(self):
        self.handle.remove()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract SFibAI feature vectors for t-SNE/UMAP-style figure reproduction."
    )
    parser.add_argument("--model_path", required=True, help="Trained SFibAI checkpoint.")
    parser.add_argument("--root_dirs", nargs="+", default=Config.ROOT_DIRS, help="Dataset roots.")
    parser.add_argument("--mode", default="val", choices=["train", "val", "test"])
    parser.add_argument("--save_path", default=str(REPO_ROOT / "artifacts" / "feature_cache" / "sfibai_features.npz"))
    parser.add_argument("--backbone", default="resnet50", choices=[
        "resnet50",
        "resnext50_32x4d",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "densenet121",
        "efficientnet_b0",
        "efficientnet_b1",
    ])
    parser.add_argument("--num_classes", type=int, default=36)
    parser.add_argument("--target_layer", default="avgpool",
                        help="Layer whose activation is flattened and exported; ResNet-50 default is avgpool.")
    parser.add_argument("--shape", nargs=2, type=int, default=Config.IMAGE_SIZE)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crop_mode", default="none", choices=["none", "fixed", "random", "mixed"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = create_model(args.backbone, num_classes=args.num_classes)
    model = load_full_checkpoint(model, args.model_path, device).to(device)
    model.eval()
    collector = ActivationCollector(model, args.target_layer)

    transform = ImageTransforms(shape=tuple(args.shape), training=False)
    dataset = SchistosomiasisDataset(
        root_dirs=args.root_dirs,
        mode=args.mode,
        transform=transform,
        crop_mode=args.crop_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    features, probs, labels = [], [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            logits = model(images)
            activation = collector.pop()
            features.append(torch.flatten(activation, 1).cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    collector.close()
    labels = np.concatenate(labels)
    label_tensor = torch.from_numpy(labels)
    scores = label_indices_to_scores(label_tensor).numpy()
    grades = clinical_grade_from_indices(label_tensor).numpy()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        features=np.concatenate(features),
        probabilities=np.concatenate(probs),
        labels=labels,
        scores=scores,
        grades=grades,
        model_path=str(Path(args.model_path)),
        target_layer=args.target_layer,
        mode=args.mode,
    )
    print(f"Saved feature archive: {save_path}")


if __name__ == "__main__":
    main()

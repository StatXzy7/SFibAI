import torchvision.models as models
import torch.nn as nn
import torch
from pathlib import Path


def _load_backbone_weights(model, checkpoint_path, skip_prefixes):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Initialization checkpoint was not found: {checkpoint_path}. "
            "Omit --init_checkpoint to train from random initialization."
        )

    pretrained_dict = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(pretrained_dict, dict) and "state_dict" in pretrained_dict:
        pretrained_dict = pretrained_dict["state_dict"]

    model_dict = model.state_dict()
    for key, value in pretrained_dict.items():
        key = key[7:] if key.startswith("module.") else key
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            continue
        if key in model_dict and model_dict[key].shape == value.shape:
            model_dict[key] = value

    model.load_state_dict(model_dict, strict=False)
    print(f"Loaded backbone weights from {checkpoint_path}")

def create_model(backbone, num_classes=3, checkpoint_path=None):
    """
    Create model based on backbone string:
    - Load custom pretrained weights from checkpoint_path, but only load backbone part;
    - The final classification layer (e.g., fc or classifier[-1]) will be reinitialized.
    """
    backbone = backbone.lower()
    
    if backbone == "resnet50":
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features

        if checkpoint_path:
            _load_backbone_weights(model, checkpoint_path, skip_prefixes=("fc.",))
        
        # Reinitialize classification layer
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif backbone == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=False)
        in_features = model.fc.in_features

        if checkpoint_path:
            _load_backbone_weights(model, checkpoint_path, skip_prefixes=("fc.",))
        
        # Reinitialize classification layer
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        if checkpoint_path:
            _load_backbone_weights(model, checkpoint_path, skip_prefixes=("classifier.1.",))

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    elif backbone == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=False)
        if checkpoint_path:
            _load_backbone_weights(model, checkpoint_path, skip_prefixes=("classifier.3.",))

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    elif backbone.startswith("efficientnet_b"):
        model = getattr(models, backbone)(pretrained=False)
        if checkpoint_path:
            _load_backbone_weights(model, checkpoint_path, skip_prefixes=("classifier.1.",))

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    elif backbone == "densenet121":
        model = models.densenet121(pretrained=False)
        if checkpoint_path:
            _load_backbone_weights(model, checkpoint_path, skip_prefixes=("classifier.",))

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    else:
        raise NotImplementedError(f"Unimplemented model: {backbone}")

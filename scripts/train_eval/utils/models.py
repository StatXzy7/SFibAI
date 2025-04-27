import torchvision.models as models
import torch.nn as nn
import torch

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

        # Load custom weights
        if checkpoint_path:
            pretrained_dict = torch.load(checkpoint_path, map_location="cpu")
            model_dict = model.state_dict()

            # Only load backbone weights (skip fc-related parameters)
            for k, v in pretrained_dict.items():
                if k.startswith('fc.'):  # Skip classification layer
                    continue
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v

            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded backbone weights from {checkpoint_path}")
        
        # Reinitialize classification layer
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif backbone == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=False)
        in_features = model.fc.in_features

        # Load custom weights
        if checkpoint_path:
            pretrained_dict = torch.load(checkpoint_path, map_location="cpu")
            model_dict = model.state_dict()

            for k, v in pretrained_dict.items():
                if k.startswith('fc.'):  # Skip classification layer
                    continue
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v

            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded backbone weights from {checkpoint_path}")
        
        # Reinitialize classification layer
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        if checkpoint_path:
            pretrained_dict = torch.load(checkpoint_path, map_location="cpu")
            model_dict = model.state_dict()

            for k, v in pretrained_dict.items():
                if k.startswith('classifier.1.'):  # Skip classification layer
                    continue
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v

            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded backbone weights from {checkpoint_path}")

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    elif backbone == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=False)
        if checkpoint_path:
            pretrained_dict = torch.load(checkpoint_path, map_location="cpu")
            model_dict = model.state_dict()

            for k, v in pretrained_dict.items():
                if k.startswith('classifier.3.'):  # Skip classification layer
                    continue
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v

            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded backbone weights from {checkpoint_path}")

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    elif backbone.startswith("efficientnet_b"):
        model = getattr(models, backbone)(pretrained=False)
        if checkpoint_path:
            pretrained_dict = torch.load(checkpoint_path, map_location="cpu")
            model_dict = model.state_dict()

            for k, v in pretrained_dict.items():
                if k.startswith('classifier.1.'):  # Skip classification layer
                    continue
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v

            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded backbone weights from {checkpoint_path}")

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    elif backbone == "densenet121":
        model = models.densenet121(pretrained=False)
        if checkpoint_path:
            pretrained_dict = torch.load(checkpoint_path, map_location="cpu")
            model_dict = model.state_dict()

            for k, v in pretrained_dict.items():
                if k.startswith('classifier.'):  # Skip classification layer
                    continue
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v

            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded backbone weights from {checkpoint_path}")

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    else:
        raise NotImplementedError(f"Unimplemented model: {backbone}")

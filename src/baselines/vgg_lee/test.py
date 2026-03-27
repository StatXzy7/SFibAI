#!/usr/bin/env python3
"""
Evaluate the reproduced Lee VGG baseline and export metrics.
Outputs binary_results.csv and confusion matrices aligned with train.py.

The original source code was not publicly accessible when this study was
conducted. This file is part of the independent re-implementation released with
this repository for fair comparison.
"""

import os
import time
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import models

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
)

warnings.filterwarnings("ignore")


def _default_paths():
    repo_root = Path(__file__).resolve().parents[3]
    data_root = repo_root / 'data' / 'baseline_data' / 'cls_raw4'
    output_root = repo_root / 'artifacts' / 'eval_results'
    return str(data_root), str(output_root)

class VGGNet(nn.Module):
    """
    VGG-Net architecture following Lee et al.; 4-class liver fibrosis (Grade 0-3).
    """
    def __init__(self, num_classes=4, input_channels=3):
        super(VGGNet, self).__init__()
        
        # VGG feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive global average pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LiverFibrosisDataset:
    """
    Four-class liver fibrosis dataset aligned with train.py.
    Resize to resize_size (256), then apply a center crop of 224.
    """
    def __init__(self, data_root, mode='train', transform=None, resize_size=(256, 256), rank=0):
        self.data_root = Path(data_root)
        self.mode = mode
        self.transform = transform
        self.resize_size = resize_size
        self.images = []
        self.labels = []
        self.rank = rank
        
        if rank == 0:
            print(f"\nLoading {mode} dataset ...")
        self._load_data()
        if rank == 0:
            print(f"Loaded {len(self.images)} samples")
        
    def _load_data(self):
        """Load image paths and labels from data_root/mode/0..3/."""
        image_dir = self.data_root / self.mode
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {image_dir}")
        
        for grade in [0, 1, 2, 3]:
            grade_dir = image_dir / str(grade)
            if not grade_dir.exists():
                if self.rank == 0:
                    print(f"Warning: grade dir missing {grade_dir}")
                continue
            for ext in ("*.jpg", "*.png"):
                for img_file in grade_dir.glob(ext):
                    self.images.append(str(img_file))
                    self.labels.append(grade)
        
        label_counts = np.bincount(self.labels, minlength=4)
        if self.rank == 0:
            print(f"Label distribution: {dict(enumerate(label_counts.tolist()))}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: cannot load image {img_path}")
            img = np.zeros((*self.resize_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.resize_size)
        
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class ImageTransforms:
    """
    Aligned with train.py (Lee et al. reproduction).
    Input is 256x256 (resized by Dataset); then center crop 224x224 + ImageNet normalization.
    Test mode uses center crop only, without extra augmentation.
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    def __init__(self, crop_size=(224, 224), training=False):
        self.crop_size = crop_size
        self.training = training
    
    def __call__(self, img):
        h, w = img.shape[:2]
        cw, ch = self.crop_size
        # center crop (same as train val)
        top = (h - ch) // 2
        left = (w - cw) // 2
        img = img[top:top + ch, left:left + cw]
        
        img = img.astype(np.float32) / 255.0
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        img = img.transpose(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32)


def evaluate_multiclass(model, dataloader, device):
    """Evaluate 4-class classification metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Multiclass metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    # Per-class AUC (one-vs-rest)
    auc_scores = []
    for i in range(4):
        if len(np.unique(all_labels)) > 1:
            try:
                auc = roc_auc_score((all_labels == i).astype(int), all_probs[:, i])
                auc_scores.append(auc)
            except:
                auc_scores.append(0.5)
        else:
            auc_scores.append(0.5)
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': bal_acc,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'kappa': kappa,
        'mcc': mcc,
        'auc_scores': auc_scores,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def _bootstrap_ci_auc(y_true, y_score, n_bootstrap=2000, alpha=0.95, seed=42):
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.5
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except Exception:
            continue
    if len(aucs) == 0:
        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
        return auc, auc, auc
    lower = np.percentile(aucs, (1 - alpha) / 2 * 100)
    upper = np.percentile(aucs, (1 + alpha) / 2 * 100)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    return auc, lower, upper


def evaluate_binary_pairs(model, dataloader, device, n_bootstrap=2000, ci_alpha=0.95, seed=42):
    """Evaluate binary and composite binary classification (9 tasks, paper Table 1)."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 9 binary/composite pairs (mainA.tex Table 1)
    binary_pairs = [
        ([0], [1], "Grade_0_vs_Grade_1"),
        ([0], [2], "Grade_0_vs_Grade_2"),
        ([0], [3], "Grade_0_vs_Grade_3"),
        ([1], [2], "Grade_1_vs_Grade_2"),
        ([1], [3], "Grade_1_vs_Grade_3"),
        ([2], [3], "Grade_2_vs_Grade_3"),
        ([0], [1, 2, 3], "Grade_0_vs_Grade_1-3"),
        ([1], [2, 3], "Grade_1_vs_Grade_2-3"),
        ([1, 2], [3], "Grade_1-2_vs_Grade_3"),
    ]
    
    results = {}
    
    for group1, group2, pair_name in binary_pairs:
        mask = np.isin(all_labels, group1 + group2)
        if not np.any(mask):
            continue
            
        y_true = all_labels[mask]
        y_pred = all_preds[mask]
        y_prob = all_probs[mask]
        
        binary_labels = np.isin(y_true, group1).astype(int)
        
        if len(group1) == 1:
            binary_probs = y_prob[:, group1[0]]
        else:
            binary_probs = np.sum(y_prob[:, group1], axis=1)
        
        binary_preds = (binary_probs >= 0.5).astype(int)
        
        try:
            if len(np.unique(binary_labels)) > 1:
                auc, auc_ci_lower, auc_ci_upper = _bootstrap_ci_auc(
                    binary_labels, binary_probs, n_bootstrap=n_bootstrap, alpha=ci_alpha, seed=seed
                )
            else:
                auc, auc_ci_lower, auc_ci_upper = 0.5, 0.5, 0.5
        except:
            auc, auc_ci_lower, auc_ci_upper = 0.5, 0.5, 0.5
        
        accuracy = accuracy_score(binary_labels, binary_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_labels, binary_preds, average='binary', zero_division=0
        )
        bal_acc = balanced_accuracy_score(binary_labels, binary_preds)
        kappa = cohen_kappa_score(binary_labels, binary_preds)
        mcc = matthews_corrcoef(binary_labels, binary_preds) if len(np.unique(binary_labels)) > 1 else 0.0
        
        cm = confusion_matrix(binary_labels, binary_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sensitivity = 0
            specificity = 0
        
        results[pair_name] = {
            'auc': auc,
            'auc_ci_lower': auc_ci_lower,
            'auc_ci_upper': auc_ci_upper,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'balanced_accuracy': bal_acc,
            'kappa': kappa,
            'mcc': mcc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'n_samples': len(binary_labels),
            'group1_count': np.sum(binary_labels),
            'group2_count': len(binary_labels) - np.sum(binary_labels)
        }
    
    return results


def _save_confusion_matrix(y_true, y_pred, out_path, classes=(0,1,2,3), normalize=True, title_prefix=""):
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_yticklabels([str(c) for c in classes])
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(f"{title_prefix} Confusion Matrix")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight')
    except OSError as e:
        print(f"Warning: failed to save confusion matrix ({out_path}): {e}. Skipping.")
    except Exception as e:
        print(f"Warning: error saving confusion matrix ({out_path}): {e}. Skipping.")
    finally:
        plt.close(fig)


def load_model(model_path, device):
    """Load a trained VGG-16 checkpoint aligned with train.py."""
    print(f"Loading model: {model_path}")

    # VGG-16 without BatchNorm (Lee et al. Eur Radiol 2020)
    vgg = models.vgg16(weights=None)
    in_features = vgg.classifier[-1].in_features
    vgg.classifier[-1] = nn.Linear(in_features, 4)
    
    # Load checkpoint weights
    if os.path.isfile(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model_state = vgg.state_dict()
            filtered_state = {}
            skipped_keys = []
            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    skipped_keys.append(k)
            
            load_msg = vgg.load_state_dict(filtered_state, strict=False)
            print(f"Loaded {len(filtered_state)} parameters")
            print(f"Skipped {len(skipped_keys)} mismatched keys")
            if hasattr(load_msg, 'missing_keys'):
                print(f"Missing keys: {len(load_msg.missing_keys)}")
            if hasattr(load_msg, 'unexpected_keys'):
                print(f"Unexpected keys: {len(load_msg.unexpected_keys)}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
    else:
        print(f"Model file not found: {model_path}")
        return None
    
    vgg = vgg.to(device)
    vgg.eval()
    return vgg


def main():
    """Load the reproduced Lee VGG baseline and run evaluation."""
    parser = argparse.ArgumentParser(description='Lee VGG baseline evaluation')

    default_data_root, default_output_dir = _default_paths()

    parser.add_argument('--data_root', type=str,
                       default=default_data_root,
                       help='Dataset root (e.g. cls_raw4)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained checkpoint (e.g. best_model.pth)')
    parser.add_argument('--output_dir', type=str,
                       default=default_output_dir,
                       help='Output directory for results')
    parser.add_argument('--resize_size', type=int, nargs=2, default=[256, 256],
                       help='Resize size before crop (Lee: 256x256)')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[224, 224],
                       help='Center crop size (Lee: 224x224)')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--auc_bootstrap', type=int, default=2000, help='Bootstrap samples for AUC CI')
    parser.add_argument('--auc_ci', type=float, default=0.95, help='AUC confidence level')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"lee_vgg_test_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Results -> {output_path}")
    
    model = load_model(args.model_path, device)
    if model is None:
        print("Model load failed; exiting.")
        return
    
    print("\nBuilding dataset (256 resize -> 224 center crop, same as train.py) ...")
    resize_sz = tuple(args.resize_size)
    crop_sz = tuple(args.crop_size)
    val_transform = ImageTransforms(crop_size=crop_sz, training=False)
    val_dataset = LiverFibrosisDataset(args.data_root, 'val', val_transform, resize_size=resize_sz)
    
    # Build data loader
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    
    print("\n=== Four-class evaluation ===")
    val_results = evaluate_multiclass(model, val_loader, device)
    print(f"Accuracy: {val_results['accuracy']:.4f} | Balanced: {val_results['balanced_accuracy']:.4f}")
    print(f"F1 (w/micro/macro): {val_results['f1_weighted']:.4f} / {val_results['f1_micro']:.4f} / {val_results['f1_macro']:.4f}")
    print(f"Kappa: {val_results['kappa']:.4f} | MCC: {val_results['mcc']:.4f}")
    print(f"Per-class AUC (one-vs-rest): {[f'{a:.4f}' for a in val_results['auc_scores']]}")
    
    cm_dir = output_path / 'confusion_matrices'
    cm_path = cm_dir / "final_confusion_matrix.png"
    _save_confusion_matrix(val_results['labels'], val_results['predictions'], cm_path,
                          classes=(0,1,2,3), normalize=True, title_prefix="Final")
    
    print("\n=== Binary-pair evaluation (9 tasks) ===")
    binary_results = evaluate_binary_pairs(model, val_loader, device, 
                                         n_bootstrap=args.auc_bootstrap, 
                                         ci_alpha=args.auc_ci, seed=args.seed)
    
    results = {
        'multiclass': val_results,
        'binary_pairs': binary_results
    }
    
    import json
    with open(output_path / 'results.json', 'w') as f:
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    binary_df = pd.DataFrame([
        {**metrics, 'pair': pair_name}
        for pair_name, metrics in binary_results.items()
    ])
    binary_df.to_csv(output_path / 'binary_results.csv', index=False)
    
    print("\nBinary-pair summary:")
    cols = ['pair','auc','auc_ci_lower','auc_ci_upper','accuracy','balanced_accuracy','precision','recall','f1','sensitivity','specificity','kappa','mcc','n_samples']
    existing = [c for c in cols if c in binary_df.columns]
    print(binary_df[existing].to_string(index=False))
    
    print(f"\nEvaluation done. Results saved to: {output_path}")
    print(f"  binary_results.csv: {output_path / 'binary_results.csv'}")
    print(f"  confusion_matrices: {cm_dir}")

if __name__ == "__main__":
    main()

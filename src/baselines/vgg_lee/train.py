# Reproduction of Lee et al. (2020):
#   "Deep learning with ultrasonography: automated classification of liver
#    fibrosis using a deep convolutional neural network"
#   European Radiology, 30:1264-1273
#
# Original method summary (Eur Radiol 2020):
#   - Architecture: VGG-16 (13 conv + 5 pool + 3 FC), ImageNet pre-trained
#   - Optimizer:    Nesterov SGD (lr=1e-4, momentum=0.9, weight_decay=5e-4)
#   - LR schedule:  StepLR with gamma=0.1
#   - Input:        Resize 256x256 -> random crop 224x224 + horizontal flip
#   - Loss:         CrossEntropy (standard, no label smoothing)
#   - Output:       4-class softmax (original: F0/F1/F23/F4; here: Grade 0-3)
#
# Adaptation notes:
#   The original paper targets METAVIR scoring for viral hepatitis (F0/F1/F23/F4).
#   Here we apply the same VGG-16 architecture to schistosomiasis-related liver
#   fibrosis grading (Grade 0/1/2/3) on our dataset for comparative evaluation.
#
# Reproducibility note:
#   The original source code was not publicly accessible when this study was
#   conducted. This implementation is an independent re-implementation written
#   for fair comparison and released with the SFibAI repository.

import os
import json
import time
import warnings
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import models

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
)

warnings.filterwarnings("ignore")

NUM_CLASSES = 4

# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    return True, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# ---------------------------------------------------------------------------
# Dataset & transforms
# ---------------------------------------------------------------------------

class LiverFibrosisDataset(Dataset):
    """Four-class liver fibrosis dataset (Grade 0/1/2/3) stored as
    ``<data_root>/<split>/<grade>/*.{jpg,png}``."""

    def __init__(self, data_root, mode='train', transform=None,
                 resize_size=(256, 256), rank=0):
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
        image_dir = self.data_root / self.mode
        if not image_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {image_dir}")

        for grade in range(NUM_CLASSES):
            grade_dir = image_dir / str(grade)
            if not grade_dir.exists():
                if self.rank == 0:
                    print(f"Warning: grade directory missing {grade_dir}")
                continue
            for ext in ("*.jpg", "*.png"):
                for img_file in grade_dir.glob(ext):
                    self.images.append(str(img_file))
                    self.labels.append(grade)

        label_counts = np.bincount(self.labels, minlength=NUM_CLASSES)
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
    """Preprocessing pipeline following Lee et al. (2020):
    resize to 256x256 -> random crop 224x224 (train) / center crop (val)
    -> horizontal flip (train only) -> ImageNet normalization."""

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, crop_size=(224, 224), training=True):
        self.crop_size = crop_size
        self.training = training

    def __call__(self, img):
        h, w = img.shape[:2]
        cw, ch = self.crop_size

        if self.training:
            top = np.random.randint(0, h - ch + 1) if h > ch else 0
            left = np.random.randint(0, w - cw + 1) if w > cw else 0
        else:
            top = (h - ch) // 2
            left = (w - cw) // 2
        img = img[top:top + ch, left:left + cw]

        if self.training and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        img = img.astype(np.float32) / 255.0
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        img = img.transpose(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32)

# ---------------------------------------------------------------------------
# Evaluation utilities (kept comprehensive for paper comparison)
# ---------------------------------------------------------------------------

def evaluate_multiclass(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0)
    p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    auc_scores = []
    for i in range(NUM_CLASSES):
        try:
            auc_scores.append(
                roc_auc_score((all_labels == i).astype(int), all_probs[:, i]))
        except Exception:
            auc_scores.append(0.5)

    return {
        'accuracy': accuracy, 'balanced_accuracy': bal_acc,
        'precision_weighted': p_w, 'recall_weighted': r_w, 'f1_weighted': f1_w,
        'precision_macro': p_ma, 'recall_macro': r_ma, 'f1_macro': f1_ma,
        'precision_micro': p_mi, 'recall_micro': r_mi, 'f1_micro': f1_mi,
        'kappa': kappa, 'mcc': mcc, 'auc_scores': auc_scores,
        'predictions': all_preds, 'labels': all_labels, 'probabilities': all_probs,
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


def evaluate_binary_pairs(model, dataloader, device,
                          n_bootstrap=2000, ci_alpha=0.95, seed=42):
    """Evaluate pairwise / composite binary classification performance."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 9 binary/composite tasks
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
        y_prob = all_probs[mask]
        binary_labels = np.isin(y_true, group1).astype(int)

        if len(group1) == 1:
            binary_probs = y_prob[:, group1[0]]
        else:
            binary_probs = np.sum(y_prob[:, group1], axis=1)

        binary_preds = (binary_probs >= 0.5).astype(int)

        try:
            if len(np.unique(binary_labels)) > 1:
                auc, auc_lo, auc_hi = _bootstrap_ci_auc(
                    binary_labels, binary_probs,
                    n_bootstrap=n_bootstrap, alpha=ci_alpha, seed=seed)
            else:
                auc, auc_lo, auc_hi = 0.5, 0.5, 0.5
        except Exception:
            auc, auc_lo, auc_hi = 0.5, 0.5, 0.5

        acc = accuracy_score(binary_labels, binary_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            binary_labels, binary_preds, average='binary', zero_division=0)
        bal_acc = balanced_accuracy_score(binary_labels, binary_preds)
        kappa = cohen_kappa_score(binary_labels, binary_preds)
        mcc = (matthews_corrcoef(binary_labels, binary_preds)
               if len(np.unique(binary_labels)) > 1 else 0.0)

        cm = confusion_matrix(binary_labels, binary_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sens, spec = 0, 0

        results[pair_name] = {
            'auc': auc, 'auc_ci_lower': auc_lo, 'auc_ci_upper': auc_hi,
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'balanced_accuracy': bal_acc, 'kappa': kappa, 'mcc': mcc,
            'sensitivity': sens, 'specificity': spec,
            'n_samples': len(binary_labels),
            'group1_count': int(np.sum(binary_labels)),
            'group2_count': int(len(binary_labels) - np.sum(binary_labels)),
        }

    return results

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device, scaler,
                rank=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if rank == 0 and batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}  '
                  f'Loss: {loss.item():.4f}  Acc: {100.*correct/total:.2f}%')

    return total_loss / len(dataloader), 100. * correct / total


def _save_confusion_matrix(y_true, y_pred, out_path,
                           classes=(0, 1, 2, 3), normalize=True,
                           title_prefix=""):
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set(xticks=tick_marks, yticks=tick_marks,
           xticklabels=[str(c) for c in classes],
           yticklabels=[str(c) for c in classes],
           ylabel='True label', xlabel='Predicted label',
           title=f"{title_prefix} Confusion Matrix")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: failed to save confusion matrix ({out_path}): {e}")
    finally:
        plt.close(fig)

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, device, args,
                rank=0, world_size=1):
    # Following Lee et al.: Nesterov SGD, StepLR, standard CrossEntropy
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma)
    scaler = GradScaler()

    best_val_acc = 0
    train_losses, train_accs, val_accs = [], [], []

    if rank == 0:
        print(f"\nStarting training for {args.epochs} epochs ...")
        print(f"  Optimizer: SGD (Nesterov, lr={args.learning_rate}, "
              f"momentum={args.momentum}, wd={args.weight_decay})")
        print(f"  Scheduler: StepLR (step_size={args.step_size}, "
              f"gamma={args.gamma})")

    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print("-" * 50)

        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, rank)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if rank == 0:
            val_results = evaluate_multiclass(model, val_loader, device)
            val_acc = val_results['accuracy'] * 100
            val_accs.append(val_acc)

            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val   - Acc: {val_acc:.2f}% | "
                  f"F1(w): {val_results['f1_weighted']:.4f} | "
                  f"Kappa: {val_results['kappa']:.4f}")

            cm_path = args.save_path / 'confusion_matrices' / \
                f"epoch_{epoch+1:03d}.png"
            _save_confusion_matrix(
                val_results['labels'], val_results['predictions'],
                cm_path, title_prefix=f"Epoch {epoch+1}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                try:
                    state = (model.module.state_dict() if hasattr(model, 'module')
                             else model.state_dict())
                    torch.save(state, args.save_path / 'best_model.pth')
                    print(f"  Saved best model (val acc {val_acc:.2f}%)")
                except Exception as e:
                    print(f"Warning: failed to save model: {e}")

        scheduler.step()

        if rank == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  LR: {lr:.6f}")

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Reproduction of Lee et al. VGG-16 for liver fibrosis '
                    'four-class classification')

    # Data
    repo_root = Path(__file__).resolve().parents[3]

    parser.add_argument('--data_root', type=str,
                        default=str(repo_root / 'data' / 'baseline_data' / 'cls_raw4'))
    parser.add_argument('--save_root', type=str,
                        default=str(repo_root / 'artifacts' / 'runs'))
    parser.add_argument('--resize_size', type=int, nargs=2, default=[256, 256],
                        help='Resize before cropping (Lee: 256x256)')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[224, 224],
                        help='Random/center crop size (Lee: 224x224)')

    # Training hyper-parameters (following Lee et al.)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Base learning rate (Lee: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (Lee: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (Lee: 0.0005)')
    parser.add_argument('--step_size', type=int, default=30,
                        help='StepLR step size in epochs')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='StepLR gamma (Lee: 0.1)')

    # Distributed
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)

    # Misc
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--auc_bootstrap', type=int, default=2000)
    parser.add_argument('--auc_ci', type=float, default=0.95)
    parser.add_argument('--pretrained_path', type=str,
                        default=str(repo_root / 'checkpoints' / 'vgg16-397923af.pth'),
                        help='Path to ImageNet pre-trained VGG-16 weights')

    args = parser.parse_args()

    # ---- distributed setup ------------------------------------------------
    is_distributed, rank, world_size, local_rank = setup_distributed()

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print(f"Device: {device}")
        if is_distributed:
            print(f"Distributed training: {world_size} GPUs")

    # ---- output directory --------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.save_path = Path(args.save_root) / f"lee_vgg_{timestamp}"
    if rank == 0:
        args.save_path.mkdir(parents=True, exist_ok=True)
        print(f"Results -> {args.save_path}")

    # ---- datasets ----------------------------------------------------------
    resize_sz = tuple(args.resize_size)
    crop_sz = tuple(args.crop_size)

    train_transform = ImageTransforms(crop_size=crop_sz, training=True)
    val_transform = ImageTransforms(crop_size=crop_sz, training=False)

    train_dataset = LiverFibrosisDataset(
        args.data_root, 'train', train_transform,
        resize_size=resize_sz, rank=rank)
    val_dataset = LiverFibrosisDataset(
        args.data_root, 'val', val_transform,
        resize_size=resize_sz, rank=rank)

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.num_workers,
            pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            sampler=val_sampler, num_workers=args.num_workers,
            pin_memory=True)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

    # ---- model (VGG-16 without BN, matching Lee et al.) --------------------
    if rank == 0:
        print("\nBuilding VGG-16 model (no BatchNorm, following Lee et al.)")
    vgg = models.vgg16(weights=None)
    in_features = vgg.classifier[-1].in_features
    vgg.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    if os.path.isfile(args.pretrained_path):
        try:
            ckpt = torch.load(args.pretrained_path, map_location='cpu')
            state_dict = (ckpt['state_dict'] if isinstance(ckpt, dict)
                          and 'state_dict' in ckpt else ckpt)

            model_state = vgg.state_dict()
            filtered = {k: v for k, v in state_dict.items()
                        if k in model_state and model_state[k].shape == v.shape}
            skipped = len(state_dict) - len(filtered)
            load_msg = vgg.load_state_dict(filtered, strict=False)
            if rank == 0:
                print(f"  Loaded {len(filtered)} params from {args.pretrained_path}")
                print(f"  Skipped {skipped} mismatched keys")
                if load_msg.missing_keys:
                    print(f"  Missing keys: {len(load_msg.missing_keys)} "
                          f"(classifier head, expected)")
        except Exception as e:
            if rank == 0:
                print(f"  Failed to load weights, using random init: {e}")
    else:
        if rank == 0:
            print(f"  Pre-trained weights not found: {args.pretrained_path}")
            print(f"  Using random initialization")

    model = vgg.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if rank == 0:
        n_params = sum(p.numel() for p in
                       (model.module if hasattr(model, 'module')
                        else model).parameters())
        print(f"  Parameters: {n_params:,}")

    # ---- train -------------------------------------------------------------
    training_results = train_model(
        model, train_loader, val_loader, device, args, rank, world_size)

    # ---- final evaluation (rank 0 only) ------------------------------------
    if rank == 0:
        print("\nLoading best model for final evaluation ...")
        best_state = torch.load(args.save_path / 'best_model.pth')
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)

        # 4-class evaluation
        print("\n=== Four-class evaluation ===")
        val_results = evaluate_multiclass(model, val_loader, device)
        print(f"Accuracy: {val_results['accuracy']:.4f} | "
              f"Balanced: {val_results['balanced_accuracy']:.4f}")
        print(f"F1 (w/mi/ma): {val_results['f1_weighted']:.4f} / "
              f"{val_results['f1_micro']:.4f} / {val_results['f1_macro']:.4f}")
        print(f"Kappa: {val_results['kappa']:.4f} | "
              f"MCC: {val_results['mcc']:.4f}")
        print(f"Per-class AUC: "
              f"{[f'{a:.4f}' for a in val_results['auc_scores']]}")

        # Binary-pair evaluation
        print("\n=== Binary-pair evaluation ===")
        binary_results = evaluate_binary_pairs(
            model, val_loader, device,
            n_bootstrap=args.auc_bootstrap, ci_alpha=args.auc_ci,
            seed=args.seed)

        # Persist results
        all_results = {
            'multiclass': val_results,
            'binary_pairs': binary_results,
            'training_history': training_results,
        }

        def _np_convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        with open(args.save_path / 'results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=_np_convert)

        binary_df = pd.DataFrame([
            {**metrics, 'pair': pair_name}
            for pair_name, metrics in binary_results.items()
        ])
        binary_df.to_csv(args.save_path / 'binary_results.csv', index=False)

        print("\nBinary-pair summary:")
        cols = ['pair', 'auc', 'auc_ci_lower', 'auc_ci_upper', 'accuracy',
                'balanced_accuracy', 'precision', 'recall', 'f1',
                'sensitivity', 'specificity', 'kappa', 'mcc', 'n_samples']
        existing = [c for c in cols if c in binary_df.columns]
        print(binary_df[existing].to_string(index=False))

        print(f"\nTraining complete. Results saved to: {args.save_path}")

    if is_distributed:
        cleanup_distributed()


if __name__ == '__main__':
    main()

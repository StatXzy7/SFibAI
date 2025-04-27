import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import label_binarize
from datetime import datetime

# Import custom modules
from config import Config
from data.dataset import SchistosomiasisDataset
from data.transforms import ImageTransforms
from utils.models import create_model

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--model_paths", nargs='+', required=True, help="Paths to model weights.")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone network name.")
    parser.add_argument("--num_classes", type=int, default=36, help="Number of classes.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for computation.")
    parser.add_argument("--root_dirs", nargs='+', default=Config.ROOT_DIRS, help="Root directories for test data.")
    parser.add_argument("--save_dir", type=str, default="./eval_results", help="Directory to save results.")
    parser.add_argument("--shape", nargs='+', type=int, default=Config.IMAGE_SIZE, help="Input image shape.")
    parser.add_argument("--mode", type=str, default="val", help="Dataset mode ('val' or 'test').")
    parser.add_argument("--crop_mode", type=str, default="none", choices=['none', 'fixed', 'random', 'mixed'], help="Crop mode.")
    return parser.parse_args()

def evaluate_model(model, data_loader, device, save_dir, model_name="model", log_file="results.log"):
    model.eval()
    
    # Initialize results storage
    total_preds = []
    total_labels = []
    total_probs = []  # For AUC computation

    # Process data in batches
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Model output
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.sum(probs * torch.arange(36, device=device), dim=1)
            preds = torch.round(preds).long()

            # Store results
            total_preds.append(preds.cpu())
            total_labels.append(labels.cpu())
            total_probs.append(probs.cpu())

    # Concatenate all results
    all_preds = torch.cat(total_preds, dim=0)
    all_labels = torch.cat(total_labels, dim=0)
    all_probs = torch.cat(total_probs, dim=0)

    # Save results to CSV
    results_df = pd.DataFrame({
        'Predicted Value': all_preds.numpy(),
        'True Value': all_labels.numpy()
    })
    output_file = os.path.join(save_dir, f"{model_name}_results.csv")
    results_df.to_csv(output_file, index=False)

    # Calculate Accuracy metrics
    strict_acc = (all_preds == all_labels).sum().item() / all_labels.shape[0]
    coarse3_acc = (torch.abs(all_preds - all_labels) <= 3).sum().item() / all_labels.shape[0]
    coarse5_acc = (torch.abs(all_preds - all_labels) <= 5).sum().item() / all_labels.shape[0]
    
    # Calculate MAE (Mean Absolute Error)
    mae = torch.abs(all_preds.float() - all_labels.float()).mean().item() / 10.0  # Divide by 10 to convert back to actual fibrosis degree

    # Boundary accuracy
    label_boundary = all_labels // 10
    pred_boundary = all_preds // 10
    boundary_acc = (label_boundary == pred_boundary).sum().item() / all_labels.shape[0]

    with open(log_file, 'a') as f:
        f.write(f"=== Results for {model_name} ===\n")
        f.write(f"Total Samples: {all_labels.shape[0]}\n")
        f.write(f"=== Fine-grained Evaluation Metrics ===\n")
        f.write(f"1) Absolute accuracy   : {strict_acc:.3f}\n")
        f.write(f"2) ±0.3 accuracy       : {coarse3_acc:.3f}\n")
        f.write(f"3) ±0.5 accuracy       : {coarse5_acc:.3f}\n")
        f.write(f"4) MAE                 : {mae:.3f}\n")
        f.write(f"\n=== Clinical Grade Evaluation Metrics ===\n")
        f.write(f"1) Boundary grade acc  : {boundary_acc:.3f}\n")

    # Plot confusion matrices
    # 36-class confusion matrix (proportions)
    cm_36 = confusion_matrix(all_labels.numpy(), all_preds.numpy(), 
                            labels=list(range(36)), normalize='true')
    plot_confusion_matrix(cm_36, [f"{i/10:.1f}" for i in range(36)], 
                         os.path.join(save_dir, f"confusion_matrix_36_{model_name}.pdf"))

    # 4-class confusion matrix (proportions)
    boundary_labels = all_labels // 10  # Map 0-35 to 0-3
    boundary_preds = all_preds // 10
    cm_4 = confusion_matrix(boundary_labels.numpy(), boundary_preds.numpy(), 
                           labels=[0, 1, 2, 3], normalize='true')
    plot_confusion_matrix(cm_4, ['F0', 'F1', 'F2', 'F3'], 
                         os.path.join(save_dir, f"confusion_matrix_4_{model_name}.pdf"),
                         title="Clinical Grade Confusion Matrix",
                         show_values=True,
                         fmt='.2f')  # Show proportions with 2 decimal places

    # Write 4-class confusion matrix results to log
    with open(log_file, 'a') as f:
        f.write("\n=== Clinical Grade Confusion Matrix ===\n")
        f.write("True\\Pred    F0      F1      F2      F3\n")
        for i, grade in enumerate(['F0', 'F1', 'F2', 'F3']):
            f.write(f"{grade:<8}    {cm_4[i][0]:.3f}  {cm_4[i][1]:.3f}  {cm_4[i][2]:.3f}  {cm_4[i][3]:.3f}\n")

    # Calculate 4-class prediction probabilities
    boundary_probs = torch.zeros((all_probs.shape[0], 4))  # Store probabilities for 4 grades
    for i in range(36):
        grade_idx = i // 10  # Map 0-35 to 0-3
        boundary_probs[:, grade_idx] += all_probs[:, i]

    # Calculate ROC-AUC for three clinically significant classification thresholds
    fpr, tpr, roc_auc = {}, {}, {}
    clinical_thresholds = [
        ('F0 vs F1-F3', lambda x: x == 0, lambda x: x > 0),  # Presence of fibrosis
        ('F0-F1 vs F2-F3', lambda x: x < 2, lambda x: x >= 2),  # Significant fibrosis
        ('F0-F2 vs F3', lambda x: x < 3, lambda x: x == 3),  # Severe fibrosis
    ]
    
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curves for each clinical threshold
    boundary_labels_np = boundary_labels.numpy()
    for idx, (threshold_name, neg_condition, pos_condition) in enumerate(clinical_thresholds):
        # Create binary labels
        binary_labels = np.zeros(len(boundary_labels_np))
        binary_labels[pos_condition(boundary_labels_np)] = 1
        
        # Calculate corresponding prediction probabilities
        if threshold_name == 'F0 vs F1-F3':
            pred_probs = 1 - boundary_probs[:, 0].numpy()  # Total probability of F1-F3
        elif threshold_name == 'F0-F1 vs F2-F3':
            pred_probs = boundary_probs[:, 2:].sum(dim=1).numpy()  # Total probability of F2-F3
        else:  # F0-F2 vs F3
            pred_probs = boundary_probs[:, 3].numpy()  # Probability of F3
        
        # Calculate ROC curve and AUC
        fpr[idx], tpr[idx], _ = roc_curve(binary_labels, pred_probs)
        roc_auc[idx] = auc(fpr[idx], tpr[idx])
        
        # Plot ROC curve
        plt.plot(fpr[idx], tpr[idx], 
                label=f'{threshold_name} (AUC = {roc_auc[idx]:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
    plt.title('Clinical Threshold ROC Curves', fontsize=24, pad=20)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"clinical_roc_auc_{model_name}.pdf"))
    plt.close()

    # Write AUC results to log
    with open(log_file, 'a') as f:
        f.write("\n=== Clinical Threshold ROC-AUC Scores ===\n")
        for idx, (threshold_name, _, _) in enumerate(clinical_thresholds):
            f.write(f"{threshold_name} AUC: {roc_auc[idx]:.3f}\n")
        f.write(f"Average Clinical AUC: {np.mean(list(roc_auc.values())):.3f}\n")

def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix", show_values=False, fmt='.2f'):
    plt.figure(figsize=(10, 8))
    if show_values:
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, square=True,
                    annot_kws={'size': 18})
    else:
        sns.heatmap(cm, annot=False, cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, square=True)
    
    plt.title(title, fontsize=24, pad=20)
    plt.xlabel("Predicted Label", fontsize=18)
    plt.ylabel("True Label", fontsize=18)
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    transform = ImageTransforms(shape=args.shape, training=False)
    dataset = SchistosomiasisDataset(root_dirs=args.root_dirs, mode=args.mode, transform=transform, crop_mode=args.crop_mode)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    for model_path in args.model_paths:
        model = create_model(backbone=args.backbone, num_classes=args.num_classes)
        model.to(device)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Model path not found: {model_path}")
            continue
        
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        evaluate_model(model, data_loader, device, save_dir, model_name=model_name, log_file=os.path.join(save_dir, f"results_{model_name}.log"))

if __name__ == "__main__":
    main()

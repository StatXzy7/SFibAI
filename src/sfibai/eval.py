import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# Import custom modules
from config import Config
from data.dataset import SchistosomiasisDataset
from data.transforms import ImageTransforms
from utils.models import create_model
from utils.binary_evaluation import evaluate_binary_scenarios
from utils.scoring import (
    class_score_values,
    clinical_grade_from_scores,
    expected_index_from_logits,
    expected_score_from_logits,
    label_indices_to_scores,
)

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
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Number of bootstrap resamples for binary AUC confidence intervals.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used for bootstrap confidence intervals.")
    return parser.parse_args()

def evaluate_model(
    model,
    data_loader,
    device,
    save_dir,
    model_name="model",
    log_file="results.log",
    n_bootstrap=2000,
    seed=42,
):
    model.eval()
    
    # Initialize results storage
    total_preds = []
    total_pred_scores = []
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
            pred_scores = expected_score_from_logits(logits)
            preds = torch.round(expected_index_from_logits(logits)).long().clamp(0, 35)

            # Store results
            total_preds.append(preds.cpu())
            total_pred_scores.append(pred_scores.cpu())
            total_labels.append(labels.cpu())
            total_probs.append(probs.cpu())

    # Concatenate all results
    all_preds = torch.cat(total_preds, dim=0)
    all_pred_scores = torch.cat(total_pred_scores, dim=0)
    all_labels = torch.cat(total_labels, dim=0)
    all_label_scores = label_indices_to_scores(all_labels)
    all_probs = torch.cat(total_probs, dim=0)

    # Save results to CSV
    results_df = pd.DataFrame({
        'Predicted Score': all_pred_scores.numpy(),
        'True Score': all_label_scores.numpy(),
        'Predicted Class Index': all_preds.numpy(),
        'True Class Index': all_labels.numpy()
    })
    output_file = os.path.join(save_dir, f"{model_name}_results.csv")
    results_df.to_csv(output_file, index=False)

    # Calculate Accuracy metrics
    strict_acc = (all_preds == all_labels).sum().item() / all_labels.shape[0]
    coarse3_acc = (torch.abs(all_pred_scores - all_label_scores) <= 0.3).sum().item() / all_labels.shape[0]
    coarse5_acc = (torch.abs(all_pred_scores - all_label_scores) <= 0.5).sum().item() / all_labels.shape[0]
    
    # Calculate MAE (Mean Absolute Error)
    mae = torch.abs(all_pred_scores - all_label_scores).mean().item()

    # Boundary accuracy
    label_boundary = clinical_grade_from_scores(all_label_scores)
    pred_boundary = clinical_grade_from_scores(all_pred_scores)
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
    boundary_labels = clinical_grade_from_scores(all_label_scores)
    boundary_preds = clinical_grade_from_scores(all_pred_scores)
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
    class_grades = clinical_grade_from_scores(class_score_values(36)).long()
    for i, grade_idx in enumerate(class_grades.tolist()):
        boundary_probs[:, grade_idx] += all_probs[:, i]

    binary_df, roc_curves = evaluate_binary_scenarios(
        labels4=boundary_labels.numpy(),
        probs4=boundary_probs.numpy(),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    binary_csv = os.path.join(save_dir, f"binary_metrics_{model_name}.csv")
    binary_df.to_csv(binary_csv, index=False)

    plt.figure(figsize=(12, 10))
    for scenario_name, (fpr, tpr, auc_value) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{scenario_name} (AUC = {auc_value:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', alpha=0.8)
    plt.title('Clinical ROC Curves for Nine Binary/Composite Scenarios', fontsize=20, pad=24)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(loc='lower right', fontsize=10, ncol=2, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"clinical_roc_auc_{model_name}.pdf"), dpi=600)
    plt.close()

    with open(log_file, 'a') as f:
        f.write("\n=== Clinical Binary/Composite Scenario Metrics ===\n")
        f.write(f"Bootstrap resamples: {n_bootstrap}; seed: {seed}\n")
        f.write(f"CSV: {binary_csv}\n")
        if not binary_df.empty:
            display_cols = [
                "scenario",
                "auc",
                "auc_ci_lower",
                "auc_ci_upper",
                "sensitivity",
                "specificity",
                "accuracy",
                "precision",
                "f1",
                "kappa",
                "n_samples",
            ]
            f.write(binary_df[display_cols].to_string(index=False))
            f.write("\n")
            f.write(f"Average Clinical AUC: {binary_df['auc'].mean():.3f}\n")

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
        evaluate_model(
            model,
            data_loader,
            device,
            save_dir,
            model_name=model_name,
            log_file=os.path.join(save_dir, f"results_{model_name}.log"),
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )

if __name__ == "__main__":
    main()

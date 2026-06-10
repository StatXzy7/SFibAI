import torch
import numpy as np
import scipy.stats
from torch.cuda.amp import autocast
from tqdm import tqdm
from .visualization import plot_confusion_matrix
from .scoring import (
    clinical_grade_from_scores,
    expected_index_from_logits,
    expected_score_from_logits,
    label_indices_to_scores,
)

def generate_soft_labels(true_labels, num_classes=36, std_dev=0.8, device='cuda'):
    soft_labels = torch.zeros(
        (len(true_labels), num_classes), dtype=torch.float32, device=device)

    for i, true_label in enumerate(true_labels):
        soft_labels[i, true_label] = 0.5  # Peak at the true label

        # Generate Gaussian distribution around the true label
        for j in range(-4, 5):
            if 0 <= true_label + j < num_classes:
                soft_labels[i, true_label + j] = scipy.stats.norm.pdf(j, scale=std_dev)

        # Normalize to make the distribution sum to 1
        soft_labels[i, :] /= torch.sum(soft_labels[i, :])

    return soft_labels

def validate(model, test_loader, device, save_folder='.', save_mat=True, epoch=0):
    model.eval()
    correct_strict = 0
    correct_coarse1 = 0
    correct_coarse2 = 0
    total = 0
    corrects = {0: 0, 1: 0, 2: 0, 3: 0}
    totals = {0: 0, 1: 0, 2: 0, 3: 0}
    pred_total = {0: 0, 1: 0, 2: 0, 3: 0}
    y_total = None
    y_hat_total = None
    thr1 = 0.3
    thr2 = 0.5
    err = 0
    
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            with autocast():
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)

            y_score = label_indices_to_scores(y)
            y_hat_score = expected_score_from_logits(y_hat)
            y_hat = torch.round(expected_index_from_logits(y_hat)).long().clamp(0, 35)
            y_grade = clinical_grade_from_scores(y_score)
            y_hat_grade = clinical_grade_from_scores(y_hat_score)

            correct_strict += (y_hat == y).sum().item()
            
            if y_total is None:
                y_total = y.detach().cpu()
                y_hat_total = y_hat.detach().cpu()
            else:
                y_total = torch.cat((y_total, y.detach().cpu()))
                y_hat_total = torch.cat((y_hat_total, y_hat.detach().cpu()))

            for grade in range(4):
                corrects[grade] += ((y_hat_grade == grade) & (y_grade == grade)).sum().item()
                totals[grade] += (y_grade == grade).sum().item()
                pred_total[grade] += (y_hat_grade == grade).sum().item()

            correct_coarse1 += (torch.abs(y_score - y_hat_score) <= thr1).sum().item()
            correct_coarse2 += (torch.abs(y_score - y_hat_score) <= thr2).sum().item()
            total += y_hat.shape[0]
            err += torch.abs(y_score - y_hat_score).sum().item()

    print(f'accuracy_strict: {correct_strict / total:.3f}, 0.3 accuracy: {correct_coarse1 / total:.3f}, 0.5 accuracy: {correct_coarse2 / total:.3f}, mae: {err/total:.3f}')
    
    print('N:', end=' ')
    for i in range(4):
        print(f'{totals[i]}', end=',')
    print()
    
    print('R:', end=' ')
    for i in range(4):
        if totals[i] == 0:
            print('nan', end=',')
        else:
            print(f'{corrects[i] / totals[i]:.2f}', end=',')
    print()
    
    print('P:', end=' ')
    for i in range(4):
        if pred_total[i] == 0:
            print('nan', end=',')
        else:
            print(f'{corrects[i] / pred_total[i]:.2f}', end=',')
    print()

    if save_mat:
        y_total = [str(round(x/10, 1)) for x in y_total.cpu().numpy()]
        y_hat_total = y_hat_total.cpu().numpy()
        y_hat_total = [str(round(int(x)/10, 1)) for x in y_hat_total]
        plot_confusion_matrix(y_total, y_hat_total, save_folder, epoch)

    model.train()
    return correct_coarse1 / total, correct_coarse2 / total, err/total

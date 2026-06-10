import torch


CLINICAL_THRESHOLDS = (0.5, 1.5, 2.5)


def class_score_values(num_classes=36, device=None, dtype=torch.float32):
    return torch.arange(num_classes, device=device, dtype=dtype) * 0.1


def expected_score_from_logits(logits):
    probs = torch.softmax(logits, dim=1)
    score_values = class_score_values(
        logits.size(1), device=logits.device, dtype=probs.dtype)
    return torch.sum(probs * score_values, dim=1)


def expected_index_from_logits(logits):
    probs = torch.softmax(logits, dim=1)
    index_values = torch.arange(logits.size(1), device=logits.device, dtype=probs.dtype)
    return torch.sum(probs * index_values, dim=1)


def label_indices_to_scores(labels):
    return labels.float() * 0.1


def clinical_grade_from_scores(scores):
    thresholds = torch.tensor(
        CLINICAL_THRESHOLDS, device=scores.device, dtype=scores.dtype)
    return torch.bucketize(scores, thresholds, right=True)


def clinical_grade_from_indices(indices):
    return clinical_grade_from_scores(indices.float() * 0.1)

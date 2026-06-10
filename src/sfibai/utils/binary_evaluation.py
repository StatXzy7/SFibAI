from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_curve,
)


@dataclass(frozen=True)
class BinaryScenario:
    name: str
    negative_grades: tuple[int, ...]
    positive_grades: tuple[int, ...]

    @property
    def file_stem(self) -> str:
        return (
            self.name.replace(" ", "_")
            .replace("/", "_")
            .replace("–", "-")
            .replace("--", "-")
        )


BINARY_SCENARIOS = (
    BinaryScenario("Grade 0 vs Grade 1", (0,), (1,)),
    BinaryScenario("Grade 0 vs Grade 2", (0,), (2,)),
    BinaryScenario("Grade 0 vs Grade 3", (0,), (3,)),
    BinaryScenario("Grade 1 vs Grade 2", (1,), (2,)),
    BinaryScenario("Grade 1 vs Grade 3", (1,), (3,)),
    BinaryScenario("Grade 2 vs Grade 3", (2,), (3,)),
    BinaryScenario("Grade 0 vs Grades 1-3", (0,), (1, 2, 3)),
    BinaryScenario("Grade 1 vs Grades 2-3", (1,), (2, 3)),
    BinaryScenario("Grades 1-2 vs Grade 3", (1, 2), (3,)),
)


def conditional_positive_probability(
    probs4: np.ndarray,
    negative_grades: Iterable[int],
    positive_grades: Iterable[int],
) -> np.ndarray:
    negative = np.asarray(list(negative_grades), dtype=int)
    positive = np.asarray(list(positive_grades), dtype=int)
    p_negative = probs4[:, negative].sum(axis=1)
    p_positive = probs4[:, positive].sum(axis=1)
    denominator = p_negative + p_positive
    return np.divide(
        p_positive,
        denominator,
        out=np.zeros_like(p_positive, dtype=float),
        where=denominator > 0,
    )


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.5

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = auc(fpr, tpr)

    rng = np.random.RandomState(seed)
    bootstrapped = []
    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        sample_idx = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[sample_idx])) < 2:
            continue
        sample_fpr, sample_tpr, _ = roc_curve(y_true[sample_idx], y_score[sample_idx])
        bootstrapped.append(auc(sample_fpr, sample_tpr))

    if not bootstrapped:
        return auc_value, auc_value, auc_value

    lower_q = (1.0 - confidence) / 2.0 * 100.0
    upper_q = (1.0 + confidence) / 2.0 * 100.0
    return (
        auc_value,
        float(np.percentile(bootstrapped, lower_q)),
        float(np.percentile(bootstrapped, upper_q)),
    )


def evaluate_binary_scenarios(
    labels4: np.ndarray,
    probs4: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray, float]]]:
    labels4 = np.asarray(labels4, dtype=int)
    probs4 = np.asarray(probs4, dtype=float)
    rows = []
    curves = {}

    for scenario in BINARY_SCENARIOS:
        scenario_grades = scenario.negative_grades + scenario.positive_grades
        mask = np.isin(labels4, scenario_grades)
        if not np.any(mask):
            continue

        y_true_grade = labels4[mask]
        y_true = np.isin(y_true_grade, scenario.positive_grades).astype(int)
        y_score = conditional_positive_probability(
            probs4[mask],
            scenario.negative_grades,
            scenario.positive_grades,
        )
        y_pred = (y_score >= 0.5).astype(int)

        auc_value, auc_lower, auc_upper = bootstrap_auc_ci(
            y_true, y_score, n_bootstrap=n_bootstrap, seed=seed
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        rows.append(
            {
                "scenario": scenario.name,
                "negative_grades": ",".join(f"F{grade}" for grade in scenario.negative_grades),
                "positive_grades": ",".join(f"F{grade}" for grade in scenario.positive_grades),
                "auc": auc_value,
                "auc_ci_lower": auc_lower,
                "auc_ci_upper": auc_upper,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "precision": precision,
                "f1": f1,
                "kappa": cohen_kappa_score(y_true, y_pred),
                "mcc": matthews_corrcoef(y_true, y_pred)
                if len(np.unique(y_true)) > 1
                else 0.0,
                "n_samples": int(len(y_true)),
                "n_negative": int(np.sum(y_true == 0)),
                "n_positive": int(np.sum(y_true == 1)),
            }
        )

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            curves[scenario.name] = (fpr, tpr, auc_value)

    return pd.DataFrame(rows), curves

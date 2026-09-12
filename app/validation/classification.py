"""
classification.py — Rival energy-bucket classification metrics.

Compares the estimator's argmax bucket (LOW/MEDIUM/HIGH) against the
ground-truth bucket derived from the hidden SoC using the same thresholds.

Class order for confusion matrix: [LOW, MEDIUM, HIGH] (index 0, 1, 2).
"""

from __future__ import annotations

from typing import List

from app.validation.models import ClassificationValidation

_CLASSES = ["LOW", "MEDIUM", "HIGH"]
_CLASS_IDX = {c: i for i, c in enumerate(_CLASSES)}


def _gt_bucket(soc_mj: float, low_edge: float, high_edge: float) -> str:
    if soc_mj < low_edge:
        return "LOW"
    if soc_mj > high_edge:
        return "HIGH"
    return "MEDIUM"


def compute_classification_metrics(
    trace_steps: List[dict],
    low_edge_mj: float = 3.0,
    high_edge_mj: float = 6.0,
) -> ClassificationValidation:
    """
    Compute accuracy, per-class F1, and confusion matrix for bucket classification.

    Only steps with both 'bucket' (predicted) and 'ground_truth_reserve_mj' (float) count.
    """
    y_true: List[str] = []
    y_pred: List[str] = []

    for step in trace_steps:
        gt = step.get("ground_truth_reserve_mj")
        pred = step.get("bucket")
        if gt is None or pred is None:
            continue
        y_true.append(_gt_bucket(float(gt), low_edge_mj, high_edge_mj))
        y_pred.append(pred)

    n = len(y_true)
    if n == 0:
        return ClassificationValidation(
            ground_truth_available=False,
            validation_type="unavailable",
            accuracy=None,
            per_class_f1=None,
            confusion_matrix=None,
            sample_count=0,
            evaluation_note="GROUND TRUTH UNAVAILABLE.",
        )

    # Accuracy
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / n

    # Confusion matrix (3×3, rows=true, cols=predicted)
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for t, p in zip(y_true, y_pred):
        ti = _CLASS_IDX.get(t, 1)
        pi = _CLASS_IDX.get(p, 1)
        cm[ti][pi] += 1

    # Per-class F1 (no sklearn dependency)
    per_class_f1: dict = {}
    for cls in _CLASSES:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1[cls] = round(f1, 4)

    return ClassificationValidation(
        ground_truth_available=True,
        validation_type="controlled_hidden_state",
        accuracy=round(accuracy, 4),
        per_class_f1=per_class_f1,
        confusion_matrix=cm,
        sample_count=n,
        evaluation_note=(
            f"Controlled hidden-state validation: {n} laps. "
            "Compares estimator argmax bucket to ground-truth bucket using same thresholds. "
            "MODELED — NOT MEASURED."
        ),
    )

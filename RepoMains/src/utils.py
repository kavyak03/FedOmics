import json
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


def save_metrics(metrics_dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)


def load_yaml(path: Union[str, Path]):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def standardize_train_val(
    X_train: np.ndarray,
    X_val: np.ndarray,
):
    """
    Z-score using train-only statistics.
    Returns:
        X_train_z, X_val_z, mean, std
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)

    X_train_z = (X_train - mean) / std
    X_val_z = (X_val - mean) / std
    return X_train_z, X_val_z, mean, std


def apply_standardization(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std


def choose_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Choose a decision threshold from training predictions.

    metric:
        - "f1"
        - "balanced_accuracy"
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.5

    candidate_thresholds = np.unique(np.round(y_prob, 4))
    if len(candidate_thresholds) == 0:
        return 0.5

    best_t = 0.5
    best_score = -np.inf

    for t in candidate_thresholds:
        y_pred = (y_prob >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            pos_mask = y_true == 1
            neg_mask = y_true == 0

            tpr = (y_pred[pos_mask] == 1).mean() if pos_mask.any() else 0.0
            tnr = (y_pred[neg_mask] == 0).mean() if neg_mask.any() else 0.0
            score = 0.5 * (tpr + tnr)

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
):
    """
    Returns:
        metrics_dict, fpr, tpr
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": None,
    }

    notes = []

    unique_true = np.unique(y_true)
    if len(unique_true) < 2:
        notes.append("Validation set contains only one class; AUROC is undefined.")
        fpr = np.array([])
        tpr = np.array([])
    else:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            notes.append("AUROC could not be computed: {0}".format(e))

        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
        except Exception as e:
            notes.append("ROC curve could not be computed: {0}".format(e))
            fpr = np.array([])
            tpr = np.array([])

    if len(np.unique(y_pred)) < 2:
        notes.append("Model predicted only one class on validation data.")

    if notes:
        metrics["notes"] = notes

    return metrics, fpr, tpr
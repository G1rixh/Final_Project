"""
Evaluation metrics for multi-label classification.

Primary   : mean AUROC across 14 labels.
Secondary : per-label AUROC + PR-AUC, micro/macro F1, precision, recall.
Optional  : Expected Calibration Error (ECE).

Robust to labels that have zero positives in a given split (AUROC undefined ->
reported as NaN and excluded from the mean, with a warning count).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score,
)

import config as C


def _safe_auroc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def _safe_ap(y_true, y_score):
    if y_true.sum() == 0:
        return np.nan
    return average_precision_score(y_true, y_score)


def per_label_table(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = C.DEFAULT_THRESHOLD) -> pd.DataFrame:
    rows = []
    y_pred = (y_prob >= threshold).astype(int)
    for i, lab in enumerate(C.LABELS):
        rows.append({
            "label": lab,
            "n_pos": int(y_true[:, i].sum()),
            "AUROC": _safe_auroc(y_true[:, i], y_prob[:, i]),
            "PR_AUC": _safe_ap(y_true[:, i], y_prob[:, i]),
            "F1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
        })
    return pd.DataFrame(rows)


def summary_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = C.DEFAULT_THRESHOLD) -> dict:
    tbl = per_label_table(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "mean_AUROC": float(np.nanmean(tbl["AUROC"])),        # PRIMARY metric
        "mean_PR_AUC": float(np.nanmean(tbl["PR_AUC"])),
        "micro_F1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "labels_without_positives": int(tbl["AUROC"].isna().sum()),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE pooled across all label predictions (flattened)."""
    yt = y_true.ravel().astype(float)
    yp = y_prob.ravel().astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(yp)
    for b in range(n_bins):
        m = (yp > bins[b]) & (yp <= bins[b + 1])
        if m.sum() == 0:
            continue
        conf = yp[m].mean()
        acc = yt[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-label threshold maximizing F1 on the validation set."""
    best = np.full(C.NUM_LABELS, C.DEFAULT_THRESHOLD)
    grid = np.linspace(0.05, 0.95, 19)
    for i in range(C.NUM_LABELS):
        if y_true[:, i].sum() == 0:
            continue
        f1s = [f1_score(y_true[:, i], (y_prob[:, i] >= t).astype(int),
                        zero_division=0) for t in grid]
        best[i] = grid[int(np.argmax(f1s))]
    return best

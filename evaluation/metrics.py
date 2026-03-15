from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def _tpr_at_fpr(
    fpr_arr: np.ndarray,
    tpr_arr: np.ndarray,
    thresholds: np.ndarray,
    target_fpr: float,
) -> tuple[float, float, float]:
    valid = np.where(fpr_arr <= target_fpr + 1e-12)[0]
    if len(valid) == 0:
        thr = float(thresholds[0]) if len(thresholds) else 0.0
        return 0.0, 1.0, thr

    best_idx = valid[np.argmax(tpr_arr[valid])]
    tpr = float(tpr_arr[best_idx])
    fnr = 1.0 - tpr
    thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.0
    return tpr, fnr, thr


def _eer_from_roc(fpr_arr: np.ndarray, tpr_arr: np.ndarray) -> float:
    fnr_arr = 1.0 - tpr_arr
    idx = int(np.argmin(np.abs(fpr_arr - fnr_arr)))
    return float((fpr_arr[idx] + fnr_arr[idx]) / 2.0)


def _best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    precision_arr, recall_arr, thresholds = precision_recall_curve(labels, scores)

    if len(thresholds) == 0:
        thr = float(np.median(scores))
        preds = (scores >= thr).astype(int)
        return thr, float(f1_score(labels, preds, zero_division=0))

    f1_arr = (
        2.0 * precision_arr[:-1] * recall_arr[:-1]
        / (precision_arr[:-1] + recall_arr[:-1] + 1e-12)
    )

    best_idx = int(np.nanargmax(f1_arr))
    return float(thresholds[best_idx]), float(f1_arr[best_idx])


def build_curve_frames(scores: np.ndarray, labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    if len(np.unique(labels)) < 2:
        roc_df = pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})
        pr_df = pd.DataFrame({"recall": [0.0, 1.0], "precision": [1.0, 0.0]})
        return roc_df, pr_df

    fpr_arr, tpr_arr, _ = roc_curve(labels, scores, pos_label=1)
    precision_arr, recall_arr, _ = precision_recall_curve(labels, scores)

    roc_df = pd.DataFrame({"fpr": fpr_arr, "tpr": tpr_arr})
    pr_df = pd.DataFrame({"recall": recall_arr, "precision": precision_arr})
    return roc_df, pr_df


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    dataset_name: str = "",
    runtime_sec: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    n_total = int(len(labels))
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())

    if n_total == 0:
        return {"dataset": dataset_name, "error": "no_eval_rows"}

    if n_pos == 0:
        logger.warning("No attack rows for %s.", dataset_name)
        return {"dataset": dataset_name, "error": "no_positives"}

    if n_neg == 0:
        logger.warning("No benign rows for %s.", dataset_name)
        return {"dataset": dataset_name, "error": "no_negatives"}

    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores, pos_label=1)
    auc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))

    tpr0, fnr0, thr0 = _tpr_at_fpr(fpr_arr, tpr_arr, thresholds, 0.0)
    tpr001, fnr001, thr001 = _tpr_at_fpr(fpr_arr, tpr_arr, thresholds, 0.001)

    eer = _eer_from_roc(fpr_arr, tpr_arr)
    best_thr, best_f1 = _best_f1_threshold(scores, labels)

    preds = (scores >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    precision_opt = float(precision_score(labels, preds, zero_division=0))
    recall_opt = float(recall_score(labels, preds, zero_division=0))

    benign_scores = scores[labels == 0]
    attack_scores = scores[labels == 1]

    metrics = {
        "dataset": dataset_name,
        "n_total": n_total,
        "n_benign": n_neg,
        "n_malicious": n_pos,
        "attack_rate": round(float(labels.mean()), 6),
        "AUC": round(auc, 6),
        "AUPRC": round(auprc, 6),
        "EER": round(eer, 6),
        "TPR_at_FPR_0": round(tpr0, 6),
        "FNR_at_FPR_0": round(fnr0, 6),
        "threshold_FPR_0": round(thr0, 6),
        "TPR_at_FPR_0001": round(tpr001, 6),
        "FNR_at_FPR_0001": round(fnr001, 6),
        "threshold_FPR_0001": round(thr001, 6),
        "threshold_opt": round(best_thr, 6),
        "F1_optimal": round(best_f1, 6),
        "Precision_opt": round(precision_opt, 6),
        "Recall_opt": round(recall_opt, 6),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "mean_score_benign": round(float(np.mean(benign_scores)), 6),
        "mean_score_attack": round(float(np.mean(attack_scores)), 6),
        "median_score_benign": round(float(np.median(benign_scores)), 6),
        "median_score_attack": round(float(np.median(attack_scores)), 6),
        "std_score_benign": round(float(np.std(benign_scores)), 6),
        "std_score_attack": round(float(np.std(attack_scores)), 6),
        "max_score": round(float(np.max(scores)), 6),
        "min_score": round(float(np.min(scores)), 6),
    }

    if runtime_sec is not None:
        metrics["runtime_sec"] = round(float(runtime_sec), 4)
        metrics["rows_per_sec"] = round(float(n_total / runtime_sec), 4) if runtime_sec > 0 else 0.0

    if extra:
        metrics.update(extra)

    return metrics


def print_metrics(metrics: dict[str, Any]) -> None:
    if "error" in metrics:
        logger.warning("Metrics unavailable for %s: %s", metrics.get("dataset", "dataset"), metrics["error"])
        return

    logger.info("-" * 60)
    logger.info("Dataset: %s", metrics["dataset"])
    logger.info(
        "Eval rows: %d | Benign: %d | Attack: %d",
        metrics["n_total"],
        metrics["n_benign"],
        metrics["n_malicious"],
    )
    logger.info(
        "AUC: %.4f | AUPRC: %.4f | EER: %.4f",
        metrics["AUC"],
        metrics["AUPRC"],
        metrics["EER"],
    )
    logger.info(
        "TPR@FPR=0: %.4f | TPR@FPR=0.001: %.4f",
        metrics["TPR_at_FPR_0"],
        metrics["TPR_at_FPR_0001"],
    )
    logger.info(
        "F1: %.4f | Precision: %.4f | Recall: %.4f",
        metrics["F1_optimal"],
        metrics["Precision_opt"],
        metrics["Recall_opt"],
    )
    logger.info(
        "TP=%d FP=%d FN=%d TN=%d",
        metrics["TP"],
        metrics["FP"],
        metrics["FN"],
        metrics["TN"],
    )
    if "runtime_sec" in metrics:
        logger.info(
            "Runtime: %.2fs | Rows/sec: %.2f",
            metrics["runtime_sec"],
            metrics["rows_per_sec"],
        )
    logger.info("-" * 60)

"""
evaluation/metrics.py
---------------------
Evaluation metrics for anomaly detection, matching the paper's protocol:

    - TPR (True Positive Rate) at FPR = 0 and FPR = 0.001
    - FNR (False Negative Rate) at FPR = 0 and FPR = 0.001
    - AUC (Area Under the ROC Curve)
    - EER (Equal Error Rate)

Additional standard metrics also computed:
    - Precision, Recall (= TPR), F1-score at optimal threshold
    - Confusion matrix

Paper reference: Mirsky et al., NDSS 2018, Section V-C.
"""

from __future__ import annotations
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def compute_metrics(
    scores:       np.ndarray,
    labels:       np.ndarray,
    dataset_name: str = "",
) -> dict:
    """
    Compute the full evaluation metrics suite.

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Raw anomaly scores from KitNET (higher = more anomalous).
    labels : np.ndarray, shape (N,), dtype int
        Ground-truth labels: 0 = benign, 1 = malicious.

    Returns
    -------
    metrics : dict
        All computed metrics, suitable for JSON serialisation.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    n_pos  = int(labels.sum())
    n_neg  = int((labels == 0).sum())
    n_total = len(labels)

    if n_pos == 0:
        logger.warning("No positive (malicious) samples — metrics undefined.")
        return {"dataset": dataset_name, "error": "no_positives"}
    if n_neg == 0:
        logger.warning("No negative (benign) samples — metrics undefined.")
        return {"dataset": dataset_name, "error": "no_negatives"}

    # ------------------------------------------------------------------ #
    # ROC curve
    # ------------------------------------------------------------------ #
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores, pos_label=1)
    auc = float(roc_auc_score(labels, scores))

    # ------------------------------------------------------------------ #
    # TPR / FNR at fixed FPR targets
    # ------------------------------------------------------------------ #
    def tpr_at_fpr(target_fpr: float) -> tuple[float, float, float]:
        """
        Find the highest TPR achievable at FPR ≤ target_fpr.
        Returns (tpr, fnr, threshold).
        """
        # Indices where FPR ≤ target_fpr
        valid = np.where(fpr_arr <= target_fpr + 1e-10)[0]
        if len(valid) == 0:
            return 0.0, 1.0, float(thresholds[0])
        # Among those, pick the one with highest TPR
        best_idx = valid[np.argmax(tpr_arr[valid])]
        tpr_ = float(tpr_arr[best_idx])
        fnr_ = 1.0 - tpr_
        thr_ = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.0
        return tpr_, fnr_, thr_

    tpr0,   fnr0,   thr0   = tpr_at_fpr(0.0)
    tpr001, fnr001, thr001 = tpr_at_fpr(0.001)

    # ------------------------------------------------------------------ #
    # Equal Error Rate (EER): point where FPR ≈ FNR
    # ------------------------------------------------------------------ #
    fnr_arr = 1.0 - tpr_arr
    diff    = np.abs(fpr_arr - fnr_arr)
    eer_idx = int(np.argmin(diff))
    eer     = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0)

    # ------------------------------------------------------------------ #
    # F1 at optimal threshold (maximise F1)
    # ------------------------------------------------------------------ #
    best_f1  = 0.0
    best_thr = 0.0
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1  = f1
            best_thr = float(thr)

    # Confusion matrix at best F1 threshold
    preds  = (scores >= best_thr).astype(int)
    tp     = int(((preds == 1) & (labels == 1)).sum())
    fp     = int(((preds == 1) & (labels == 0)).sum())
    fn     = int(((preds == 0) & (labels == 1)).sum())
    tn     = int(((preds == 0) & (labels == 0)).sum())
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "dataset":           dataset_name,
        "n_total":           n_total,
        "n_malicious":       n_pos,
        "n_benign":          n_neg,
        # Paper metrics
        "TPR_at_FPR_0":     round(tpr0,   6),
        "FNR_at_FPR_0":     round(fnr0,   6),
        "threshold_FPR_0":  round(thr0,   6),
        "TPR_at_FPR_0001":  round(tpr001, 6),
        "FNR_at_FPR_0001":  round(fnr001, 6),
        "threshold_FPR_0001": round(thr001, 6),
        "AUC":              round(auc, 6),
        "EER":              round(eer, 6),
        # Additional
        "F1_optimal":       round(best_f1, 6),
        "Precision_opt":    round(prec,    6),
        "Recall_opt":       round(rec,     6),
        "threshold_opt":    round(best_thr, 6),
        "TP":  tp, "FP":  fp,
        "FN":  fn, "TN":  tn,
    }


def print_metrics(m: dict) -> None:
    """Pretty-print a metrics dict to the logger."""
    if "error" in m:
        logger.warning("Metrics unavailable: %s", m["error"])
        return

    logger.info("─" * 50)
    logger.info("Dataset      : %s", m.get("dataset", "—"))
    logger.info("Total        : %d  |  Malicious: %d  |  Benign: %d",
                m["n_total"], m["n_malicious"], m["n_benign"])
    logger.info("─" * 50)
    logger.info("TPR @ FPR=0      : %.4f   (FNR=%.4f)",
                m["TPR_at_FPR_0"], m["FNR_at_FPR_0"])
    logger.info("TPR @ FPR=0.001  : %.4f   (FNR=%.4f)",
                m["TPR_at_FPR_0001"], m["FNR_at_FPR_0001"])
    logger.info("AUC              : %.4f", m["AUC"])
    logger.info("EER              : %.4f", m["EER"])
    logger.info("F1 (optimal thr) : %.4f   (P=%.4f  R=%.4f)",
                m["F1_optimal"], m["Precision_opt"], m["Recall_opt"])
    logger.info("Confusion  TP=%-6d FP=%-6d FN=%-6d TN=%d",
                m["TP"], m["FP"], m["FN"], m["TN"])
    logger.info("─" * 50)

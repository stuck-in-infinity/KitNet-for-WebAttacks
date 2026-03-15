"""
evaluation/plot_results.py
---------------------------
Plotting utilities to visualise Kitsune evaluation results.

Generates:
    1. ROC curves (one per dataset / attack type)
    2. RMSE anomaly score distributions (benign vs malicious)
    3. Summary bar chart of AUC across all datasets
    4. TPR comparison table figure

Usage
-----
    python -m evaluation.plot_results \
        --results_dir ./results/phase1 \
        --output_dir  ./results/plots
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ======================================================================= #
#  ROC curves
# ======================================================================= #

def plot_roc_curves(
    results_dir: Path,
    output_dir:  Path,
    title_prefix: str = "Kitsune"
) -> None:
    """
    Plot one ROC curve per dataset sub-directory in results_dir.
    Each sub-directory must contain scores.npy and labels.npy.
    """
    datasets = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir()
        and (d / "scores.npy").exists()
        and (d / "labels.npy").exists()
    ])

    if not datasets:
        logger.warning("No result directories found in %s", results_dir)
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colours = plt.cm.tab10(np.linspace(0, 0.9, len(datasets)))

    for ds_dir, colour in zip(datasets, colours):
        scores = np.load(ds_dir / "scores.npy")
        labels = np.load(ds_dir / "labels.npy")

        if labels.sum() == 0:
            logger.warning("No malicious samples in %s — skipping ROC.", ds_dir.name)
            continue

        fpr, tpr, _ = roc_curve(labels, scores)

        # Load AUC from saved metrics
        metrics_path = ds_dir / "metrics.json"
        auc_str = ""
        if metrics_path.exists():
            with open(metrics_path) as fh:
                m = json.load(fh)
            auc_str = f" (AUC={m.get('AUC', 0):.3f})"

        ax.plot(fpr, tpr, color=colour,
                label=f"{ds_dir.name}{auc_str}", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"{title_prefix} — ROC Curves", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    out_path = output_dir / "roc_curves.pdf"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curves saved to %s", out_path)


# ======================================================================= #
#  Score distributions
# ======================================================================= #

def plot_score_distributions(
    results_dir: Path,
    output_dir:  Path,
) -> None:
    """
    For each dataset, plot the anomaly score distributions for benign
    vs malicious traffic using log-scale histograms.
    """
    datasets = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir()
        and (d / "scores.npy").exists()
        and (d / "labels.npy").exists()
    ])

    for ds_dir in datasets:
        scores = np.load(ds_dir / "scores.npy")
        labels = np.load(ds_dir / "labels.npy")

        benign_scores  = scores[labels == 0]
        attack_scores  = scores[labels == 1]

        if len(benign_scores) == 0 or len(attack_scores) == 0:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))

        bins = np.logspace(
            np.log10(max(scores.min(), 1e-8)),
            np.log10(scores.max() + 1e-8),
            60
        )

        ax.hist(benign_scores, bins=bins, alpha=0.6,
                color="steelblue", label="Benign", density=True)
        ax.hist(attack_scores, bins=bins, alpha=0.6,
                color="tomato",    label="Malicious", density=True)

        ax.set_xscale("log")
        ax.set_xlabel("Anomaly Score (RMSE)", fontsize=11)
        ax.set_ylabel("Density",              fontsize=11)
        ax.set_title(f"{ds_dir.name} — Score Distribution", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        out_path = output_dir / f"dist_{ds_dir.name}.pdf"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Score dist saved: %s", out_path)


# ======================================================================= #
#  AUC bar chart
# ======================================================================= #

def plot_auc_summary(
    results_dir: Path,
    output_dir:  Path,
    title:       str = "AUC per Attack Type",
) -> None:
    """
    Bar chart of AUC for every dataset found in results_dir.
    """
    names, aucs = [], []
    for ds_dir in sorted(results_dir.iterdir()):
        mp = ds_dir / "metrics.json"
        if mp.exists():
            with open(mp) as fh:
                m = json.load(fh)
            if "AUC" in m:
                names.append(ds_dir.name)
                aucs.append(m["AUC"])

    if not names:
        logger.warning("No metrics.json files found.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.1), 4))
    x = np.arange(len(names))
    bars = ax.bar(x, aucs, color="steelblue", edgecolor="white", width=0.6)

    # Annotate value on each bar
    for bar, val in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_ylim([0, 1.08])
    ax.set_title(title, fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)

    out_path = output_dir / "auc_summary.pdf"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("AUC summary saved to %s", out_path)


# ======================================================================= #
#  CLI
# ======================================================================= #

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot Kitsune evaluation results."
    )
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--output_dir",  type=Path, required=True)
    parser.add_argument("--title_prefix", type=str, default="Kitsune")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curves(args.results_dir, args.output_dir, args.title_prefix)
    plot_score_distributions(args.results_dir, args.output_dir)
    plot_auc_summary(args.results_dir, args.output_dir,
                     title=f"{args.title_prefix} — AUC per Attack Type")


if __name__ == "__main__":
    main()

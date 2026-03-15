from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _result_dirs(results_dir: Path) -> list[Path]:
    return sorted(
        [
            d for d in results_dir.iterdir()
            if d.is_dir()
            and (d / "metrics.json").exists()
            and (d / "scores.csv").exists()
        ]
    )


def load_summary_metrics(results_dir: Path) -> pd.DataFrame:
    rows = []

    for ds_dir in _result_dirs(results_dir):
        with open(ds_dir / "metrics.json") as fh:
            rows.append(json.load(fh))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def plot_auc_summary(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = metrics_df.sort_values("AUC", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=plot_df, x="dataset", y="AUC", ax=ax)
    ax.set_title("KitNET replication: AUROC by dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("AUROC")
    ax.tick_params(axis="x", rotation=35)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_dir / "summary_auc.png", dpi=180)
    plt.close(fig)


def plot_auprc_summary(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = metrics_df.sort_values("AUPRC", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=plot_df, x="dataset", y="AUPRC", ax=ax)
    ax.set_title("KitNET replication: AUPRC by dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("AUPRC")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(output_dir / "summary_auprc.png", dpi=180)
    plt.close(fig)


def plot_runtime_summary(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    if "runtime_sec" not in metrics_df.columns:
        return

    plot_df = metrics_df.sort_values("runtime_sec", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=plot_df, x="dataset", y="runtime_sec", ax=ax)
    ax.set_title("KitNET replication: runtime by dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(output_dir / "summary_runtime.png", dpi=180)
    plt.close(fig)


def plot_combined_roc(results_dir: Path, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    for ds_dir in _result_dirs(results_dir):
        roc_path = ds_dir / "roc_curve.csv"
        if not roc_path.exists():
            continue

        roc_df = pd.read_csv(roc_path)
        with open(ds_dir / "metrics.json") as fh:
            metrics = json.load(fh)

        auc = metrics.get("AUC", np.nan)
        ax.plot(roc_df["fpr"], roc_df["tpr"], label=f"{ds_dir.name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
    ax.set_title("KitNET replication: ROC curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "combined_roc.png", dpi=180)
    plt.close(fig)


def plot_combined_pr(results_dir: Path, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    for ds_dir in _result_dirs(results_dir):
        pr_path = ds_dir / "pr_curve.csv"
        if not pr_path.exists():
            continue

        pr_df = pd.read_csv(pr_path)
        with open(ds_dir / "metrics.json") as fh:
            metrics = json.load(fh)

        auprc = metrics.get("AUPRC", np.nan)
        ax.plot(pr_df["recall"], pr_df["precision"], label=f"{ds_dir.name} (AP={auprc:.3f})")

    ax.set_title("KitNET replication: PR curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(output_dir / "combined_pr.png", dpi=180)
    plt.close(fig)


def plot_score_distributions(results_dir: Path, output_dir: Path) -> None:
    for ds_dir in _result_dirs(results_dir):
        score_df = pd.read_csv(ds_dir / "scores.csv")

        if score_df["label"].nunique() < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            data=score_df,
            x="score",
            hue="label",
            bins=100,
            stat="density",
            common_norm=False,
            element="step",
            ax=ax,
        )
        ax.set_title(f"{ds_dir.name}: score distribution")
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Density")
        ax.legend(["Benign", "Attack"])

        fig.tight_layout()
        fig.savefig(output_dir / f"{ds_dir.name}_score_dist.png", dpi=180)
        plt.close(fig)


def plot_score_timelines(results_dir: Path, output_dir: Path, max_points: int = 50000) -> None:
    for ds_dir in _result_dirs(results_dir):
        score_df = pd.read_csv(ds_dir / "scores.csv")

        if score_df.empty:
            continue

        step = max(1, len(score_df) // max_points)
        plot_df = score_df.iloc[::step].copy()

        fig, ax = plt.subplots(figsize=(14, 4))

        benign = plot_df[plot_df["label"] == 0]
        attack = plot_df[plot_df["label"] == 1]

        ax.scatter(benign["row_index"], benign["score"], s=4, alpha=0.35, label="Benign")
        ax.scatter(attack["row_index"], attack["score"], s=6, alpha=0.50, label="Attack")

        ax.set_title(f"{ds_dir.name}: anomaly score over stream")
        ax.set_xlabel("Row index")
        ax.set_ylabel("Score")
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / f"{ds_dir.name}_timeline.png", dpi=180)
        plt.close(fig)


def make_all_plots(results_dir: str | Path, output_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_summary_metrics(results_dir)
    if metrics_df.empty:
        logger.warning("I could not find any finished result folders in %s", results_dir)
        return

    plot_auc_summary(metrics_df, output_dir)
    plot_auprc_summary(metrics_df, output_dir)
    plot_runtime_summary(metrics_df, output_dir)
    plot_combined_roc(results_dir, output_dir)
    plot_combined_pr(results_dir, output_dir)
    plot_score_distributions(results_dir, output_dir)
    plot_score_timelines(results_dir, output_dir)

    metrics_df.sort_values("AUC", ascending=False).to_csv(
        output_dir / "summary_metrics_sorted.csv",
        index=False,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot KitNET replication results.")
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)

    make_all_plots(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()

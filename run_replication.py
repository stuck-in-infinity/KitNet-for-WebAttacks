"""Replicates KitNET on pre-extracted feature CSV datasets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.feature_mapper import FeatureMapper
from core.kitnet import KitNET
from dataset_reader import (
    EXPECTED_N_FEATURES,
    DatasetPair,
    PairedCSVDatasetReader,
    discover_dataset_pairs,
)
from evaluation.metrics import build_curve_frames, compute_metrics, print_metrics
from evaluation.plot_results import make_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


class KitNETFromExtractedCSV:
    """Runs FeatureMapper + KitNET over precomputed feature rows."""

    def __init__(
        self,
        fm_grace: int = 5000,
        ad_grace: int = 50000,
        max_cluster_size: int = 10,
        n_features: int = EXPECTED_N_FEATURES,
        beta: float = 0.75,
        lr: float = 0.1,
    ):
        self.fm_grace = fm_grace
        self.ad_grace = ad_grace
        self.max_cluster_size = max_cluster_size
        self.n_features = n_features
        self.beta = beta
        self.lr = lr

        self.fm = FeatureMapper(
            n_features=n_features,
            max_cluster_size=max_cluster_size,
        )

        self.ad: KitNET | None = None
        self._fm_fitted = False
        self._n_seen = 0
        self.phi = 0.0

    @property
    def warmup_rows(self) -> int:
        return self.fm_grace + self.ad_grace

    def process(self, features: np.ndarray) -> tuple[float | None, str]:
        row_idx = self._n_seen
        self._n_seen += 1

        if row_idx < self.fm_grace:
            # First phase: only teach the mapper the correlation structure.
            self.fm.update(features)
            return None, "fm_train"

        if not self._fm_fitted:
            # Freeze the mapping before any autoencoder training happens.
            logger.info("Fitting the FeatureMapper after %d rows.", self.fm_grace)
            self.fm.fit()
            self.ad = KitNET(
                cluster_sizes=self.fm.cluster_sizes,
                beta=self.beta,
                lr=self.lr,
            )
            self._fm_fitted = True
            logger.info(
                "FeatureMapper ready: k=%d, cluster sizes=%s",
                self.fm.n_clusters,
                self.fm.cluster_sizes,
            )

        sub_instances = self.fm.transform(features)

        if row_idx < self.warmup_rows:
            # Warmup KitNET but do not score yet; track max phi for thresholding.
            score = float(self.ad.train(sub_instances))
            self.phi = max(self.phi, score)
            return None, "ad_train"

        # Steady-state scoring after both grace periods finish.
        score = float(self.ad.execute(sub_instances))
        return score, "exec"


def sanitize_dataset_name(name: str) -> str:
    return name.replace(" ", "_")


def run_dataset(
    pair: DatasetPair,
    output_dir: Path,
    fm_grace: int,
    ad_grace: int,
    max_cluster_size: int,
    beta: float,
    lr: float,
    expected_n_features: int,
    log_every: int = 100_000,
) -> dict:
    logger.info("=" * 70)
    logger.info("Dataset: %s", pair.name)
    logger.info("Features: %s", pair.features_path.name)
    logger.info("Labels:   %s", pair.labels_path.name)

    reader = PairedCSVDatasetReader(
        features_path=pair.features_path,
        labels_path=pair.labels_path,
        expected_n_features=expected_n_features,
    )

    total_rows = reader.count_rows()

    pipeline = KitNETFromExtractedCSV(
        fm_grace=fm_grace,
        ad_grace=ad_grace,
        max_cluster_size=max_cluster_size,
        n_features=expected_n_features,
        beta=beta,
        lr=lr,
    )

    phase_counts = {"fm_train": 0, "ad_train": 0, "exec": 0}
    score_rows: list[dict] = []

    start_time = time.time()

    for row_idx, features, label in reader:
        score, phase = pipeline.process(features)
        phase_counts[phase] += 1

        if phase == "exec":
            # Only commit rows once the system is fully warmed up.
            score_rows.append(
                {
                    "row_index": row_idx,
                    "label": int(label),
                    "score": float(score),
                }
            )

        if row_idx > 0 and row_idx % log_every == 0:
            logger.info(
                "Row %d/%d | fm=%d ad=%d exec=%d",
                row_idx,
                total_rows,
                phase_counts["fm_train"],
                phase_counts["ad_train"],
                phase_counts["exec"],
            )

    runtime_sec = time.time() - start_time

    if not score_rows:
        logger.warning("Finished %s but there were no eval rows to score.", pair.name)
        return {"dataset": pair.name, "error": "no_eval_rows"}

    scores_df = pd.DataFrame(score_rows)
    scores = scores_df["score"].to_numpy(dtype=np.float64)
    labels = scores_df["label"].to_numpy(dtype=np.int32)

    metrics = compute_metrics(
        scores=scores,
        labels=labels,
        dataset_name=pair.name,
        runtime_sec=runtime_sec,
        extra={
            "original_rows_seen": int(total_rows),
            "FMgrace": int(fm_grace),
            "ADgrace": int(ad_grace),
            "warmup_rows": int(pipeline.warmup_rows),
            "max_cluster_size": int(max_cluster_size),
            "n_clusters": int(pipeline.fm.n_clusters),
            "cluster_sizes": pipeline.fm.cluster_sizes,
            "phase_fm_rows": int(phase_counts["fm_train"]),
            "phase_ad_rows": int(phase_counts["ad_train"]),
            "phase_exec_rows": int(phase_counts["exec"]),
            "phi_train_max": round(float(pipeline.phi), 6),
        },
    )

    print_metrics(metrics)

    dataset_dir = output_dir / sanitize_dataset_name(pair.name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    scores_df.to_csv(dataset_dir / "scores.csv", index=False)
    np.save(dataset_dir / "scores.npy", scores)
    np.save(dataset_dir / "labels.npy", labels)

    roc_df, pr_df = build_curve_frames(scores=scores, labels=labels)
    roc_df.to_csv(dataset_dir / "roc_curve.csv", index=False)
    pr_df.to_csv(dataset_dir / "pr_curve.csv", index=False)

    with open(dataset_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Saved dataset results to %s", dataset_dir)
    return metrics


def build_dataset_list(
    sample_dir: Path | None,
    features_path: Path | None,
    labels_path: Path | None,
) -> list[DatasetPair]:
    if sample_dir is not None:
        pairs = discover_dataset_pairs(sample_dir)
        if not pairs:
            raise FileNotFoundError(f"No dataset pairs found in {sample_dir}")
        return pairs

    if features_path is None or labels_path is None:
        raise ValueError("Both --features and --labels are required together.")

    return [
        DatasetPair(
            name=features_path.stem.replace("_dataset", ""),
            features_path=features_path,
            labels_path=labels_path,
        )
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Replicate KitNET on sampled Kitsune feature CSV datasets."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--sample_dir",
        type=Path,
        help="Directory containing sampled *_dataset.csv and *_labels.csv files.",
    )
    src.add_argument(
        "--features",
        type=Path,
        help="Path to one sampled features CSV.",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to one sampled labels CSV (use with --features).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Directory for per-dataset results.",
    )
    parser.add_argument(
        "--fm_grace",
        type=int,
        default=5000,
        help="Rows used to train the FeatureMapper.",
    )
    parser.add_argument(
        "--ad_grace",
        type=int,
        default=50000,
        help="Rows used to train KitNET after the mapper is fixed.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=10,
        help="Max features per autoencoder input.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.75,
        help="Hidden-layer compression ratio.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for the autoencoders.",
    )
    parser.add_argument(
        "--expected_n_features",
        type=int,
        default=EXPECTED_N_FEATURES,
        help="Expected feature count per row.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip plot generation at the end.",
    )

    args = parser.parse_args(argv)

    if args.features is not None and args.labels is None:
        parser.error("--labels is required when you use --features")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_pairs = build_dataset_list(
        sample_dir=args.sample_dir,
        features_path=args.features,
        labels_path=args.labels,
    )

    logger.info("Found %d dataset(s) to run.", len(dataset_pairs))

    all_metrics: list[dict] = []

    for pair in dataset_pairs:
        metrics = run_dataset(
            pair=pair,
            output_dir=args.output_dir,
            fm_grace=args.fm_grace,
            ad_grace=args.ad_grace,
            max_cluster_size=args.m,
            beta=args.beta,
            lr=args.lr,
            expected_n_features=args.expected_n_features,
        )
        all_metrics.append(metrics)

    summary_df = pd.DataFrame(all_metrics)
    summary_csv = args.output_dir / "summary_metrics.csv"
    summary_json = args.output_dir / "summary_metrics.json"

    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w") as fh:
        json.dump(all_metrics, fh, indent=2)

    logger.info("Saved the summary table to %s", summary_csv)
    logger.info("Saved the summary json to %s", summary_json)

    if not args.skip_plots:
        plots_dir = args.output_dir / "_plots"
        make_all_plots(args.output_dir, plots_dir)
        logger.info("Saved plots to %s", plots_dir)


if __name__ == "__main__":
    main()

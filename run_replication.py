"""
Phase 1: Replicate Kitsune on KitNET pre-extracted feature CSV datasets.
------------------------------------------------------------
    python run_replication.py \
        --dataset_dir dataset/ \
        --n_train 400000 \
        --m 10 \
        --output_dir results/
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.feature_mapper import FeatureMapper
from core.kitnet import KitNET
from dataset_reader import CSVDatasetReader, EXPECTED_N_FEATURES
from evaluation.metrics import compute_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



class KitNETFromCSV:
    """
    Kitsune pipeline for pre-extracted feature CSVs.

    Parameters
    ----------
    n_train          : int   rows used for FM fitting + KitNET training
    max_cluster_size : int   m — max features per autoencoder (default 10)
    n_features       : int   feature dimensionality (default 115)
    beta             : float hidden-layer compression ratio (default 0.75)
    lr               : float SGD learning rate (default 0.1)
    """

    def __init__(
        self,
        n_train:          int,
        max_cluster_size: int   = 10,
        n_features:       int   = EXPECTED_N_FEATURES,
        beta:             float = 0.75,
        lr:               float = 0.1,
    ):
        self.n_train = n_train
        self.fm      = FeatureMapper(
            n_features=n_features,
            max_cluster_size=max_cluster_size,
        )
        self.ad: KitNET | None = None
        self.beta = beta
        self.lr   = lr
        self.phi: float = 0.0

        self._n_seen   = 0
        self._in_train = True

    def process(self, features: np.ndarray) -> float | None:
        """
        Process one pre-extracted feature vector.

        Returns
        -------
        float  — anomaly score (exec-mode)
        None   — during train-mode
        """
        self._n_seen += 1

        if self._in_train:
            self.fm.update(features)

            if self._n_seen == self.n_train:
                # ---- Transition from train to exec ----
                logger.info(
                    "Fitting FeatureMapper after %d training rows...",
                    self.n_train,
                )
                self.fm.fit()
                logger.info(
                    "FeatureMapper fitted: k=%d clusters, sizes=%s",
                    self.fm.n_clusters, self.fm.cluster_sizes,
                )
                self.ad = KitNET(
                    cluster_sizes=self.fm.cluster_sizes,
                    beta=self.beta,
                    lr=self.lr,
                )
                self._in_train = False
                # Train KitNET on this transition instance
                vi    = self.fm.transform(features)
                score = self.ad.train(vi)
                self.phi = max(self.phi, score)

            return None

        else:
            # Exec-mode: score the instance without updating weights
            vi    = self.fm.transform(features)
            score = self.ad.execute(vi)
            return score


def run_dataset(
    dataset_name:  str,
    features_path: Path,
    ts_path:       Path | None,
    n_train:       int,
    attack_start:  int,
    m:             int,
    output_dir:    Path,
) -> dict:
    """Run the pipeline on one feature CSV and compute detection metrics."""

    logger.info("=" * 60)
    logger.info("Dataset      : %s", dataset_name)
    logger.info("Features CSV : %s", features_path.name)
    logger.info("Timestamps   : %s",
                ts_path.name if ts_path else "synthetic (row index)")
    logger.info(
        "n_train: %d  |  attack_start: %d  |  m: %d",
        n_train, attack_start, m,
    )

    pipeline = KitNETFromCSV(n_train=n_train, max_cluster_size=m)
    reader   = CSVDatasetReader(features_path, ts_path)

    scores_list: list[float] = []
    labels_list: list[int]   = []
    row_idx = 0

    for features, _ in reader:
        score = pipeline.process(features)

        if score is not None:
            # Label: 1 if this row is in the attack window, else 0
            label = 1 if row_idx >= attack_start else 0
            scores_list.append(score)
            labels_list.append(label)

            if len(scores_list) % 50_000 == 0:
                logger.info(
                    "  [exec]  %d rows scored | malicious so far: %d",
                    len(scores_list), int(np.sum(labels_list)),
                )
        else:
            if row_idx % 100_000 == 0 and row_idx > 0:
                logger.info(
                    "  [train] row %d / %d", row_idx, n_train
                )

        row_idx += 1

    logger.info(
        "Finished: %d train rows | %d eval rows | %d labelled malicious",
        n_train, len(scores_list), int(np.sum(labels_list)),
    )

    if not scores_list:
        logger.warning("No exec-mode rows — nothing to evaluate.")
        return {}

    scores = np.array(scores_list, dtype=np.float64)
    labels = np.array(labels_list, dtype=np.int32)

    # Compute and display metrics
    metrics = compute_metrics(scores, labels, dataset_name=dataset_name)
    print_metrics(metrics)

    # Save outputs
    out_dir = output_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "scores.npy", scores)
    np.save(out_dir / "labels.npy", labels)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Results saved → %s", out_dir)
    return metrics


def _discover_pairs(
    dataset_dir: Path,
) -> list[tuple[str, Path, Path | None]]:
    """
    Scan dataset_dir for feature/timestamp CSV pairs.

    KitNET naming convention:
        mirai3.csv    → feature file
        mirai3_ts.csv → paired timestamp file

    Any CSV whose stem does NOT end in '_ts' is treated as a feature file.
    The matching timestamp file <stem>_ts.csv is used if it exists.
    """
    pairs = []
    for feat_csv in sorted(dataset_dir.glob("*.csv")):
        if feat_csv.stem.endswith("_ts"):
            continue
        ts_csv  = dataset_dir / (feat_csv.stem + "_ts.csv")
        ts_path = ts_csv if ts_csv.exists() else None
        pairs.append((feat_csv.stem, feat_csv, ts_path))
    return pairs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 — Kitsune replication on KitNET pre-extracted CSV datasets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input: single file pair OR directory
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--features", type=Path,
        metavar="FEATURES_CSV",
        help="Path to a features CSV  (e.g.  dataset/mirai3.csv).",
    )
    src.add_argument(
        "--dataset_dir", type=Path,
        metavar="DIR",
        help="Directory containing *.csv feature files (auto-discovers pairs).",
    )

    parser.add_argument(
        "--timestamps", type=Path, default=None,
        metavar="TIMESTAMPS_CSV",
        help="Timestamps CSV paired with --features  (e.g.  dataset/mirai3_ts.csv).",
    )
    parser.add_argument(
        "--n_train", type=int, default=400_000,
        help="Rows used for training (default: 400,000).",
    )
    parser.add_argument(
        "--attack_start", type=int, default=None,
        help=(
            "Row index (0-based, inclusive) where the attack begins "
            "in exec-mode.  Defaults to n_train."
        ),
    )
    parser.add_argument(
        "--m", type=int, default=10,
        help="Max features per autoencoder input  (default: 10).",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results"),
        help="Directory to write results  (default: ./results).",
    )

    args   = parser.parse_args(argv)
    attack = args.attack_start if args.attack_start is not None else args.n_train
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build the list of datasets to process
    if args.features:
        datasets: list[tuple[str, Path, Path | None]] = [(
            args.features.stem,
            args.features,
            args.timestamps,
        )]
    else:
        datasets = _discover_pairs(args.dataset_dir)
        if not datasets:
            logger.error("No feature CSV files found in %s", args.dataset_dir)
            sys.exit(1)
        logger.info(
            "Discovered %d dataset(s) in %s", len(datasets), args.dataset_dir
        )

    # Run
    all_metrics: dict = {}
    for name, feat_csv, ts_csv in datasets:
        m = run_dataset(
            dataset_name=name,
            features_path=feat_csv,
            ts_path=ts_csv,
            n_train=args.n_train,
            attack_start=attack,
            m=args.m,
            output_dir=args.output_dir,
        )
        all_metrics[name] = m

    # Summary
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    logger.info("Summary → %s", summary_path)


if __name__ == "__main__":
    main()
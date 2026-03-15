from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

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
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
_log = logging.getLogger(__name__)


class AnomalyDetectionPipeline:
    # Three-phase pipeline: FeatureMapper training → KitNET training → inference.
    # No score is returned during the two warm-up phases.

    def __init__(
        self,
        fm_grace:         int,
        ad_grace:         int,
        max_cluster_size: int   = 10,
        num_features:     int   = EXPECTED_N_FEATURES,
        beta_ratio:       float = 0.75,
        step_size:        float = 0.1,
    ) -> None:
        self.fm_grace  = fm_grace
        self.ad_grace  = ad_grace

        self.feat_mapper = FeatureMapper(
            n_features=num_features,
            max_cluster_size=max_cluster_size,
        )
        self.detector: Optional[KitNET] = None
        self._mapper_ready    = False
        self._rows_seen       = 0
        self.peak_train_score: float = 0.0
        self._beta = beta_ratio
        self._lr   = step_size

    @property
    def warmup_length(self) -> int:
        return self.fm_grace + self.ad_grace

    def step(self, feature_vec: np.ndarray) -> Tuple[Optional[float], str]:
        idx = self._rows_seen
        self._rows_seen += 1

        # Phase 1 — build feature correlation matrix
        if idx < self.fm_grace:
            self.feat_mapper.update(feature_vec)
            return None, "fm_train"

        # Transition — fit mapper once, then initialise KitNET
        if not self._mapper_ready:
            _log.info("Fitting FeatureMapper on %d rows...", self.fm_grace)
            self.feat_mapper.fit()
            self.detector = KitNET(
                cluster_sizes=self.feat_mapper.cluster_sizes,
                beta=self._beta,
                lr=self._lr,
            )
            self._mapper_ready = True
            _log.info(
                "FeatureMapper ready — k=%d clusters: %s",
                self.feat_mapper.n_clusters,
                self.feat_mapper.cluster_sizes,
            )

        sub_vecs = self.feat_mapper.transform(feature_vec)

        # Phase 2 — train KitNET autoencoders
        if idx < self.warmup_length:
            score = float(self.detector.train(sub_vecs))
            self.peak_train_score = max(self.peak_train_score, score)
            return None, "ad_train"

        # Phase 3 — inference
        return float(self.detector.execute(sub_vecs)), "exec"


def _safe_dirname(name: str) -> str:
    return name.replace(" ", "_")


def evaluate_single_dataset(
    pair:          DatasetPair,
    output_root:   Path,
    fm_grace:      int,
    ad_grace:      int,
    cluster_limit: int,
    beta:          float,
    learn_rate:    float,
    n_features:    int,
    log_every:     int = 100_000,
) -> Dict:
    _log.info("*" * 70)
    _log.info("Dataset  : %s", pair.name)
    _log.info("Features : %s", pair.features_path.name)
    _log.info("Labels   : %s", pair.labels_path.name)

    reader     = PairedCSVDatasetReader(pair.features_path, pair.labels_path, n_features)
    total_rows = reader.count_rows()

    pipeline = AnomalyDetectionPipeline(
        fm_grace=fm_grace, ad_grace=ad_grace,
        max_cluster_size=cluster_limit,
        num_features=n_features, beta_ratio=beta, step_size=learn_rate,
    )

    phase_counts: Dict[str, int] = {"fm_train": 0, "ad_train": 0, "exec": 0}
    scored_rows:  List[Dict]     = []
    t_start = time.time()

    for idx, feat_vec, true_label in reader:
        score, phase = pipeline.step(feat_vec)
        phase_counts[phase] += 1

        if phase == "exec":
            scored_rows.append({"row_index": idx, "label": int(true_label), "score": float(score)})

        if idx > 0 and idx % log_every == 0:
            _log.info(
                "Row %d / %d  |  fm=%d  ad=%d  exec=%d",
                idx, total_rows,
                phase_counts["fm_train"], phase_counts["ad_train"], phase_counts["exec"],
            )

    elapsed = time.time() - t_start

    if not scored_rows:
        _log.warning("No scored rows for '%s'.", pair.name)
        return {"dataset": pair.name, "error": "no_eval_rows"}

    results_df = pd.DataFrame(scored_rows)
    score_arr  = results_df["score"].to_numpy(dtype=np.float64)
    label_arr  = results_df["label"].to_numpy(dtype=np.int32)

    metrics = compute_metrics(
        scores=score_arr, labels=label_arr,
        dataset_name=pair.name, runtime_sec=elapsed,
        extra={
            "total_rows":       int(total_rows),
            "fm_grace":         int(fm_grace),
            "ad_grace":         int(ad_grace),
            "warmup_rows":      int(pipeline.warmup_length),
            "max_cluster_size": int(cluster_limit),
            "n_clusters":       int(pipeline.feat_mapper.n_clusters),
            "cluster_sizes":    pipeline.feat_mapper.cluster_sizes,
            "fm_train_rows":    int(phase_counts["fm_train"]),
            "ad_train_rows":    int(phase_counts["ad_train"]),
            "exec_rows":        int(phase_counts["exec"]),
            "peak_train_score": round(float(pipeline.peak_train_score), 6),
        },
    )

    print_metrics(metrics)

    target_dir = output_root / _safe_dirname(pair.name)
    target_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(target_dir / "scores.csv", index=False)
    np.save(target_dir / "scores.npy", score_arr)
    np.save(target_dir / "labels.npy", label_arr)

    roc_df, pr_df = build_curve_frames(scores=score_arr, labels=label_arr)
    roc_df.to_csv(target_dir / "roc_curve.csv", index=False)
    pr_df.to_csv( target_dir / "pr_curve.csv",  index=False)

    with open(target_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    _log.info("Artefacts written → %s", target_dir)
    return metrics


def resolve_datasets(
    scan_dir:      Optional[Path],
    features_file: Optional[Path],
    labels_file:   Optional[Path],
) -> List[DatasetPair]:
    if scan_dir is not None:
        pairs = discover_dataset_pairs(scan_dir)
        if not pairs:
            raise FileNotFoundError(f"No dataset pairs found in '{scan_dir}'.")
        return pairs

    if not features_file or not labels_file:
        raise ValueError("Provide both --features and --labels when not using --sample_dir.")

    stem = features_file.stem
    if stem.endswith("_dataset"):
        stem = stem[:-8]
    return [DatasetPair(name=stem, features_path=features_file, labels_path=labels_file)]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run KitNET anomaly detection on pre-extracted CSV datasets."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample_dir", type=Path,
                     help="Directory with *_dataset.csv / *_labels.csv pairs.")
    src.add_argument("--features",   type=Path, help="Path to a features CSV.")

    parser.add_argument("--labels",      type=Path,  default=None)
    parser.add_argument("--output_dir",  type=Path,  default=Path("results"))
    parser.add_argument("--fm_grace",    type=int,   default=5_000)
    parser.add_argument("--ad_grace",    type=int,   default=50_000)
    parser.add_argument("--m",           type=int,   default=10)
    parser.add_argument("--beta",        type=float, default=0.75)
    parser.add_argument("--lr",          type=float, default=0.1)
    parser.add_argument("--expected_n_features", type=int, default=EXPECTED_N_FEATURES)
    parser.add_argument("--skip_plots",  action="store_true")

    args = parser.parse_args(argv)

    if args.features is not None and args.labels is None:
        parser.error("--labels is required when --features is provided.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_list = resolve_datasets(args.sample_dir, args.features, args.labels)
    _log.info("Datasets to process: %d", len(dataset_list))

    all_metrics: List[Dict] = []
    for ds in dataset_list:
        result = evaluate_single_dataset(
            pair=ds, output_root=args.output_dir,
            fm_grace=args.fm_grace, ad_grace=args.ad_grace,
            cluster_limit=args.m, beta=args.beta,
            learn_rate=args.lr, n_features=args.expected_n_features,
        )
        all_metrics.append(result)

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    with open(args.output_dir / "summary_metrics.json", "w") as fh:
        json.dump(all_metrics, fh, indent=2)

    _log.info("Summary → %s", args.output_dir / "summary_metrics.json")

    if not args.skip_plots:
        plots_dir = args.output_dir / "_plots"
        make_all_plots(args.output_dir, plots_dir)
        _log.info("Plots → %s", plots_dir)


if __name__ == "__main__":
    main()
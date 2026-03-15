from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_N_FEATURES = 115

_BENIGN_TOKENS = {
    "0",
    "0.0",
    "benign",
    "normal",
    "false",
    "no",
    "clean",
    "legit",
}


@dataclass(frozen=True)
class DatasetPair:
    name: str
    features_path: Path
    labels_path: Path


def discover_dataset_pairs(sample_dir: str | Path) -> list[DatasetPair]:
    sample_dir = Path(sample_dir)
    if not sample_dir.exists():
        raise FileNotFoundError(f"Directory not found: {sample_dir}")

    pairs: list[DatasetPair] = []

    for features_path in sorted(sample_dir.glob("*_dataset.csv")):
        labels_path = features_path.with_name(
            features_path.name.replace("_dataset.csv", "_labels.csv")
        )

        if not labels_path.exists():
            logger.warning("Skipped %s because labels were missing.", features_path.name)
            continue

        name = features_path.stem.replace("_dataset", "")
        pairs.append(
            DatasetPair(
                name=name,
                features_path=features_path,
                labels_path=labels_path,
            )
        )

    return pairs


class PairedCSVDatasetReader:
    """Streams paired feature and label rows from two CSV files."""

    def __init__(
        self,
        features_path: str | Path,
        labels_path: str | Path,
        expected_n_features: int = EXPECTED_N_FEATURES,
    ):
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)
        self.expected_n_features = expected_n_features
        self._n_features: int | None = None

        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

    def __iter__(self):
        with open(self.features_path, newline="") as feat_fh, open(self.labels_path, newline="") as lab_fh:
            feat_reader = csv.reader(feat_fh)
            lab_reader = csv.reader(lab_fh)

            row_idx = 0

            while True:
                feat_row = self._next_non_empty_row(feat_reader)
                lab_row = self._next_non_empty_row(lab_reader)

                if feat_row is None and lab_row is None:
                    break

                if feat_row is None or lab_row is None:
                    raise ValueError(
                        f"Feature/label length mismatch near row {row_idx} "
                        f"for {self.features_path.name}"
                    )

                features = self._parse_features(feat_row, row_idx)
                label = self._parse_label(lab_row, row_idx)

                yield row_idx, features, label
                row_idx += 1

    def count_rows(self) -> int:
        n_rows = 0
        with open(self.features_path, newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if row:
                    n_rows += 1
        return n_rows

    @staticmethod
    def _next_non_empty_row(reader):
        for row in reader:
            if row:
                return row
        return None

    def _parse_features(self, row: list[str], row_idx: int) -> np.ndarray:
        try:
            features = np.asarray(row, dtype=np.float32)
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric feature row at index {row_idx} in {self.features_path.name}"
            ) from exc

        if self._n_features is None:
            self._n_features = len(features)
            if self._n_features != self.expected_n_features:
                logger.warning(
                    "Found %d features in %s, Expected %d.",
                    self._n_features,
                    self.features_path.name,
                    self.expected_n_features,
                )

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def _parse_label(self, row: list[str], row_idx: int) -> int:
        raw = str(row[0]).strip()

        if raw == "":
            raise ValueError(
                f"Empty label at row {row_idx} in {self.labels_path.name}"
            )

        try:
            return int(float(raw) > 0.0)
        except ValueError:
            token = raw.lower()
            if token in _BENIGN_TOKENS:
                return 0
            return 1

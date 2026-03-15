from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

EXPECTED_N_FEATURES = 115

# String tokens that unambiguously indicate a benign sample
_BENIGN_VOCAB = frozenset({
    "0", "0.0", "benign", "normal", "false", "no", "clean", "legit",
})


@dataclass(frozen=True)
class DatasetPair:
    name:          str
    features_path: Path
    labels_path:   Path


def discover_dataset_pairs(search_dir: str | Path) -> list[DatasetPair]:
    # Finds all *_dataset.csv / *_labels.csv pairs in a directory.
    search_dir = Path(search_dir)
    if not search_dir.exists():
        raise FileNotFoundError(f"Directory not found: {search_dir}")

    found: list[DatasetPair] = []
    for feat_file in sorted(search_dir.glob("*_dataset.csv")):
        label_file = feat_file.with_name(
            feat_file.name.replace("_dataset.csv", "_labels.csv")
        )
        if not label_file.exists():
            log.warning("Skipping '%s': no matching label file.", feat_file.name)
            continue
        stem = feat_file.stem.replace("_dataset", "")
        found.append(DatasetPair(name=stem, features_path=feat_file, labels_path=label_file))
    return found


class PairedCSVDatasetReader:
    # Streams paired (feature_vector, label) rows from two aligned CSV files.

    def __init__(
        self,
        features_path:       str | Path,
        labels_path:         str | Path,
        expected_n_features: int = EXPECTED_N_FEATURES,
    ):
        self.features_path       = Path(features_path)
        self.labels_path         = Path(labels_path)
        self.expected_n_features = expected_n_features
        self._detected_width:    int | None = None

        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

    def __iter__(self):
        with (
            open(self.features_path, newline="") as feat_fh,
            open(self.labels_path,   newline="") as lbl_fh,
        ):
            feat_reader = csv.reader(feat_fh)
            lbl_reader  = csv.reader(lbl_fh)
            row_num     = 0

            while True:
                feat_row = self._advance(feat_reader)
                lbl_row  = self._advance(lbl_reader)

                if feat_row is None and lbl_row is None:
                    break

                if feat_row is None or lbl_row is None:
                    raise ValueError(
                        f"Row count mismatch near row {row_num} in '{self.features_path.name}'."
                    )

                yield row_num, self._to_features(feat_row, row_num), self._to_label(lbl_row, row_num)
                row_num += 1

    def count_rows(self) -> int:
        total = 0
        with open(self.features_path, newline="") as fh:
            for row in csv.reader(fh):
                if row:
                    total += 1
        return total

    @staticmethod
    def _advance(reader) -> list[str] | None:
        for row in reader:
            if row:
                return row
        return None

    def _to_features(self, row: list[str], row_num: int) -> np.ndarray:
        try:
            vec = np.asarray(row, dtype=np.float32)
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric value in feature row {row_num} of '{self.features_path.name}'."
            ) from exc

        if self._detected_width is None:
            self._detected_width = len(vec)
            if self._detected_width != self.expected_n_features:
                log.warning(
                    "'%s' has %d features; expected %d.",
                    self.features_path.name, self._detected_width, self.expected_n_features,
                )

        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    def _to_label(self, row: list[str], row_num: int) -> int:
        raw = str(row[0]).strip()
        if not raw:
            raise ValueError(f"Empty label at row {row_num} in '{self.labels_path.name}'.")

        try:
            return int(float(raw) > 0.0)
        except ValueError:
            return 0 if raw.lower() in _BENIGN_VOCAB else 1
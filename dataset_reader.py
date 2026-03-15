from __future__ import annotations
import csv
import logging
from pathlib import Path
 
import numpy as np
 
logger = logging.getLogger(__name__)
 
# Expected number of features in a KitNET feature CSV
EXPECTED_N_FEATURES = 115
 
 
class CSVDatasetReader:
 
    def __init__(
        self,
        features_path:   str | Path,
        timestamps_path: str | Path | None = None,
    ):
        self.features_path   = Path(features_path)
        self.timestamps_path = Path(timestamps_path) if timestamps_path else None
        self._n_features: int | None = None
 
 
    def __iter__(self):
        """
        Yields (feature_vector, timestamp) for each row.
        feature_vector : np.ndarray, shape (n_features,), dtype float32
        timestamp      : float
        """
        ts_iter = self._open_timestamps()
 
        with open(self.features_path, newline="") as feat_fh:
            reader = csv.reader(feat_fh)
            for row_idx, row in enumerate(reader):
                if not row:
                    continue
 
                # Parse feature vector
                try:
                    feat = np.array(row, dtype=np.float32)
                except ValueError:
                    # Skip header rows or malformed lines
                    logger.debug("Skipping non-numeric row %d", row_idx)
                    continue
 
                # Validate dimensionality on first real row
                if self._n_features is None:
                    self._n_features = len(feat)
                    if self._n_features != EXPECTED_N_FEATURES:
                        logger.warning(
                            "Feature CSV has %d columns; expected %d. "
                            "Proceeding anyway.",
                            self._n_features, EXPECTED_N_FEATURES,
                        )
 
                feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
 
                # Get matching timestamp
                timestamp = next(ts_iter, float(row_idx))
 
                yield feat, float(timestamp)
 
    def _open_timestamps(self):

        if self.timestamps_path is None or not self.timestamps_path.exists():
            # Synthetic timestamps: 0.0, 1.0, 2.0, ...
            def _counter():
                i = 0
                while True:
                    yield float(i)
                    i += 1
            return _counter()
 
        def _ts_gen():
            with open(self.timestamps_path, newline="") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if not row:
                        continue
                    try:
                        yield float(row[0])
                    except (ValueError, IndexError):
                        yield 0.0
 
        return _ts_gen()
 
 
    def count_rows(self) -> int:
        """Count the number of data rows in the features CSV."""
        n = 0
        with open(self.features_path, newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if row:
                    try:
                        float(row[0])   # skip non-numeric header
                        n += 1
                    except ValueError:
                        pass
        return n
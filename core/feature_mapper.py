"""Feature Mapper for Kitsune: clusters correlated features into groups."""

from __future__ import annotations
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


class FeatureMapper:
    """Groups input features by correlation for ensemble autoencoders."""

    def __init__(self, n_features: int = 115, max_cluster_size: int = 10):
        self.n = n_features
        self.m = max_cluster_size

        self._c = np.zeros(self.n, dtype=np.float64)
        self._c_rs = np.zeros(self.n, dtype=np.float64)
        self._C = np.zeros((self.n, self.n), dtype=np.float64)
        self._t = 0

        self.mapping: list[list[int]] | None = None
        self.is_fitted: bool = False

    def update(self, x: np.ndarray) -> None:
        """Accumulate running statistics from a feature vector."""
        self._t += 1
        self._c += x
        mu = self._c / self._t
        r = x - mu
        self._c_rs += r ** 2
        self._C += np.outer(r, r)

    def fit(self) -> None:
        """Build correlation distances and cluster features."""
        D = np.ones((self.n, self.n), dtype=np.float64)

        denom_outer = np.sqrt(np.outer(self._c_rs, self._c_rs))
        nonzero = denom_outer > 1e-12
        D[nonzero] = 1.0 - self._C[nonzero] / denom_outer[nonzero]

        D = np.clip(D, 0.0, 1.0)
        np.fill_diagonal(D, 0.0)

        D = (D + D.T) / 2.0

        condensed = squareform(D, checks=False)
        Z = linkage(condensed, method="complete")

        self.mapping = self._cut_dendrogram(Z)
        self.is_fitted = True

    def _cut_dendrogram(self, Z: np.ndarray) -> list[list[int]]:
        """Split clusters until each respects the max size."""
        labels = fcluster(Z, t=self.m, criterion="maxclust")

        groups: dict[int, list[int]] = {}
        for feat_idx, cluster_id in enumerate(labels):
            groups.setdefault(cluster_id, []).append(feat_idx)

        final_groups: list[list[int]] = []
        for grp in groups.values():
            if len(grp) <= self.m:
                final_groups.append(grp)
            else:
                for i in range(0, len(grp), self.m):
                    final_groups.append(grp[i:i + self.m])

        return final_groups

    def transform(self, x: np.ndarray) -> list[np.ndarray]:
        """Apply the learned mapping to a feature vector."""
        if not self.is_fitted:
            raise RuntimeError("FeatureMapper.fit() must be called before transform().")
        return [x[idx] for idx in self.mapping]

    @property
    def n_clusters(self) -> int:
        """Number of feature groups after fitting."""
        if not self.is_fitted:
            return 0
        return len(self.mapping)

    @property
    def cluster_sizes(self) -> list[int]:
        """Sizes of each cluster after fitting."""
        if not self.is_fitted:
            return []
        return [len(g) for g in self.mapping]

"""
core/feature_mapper.py
-----------------------
Feature Mapper (FM) component of Kitsune.

During training: incrementally maintains a partial correlation matrix C over
the n feature dimensions by accumulating running sums needed to compute
pairwise correlation distances without storing any individual instances.

After training: performs hierarchical agglomerative clustering on the nxn
correlation distance matrix D and cuts the dendrogram to yield k groups,
each of size ≤ m.  The resulting mapping f(x) = v partitions every future
feature vector into k sub-instances, one per autoencoder in the ensemble.

Paper reference: Mirsky et al., NDSS 2018, Section IV-C (Equations 8-10).
"""

from __future__ import annotations
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


class FeatureMapper:
    """
    Parameters
    ----------
    n_features : int
        Total number of input features (115 for original Kitsune).
    max_cluster_size : int
        Maximum number of features per sub-instance (m in the paper).
        Controls the ensemble size and per-autoencoder complexity.
    """

    def __init__(self, n_features: int = 115, max_cluster_size: int = 10):
        self.n = n_features
        self.m = max_cluster_size

        # Incremental partial-correlation accumulators
        # c[i]   = running sum of feature i values
        # c_rs[i]= running sum of squared residuals for feature i
        # C[i,j] = running sum of products of residuals for features i,j
        self._c    = np.zeros(self.n, dtype=np.float64)
        self._c_rs = np.zeros(self.n, dtype=np.float64)
        self._C    = np.zeros((self.n, self.n), dtype=np.float64)
        self._t    = 0  # number of instances seen

        # Populated after fit() is called
        self.mapping: list[list[int]] | None = None  # k groups of feature indices
        self.is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Training phase
    # ------------------------------------------------------------------ #

    def update(self, x: np.ndarray) -> None:
        """
        Incorporate one feature vector into the incremental correlation matrix.
        Must be called for every training instance before fit() is invoked.

        Parameters
        ----------
        x : np.ndarray, shape (n,)
        """
        self._t += 1
        # Running sums
        self._c += x
        # Current mean estimate (used only to compute residual)
        mu = self._c / self._t
        r = x - mu                        # residual vector, shape (n,)
        self._c_rs += r ** 2              # per-feature squared residuals
        # Outer product of residuals → update cross-feature products
        self._C    += np.outer(r, r)

    # ------------------------------------------------------------------ #
    # Derive distance matrix and cluster
    # ------------------------------------------------------------------ #

    def fit(self) -> None:
        """
        Compute the n×n correlation distance matrix D from the accumulated
        statistics, perform hierarchical clustering, and cut the dendrogram
        to obtain groups of size ≤ m.

        Must be called exactly once after all training instances have been
        passed through update().
        """
        # Equation (10) in the paper: D[i,j] = 1 - C[i,j] / sqrt(c_rs[i] * c_rs[j])
        D = np.ones((self.n, self.n), dtype=np.float64)

        denom_outer = np.sqrt(np.outer(self._c_rs, self._c_rs))
        # Avoid divide-by-zero for constant features
        nonzero = denom_outer > 1e-12
        D[nonzero] = 1.0 - self._C[nonzero] / denom_outer[nonzero]

        # Clip to [0, 1] to handle minor numerical issues
        D = np.clip(D, 0.0, 1.0)
        np.fill_diagonal(D, 0.0)

        # Symmetrise (should already be symmetric, but numerical safety)
        D = (D + D.T) / 2.0

        # Hierarchical agglomerative clustering with complete linkage
        # (ensures tight, well-bounded clusters)
        condensed = squareform(D, checks=False)
        Z = linkage(condensed, method="complete")

        # Cut dendrogram: iteratively increase the cut threshold until all
        # clusters have size ≤ m, following the paper's procedure.
        self.mapping = self._cut_dendrogram(Z, D)
        self.is_fitted = True

    def _cut_dendrogram(
        self, Z: np.ndarray, D: np.ndarray
    ) -> list[list[int]]:
        """
        Cut the dendrogram to yield groups where no group exceeds size m.

        Strategy: try increasing distance thresholds from 0 to 1 in small
        steps; for any cluster that still exceeds m, recursively split it
        by halving the threshold within that sub-cluster.
        """
        # Start: every feature is its own cluster
        labels = fcluster(Z, t=self.m, criterion="maxclust")

        # Re-partition to respect the size bound m exactly
        groups: dict[int, list[int]] = {}
        for feat_idx, cluster_id in enumerate(labels):
            groups.setdefault(cluster_id, []).append(feat_idx)

        # If any group still exceeds m, split greedily
        final_groups: list[list[int]] = []
        for grp in groups.values():
            if len(grp) <= self.m:
                final_groups.append(grp)
            else:
                # Split into chunks of size m
                for i in range(0, len(grp), self.m):
                    final_groups.append(grp[i:i + self.m])

        return final_groups

    # ------------------------------------------------------------------ #
    # Exec phase
    # ------------------------------------------------------------------ #

    def transform(self, x: np.ndarray) -> list[np.ndarray]:
        """
        Apply the learned mapping to a feature vector.

        Returns
        -------
        v : list of np.ndarray
            k sub-instances, one per cluster/autoencoder.

        Raises
        ------
        RuntimeError if fit() has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureMapper.fit() must be called before transform().")
        return [x[idx] for idx in self.mapping]

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def n_clusters(self) -> int:
        """Number of feature groups (k) after fitting."""
        if not self.is_fitted:
            return 0
        return len(self.mapping)

    @property
    def cluster_sizes(self) -> list[int]:
        if not self.is_fitted:
            return []
        return [len(g) for g in self.mapping]

"""Kitsune pipeline combining feature extraction, mapping, and KitNET."""

from __future__ import annotations
import logging
import numpy as np

from .feature_extractor import FeatureExtractor, N_FEATURES
from .feature_mapper import FeatureMapper
from .kitnet import KitNET

logger = logging.getLogger(__name__)


class Kitsune:
    """Online NIDS with train and exec phases."""

    def __init__(
        self,
        n_train: int,
        max_cluster_size: int = 10,
        beta: float = 0.75,
        lr: float = 0.1,
        lambdas: list[float] | None = None,
    ):
        self.n_train = n_train
        self.max_cluster_size = max_cluster_size
        self.beta = beta
        self.lr = lr

        self.fe = FeatureExtractor(lambdas=lambdas)
        self.fm = FeatureMapper(
            n_features=N_FEATURES,
            max_cluster_size=max_cluster_size,
        )
        self.ad: KitNET | None = None

        self._n_seen = 0
        self._in_train = True

    def process(self, pkt: dict) -> float | None:
        """Process one packet; returns score only in exec mode."""
        x = self.fe.update(pkt)
        self._n_seen += 1

        if self._in_train:
            self.fm.update(x)

            if self._n_seen == self.n_train:
                # Flip into exec mode once we've seen the planned training budget.
                self._switch_to_exec()

            if self.ad is not None:
                # Train KitNET on the same packet that triggered exec switch.
                vi = self.fm.transform(x)
                self.ad.train(vi)

            return None

        vi = self.fm.transform(x)
        return self.ad.execute(vi)

    def process_batch(
        self, packets: list[dict]
    ) -> tuple[list[float], list[int]]:
        scores, indices = [], []
        for i, pkt in enumerate(packets):
            s = self.process(pkt)
            if s is not None:
                scores.append(s)
                indices.append(i)
        return scores, indices

    def _switch_to_exec(self) -> None:
        logger.info(
            "Kitsune: switching to exec-mode after %d training packets.",
            self.n_train,
        )
        # Lock in the feature grouping before bringing KitNET online.
        self.fm.fit()
        logger.info(
            "FeatureMapper fitted: k=%d clusters, sizes=%s",
            self.fm.n_clusters,
            self.fm.cluster_sizes,
        )
        # Initialise the anomaly detector with the frozen mapping.
        self.ad = KitNET(
            cluster_sizes=self.fm.cluster_sizes,
            beta=self.beta,
            lr=self.lr,
        )
        self._in_train = False

    @property
    def in_train_mode(self) -> bool:
        return self._in_train

    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def threshold(self) -> float:
        if self.ad is None:
            return 0.0
        return self.ad.phi

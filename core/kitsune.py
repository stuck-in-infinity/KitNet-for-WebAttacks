"""
core/kitsune.py
---------------
Main Kitsune pipeline.

Integrates the Feature Extractor (FE), Feature Mapper (FM), and
KitNET Anomaly Detector (AD) into a single plug-and-play NIDS.

Two operating modes mirror the paper exactly:
    train-mode  : FE extracts features → FM accumulates correlation stats
                  → KitNET trains autoencoders online.
                  Continues until n_train instances have been processed.

    exec-mode   : FE extracts features (using trained channel statistics)
                  → FM maps to sub-instances (fixed mapping)
                  → KitNET executes (inference only, weights frozen).
                  Returns anomaly score for every packet.

Paper reference: Mirsky et al., NDSS 2018, Section IV and Algorithm 4/5.
"""

from __future__ import annotations
import logging
import numpy as np

from .feature_extractor import FeatureExtractor, N_FEATURES
from .feature_mapper import FeatureMapper
from .kitnet import KitNET

logger = logging.getLogger(__name__)


class Kitsune:
    """
    Parameters
    ----------
    n_train : int
        Number of packets to use for training (train-mode).
        After n_train packets the pipeline automatically switches to exec-mode.
    max_cluster_size : int
        m parameter — maximum features per autoencoder input (default 10).
    beta : float
        Autoencoder hidden-layer compression ratio (default 3/4).
    lr : float
        SGD learning rate for KitNET autoencoders (default 0.1).
    lambdas : list of float | None
        Decay values for the FE. Default: [5, 3, 1, 0.1, 0.01].
    """

    def __init__(
        self,
        n_train:          int,
        max_cluster_size: int   = 10,
        beta:             float = 0.75,
        lr:               float = 0.1,
        lambdas:          list[float] | None = None,
    ):
        self.n_train          = n_train
        self.max_cluster_size = max_cluster_size
        self.beta             = beta
        self.lr               = lr

        # Components
        self.fe = FeatureExtractor(lambdas=lambdas)
        self.fm = FeatureMapper(
            n_features=N_FEATURES,
            max_cluster_size=max_cluster_size,
        )
        self.ad: KitNET | None = None   # initialised after FM fitting

        # State
        self._n_seen   = 0
        self._in_train = True           # True while still in train-mode

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def process(self, pkt: dict) -> float | None:
        """
        Process one packet.

        Parameters
        ----------
        pkt : dict
            Keys: src_mac, src_ip, dst_ip, src_port, dst_port, proto,
                  size, time.  (See FeatureExtractor.update for details.)

        Returns
        -------
        score : float or None
            Anomaly score in [0, ∞) during exec-mode.
            Returns None during train-mode (model is being trained).
        """
        # Step 1: Extract 115-dim feature vector
        x = self.fe.update(pkt)

        self._n_seen += 1

        if self._in_train:
            # ---- Train-mode ----
            self.fm.update(x)

            # Transition to exec-mode after n_train packets
            if self._n_seen == self.n_train:
                self._switch_to_exec()

            # After AD is initialised, also train KitNET on this instance.
            # (KitNET is initialised at the exact moment FM fits, i.e. at
            #  the n_train-th packet, so training starts from that packet.)
            if self.ad is not None:
                vi = self.fm.transform(x)
                self.ad.train(vi)

            return None  # no score during train-mode

        else:
            # ---- Exec-mode ----
            vi    = self.fm.transform(x)
            score = self.ad.execute(vi)
            return score

    def process_batch(
        self, packets: list[dict]
    ) -> tuple[list[float], list[int]]:
        """
        Convenience wrapper: process a list of packets.

        Returns
        -------
        scores  : list of float
            Anomaly scores (only for exec-mode packets).
        indices : list of int
            Corresponding packet indices in the input list.
        """
        scores, indices = [], []
        for i, pkt in enumerate(packets):
            s = self.process(pkt)
            if s is not None:
                scores.append(s)
                indices.append(i)
        return scores, indices

    # ------------------------------------------------------------------ #
    # Mode transition
    # ------------------------------------------------------------------ #

    def _switch_to_exec(self) -> None:
        """Called exactly once: fit FM then initialise KitNET."""
        logger.info(
            "Kitsune: switching to exec-mode after %d training packets.",
            self.n_train,
        )
        self.fm.fit()
        logger.info(
            "FeatureMapper fitted: k=%d clusters, sizes=%s",
            self.fm.n_clusters,
            self.fm.cluster_sizes,
        )
        self.ad = KitNET(
            cluster_sizes=self.fm.cluster_sizes,
            beta=self.beta,
            lr=self.lr,
        )
        self._in_train = False

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def in_train_mode(self) -> bool:
        return self._in_train

    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def threshold(self) -> float:
        """φ — maximum RMSE seen during training (naive threshold)."""
        if self.ad is None:
            return 0.0
        return self.ad.phi

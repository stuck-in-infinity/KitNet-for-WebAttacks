"""
core/kitnet.py
--------------
KitNET (Kitsune NETwork) Anomaly Detector.

Implements the two-layer stacked autoencoder ensemble described in
Mirsky et al., NDSS 2018, Section IV-D.

Architecture
------------
Ensemble Layer L^(1): k three-layer autoencoders theta_1 ... theta_k
    - theta_i receives sub-instance v_i of dimension dim(v_i)
    - hidden layer: ceil(beta * dim(v_i)) neurons, beta = 3/4
    - output: RMSE reconstruction error z~[i]

Output Layer L^(2): single autoencoder theta_0
    - receives 0-1 normalised RMSE vector z from L^(1)
    - hidden layer: ceil(k * beta) neurons
    - output: final anomaly score s = RMSE(z', y_0)

Training: online SGD, one weight update per instance (no batch, no replay).
Activation: sigmoid throughout.
Normalisation: per-feature 0-1 normalisation using running min/max,
               updated ONLY during train-mode and frozen during exec-mode.
"""

from __future__ import annotations
import numpy as np


# ======================================================================= #
#  Autoencoder (3-layer: input -> hidden -> output)
# ======================================================================= #

class Autoencoder:
    """
    A single three-layer autoencoder trained with online SGD.

    Parameters
    ----------
    n_visible : int
        Number of input (and output) neurons.
    n_hidden  : int
        Number of hidden neurons.  Paper uses ceil(beta * n_visible), beta=3/4.
    lr        : float
        SGD learning rate (default 0.1 as used in the paper).
    """

    def __init__(self, n_visible: int, n_hidden: int, lr: float = 0.1):
        self.n_v = n_visible
        self.n_h = n_hidden
        self.lr  = lr

        # Weight initialisation: U(-1/n_visible, 1/n_visible) per paper
        lim = 1.0 / max(n_visible, 1)
        rng = np.random.default_rng()
        self.W1 = rng.uniform(-lim, lim, (n_hidden, n_visible)).astype(np.float64)
        self.b1 = np.zeros(n_hidden, dtype=np.float64)
        self.W2 = rng.uniform(-lim, lim, (n_visible, n_hidden)).astype(np.float64)
        self.b2 = np.zeros(n_visible, dtype=np.float64)

        # Running min/max for 0-1 normalisation — frozen after training
        self._min = np.full(n_visible, np.inf,  dtype=np.float64)
        self._max = np.full(n_visible, -np.inf, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Normalisation
    # ------------------------------------------------------------------ #

    def _update_norm(self, x: np.ndarray) -> None:
        """Update running min/max (train-mode only)."""
        self._min = np.minimum(self._min, x)
        self._max = np.maximum(self._max, x)

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        """Apply 0-1 normalisation using current min/max."""
        denom = self._max - self._min
        denom[denom == 0] = 1.0
        return (x - self._min) / denom

    # ------------------------------------------------------------------ #
    # Activation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Clip to prevent overflow in exp, then apply stable sigmoid
        z = np.clip(z, -500.0, 500.0)
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z))
        )

    @staticmethod
    def _sigmoid_deriv(a: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid given the activation value a = sigmoid(z)."""
        return a * (1.0 - a)

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def _forward(self, x_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (hidden activations, output activations).
        """
        h = self._sigmoid(self.W1 @ x_norm + self.b1)
        y = self._sigmoid(self.W2 @ h    + self.b2)
        return h, y

    @staticmethod
    def _rmse(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((x - y) ** 2)))

    # ------------------------------------------------------------------ #
    # Train one instance (online SGD + backpropagation)
    # ------------------------------------------------------------------ #

    def train(self, x: np.ndarray) -> float:
        """
        Train on a single instance x.
        Returns reconstruction RMSE (before weight update).

        Parameters
        ----------
        x : np.ndarray, shape (n_visible,)
        """
        self._update_norm(x)
        x_norm = self._normalise(x)

        # Forward pass
        h, y = self._forward(x_norm)

        # Reconstruction error (used as the signal for L^(2))
        rmse = self._rmse(x_norm, y)

        # Backpropagation
        # Output layer delta
        delta2 = (y - x_norm) * self._sigmoid_deriv(y)          # (n_v,)
        # Hidden layer delta
        delta1 = (self.W2.T @ delta2) * self._sigmoid_deriv(h)  # (n_h,)

        # Weight updates (gradient descent)
        self.W2 -= self.lr * np.outer(delta2, h)
        self.b2 -= self.lr * delta2
        self.W1 -= self.lr * np.outer(delta1, x_norm)
        self.b1 -= self.lr * delta1

        return rmse

    # ------------------------------------------------------------------ #
    # Execute (inference only)
    # ------------------------------------------------------------------ #

    def execute(self, x: np.ndarray) -> float:
        """
        Compute reconstruction RMSE for x without updating weights.

        Parameters
        ----------
        x : np.ndarray, shape (n_visible,)
        """
        x_norm = self._normalise(x)
        _, y   = self._forward(x_norm)
        return self._rmse(x_norm, y)


# ======================================================================= #
#  KitNET
# ======================================================================= #

class KitNET:
    """
    KitNET anomaly detector.

    Parameters
    ----------
    cluster_sizes : list of int
        Sizes of the k feature sub-instances produced by the FeatureMapper.
        This defines the Ensemble Layer architecture.
    beta : float
        Compression ratio for hidden layers (default 3/4).
    lr : float
        SGD learning rate (default 0.1).
    """

    def __init__(
        self,
        cluster_sizes: list[int],
        beta: float = 0.75,
        lr:   float = 0.1,
    ):
        self.k    = len(cluster_sizes)
        self.beta = beta
        self.lr   = lr

        # ---- Ensemble Layer L^(1) ----------------------------------------
        self.ensemble: list[Autoencoder] = []
        for sz in cluster_sizes:
            n_h = max(1, int(np.ceil(beta * sz)))
            self.ensemble.append(Autoencoder(sz, n_h, lr))

        # ---- Output Layer L^(2) ------------------------------------------
        n_h_out = max(1, int(np.ceil(beta * self.k)))
        self.output_ae = Autoencoder(self.k, n_h_out, lr)

        # Maximum RMSE score seen during training (used to set threshold φ)
        self.phi: float = 0.0

    # ------------------------------------------------------------------ #
    # Train one instance
    # ------------------------------------------------------------------ #

    def train(self, sub_instances: list[np.ndarray]) -> float:
        """
        Train KitNET on one packet's sub-instances.

        Parameters
        ----------
        sub_instances : list of np.ndarray
            k sub-instances from FeatureMapper.transform().

        Returns
        -------
        score : float
            Anomaly score for this instance.
        """
        # Step 1: Train Ensemble Layer, collect RMSE error signals
        z = np.zeros(self.k, dtype=np.float64)
        for i, (ae, vi) in enumerate(zip(self.ensemble, sub_instances)):
            z[i] = ae.train(vi)

        # Step 2: Train Output Layer on the RMSE vector
        score = self.output_ae.train(z)

        # Track maximum score during training for threshold φ
        self.phi = max(self.phi, score)
        return score

    # ------------------------------------------------------------------ #
    # Execute (inference)
    # ------------------------------------------------------------------ #

    def execute(self, sub_instances: list[np.ndarray]) -> float:
        """
        Compute anomaly score for one instance without updating weights.

        Parameters
        ----------
        sub_instances : list of np.ndarray

        Returns
        -------
        score : float  ∈ [0, ∞)
            Higher score = more anomalous.
        """
        z = np.zeros(self.k, dtype=np.float64)
        for i, (ae, vi) in enumerate(zip(self.ensemble, sub_instances)):
            z[i] = ae.execute(vi)
        return self.output_ae.execute(z)

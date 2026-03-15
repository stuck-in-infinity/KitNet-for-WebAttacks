"""KitNET anomaly detector with stacked autoencoders."""

from __future__ import annotations
import numpy as np


class Autoencoder:
    """Three-layer autoencoder trained online with SGD."""

    def __init__(self, n_visible: int, n_hidden: int, lr: float = 0.1):
        self.n_v = n_visible
        self.n_h = n_hidden
        self.lr = lr

        lim = 1.0 / max(n_visible, 1)
        rng = np.random.default_rng()
        self.W1 = rng.uniform(-lim, lim, (n_hidden, n_visible)).astype(np.float64)
        self.b1 = np.zeros(n_hidden, dtype=np.float64)
        self.W2 = rng.uniform(-lim, lim, (n_visible, n_hidden)).astype(np.float64)
        self.b2 = np.zeros(n_visible, dtype=np.float64)

        self._min = np.full(n_visible, np.inf, dtype=np.float64)
        self._max = np.full(n_visible, -np.inf, dtype=np.float64)

    def _update_norm(self, x: np.ndarray) -> None:
        # Expand running bounds so later normalisation stays in-range.
        self._min = np.minimum(self._min, x)
        self._max = np.maximum(self._max, x)

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        denom = self._max - self._min
        denom[denom == 0] = 1.0
        return (x - self._min) / denom

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500.0, 500.0)
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    @staticmethod
    def _sigmoid_deriv(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    def _forward(self, x_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = self._sigmoid(self.W1 @ x_norm + self.b1)
        y = self._sigmoid(self.W2 @ h + self.b2)
        return h, y

    @staticmethod
    def _rmse(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((x - y) ** 2)))

    def train(self, x: np.ndarray) -> float:
        self._update_norm(x)
        x_norm = self._normalise(x)

        h, y = self._forward(x_norm)
        rmse = self._rmse(x_norm, y)

        # Backprop through the two-layer stack with the current sample.
        delta2 = (y - x_norm) * self._sigmoid_deriv(y)
        delta1 = (self.W2.T @ delta2) * self._sigmoid_deriv(h)

        self.W2 -= self.lr * np.outer(delta2, h)
        self.b2 -= self.lr * delta2
        self.W1 -= self.lr * np.outer(delta1, x_norm)
        self.b1 -= self.lr * delta1

        return rmse

    def execute(self, x: np.ndarray) -> float:
        x_norm = self._normalise(x)
        _, y = self._forward(x_norm)
        return self._rmse(x_norm, y)


class KitNET:
    """Ensemble of autoencoders producing an anomaly score."""

    def __init__(
        self,
        cluster_sizes: list[int],
        beta: float = 0.75,
        lr: float = 0.1,
    ):
        self.k = len(cluster_sizes)
        self.beta = beta
        self.lr = lr

        self.ensemble: list[Autoencoder] = []
        for sz in cluster_sizes:
            n_h = max(1, int(np.ceil(beta * sz)))
            self.ensemble.append(Autoencoder(sz, n_h, lr))

        n_h_out = max(1, int(np.ceil(beta * self.k)))
        self.output_ae = Autoencoder(self.k, n_h_out, lr)
        self.phi: float = 0.0

    def train(self, sub_instances: list[np.ndarray]) -> float:
        z = np.zeros(self.k, dtype=np.float64)
        for i, (ae, vi) in enumerate(zip(self.ensemble, sub_instances)):
            # Each base autoencoder learns its slice of the feature vector.
            z[i] = ae.train(vi)

        # Output autoencoder learns to reconstruct the ensemble errors.
        score = self.output_ae.train(z)
        self.phi = max(self.phi, score)
        return score

    def execute(self, sub_instances: list[np.ndarray]) -> float:
        z = np.zeros(self.k, dtype=np.float64)
        for i, (ae, vi) in enumerate(zip(self.ensemble, sub_instances)):
            z[i] = ae.execute(vi)
        return self.output_ae.execute(z)

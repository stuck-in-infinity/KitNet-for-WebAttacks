from __future__ import annotations
import numpy as np


class Autoencoder:
    # Three-layer autoencoder trained online with SGD.
    # Normalisation bounds are recorded during training and frozen at inference.

    def __init__(self, n_visible: int, n_hidden: int, lr: float = 0.1):
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.lr        = lr

        bound = 1.0 / max(n_visible, 1)
        rng   = np.random.default_rng()
        self.W_enc = rng.uniform(-bound, bound, (n_hidden,  n_visible)).astype(np.float64)
        self.b_enc = np.zeros(n_hidden,  dtype=np.float64)
        self.W_dec = rng.uniform(-bound, bound, (n_visible, n_hidden )).astype(np.float64)
        self.b_dec = np.zeros(n_visible, dtype=np.float64)

        self._feat_min = np.full(n_visible,  np.inf, dtype=np.float64)
        self._feat_max = np.full(n_visible, -np.inf, dtype=np.float64)

    def _record_bounds(self, x: np.ndarray) -> None:
        self._feat_min = np.minimum(self._feat_min, x)
        self._feat_max = np.maximum(self._feat_max, x)

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        span = self._feat_max - self._feat_min
        span[span == 0] = 1.0
        return (x - self._feat_min) / span

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500.0, 500.0)
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    @staticmethod
    def _sigmoid_grad(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    def _forward(self, x_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = self._sigmoid(self.W_enc @ x_norm + self.b_enc)
        output = self._sigmoid(self.W_dec @ hidden  + self.b_dec)
        return hidden, output

    @staticmethod
    def _rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
        return float(np.sqrt(np.mean((original - reconstructed) ** 2)))

    def train(self, x: np.ndarray) -> float:
        self._record_bounds(x)
        x_norm = self._normalise(x)
        hidden, output = self._forward(x_norm)
        error = self._rmse(x_norm, output)

        # Backpropagation through output then hidden layer
        out_delta    = (output - x_norm) * self._sigmoid_grad(output)
        hidden_delta = (self.W_dec.T @ out_delta) * self._sigmoid_grad(hidden)

        self.W_dec -= self.lr * np.outer(out_delta,    hidden)
        self.b_dec -= self.lr * out_delta
        self.W_enc -= self.lr * np.outer(hidden_delta, x_norm)
        self.b_enc -= self.lr * hidden_delta

        return error

    def execute(self, x: np.ndarray) -> float:
        # Inference only — no weight update.
        x_norm = self._normalise(x)
        _, output = self._forward(x_norm)
        return self._rmse(x_norm, output)

    # Backward-compatible aliases
    @property
    def W1(self) -> np.ndarray:
        return self.W_enc

    @property
    def W2(self) -> np.ndarray:
        return self.W_dec

    @property
    def n_v(self) -> int:
        return self.n_visible

    @property
    def n_h(self) -> int:
        return self.n_hidden


class KitNET:
    # Ensemble of small autoencoders (one per feature cluster) plus
    # an output autoencoder that aggregates their RMSE errors.

    def __init__(self, cluster_sizes: list[int], beta: float = 0.75, lr: float = 0.1):
        self.k    = len(cluster_sizes)
        self.beta = beta
        self.lr   = lr

        self.ensemble: list[Autoencoder] = [
            Autoencoder(sz, max(1, int(np.ceil(beta * sz))), lr)
            for sz in cluster_sizes
        ]
        self.output_layer = Autoencoder(self.k, max(1, int(np.ceil(beta * self.k))), lr)
        self.phi: float = 0.0

    def train(self, sub_instances: list[np.ndarray]) -> float:
        error_signal = np.zeros(self.k, dtype=np.float64)
        for i, (ae, vi) in enumerate(zip(self.ensemble, sub_instances)):
            error_signal[i] = ae.train(vi)
        score    = self.output_layer.train(error_signal)
        self.phi = max(self.phi, score)
        return score

    def execute(self, sub_instances: list[np.ndarray]) -> float:
        error_signal = np.zeros(self.k, dtype=np.float64)
        for i, (ae, vi) in enumerate(zip(self.ensemble, sub_instances)):
            error_signal[i] = ae.execute(vi)
        return self.output_layer.execute(error_signal)

    @property
    def output_ae(self) -> Autoencoder:
        return self.output_layer

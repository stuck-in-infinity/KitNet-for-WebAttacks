import numpy as np


class IncStat1D:
    # Tracks exponentially-decayed statistics for one data stream.
    # Stores weight, linear sum, and squared sum updated per observation.

    def __init__(self, decay_factor: float):
        self.decay_factor = decay_factor
        self.weight:    float = 0.0
        self.lin_sum:   float = 0.0
        self.sq_sum:    float = 0.0
        self.last_time: float = -1.0

    def _decay(self, t_now: float) -> None:
        # Downweight all accumulators before inserting a new value.
        if self.last_time < 0:
            return
        gap   = max(0.0, t_now - self.last_time)
        gamma = pow(2.0, -self.decay_factor * gap)
        self.weight  *= gamma
        self.lin_sum *= gamma
        self.sq_sum  *= gamma

    def update(self, value: float, t_now: float) -> None:
        self._decay(t_now)
        self.weight   += 1.0
        self.lin_sum  += value
        self.sq_sum   += value * value
        self.last_time = t_now

    # Alias kept for backward compatibility
    def insert(self, value: float, t_now: float) -> None:
        self.update(value, t_now)

    @property
    def mean(self) -> float:
        return 0.0 if self.weight == 0 else self.lin_sum / self.weight

    @property
    def variance(self) -> float:
        if self.weight == 0:
            return 0.0
        return abs(self.sq_sum / self.weight - (self.lin_sum / self.weight) ** 2)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def residual(self, value: float) -> float:
        return value - self.mean

    # Backward-compatible aliases
    @property
    def w(self) -> float:
        return self.weight

    @property
    def var(self) -> float:
        return self.variance

    @property
    def LS(self) -> float:
        return self.lin_sum

    @property
    def SS(self) -> float:
        return self.sq_sum

    @property
    def T_last(self) -> float:
        return self.last_time


class IncStat2D:
    # Tracks cross-channel statistics between two IncStat1D streams.
    # Maintains a decayed sum of paired residual products.

    def __init__(self, stream_a: IncStat1D, stream_b: IncStat1D, decay_factor: float):
        self.stream_a     = stream_a
        self.stream_b     = stream_b
        self.decay_factor = decay_factor
        self.resid_prod:  float = 0.0
        self.last_time:   float = -1.0

    def update(self, val_a: float, val_b: float, t_now: float) -> None:
        if self.last_time >= 0:
            gap   = max(0.0, t_now - self.last_time)
            gamma = pow(2.0, -self.decay_factor * gap)
            self.resid_prod *= gamma
        self.resid_prod += self.stream_a.residual(val_a) * self.stream_b.residual(val_b)
        self.last_time   = t_now

    def insert(self, val_a: float, val_b: float, t_now: float) -> None:
        self.update(val_a, val_b, t_now)

    @property
    def magnitude(self) -> float:
        return float(np.sqrt(self.stream_a.mean ** 2 + self.stream_b.mean ** 2))

    @property
    def radius(self) -> float:
        return float(np.sqrt(self.stream_a.variance + self.stream_b.variance))

    @property
    def covariance(self) -> float:
        total = self.stream_a.weight + self.stream_b.weight
        return 0.0 if total == 0 else self.resid_prod / total

    @property
    def correlation(self) -> float:
        denom = self.stream_a.std * self.stream_b.std
        return 0.0 if denom == 0 else self.covariance / denom

    # Backward-compatible aliases
    @property
    def cov(self) -> float:
        return self.covariance

    @property
    def pcc(self) -> float:
        return self.correlation

    @property
    def SR(self) -> float:
        return self.resid_prod

    @property
    def s1(self) -> IncStat1D:
        return self.stream_a

    @property
    def s2(self) -> IncStat1D:
        return self.stream_b
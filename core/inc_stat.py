"""Damped incremental 1D and 2D statistics for Kitsune."""

import numpy as np


class IncStat1D:
    """Exponentially weighted statistics for a single stream."""

    def __init__(self, decay: float):
        self.decay = decay
        self.w: float = 0.0
        self.LS: float = 0.0
        self.SS: float = 0.0
        self.T_last: float = -1.0

    def _apply_decay(self, t_cur: float) -> None:
        if self.T_last < 0:
            return
        elapsed = max(0.0, t_cur - self.T_last)
        gamma = pow(2.0, -self.decay * elapsed)
        self.w *= gamma
        self.LS *= gamma
        self.SS *= gamma

    def insert(self, x: float, t_cur: float) -> None:
        self._apply_decay(t_cur)
        self.w += 1.0
        self.LS += x
        self.SS += x * x
        self.T_last = t_cur

    @property
    def mean(self) -> float:
        return 0.0 if self.w == 0 else self.LS / self.w

    @property
    def var(self) -> float:
        if self.w == 0:
            return 0.0
        v = self.SS / self.w - (self.LS / self.w) ** 2
        return abs(v)

    @property
    def std(self) -> float:
        return np.sqrt(self.var)

    def residual(self, x: float) -> float:
        return x - self.mean


class IncStat2D:
    """Cross statistics between two IncStat1D streams."""

    def __init__(self, s1: IncStat1D, s2: IncStat1D, decay: float):
        self.s1 = s1
        self.s2 = s2
        self.decay = decay
        self.SR: float = 0.0
        self.T_last: float = -1.0

    def insert(self, x1: float, x2: float, t_cur: float) -> None:
        if self.T_last >= 0:
            elapsed = max(0.0, t_cur - self.T_last)
            gamma = pow(2.0, -self.decay * elapsed)
            self.SR *= gamma
        r1 = self.s1.residual(x1)
        r2 = self.s2.residual(x2)
        self.SR += r1 * r2
        self.T_last = t_cur

    @property
    def magnitude(self) -> float:
        return np.sqrt(self.s1.mean ** 2 + self.s2.mean ** 2)

    @property
    def radius(self) -> float:
        return np.sqrt(self.s1.var + self.s2.var)

    @property
    def cov(self) -> float:
        denom = self.s1.w + self.s2.w
        return 0.0 if denom == 0 else self.SR / denom

    @property
    def pcc(self) -> float:
        denom = self.s1.std * self.s2.std
        return 0.0 if denom == 0 else self.cov / denom

"""
core/inc_stat.py
----------------
Damped Incremental Statistics (1D and 2D) as described in
Mirsky et al., "Kitsune", NDSS 2018, Section IV-B.

Each channel maintains a tuple (w, LS, SS, SR_ij, T_last) and applies
an exponential decay d_lambda(t) = 2^(-lambda * elapsed) before inserting
every new observation, giving O(1) time and O(1) memory per update.
"""

import numpy as np


class IncStat1D:
    """
    One-dimensional damped incremental statistic for a single channel.
    Tracks: weight (w), linear sum (LS), squared sum (SS), last timestamp.
    Provides: mean, variance, std.
    """

    def __init__(self, decay: float):
        """
        Parameters
        ----------
        decay : float
            Lambda (λ > 0). Larger values = faster forgetting.
        """
        self.decay = decay
        self.w: float = 0.0        # current weight (dampened count)
        self.LS: float = 0.0       # dampened linear sum
        self.SS: float = 0.0       # dampened squared sum
        self.T_last: float = -1.0  # timestamp of last update

    def _apply_decay(self, t_cur: float) -> None:
        """Exponentially downweight all accumulated sums before inserting."""
        if self.T_last < 0:
            return
        elapsed = t_cur - self.T_last
        if elapsed < 0:
            elapsed = 0.0
        gamma = pow(2.0, -self.decay * elapsed)
        self.w  *= gamma
        self.LS *= gamma
        self.SS *= gamma

    def insert(self, x: float, t_cur: float) -> None:
        """Insert a new observation x at time t_cur."""
        self._apply_decay(t_cur)
        self.w   += 1.0
        self.LS  += x
        self.SS  += x * x
        self.T_last = t_cur

    # ------------------------------------------------------------------ #
    # Derived statistics (all O(1))
    # ------------------------------------------------------------------ #

    @property
    def mean(self) -> float:
        if self.w == 0:
            return 0.0
        return self.LS / self.w

    @property
    def var(self) -> float:
        if self.w == 0:
            return 0.0
        v = self.SS / self.w - (self.LS / self.w) ** 2
        return abs(v)   # numerical noise can make it slightly negative

    @property
    def std(self) -> float:
        return np.sqrt(self.var)

    def residual(self, x: float) -> float:
        """Residual of x with respect to current mean."""
        return x - self.mean


class IncStat2D:
    """
    Two-dimensional cross-channel statistic between two IncStat1D channels.
    Tracks: SR_ij (sum of residual products) to compute covariance /
    correlation, plus radius and magnitude.

    Paper reference: Table I and Section IV-B.
    """

    def __init__(self, s1: IncStat1D, s2: IncStat1D, decay: float):
        self.s1 = s1
        self.s2 = s2
        self.decay = decay
        self.SR: float = 0.0       # sum of residual products
        self.T_last: float = -1.0

    def insert(self, x1: float, x2: float, t_cur: float) -> None:
        """
        Update cross-channel statistic.  The two 1D stats MUST be updated
        before calling this method for the same timestamp.
        """
        if self.T_last >= 0:
            elapsed = max(0.0, t_cur - self.T_last)
            gamma = pow(2.0, -self.decay * elapsed)
            self.SR *= gamma
        r1 = self.s1.residual(x1)
        r2 = self.s2.residual(x2)
        self.SR += r1 * r2
        self.T_last = t_cur

    # ------------------------------------------------------------------ #
    # Derived 2D statistics
    # ------------------------------------------------------------------ #

    @property
    def magnitude(self) -> float:
        """||S_i, S_j|| = sqrt(mu_i^2 + mu_j^2)"""
        return np.sqrt(self.s1.mean ** 2 + self.s2.mean ** 2)

    @property
    def radius(self) -> float:
        """R_{S_i S_j} = sqrt(sigma_i^2 + sigma_j^2)"""
        return np.sqrt(self.s1.var + self.s2.var)

    @property
    def cov(self) -> float:
        """Approx. covariance: SR / (w_i + w_j)"""
        denom = self.s1.w + self.s2.w
        if denom == 0:
            return 0.0
        return self.SR / denom

    @property
    def pcc(self) -> float:
        """Pearson correlation coefficient: Cov / (sigma_i * sigma_j)"""
        denom = self.s1.std * self.s2.std
        if denom == 0:
            return 0.0
        return self.cov / denom

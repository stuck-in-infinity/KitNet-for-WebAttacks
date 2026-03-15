"""
core/feature_extractor.py
--------------------------
Feature Extractor (FE) component of Kitsune.

Extracts 115 statistical features per packet across 5 time-windows
(lambda = 5, 3, 1, 0.1, 0.01 seconds) and 4 channel types:
    - SrcMAC-IP  : identified by (src_mac, src_ip)
    - SrcIP      : identified by src_ip
    - Channel    : identified by (src_ip, dst_ip)
    - Socket     : identified by (src_ip, dst_ip, src_port, dst_port, proto)

Per window, 23 statistics are extracted (see Table II of the paper):
    8  size statistics       (mu, sigma for 4 channel types)
    8  2D size statistics    (magnitude, radius, cov, pcc for Channel+Socket)
    4  count statistics      (weight for 4 channel types)
    3  jitter statistics     (weight, mu, sigma for Channel)
    ---
    23 features × 5 windows = 115 total

Paper reference: Mirsky et al., NDSS 2018, Section IV-B and Table II.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict
from .inc_stat import IncStat1D, IncStat2D

# Five decay (lambda) values used in the paper
LAMBDAS = [5.0, 3.0, 1.0, 0.1, 0.01]

# Number of features per time window
STATS_PER_WINDOW = 23

# Total feature dimensionality
N_FEATURES = len(LAMBDAS) * STATS_PER_WINDOW  # = 115


class ChannelStats:
    """
    Holds all IncStat1D objects for a single (channel_key, lambda) pair,
    tracking packet-size and jitter (inter-arrival time).
    """

    def __init__(self, decay: float):
        self.size  = IncStat1D(decay)   # packet size stream S_i
        self.jitter = IncStat1D(decay)  # inter-arrival time stream

    def update(self, pkt_size: float, t_cur: float) -> None:
        # Compute jitter before updating (needs last timestamp)
        if self.size.T_last >= 0:
            jitter_val = t_cur - self.size.T_last
        else:
            jitter_val = 0.0
        self.size.insert(pkt_size, t_cur)
        self.jitter.insert(jitter_val, t_cur)


class FeatureExtractor:
    """
    Stateful feature extractor.  Call update(pkt) for each arriving packet;
    returns a 115-dimensional numpy feature vector.

    Parameters
    ----------
    lambdas : list of float
        Decay factors. Default: [5, 3, 1, 0.1, 0.01] as in the paper.
    """

    def __init__(self, lambdas: list[float] = None):
        self.lambdas = lambdas if lambdas is not None else LAMBDAS

        # Nested dicts: channel_key -> lambda_idx -> ChannelStats
        self._stats: dict[str, dict[int, ChannelStats]] = defaultdict(dict)

        # 2D cross-channel stats: (chan_key_i, chan_key_j, lam_idx) -> IncStat2D
        self._stats2d: dict[tuple, IncStat2D] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_1d(self, key: str, lam_idx: int, decay: float) -> ChannelStats:
        if lam_idx not in self._stats[key]:
            self._stats[key][lam_idx] = ChannelStats(decay)
        return self._stats[key][lam_idx]

    def _get_2d(
        self, k1: str, k2: str, lam_idx: int,
        s1: IncStat1D, s2: IncStat1D, decay: float
    ) -> IncStat2D:
        key = (k1, k2, lam_idx)
        if key not in self._stats2d:
            self._stats2d[key] = IncStat2D(s1, s2, decay)
        return self._stats2d[key]

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def update(self, pkt: dict) -> np.ndarray:
        """
        Process one packet and return its 115-dim feature vector.

        Expected keys in pkt
        --------------------
        src_mac  : str   e.g. "aa:bb:cc:dd:ee:ff"
        src_ip   : str   e.g. "192.168.1.1"
        dst_ip   : str
        src_port : int   (0 if not TCP/UDP)
        dst_port : int   (0 if not TCP/UDP)
        proto    : str   e.g. "TCP", "UDP", "ICMP"
        size     : float packet size in bytes
        time     : float Unix timestamp (seconds, float)
        """
        src_mac  = pkt.get("src_mac",  "00:00:00:00:00:00")
        src_ip   = pkt.get("src_ip",   "0.0.0.0")
        dst_ip   = pkt.get("dst_ip",   "0.0.0.0")
        src_port = pkt.get("src_port", 0)
        dst_port = pkt.get("dst_port", 0)
        proto    = pkt.get("proto",    "OTHER")
        size     = float(pkt.get("size", 0))
        t        = float(pkt["time"])

        # Build channel keys (strings used as dict keys)
        key_mac_ip  = f"{src_mac}:{src_ip}"
        key_src_ip  = src_ip
        key_channel = f"{src_ip}->{dst_ip}"
        key_socket  = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}:{proto}"

        features = []

        for li, lam in enumerate(self.lambdas):
            cs_mac   = self._get_1d(key_mac_ip,  li, lam)
            cs_src   = self._get_1d(key_src_ip,  li, lam)
            cs_chan  = self._get_1d(key_channel, li, lam)
            cs_sock  = self._get_1d(key_socket,  li, lam)

            # --- Update all 1D stats BEFORE reading residuals for 2D ---
            cs_mac.update(size, t)
            cs_src.update(size, t)
            cs_chan.update(size, t)
            cs_sock.update(size, t)

            # ----------------------------------------------------------
            # 8 size statistics: (mu, sigma) × 4 channels
            # ----------------------------------------------------------
            f_size = [
                cs_mac.size.mean,  cs_mac.size.std,
                cs_src.size.mean,  cs_src.size.std,
                cs_chan.size.mean, cs_chan.size.std,
                cs_sock.size.mean, cs_sock.size.std,
            ]

            # ----------------------------------------------------------
            # 8 2D size statistics: (magnitude, radius, cov, pcc)
            #   for (Channel, Socket) pair
            # ----------------------------------------------------------
            s2d_chan_sock = self._get_2d(
                key_channel, key_socket, li,
                cs_chan.size, cs_sock.size, lam
            )
            s2d_chan_sock.insert(size, size, t)

            f_size2d = [
                s2d_chan_sock.magnitude,
                s2d_chan_sock.radius,
                s2d_chan_sock.cov,
                s2d_chan_sock.pcc,
                # Second pair: SrcMAC-IP vs SrcIP (inbound/outbound relation)
                self._get_2d(key_mac_ip, key_src_ip, li,
                             cs_mac.size, cs_src.size, lam).magnitude,
                self._get_2d(key_mac_ip, key_src_ip, li,
                             cs_mac.size, cs_src.size, lam).radius,
                self._get_2d(key_mac_ip, key_src_ip, li,
                             cs_mac.size, cs_src.size, lam).cov,
                self._get_2d(key_mac_ip, key_src_ip, li,
                             cs_mac.size, cs_src.size, lam).pcc,
            ]
            # Keep insertion for the second pair too
            self._get_2d(key_mac_ip, key_src_ip, li,
                         cs_mac.size, cs_src.size, lam).insert(size, size, t)

            # ----------------------------------------------------------
            # 4 count statistics: weight × 4 channels
            # ----------------------------------------------------------
            f_count = [
                cs_mac.size.w,
                cs_src.size.w,
                cs_chan.size.w,
                cs_sock.size.w,
            ]

            # ----------------------------------------------------------
            # 3 jitter statistics: (weight, mu, sigma) for Channel only
            # ----------------------------------------------------------
            f_jitter = [
                cs_chan.jitter.w,
                cs_chan.jitter.mean,
                cs_chan.jitter.std,
            ]

            features.extend(f_size)    # 8
            features.extend(f_size2d)  # 8
            features.extend(f_count)   # 4
            features.extend(f_jitter)  # 3
            # = 23 per window

        return np.array(features, dtype=np.float32)  # shape (115,)

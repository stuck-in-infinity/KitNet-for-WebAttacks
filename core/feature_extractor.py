"""Extracts 115 Kitsune packet features across multiple decay windows."""

from __future__ import annotations
import numpy as np
from collections import defaultdict
from .inc_stat import IncStat1D, IncStat2D

LAMBDAS = [5.0, 3.0, 1.0, 0.1, 0.01]
STATS_PER_WINDOW = 23
N_FEATURES = len(LAMBDAS) * STATS_PER_WINDOW  # 115


class ChannelStats:
    """Incremental size and jitter stats for one channel."""

    def __init__(self, decay: float):
        self.size = IncStat1D(decay)
        self.jitter = IncStat1D(decay)

    def update(self, pkt_size: float, t_cur: float) -> None:
        jitter_val = t_cur - self.size.T_last if self.size.T_last >= 0 else 0.0
        self.size.insert(pkt_size, t_cur)
        self.jitter.insert(jitter_val, t_cur)


class FeatureExtractor:
    """Stateful extractor returning a 115-D feature vector per packet."""

    def __init__(self, lambdas: list[float] = None):
        self.lambdas = lambdas if lambdas is not None else LAMBDAS
        self._stats: dict[str, dict[int, ChannelStats]] = defaultdict(dict)
        self._stats2d: dict[tuple, IncStat2D] = {}

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

    def update(self, pkt: dict) -> np.ndarray:
        """Process one packet and return its 115-D feature vector."""
        src_mac  = pkt.get("src_mac",  "00:00:00:00:00:00")
        src_ip   = pkt.get("src_ip",   "0.0.0.0")
        dst_ip   = pkt.get("dst_ip",   "0.0.0.0")
        src_port = pkt.get("src_port", 0)
        dst_port = pkt.get("dst_port", 0)
        proto    = pkt.get("proto",    "OTHER")
        size     = float(pkt.get("size", 0))
        t        = float(pkt["time"])

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

            cs_mac.update(size, t)
            cs_src.update(size, t)
            cs_chan.update(size, t)
            cs_sock.update(size, t)

            f_size = [
                cs_mac.size.mean,  cs_mac.size.std,
                cs_src.size.mean,  cs_src.size.std,
                cs_chan.size.mean, cs_chan.size.std,
                cs_sock.size.mean, cs_sock.size.std,
            ]

            s2d_chan_sock = self._get_2d(
                key_channel, key_socket, li,
                cs_chan.size, cs_sock.size, lam
            )
            s2d_chan_sock.insert(size, size, t)

            mac_src = self._get_2d(key_mac_ip, key_src_ip, li,
                                   cs_mac.size, cs_src.size, lam)
            mac_src.insert(size, size, t)

            f_size2d = [
                s2d_chan_sock.magnitude,
                s2d_chan_sock.radius,
                s2d_chan_sock.cov,
                s2d_chan_sock.pcc,
                mac_src.magnitude,
                mac_src.radius,
                mac_src.cov,
                mac_src.pcc,
            ]

            f_count = [
                cs_mac.size.w,
                cs_src.size.w,
                cs_chan.size.w,
                cs_sock.size.w,
            ]

            f_jitter = [
                cs_chan.jitter.w,
                cs_chan.jitter.mean,
                cs_chan.jitter.std,
            ]

            features.extend(f_size)
            features.extend(f_size2d)
            features.extend(f_count)
            features.extend(f_jitter)

        return np.array(features, dtype=np.float32)

"""Causal, deterministic inputs shared by offline fitting and live inference."""
import numpy as np

HORIZON = 5
FEATURE_SCHEMA = "service-v8-observable-features-v1"


class Features:
    def __init__(self):
        self.ewma = None
        self.previous = None
        self.step = None

    def update(self, frame):
        q = np.asarray(frame["rx_ratio"], dtype=np.float32)
        positions = np.asarray(frame["positions"], dtype=np.float32)
        velocity = np.asarray(frame["velocities"], dtype=np.float32)
        n = len(q)
        if q.shape != (n, n) or positions.shape != (n, 3) or velocity.shape != (n, 3):
            raise ValueError("Invalid observable dimensions")
        if not all(np.isfinite(x).all() for x in (q, positions, velocity)) or np.any((q < 0) | (q > 1)):
            raise ValueError("Invalid observable values")
        if self.step is not None and frame["step"] != self.step + 1:
            raise ValueError("Frames must be consecutive and causal")
        trend = np.zeros_like(q) if self.previous is None else q - self.previous
        self.ewma = q.copy() if self.ewma is None else .5 * q + .5 * self.ewma
        self.previous = q.copy()
        self.step = frame["step"]
        offdiag = ~np.eye(n, dtype=bool)
        adjacency = (q >= .1) & offdiag
        distance_vectors = positions[:, None, :] - positions[None, :, :]
        velocity_vectors = velocity[:, None, :] - velocity[None, :, :]
        distance = np.linalg.norm(distance_vectors, axis=-1)
        relative_speed = np.linalg.norm(velocity_vectors, axis=-1)
        radial = (distance_vectors * velocity_vectors).sum(axis=-1) / np.maximum(distance, 1e-6)
        rssi = np.asarray(frame["rssi"], dtype=np.float32)
        measured = np.asarray(frame["rssi_mask"], dtype=bool)
        if rssi.shape != q.shape or measured.shape != q.shape or not np.isfinite(rssi).all():
            raise ValueError("Invalid RSSI observations")
        # Missing measurements are encoded deterministically, not filled from future data.
        rssi = np.where(measured, rssi, -100.)
        node = np.concatenate(((positions - positions.mean(axis=0)) / 240., velocity / 10.,
                               adjacency.sum(axis=1, keepdims=True) / max(n - 1, 1),
                               q.sum(axis=1, keepdims=True) / max(n - 1, 1)), axis=1)
        edge = np.stack((distance / 240., (rssi + 90.) / 40., q, self.ewma,
                         trend, radial / 10., relative_speed / 10.), axis=-1)
        edge[~offdiag] = 0
        return node.astype(np.float32), edge.astype(np.float32), adjacency.astype(np.float32)

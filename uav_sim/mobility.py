from __future__ import annotations

import math
import random
from typing import Iterable, Tuple

import config

# Khi UAV chạm vào giới hạn không gian, phản xạ lại và đảo chiều vận tốc tương ứng
def reflect_position(pos: float, vel: float, lower: float, upper: float) -> tuple[float, float]:
    while pos < lower or pos > upper:
        if pos < lower:
            pos = lower + (lower - pos)
            vel = -vel
        elif pos > upper:
            pos = upper - (pos - upper)
            vel = -vel
    return pos, vel

# Cập nhật vị trí của tất cả UAV dựa trên vận tốc và thời gian dt, đồng thời xử lý phản xạ nếu chạm giới hạn
def update_positions(uavs: Iterable[object], dt: float) -> None:
    if config.MOBILITY_MODEL == "gauss-markov":
        _apply_gauss_markov_velocity(uavs)

    for uav in uavs:
        uav.x += uav.vx * dt
        uav.y += uav.vy * dt
        uav.z += uav.vz * dt

        uav.x, uav.vx = reflect_position(uav.x, uav.vx, *config.X_LIMIT)
        uav.y, uav.vy = reflect_position(uav.y, uav.vy, *config.Y_LIMIT)
        uav.z, uav.vz = reflect_position(uav.z, uav.vz, *config.Z_LIMIT)


def _apply_gauss_markov_velocity(uavs: Iterable[object]) -> None:
    alpha = config.GAUSS_MARKOV_ALPHA
    mean_vx, mean_vy, mean_vz = config.GAUSS_MARKOV_MEAN_VELOCITY
    stddev = config.GAUSS_MARKOV_STDDEV

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("GAUSS_MARKOV_ALPHA must be between 0.0 and 1.0")
    if stddev < 0.0:
        raise ValueError("GAUSS_MARKOV_STDDEV must be non-negative")

    blend = math.sqrt(max(0.0, 1.0 - alpha * alpha))

    for uav in uavs:
        uav.vx = alpha * uav.vx + (1.0 - alpha) * mean_vx + blend * stddev * random.gauss(0.0, 1.0)
        uav.vy = alpha * uav.vy + (1.0 - alpha) * mean_vy + blend * stddev * random.gauss(0.0, 1.0)
        uav.vz = alpha * uav.vz + (1.0 - alpha) * mean_vz + blend * stddev * random.gauss(0.0, 1.0)

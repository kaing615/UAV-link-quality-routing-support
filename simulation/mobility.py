from __future__ import annotations

import math
import random
from typing import Iterable

import config


def reflect_position(pos: float, vel: float, lower: float, upper: float) -> tuple[float, float]:
    while pos < lower or pos > upper:
        if pos < lower:
            pos = lower + (lower - pos)
            vel = -vel
        elif pos > upper:
            pos = upper - (pos - upper)
            vel = -vel
    return pos, vel


def update_positions(uavs: Iterable[object], dt: float) -> None:
    model = getattr(config, "MOBILITY_MODEL", "gauss-markov").lower()

    if model == "gauss-markov":
        _apply_gauss_markov_velocity(uavs)
    elif model in {"random-waypoint", "random_waypoint", "rwp"}:
        _apply_random_waypoint(uavs, dt)
    else:
        raise ValueError(f"Unsupported MOBILITY_MODEL: {config.MOBILITY_MODEL}")

    for uav in uavs:
        uav.x += uav.vx * dt
        uav.y += uav.vy * dt
        uav.z += uav.vz * dt

        uav.x, uav.vx = reflect_position(uav.x, uav.vx, *config.X_LIMIT)
        uav.y, uav.vy = reflect_position(uav.y, uav.vy, *config.Y_LIMIT)
        uav.z, uav.vz = reflect_position(uav.z, uav.vz, *config.Z_LIMIT)


def _apply_gauss_markov_velocity(uavs: Iterable[object]) -> None:
    alpha = float(config.GAUSS_MARKOV_ALPHA)
    mean_vx, mean_vy, mean_vz = config.GAUSS_MARKOV_MEAN_VELOCITY
    stddev = float(config.GAUSS_MARKOV_STDDEV)

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("GAUSS_MARKOV_ALPHA must be between 0.0 and 1.0")
    if stddev < 0.0:
        raise ValueError("GAUSS_MARKOV_STDDEV must be non-negative")

    blend = math.sqrt(max(0.0, 1.0 - alpha * alpha))

    for uav in uavs:
        uav.vx = alpha * uav.vx + (1.0 - alpha) * mean_vx + blend * stddev * random.gauss(0.0, 1.0)
        uav.vy = alpha * uav.vy + (1.0 - alpha) * mean_vy + blend * stddev * random.gauss(0.0, 1.0)
        uav.vz = alpha * uav.vz + (1.0 - alpha) * mean_vz + blend * stddev * random.gauss(0.0, 1.0)


def _apply_random_waypoint(uavs: Iterable[object], dt: float) -> None:
    min_speed, max_speed = getattr(config, "RWP_SPEED_RANGE", (4.0, 12.0))
    pause_steps = int(getattr(config, "RWP_PAUSE_STEPS", 0))
    reach_threshold = float(getattr(config, "RWP_REACH_THRESHOLD", 2.0))

    if min_speed <= 0.0 or max_speed < min_speed:
        raise ValueError("RWP_SPEED_RANGE must satisfy 0 < min_speed <= max_speed")
    if pause_steps < 0:
        raise ValueError("RWP_PAUSE_STEPS must be >= 0")
    if reach_threshold < 0.0:
        raise ValueError("RWP_REACH_THRESHOLD must be >= 0")

    for uav in uavs:
        if not hasattr(uav, "_rwp_target"):
            _reset_rwp_state(uav, min_speed, max_speed)

        if uav._rwp_pause_remaining > 0:
            uav.vx = 0.0
            uav.vy = 0.0
            uav.vz = 0.0
            uav._rwp_pause_remaining -= 1
            continue

        tx, ty, tz = uav._rwp_target
        dx = tx - uav.x
        dy = ty - uav.y
        dz = tz - uav.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance <= reach_threshold:
            uav.x, uav.y, uav.z = tx, ty, tz
            uav.vx = 0.0
            uav.vy = 0.0
            uav.vz = 0.0

            _reset_rwp_state(uav, min_speed, max_speed)
            uav._rwp_pause_remaining = pause_steps
            continue

        speed = min(uav._rwp_speed, distance / max(dt, 1e-9))
        inv = 1.0 / distance

        uav.vx = speed * dx * inv
        uav.vy = speed * dy * inv
        uav.vz = speed * dz * inv


def _reset_rwp_state(uav: object, min_speed: float, max_speed: float) -> None:
    uav._rwp_target = _sample_waypoint()
    uav._rwp_speed = random.uniform(min_speed, max_speed)
    uav._rwp_pause_remaining = 0


def _sample_waypoint() -> tuple[float, float, float]:
    return (
        random.uniform(*config.X_LIMIT),
        random.uniform(*config.Y_LIMIT),
        random.uniform(*config.Z_LIMIT),
    )

"""Potential based reward computed only from consecutive observed states."""

from __future__ import annotations

from collections.abc import Mapping
from math import dist
from typing import Any

from .config import RecoveryConfig


def _mean_mst_gap(positions: list[list[float]], config: RecoveryConfig) -> float:
    """Prim's MST over physical distances, with each edge gap capped."""
    n = config.num_uavs
    included = {0}
    total = 0.0
    while len(included) < n:
        edge = min((dist(positions[i], positions[j]), j) for i in included for j in range(n) if j not in included)
        distance, node = edge
        gap = max(0.0, distance / (config.reward.range_fraction * config.comm_range) - 1.0)
        total += min(config.reward.gap_cap, gap)
        included.add(node)
    return total / (n - 1)


def potential(frame: Mapping[str, Any], config: RecoveryConfig) -> float:
    component_term = -config.reward.component_weight * (int(frame["components"]) - 1) / (config.num_uavs - 1)
    return component_term - _mean_mst_gap(frame["positions"], config)


def is_success(frame: Mapping[str, Any], config: RecoveryConfig) -> bool:
    return (
        frame["components"] == 1
        and frame["source_dest_connected"]
        and frame["olsr_route_usable"]
        and frame["connected_streak"] >= config.confirm_windows
    )


def reward_transition(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    config: RecoveryConfig,
    already_succeeded: bool = False,
) -> tuple[float, dict[str, float]]:
    weights = config.reward
    shaping = weights.gamma * potential(current, config) - potential(previous, config)
    moved = sum(max(0.0, float(current["travel"][i]) - float(previous["travel"][i])) for i in config.relay_ids)
    travel_norm = moved / (len(config.relay_ids) * config.max_speed)
    components = {
        "potential_previous": potential(previous, config),
        "potential_current": potential(current, config),
        "potential_shaping": shaping,
        "time_penalty": -weights.time_penalty,
        "travel_penalty": -weights.travel_penalty * travel_norm,
        "travel_norm": travel_norm,
        "safety_penalty": -weights.rejection_penalty if current["safety_rejected"] else 0.0,
        "success_bonus": weights.success_bonus if is_success(current, config) and not already_succeeded else 0.0,
    }
    reward = sum(
        components[key]
        for key in ("potential_shaping", "time_penalty", "travel_penalty", "safety_penalty", "success_bonus")
    )
    return reward, components

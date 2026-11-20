"""Untuned, accepted path-service-control-v1 constants."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PathServiceConfig:
    method_version: str = 'path-service-control-v1'
    risk_threshold: float = .35
    routing_delay_s: float = 3.
    rho: float = .50
    service_ratio: float = .95
    service_windows: int = 3
    lookahead_cap_s: float = 20.
    min_motion_m: float = 1.
    gap_improvement: float = 1e-3


POLICIES = ('hold', 'reactive', 'geometry', 'predictive-simple', 'predictive-learned',
            'path-service', 'path-service-no-service', 'path-service-no-deadline', 'path-service-no-budget')

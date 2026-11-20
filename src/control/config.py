"""Configuration shared by the simulator bridge, Gym environment and reports."""

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class RewardWeights:
    gamma: float = 0.99
    component_weight: float = 2.0
    gap_cap: float = 3.0
    range_fraction: float = 0.65
    time_penalty: float = 0.1
    travel_penalty: float = 0.05
    rejection_penalty: float = 1.0
    success_bonus: float = 20.0

    def __post_init__(self) -> None:
        values = (
            self.gamma,
            self.component_weight,
            self.gap_cap,
            self.range_fraction,
            self.time_penalty,
            self.travel_penalty,
            self.rejection_penalty,
            self.success_bonus,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Reward weights must be finite")
        if not 0 <= self.gamma <= 1 or self.range_fraction <= 0 or min(values[1:3] + values[4:]) < 0:
            raise ValueError("Invalid reward weights")


@dataclass(frozen=True)
class RecoveryConfig:
    num_uavs: int = 8
    steps: int = 120
    source: int = 0
    destination: int = 4
    max_speed: float = 8.0
    budget: float = 400.0
    min_separation: float = 5.0
    comm_range: float = 240.0
    x_max: float = 800.0
    y_max: float = 800.0
    z_min: float = 50.0
    z_max: float = 150.0
    confirm_windows: int = 3
    warmup: float = 10.0
    random_layout: bool = True
    reward: RewardWeights = field(default_factory=RewardWeights)

    def __post_init__(self) -> None:
        if (
            self.num_uavs < 3
            or not 0 <= self.source < self.num_uavs
            or not 0 <= self.destination < self.num_uavs
            or self.source == self.destination
        ):
            raise ValueError("RecoveryConfig requires at least 3 UAVs and distinct valid endpoints")
        if self.steps < 1 or self.confirm_windows < 1 or self.warmup < 0:
            raise ValueError("Invalid recovery duration or warmup")
        measures = (
            self.max_speed,
            self.budget,
            self.min_separation,
            self.comm_range,
            self.x_max,
            self.y_max,
            self.z_min,
            self.z_max,
        )
        if not all(isfinite(value) for value in measures):
            raise ValueError("Recovery geometry and motion limits must be finite")
        if (
            min(self.max_speed, self.min_separation, self.comm_range, self.x_max, self.y_max) <= 0
            or self.budget < 0
            or self.z_max <= self.z_min
        ):
            raise ValueError("Invalid recovery geometry or motion limits")

    @property
    def relay_ids(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_uavs) if i not in (self.source, self.destination))

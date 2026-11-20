"""Independent prevention defaults; the recovery observation/reward ABI is unchanged."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PreventionConfig:
    num_uavs: int = 8
    steps: int = 120
    source: int = 0
    destination: int = 4
    max_speed: float = 8.0
    budget: float = 400.0
    min_separation: float = 5.0
    comm_range: float = 240.0
    x_max: float = 1600.0
    y_max: float = 1200.0
    z_min: float = 50.0
    z_max: float = 150.0
    confirm_windows: int = 3
    warmup: float = 10.0
    random_layout: bool = True
    mission_speed: float = 3.0
    mission_start_step: int = 10
    initial_spacing_fraction: float = 0.55

    def __post_init__(self):
        integers = (self.num_uavs, self.steps, self.source, self.destination, self.confirm_windows, self.mission_start_step)
        if any(type(v) is not int for v in integers):
            raise ValueError('Prevention counts, endpoints and mission start must be integers')
        if (self.num_uavs < 3 or self.steps < 1 or self.confirm_windows < 1 or self.mission_start_step < 0
            or not 0 <= self.source < self.num_uavs or not 0 <= self.destination < self.num_uavs
            or self.source == self.destination):
            raise ValueError('Invalid prevention duration or endpoints')
        positive = (self.max_speed, self.min_separation, self.comm_range, self.x_max, self.y_max, self.mission_speed)
        numbers = positive + (self.budget, self.z_min, self.z_max, self.warmup, self.initial_spacing_fraction)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in numbers):
            raise ValueError('Prevention geometry and motion must be finite numbers')
        if min(positive) <= 0 or self.budget < 0 or self.warmup < 0 or self.z_max <= self.z_min:
            raise ValueError('Invalid prevention geometry or motion')
        if type(self.random_layout) is not bool or not 0 < self.initial_spacing_fraction < 1:
            raise ValueError('Invalid prevention random layout or spacing fraction')
        spacing = self.initial_spacing_fraction * self.comm_range
        jitter = min(5.0, spacing * .03) if self.random_layout else 0.0
        if (spacing*(self.num_uavs-1)+2*jitter > self.x_max or spacing-2*3**.5*jitter < self.min_separation
            or 2*jitter > min(self.y_max, self.z_max-self.z_min)):
            raise ValueError('Initially connected prevention chain does not fit safe geometry')

    @property
    def relay_ids(self):
        return tuple(i for i in range(self.num_uavs) if i not in (self.source, self.destination))

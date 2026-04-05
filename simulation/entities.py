from __future__ import annotations

import math
import random
from dataclasses import dataclass

import config


@dataclass
class UAV:
    node_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)


def random_nonzero_velocity() -> tuple[float, float, float]:
    low, high = config.VELOCITY_COMPONENT_RANGE
    while True:
        vx = random.uniform(low, high)
        vy = random.uniform(low, high)
        vz = random.uniform(low, high)
        if abs(vx) + abs(vy) + abs(vz) > 1e-9:
            return vx, vy, vz


def initialize_uavs() -> list[UAV]:
    random.seed(config.SEED)
    uavs: list[UAV] = []

    for i in range(config.NUM_UAVS):
        x = random.uniform(*config.X_LIMIT)
        y = random.uniform(*config.Y_LIMIT)
        z = random.uniform(*config.Z_LIMIT)
        vx, vy, vz = random_nonzero_velocity()

        uavs.append(
            UAV(
                node_id=i,
                x=x,
                y=y,
                z=z,
                vx=vx,
                vy=vy,
                vz=vz,
            )
        )

    return uavs

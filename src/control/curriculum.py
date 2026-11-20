"""Episode-level, training-only recovery separation curriculum."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import gymnasium as gym
import numpy as np

from .environment import Ns3RecoveryEnv


class CurriculumWrapper(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Select separation at reset using the number of completed transitions.

    The first quarter uses 1.65 times communication range, the second quarter
    uses 2.0 times range, and later episodes use nominal simulator sampling.
    Resetting never resets the transition count. The wrapper owns the
    ``group_separation`` and ``curriculum_stage`` reset options; other options,
    including explicit scenario seeds, pass through unchanged.
    """

    def __init__(self, env: Ns3RecoveryEnv, *, total_timesteps: int, enabled: bool = True) -> None:
        super().__init__(env)
        if isinstance(total_timesteps, bool) or not isinstance(total_timesteps, Integral) or total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")
        if enabled and not env.unwrapped.training:
            raise ValueError("Curriculum is available only for training environments")
        self.total_timesteps = int(total_timesteps)
        self.enabled = enabled
        self.transition_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        stage, separation = "nominal", 0.0
        if self.enabled:
            if 4 * self.transition_count < self.total_timesteps:
                stage, separation = "easy", 1.65 * self.env.unwrapped.config.comm_range
            elif 2 * self.transition_count < self.total_timesteps:
                stage, separation = "intermediate", 2.0 * self.env.unwrapped.config.comm_range
        episode_options = dict(options or {})
        episode_options.update(group_separation=separation, curriculum_stage=stage)
        return self.env.reset(seed=seed, options=episode_options)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        transition = self.env.step(action)
        self.transition_count += 1
        return transition

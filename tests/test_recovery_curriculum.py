"""Training curriculum stages follow transitions, changing only at reset."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from src.control.config import RecoveryConfig
from src.control.curriculum import CurriculumWrapper


class RecordingEnv(gym.Env):
    def __init__(self, *, training: bool = True) -> None:
        self.training = training
        self.config = RecoveryConfig()
        self.observation_space = self.action_space = spaces.Box(-1, 1, (1,))
        self.resets = []

    def reset(self, *, seed=None, options=None):
        self.resets.append((seed, options))
        return np.zeros(1), dict(options or {})

    def step(self, action):
        return np.zeros(1), 0.0, False, False, dict(self.resets[-1][1] or {})


def test_stages_change_at_quarters_only_on_reset_and_counter_persists() -> None:
    env = RecordingEnv()
    wrapper = CurriculumWrapper(env, total_timesteps=8, enabled=True)
    options = {"layout_seed": 11, "radio_seed": 22, "custom_option": "preserved"}
    _, info = wrapper.reset(seed=7, options=options)
    assert info["curriculum_stage"] == "easy" and info["group_separation"] == 396.0
    assert env.resets[-1][0] == 7
    assert all(env.resets[-1][1][key] == value for key, value in options.items())
    assert "group_separation" not in options
    for _ in range(2):
        _, _, _, _, info = wrapper.step(np.zeros(1))
        assert info["curriculum_stage"] == "easy"
    _, info = wrapper.reset()
    assert info["curriculum_stage"] == "intermediate" and info["group_separation"] == 480.0
    for _ in range(2):
        wrapper.step(np.zeros(1))
    _, info = wrapper.reset()
    assert info["curriculum_stage"] == "nominal" and info["group_separation"] == 0.0
    for _ in range(8):
        wrapper.step(np.zeros(1))
    assert wrapper.reset()[1]["curriculum_stage"] == "nominal"
    assert wrapper.transition_count == 12


def test_nondivisible_budget_uses_fractional_boundaries() -> None:
    wrapper = CurriculumWrapper(RecordingEnv(), total_timesteps=5)
    stages = []
    for _ in range(6):
        stages.append(wrapper.reset()[1]["curriculum_stage"])
        wrapper.step(np.zeros(1))
    assert stages == ["easy", "easy", "intermediate", "nominal", "nominal", "nominal"]


def test_disabled_stays_nominal() -> None:
    wrapper = CurriculumWrapper(RecordingEnv(), total_timesteps=8, enabled=False)
    for _ in range(9):
        _, info = wrapper.reset()
        assert info["curriculum_stage"] == "nominal" and info["group_separation"] == 0
        wrapper.step(np.zeros(1))


def test_enabled_curriculum_refuses_evaluation() -> None:
    with pytest.raises(ValueError, match="training"):
        CurriculumWrapper(RecordingEnv(training=False), total_timesteps=8)


@pytest.mark.parametrize("total_timesteps", [0, -1, True, 1.5])
def test_invalid_total_timesteps(total_timesteps) -> None:
    with pytest.raises(ValueError, match="total_timesteps"):
        CurriculumWrapper(RecordingEnv(), total_timesteps=total_timesteps)


def test_failed_step_does_not_advance_transition_count() -> None:
    class FailingEnv(RecordingEnv):
        def step(self, action):
            raise RuntimeError("simulator failure")

    wrapper = CurriculumWrapper(FailingEnv(), total_timesteps=8)
    wrapper.reset()
    with pytest.raises(RuntimeError, match="simulator failure"):
        wrapper.step(np.zeros(1))
    assert wrapper.transition_count == 0

"""Contract tests for the synchronous ns-3 recovery environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from src.control.bridge import ProtocolError, SimulatorTimeout, validate_frame
from src.control.config import RecoveryConfig
from src.control.curriculum import CurriculumWrapper
from src.control.environment import Ns3RecoveryEnv
from src.control.reward import is_success, potential, reward_transition


def frame(step: int, *, n: int = 4, connected: bool = False, terminal: bool = False) -> dict:
    positions = [[float(i * 250), 100.0, 100.0] for i in range(n)]
    return {
        "version": 1,
        "step": step,
        "positions": positions,
        "velocities": [[0.0, 0.0, 0.0] for _ in range(n)],
        "travel": [0.0] * n,
        "rx_ratio": [[0.0] * n for _ in range(n)],
        "rssi": [[0.0] * n for _ in range(n)],
        "rssi_mask": [[False] * n for _ in range(n)],
        "components": 1 if connected else 2,
        "source_dest_connected": connected,
        "olsr_route_usable": connected,
        "connected_streak": 3 if connected else 0,
        "safety_rejected": False,
        "terminal": terminal,
    }


@pytest.fixture
def config() -> RecoveryConfig:
    return RecoveryConfig(num_uavs=4, steps=2, source=0, destination=3, confirm_windows=3)


def child(
    tmp_path: Path, frames: list[dict], *, stall: bool = False, stdout_diagnostic: bool = False, exit_code: int = 0
) -> list[str]:
    script = tmp_path / "simulator_child.py"
    script.write_text(
        "import json,sys,time\n"
        f"frames = {frames!r}\n"
        "print('diagnostic line', file=sys.stderr, flush=True)\n"
        f"if {stdout_diagnostic!r}: print('ns-3 startup notice', flush=True)\n"
        "print('UAVRL ' + json.dumps(frames[0]), flush=True)\n"
        "for line in sys.stdin:\n"
        "    if line.startswith('UAVQUIT'):\n"
        "        break\n"
        f"    if {stall!r}:\n"
        "        time.sleep(60)\n"
        "    else:\n"
        "        step = int(line.split()[2])\n"
        "        print('UAVRL ' + json.dumps(frames[step + 1]), flush=True)\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    return [sys.executable, "-u", str(script)]


def test_potential_and_success_bonus_only_once(config: RecoveryConfig) -> None:
    first = frame(0)
    success = frame(1, connected=True)
    assert potential(first, config) < potential(success, config)
    reward, parts = reward_transition(first, success, config)
    later, later_parts = reward_transition(success, frame(2, connected=True), config, already_succeeded=True)
    assert reward > 19
    assert parts["success_bonus"] == 20
    assert later_parts["success_bonus"] == 0
    assert later < reward


def test_measured_recovery_and_actual_motion_penalties(config: RecoveryConfig) -> None:
    first = frame(0)
    old_route = frame(1, connected=True)
    old_route["components"] = 2
    assert not is_success(old_route, config)
    moved = frame(1)
    moved["travel"][1] = 8.0
    moved["safety_rejected"] = True
    _, parts = reward_transition(first, moved, config)
    assert parts["travel_norm"] == pytest.approx(0.5)
    assert parts["travel_penalty"] == pytest.approx(-0.025)
    assert parts["safety_penalty"] == -1.0


def test_validation_rejects_wrong_shape_version_and_nonfinite(config: RecoveryConfig) -> None:
    bad = frame(0)
    bad["version"] = 2
    with pytest.raises(ProtocolError):
        validate_frame(bad, config, expected_step=0)
    bad = frame(0)
    bad["rssi"] = [[0.0] * 3] * 4
    with pytest.raises(ProtocolError):
        validate_frame(bad, config, expected_step=0)
    bad = frame(0)
    bad["positions"][0][0] = float("nan")
    with pytest.raises(ProtocolError):
        validate_frame(bad, config, expected_step=0)
    with pytest.raises(ProtocolError):
        validate_frame(frame(2), config, expected_step=0)
    numeric_booleans = frame(0)
    numeric_booleans["rssi_mask"][0][1] = 1
    numeric_booleans["terminal"] = 0
    checked = validate_frame(numeric_booleans, config, expected_step=0)
    assert checked["rssi_mask"][0][1] is True and checked["terminal"] is False


def test_horizon_is_censored_and_actions_are_checked(config: RecoveryConfig, tmp_path: Path) -> None:
    env = Ns3RecoveryEnv(
        config=config, launcher=child(tmp_path, [frame(0), frame(1), frame(2, terminal=True)]), output_root=tmp_path
    )
    try:
        observation, info = env.reset(options={"layout_seed": 11, "radio_seed": 22})
        assert env.observation_space.contains(observation)
        assert (info["layout_seed"], info["radio_seed"]) == (11, 22)
        with pytest.raises(ValueError):
            env.step(np.array([np.nan] * 6, dtype=np.float32))
        with pytest.raises(ValueError):
            env.step(np.zeros(5, dtype=np.float32))
        env.step(np.zeros(6, dtype=np.float32))
        _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert not terminated and truncated
        assert info["terminal_reason"] == "horizon"
        assert info["recovery_time_s"] is None
        assert info["censored"] is True
    finally:
        env.close()


def test_timeout_kills_child(config: RecoveryConfig, tmp_path: Path) -> None:
    env = Ns3RecoveryEnv(
        config=config, launcher=child(tmp_path, [frame(0), frame(1)], stall=True), output_root=tmp_path, timeout_s=1.0
    )
    env.reset()
    process = env.bridge.process
    with pytest.raises(SimulatorTimeout):
        env.step(np.zeros(6, dtype=np.float32))
    assert process.poll() is not None
    env.close()


def test_initial_connected_frame_is_retried(config: RecoveryConfig, tmp_path: Path) -> None:
    env = Ns3RecoveryEnv(
        config=config, launcher=child(tmp_path, [frame(0, connected=True)]), output_root=tmp_path, max_reset_attempts=2
    )
    with pytest.raises(RuntimeError, match="fragmented"):
        env.reset()
    assert env.bridge is None


def test_explicit_scenario_seeds_are_never_replaced(config: RecoveryConfig, tmp_path: Path) -> None:
    env = Ns3RecoveryEnv(
        config=config, launcher=child(tmp_path, [frame(0, connected=True)]), output_root=tmp_path, max_reset_attempts=5
    )
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        env.reset(options={"layout_seed": 10, "radio_seed": 20})
    assert len(list(tmp_path.glob("episode_*_attempt_*"))) == 1


def test_diagnostics_do_not_count_as_frames_and_nonzero_exit_is_error(config: RecoveryConfig, tmp_path: Path) -> None:
    env = Ns3RecoveryEnv(
        config=config,
        launcher=child(tmp_path, [frame(0), frame(1), frame(2, terminal=True)], stdout_diagnostic=True, exit_code=7),
        output_root=tmp_path,
    )
    env.reset()
    env.step(np.zeros(6, dtype=np.float32))
    with pytest.raises(ProtocolError, match="code 7"):
        env.step(np.zeros(6, dtype=np.float32))
    env.close()


def test_evaluation_continues_after_budget_exhaustion(config: RecoveryConfig, tmp_path: Path) -> None:
    zero_budget = RecoveryConfig(num_uavs=4, steps=2, source=0, destination=3, budget=0)
    env = Ns3RecoveryEnv(
        config=zero_budget,
        launcher=child(tmp_path, [frame(0), frame(1), frame(2, terminal=True)]),
        output_root=tmp_path,
        training=False,
    )
    env.reset()
    _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
    assert not terminated and not truncated and info["budget_exhausted"]
    _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
    assert not terminated and truncated
    assert info["terminal_reason"] == "budget_exhausted"
    assert info["censored"]


@pytest.mark.parametrize("budget", [0.0, 8.0])
def test_training_budget_exhaustion_continues_to_horizon(tmp_path: Path, budget: float) -> None:
    config = RecoveryConfig(num_uavs=4, steps=2, source=0, destination=3, budget=budget)
    frames = [frame(0), frame(1), frame(2, terminal=True)]
    for current in frames[1:]:
        current["travel"][1] = current["travel"][2] = budget
    env = Ns3RecoveryEnv(config=config, launcher=child(tmp_path, frames), output_root=tmp_path)
    try:
        env.reset()
        _, _, terminated, truncated, info = env.step(np.zeros(6))
        assert not terminated and not truncated
        assert info["budget_exhausted"] and info["terminal_reason"] is None
        assert env.bridge is not None
        _, _, terminated, truncated, info = env.step(np.zeros(6))
        assert not terminated and truncated and info["censored"]
        assert info["terminal_reason"] == "budget_exhausted"
    finally:
        env.close()


@pytest.mark.parametrize("training", [True, False])
def test_early_simulator_terminal_is_error_even_when_budget_exhausted(tmp_path: Path, training: bool) -> None:
    config = RecoveryConfig(num_uavs=4, steps=2, source=0, destination=3, budget=0)
    env = Ns3RecoveryEnv(
        config=config, launcher=child(tmp_path, [frame(0), frame(1, terminal=True)]),
        output_root=tmp_path, training=training,
    )
    try:
        env.reset()
        with pytest.raises(ProtocolError, match="before the configured horizon"):
            env.step(np.zeros(6))
        assert env.bridge is None
    finally:
        env.close()


@pytest.mark.parametrize("training", [True, False])
def test_success_terminates_only_training(config: RecoveryConfig, tmp_path: Path, training: bool) -> None:
    env = Ns3RecoveryEnv(
        config=config,
        launcher=child(tmp_path, [frame(0), frame(1, connected=True), frame(2, connected=True, terminal=True)]),
        output_root=tmp_path, training=training,
    )
    try:
        env.reset()
        _, _, terminated, truncated, info = env.step(np.zeros(6))
        assert terminated == training and not truncated
        assert info["success"] and info["recovery_time_s"] == 1
        if not training:
            _, _, terminated, truncated, info = env.step(np.zeros(6))
            assert not terminated and truncated and info["recovery_time_s"] == 1
    finally:
        env.close()


@pytest.mark.parametrize("separation", [0.0, 396.0, 480.0, 800.0])
def test_reset_separation_is_logged_and_passed_to_simulator(
    config: RecoveryConfig, tmp_path: Path, separation: float
) -> None:
    env = Ns3RecoveryEnv(config=config, launcher=child(tmp_path, [frame(0)]), output_root=tmp_path)
    try:
        _, info = env.reset(options={"group_separation": separation, "curriculum_stage": "nominal"})
        saved = json.loads((env.episode_output_dir / "config.json").read_text(encoding="utf-8"))
        assert info["group_separation"] == saved["group_separation"] == separation
        assert info["curriculum_stage"] == saved["curriculum_stage"] == "nominal"
        assert f"--groupSeparation={separation}" in saved["command"]
        assert "group_separation" not in saved["config"]
    finally:
        env.close()


@pytest.mark.parametrize("separation", [-1.0, 360.0, 350.0, 801.0, float("nan"), float("inf"), True, "396"])
def test_reset_rejects_invalid_separation(config: RecoveryConfig, tmp_path: Path, separation: object) -> None:
    env = Ns3RecoveryEnv(config=config, launcher=["unused"], output_root=tmp_path)
    with pytest.raises(ValueError, match="separation"):
        env.reset(options={"group_separation": separation})
    assert env.bridge is None


def test_reset_separation_checks_simulator_compact_group_geometry(tmp_path: Path) -> None:
    config = RecoveryConfig(num_uavs=4, steps=2, source=0, destination=3, min_separation=40)
    env = Ns3RecoveryEnv(config=config, launcher=["unused"], output_root=tmp_path)
    with pytest.raises(ValueError, match="separation"):
        env.reset(options={"group_separation": 396})


@pytest.mark.parametrize("options", [{"group_separation": 396}, {"curriculum_stage": "easy"}])
def test_evaluation_refuses_curriculum_overrides(config: RecoveryConfig, tmp_path: Path, options: dict) -> None:
    env = Ns3RecoveryEnv(config=config, launcher=["unused"], output_root=tmp_path, training=False)
    with pytest.raises(ValueError, match="evaluation"):
        env.reset(options=options)
    assert env.bridge is None


def test_outer_seed_wrapper_preserves_curriculum_and_episode_metadata(config: RecoveryConfig, tmp_path: Path) -> None:
    class SeedWrapper(gym.Wrapper):
        def reset(self, *, seed=None, options=None):
            seeded_options = dict(options or {}, layout_seed=11, radio_seed=22)
            return self.env.reset(seed=seed, options=seeded_options)

    base = Ns3RecoveryEnv(
        config=config, launcher=child(tmp_path, [frame(0), frame(1), frame(2, terminal=True)]), output_root=tmp_path
    )
    curriculum = CurriculumWrapper(base, total_timesteps=8)
    wrapped = SeedWrapper(curriculum)
    try:
        _, info = wrapped.reset()
        assert (info["layout_seed"], info["radio_seed"]) == (11, 22)
        assert info["curriculum_stage"] == "easy" and info["group_separation"] == 396
        for _ in range(2):
            _, _, _, _, info = wrapped.step(np.zeros(6))
            assert info["curriculum_stage"] == "easy"
        _, info = wrapped.reset()
        saved = json.loads((base.episode_output_dir / "config.json").read_text(encoding="utf-8"))
        assert info["curriculum_stage"] == saved["curriculum_stage"] == "intermediate"
        assert info["group_separation"] == saved["group_separation"] == 480
        assert (saved["layout_seed"], saved["radio_seed"]) == (11, 22)
    finally:
        wrapped.close()

"""Real radio/motion checks; requires the separately built recovery simulator."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from src.control.bridge import make_launcher, simulator_output_path
from src.control.config import RecoveryConfig
from src.control.environment import Ns3RecoveryEnv

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/recovery/uav-olsr-dataset"
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="Build the recovery ns-3 binary first")


def launcher() -> list[str]:
    return make_launcher(BINARY, "Ubuntu" if os.name == "nt" else None)


def test_live_zero_action_preserves_positions_and_censors_failure(tmp_path: Path):
    env = Ns3RecoveryEnv(RecoveryConfig(steps=5), launcher(), tmp_path, training=False)
    try:
        env.reset(seed=17, options={"layout_seed": 17, "radio_seed": 71})
        initial = np.array(env.last_frame["positions"])
        for _ in range(5):
            _, _, terminated, truncated, info = env.step(np.zeros(18, dtype=np.float32))
            np.testing.assert_array_equal(env.last_frame["positions"], initial)
        assert not terminated and truncated
        assert info["censored"] and info["recovery_time_s"] is None
        assert sum(info["travel"]) == 0
        with (env.episode_output_dir / "closed_loop_metrics.csv").open() as stream:
            metrics = next(csv.DictReader(stream))
        # Five actual commanded intervals plus the initial observed snapshot.
        offered = int(np.ceil((5 - 0.01) / (512 * 8 / 256000)))
        assert int(metrics["offered_packets"]) == offered
        assert int(metrics["rx_packets"]) == 0
    finally:
        env.close()


def test_live_motion_respects_fixed_endpoints_speed_and_budget(tmp_path: Path):
    config = RecoveryConfig(steps=5, budget=2, random_layout=False)
    env = Ns3RecoveryEnv(config, launcher(), tmp_path, training=False)
    try:
        env.reset(seed=7)
        initial = np.array(env.last_frame["positions"])
        action = np.tile([0, 1, 0], 6).astype(np.float32)
        for _ in range(5):
            _, _, _, _, info = env.step(action)
            current = np.array(env.last_frame["positions"])
            np.testing.assert_array_equal(current[[0, 4]], initial[[0, 4]])
            assert max(info["travel"]) <= 2 + 1e-6
        assert np.max(np.linalg.norm(current - initial, axis=1)) == pytest.approx(2)
        assert info["step"] == 5  # evaluation observes the full horizon after budget depletion
    finally:
        env.close()


def test_existing_partition_scenario_retains_event_metadata(tmp_path: Path):
    subprocess.run(
        launcher()
        + [
            "--numUavs=20",
            "--timeSteps=12",
            "--seed=7",
            "--commRange=240",
            "--xMax=800",
            "--yMax=800",
            "--scenarioMode=partition-reconnect",
            "--enableAnim=false",
            f"--outputDir={simulator_output_path(tmp_path, launcher())}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    scenario = json.loads((tmp_path / "scenario.json").read_text())
    assert scenario["mobility_model"] == "event-controlled-waypoint"
    assert (scenario["partition_start_step"], scenario["partition_end_step"], scenario["reconnect_step"]) == (3, 8, 10)


def test_standalone_metrics_stop_counting_when_traffic_stops(tmp_path: Path):
    command = launcher()
    subprocess.run(
        command
        + [
            "--numUavs=8",
            "--timeSteps=6",
            "--seed=7",
            "--commRange=240",
            "--xMax=800",
            "--yMax=800",
            "--scenarioMode=relay-recovery",
            "--recoveryPolicy=none",
            "--enableDataFlow=true",
            "--enableAnim=false",
            f"--outputDir={simulator_output_path(tmp_path, command)}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    with (tmp_path / "closed_loop_metrics.csv").open() as stream:
        row = next(csv.DictReader(stream))
    # App stops one second before the standalone simulator's cleanup interval.
    assert int(row["offered_packets"]) == int(np.ceil(4.99 / 0.016))


def test_live_same_seeds_and_actions_reproduce_observations(tmp_path: Path):
    snapshots = []
    for run in range(2):
        env = Ns3RecoveryEnv(RecoveryConfig(steps=3), launcher(), tmp_path / str(run), training=False)
        try:
            obs, _ = env.reset(seed=17, options={"layout_seed": 99, "radio_seed": 121})
            observations = [obs]
            for _ in range(3):
                obs, *_ = env.step(np.zeros(18, dtype=np.float32))
                observations.append(obs)
            snapshots.append(np.stack(observations))
        finally:
            env.close()
    np.testing.assert_array_equal(*snapshots)


@pytest.mark.parametrize('stage,factor', [('easy', 1.65), ('intermediate', 2.0), ('nominal', 0)])
def test_live_training_curriculum_keeps_budget_failure_until_horizon(tmp_path: Path, stage: str, factor: float):
    config = RecoveryConfig(steps=3, budget=0)
    env = Ns3RecoveryEnv(config, launcher(), tmp_path, training=True)
    try:
        _, info = env.reset(options={
            'layout_seed': 200000, 'radio_seed': 200001,
            'group_separation': factor * config.comm_range, 'curriculum_stage': stage,
        })
        assert info['components'] > 1
        assert info['curriculum_stage'] == stage
        initial = np.array(env.last_frame['positions'])
        if factor:
            assert np.linalg.norm(initial[0] - initial[4]) == pytest.approx(factor * config.comm_range)
        for step in range(1, 4):
            _, _, terminated, truncated, info = env.step(np.zeros(18, dtype=np.float32))
            assert not terminated
            assert truncated == (step == 3)
            assert info['budget_exhausted']
            np.testing.assert_array_equal(env.last_frame['positions'], initial)
        assert info['terminal_reason'] == 'budget_exhausted' and info['censored']
    finally:
        env.close()

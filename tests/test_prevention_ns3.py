"""Live measured radio and motion behavior for the prevention scenario."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.control.bridge import make_launcher, simulator_output_path
from src.control.prevention.config import PreventionConfig
from src.control.prevention.environment import Ns3PreventionEnv

BINARY = Path(__file__).resolve().parents[1] / 'build/recovery/uav-olsr-dataset'
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason='Build the ns-3 simulator first')


def make_env(tmp_path, **kwargs):
    return Ns3PreventionEnv(PreventionConfig(**kwargs), make_launcher(BINARY, 'Ubuntu' if os.name == 'nt' else None), tmp_path)


def raw_prevention(output):
    launcher = make_launcher(BINARY, 'Ubuntu' if os.name == 'nt' else None)
    return subprocess.run(launcher + ['--scenarioMode=relay-prevention', '--numUavs=8',
        '--timeSteps=2', '--xMax=1600', '--yMax=1200', '--commRange=240',
        '--enableAnim=false', f'--outputDir={simulator_output_path(output, launcher)}'],
        capture_output=True, text=True, timeout=45)


def test_raw_cli_refuses_repeated_output_without_changing_artifacts(tmp_path):
    output = tmp_path/'empty'
    output.mkdir()
    first = raw_prevention(output)
    assert first.returncode == 0, first.stderr
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    assert 'scenario.json' in original and 'nodes.csv' in original
    repeated = raw_prevention(output)
    assert repeated.returncode != 0
    assert 'empty' in repeated.stderr.lower()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original


def test_raw_cli_refuses_nonempty_and_file_paths_without_changes(tmp_path):
    output = tmp_path/'sentinel'
    output.mkdir()
    (output/'sentinel.bin').write_bytes(b'untouched evidence\x00\xff')
    (output/'nodes.csv').write_bytes(b'prior nodes\n')
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    result = raw_prevention(output)
    assert result.returncode != 0
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original
    path = tmp_path/'file'
    path.write_bytes(b'file evidence')
    assert raw_prevention(path).returncode != 0
    assert path.read_bytes() == b'file evidence'


def test_raw_cli_new_directory_and_environment_reserved_directory_launch(tmp_path):
    output = tmp_path/'new'
    result = raw_prevention(output)
    assert result.returncode == 0, result.stderr
    assert (output/'scenario.json').is_file()
    env = make_env(tmp_path/'env', random_layout=False, steps=1)
    try:
        env.reset(seed=7)
        env.step(np.zeros(18))
        assert (env.episode_output_dir/'scenario.json').is_file()
        assert (env.episode_output_dir/'config.json').is_file()
        assert not (env.episode_output_dir/'.prevention-reservation').exists()
        saved = json.loads((env.episode_output_dir/'config.json').read_text())
        reservation = next(arg.split('=', 1)[1] for arg in saved['command'] if arg.startswith('--preventionOutputReservation='))
        # A copied/recreated ownership marker cannot authorize replacing a run.
        (env.episode_output_dir/'.prevention-reservation').write_bytes((reservation+'\n').encode('ascii'))
        original = {p.name: p.read_bytes() for p in env.episode_output_dir.iterdir()}
        repeated = subprocess.run(saved['command'], capture_output=True, text=True, timeout=45)
        assert repeated.returncode != 0
        assert {p.name: p.read_bytes() for p in env.episode_output_dir.iterdir()} == original
    finally: env.close()


def test_fixed_hold_starts_connected_moves_anchors_after_boundary_and_partitions(tmp_path):
    env = make_env(tmp_path, random_layout=False)
    try:
        env.reset(options=dict(layout_seed=17, radio_seed=71))
        first = np.array(env.last_frame['positions'])
        spacing = 132
        assert first[0] == pytest.approx([338, 600, 100])
        assert first[4] == pytest.approx([1262, 600, 100])
        assert sorted(first[:, 0]) == pytest.approx([338+i*spacing for i in range(8)])
        disconnected = False
        for step in range(1, 121):
            _, _, terminated, truncated, info = env.step(np.zeros(18))
            current = np.array(env.last_frame['positions'])
            if step <= 10: np.testing.assert_array_equal(current, first)
            if step == 11:
                assert current[0, 1] == 603 and current[4, 1] == 597
            np.testing.assert_array_equal(current[list(env.config.relay_ids)], first[list(env.config.relay_ids)])
            disconnected |= info['components'] > 1
            assert not terminated and truncated == (step == 120)
        assert disconnected
        assert info['travel'][0] == pytest.approx(330)
        assert info['travel'][4] == pytest.approx(330)
        scenario = json.loads((env.episode_output_dir/'scenario.json').read_text())
        assert scenario['mobility_model'] == 'bounded-relay-prevention'
        assert scenario['prevention_actual_mission_speed_mps'] == 3
        assert scenario['prevention_spacing_m'] == 132
    finally: env.close()


def test_mission_stops_at_bounds_and_remapped_endpoints_remain_independent(tmp_path):
    env = make_env(tmp_path, random_layout=False, source=2, destination=7,
                   y_max=8, mission_start_step=0, steps=5, budget=0)
    try:
        env.reset(seed=7)
        initial = np.array(env.last_frame['positions'])
        for _ in range(5): env.step(np.zeros(18))
        current = np.array(env.last_frame['positions'])
        assert current[2, 1] == pytest.approx(8) and current[7, 1] == pytest.approx(0, abs=1e-6)
        np.testing.assert_array_equal(current[list(env.config.relay_ids)], initial[list(env.config.relay_ids)])
        assert env.last_frame['travel'][2] == pytest.approx(4)
        assert env.last_frame['travel'][7] == pytest.approx(4)
    finally: env.close()


def test_swept_safety_accounts_for_moving_anchor(tmp_path):
    # Moving relay 1 diagonally left/up stays >6m from the old anchor position,
    # but passes <5m from the anchor's +3m y trajectory. Its batch is rejected.
    env = make_env(tmp_path, random_layout=False, initial_spacing_fraction=.0375,
                   mission_start_step=0, steps=1)
    try:
        env.reset(seed=7)
        initial = np.array(env.last_frame['positions'])
        action = np.zeros(18)
        action[:3] = [-1, 1, 0]
        _, _, _, _, info = env.step(action)
        assert info['safety_rejected']
        current = np.array(env.last_frame['positions'])
        np.testing.assert_array_equal(current[list(env.config.relay_ids)], initial[list(env.config.relay_ids)])
        assert current[0, 1] == 603 and current[4, 1] == 597
    finally: env.close()


def test_actions_change_measured_radio_and_respect_euclidean_speed(tmp_path):
    runs = []
    for moving in (False, True):
        env = make_env(tmp_path/str(moving), random_layout=False, steps=32)
        try:
            env.reset(options=dict(layout_seed=7, radio_seed=71))
            previous = np.array(env.last_frame['positions'])
            for _ in range(32):
                env.step(np.tile([1, 1, 1], 6) if moving else np.zeros(18))
                current = np.array(env.last_frame['positions'])
                assert max(np.linalg.norm((current-previous)[list(env.config.relay_ids)], axis=1)) <= 8+1e-6
                previous = current
            runs.append(env.last_frame['rx_ratio'])
        finally: env.close()
    assert runs[0] != runs[1]


def test_same_seeds_actions_reproduce_frames_and_randomization_is_recorded(tmp_path):
    runs = []
    for run in range(2):
        env = make_env(tmp_path/str(run), steps=12)
        try:
            env.reset(options=dict(layout_seed=99, radio_seed=121))
            frames = [env.last_frame]
            for _ in range(12):
                env.step(np.tile([0, .25, 0], 6))
                frames.append(env.last_frame)
            runs.append(frames)
            meta = json.loads((env.episode_output_dir/'scenario.json').read_text())
            assert .85 <= meta['prevention_mission_speed_multiplier'] <= 1.15
            assert meta['prevention_jitter_max_m'] <= 5
        finally: env.close()
    assert runs[0] == runs[1]


@pytest.mark.parametrize('budget', [0, 2])
def test_relay_limits_do_not_stop_mission_anchors_or_end_episode(tmp_path, budget):
    env = make_env(tmp_path, random_layout=False, budget=budget, steps=15)
    try:
        env.reset(seed=7)
        initial = np.array(env.last_frame['positions'])
        for step in range(1, 16):
            _, _, terminated, truncated, info = env.step(np.tile([0, 1, 0], 6))
            pos = np.array(env.last_frame['positions'])
            assert max(info['travel'][i] for i in env.config.relay_ids) <= budget+1e-6
            assert not terminated and truncated == (step == 15)
        assert pos[0, 1] == initial[0, 1]+15 and pos[4, 1] == initial[4, 1]-15
    finally: env.close()

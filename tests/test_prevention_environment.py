"""Public prevention environment contract, using a real protocol child process."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.control.bridge import ProtocolError
from src.control.prevention.config import PreventionConfig
from src.control.prevention.environment import Ns3PreventionEnv


def frame(step, connected=True, terminal=False):
    n = 4
    return dict(version=1, step=step, positions=[[100 + i*100, 100, 100] for i in range(n)],
                velocities=[[0, 0, 0] for _ in range(n)], travel=[0]*n,
                rx_ratio=[[0]*n for _ in range(n)], rssi=[[0]*n for _ in range(n)],
                rssi_mask=[[False]*n for _ in range(n)], components=1 if connected else 2,
                source_dest_connected=connected, olsr_route_usable=connected,
                connected_streak=0, safety_rejected=False, terminal=terminal)


def child(tmp_path, frames):
    path = tmp_path / 'child.py'
    path.write_text('import json,sys\n' + f'frames={frames!r}\n'
                    "print('UAVRL '+json.dumps(frames[0]),flush=True)\n"
                    'for line in sys.stdin:\n'
                    " if line.startswith('UAVQUIT'): break\n"
                    " print('UAVRL '+json.dumps(frames[int(line.split()[2])+1]),flush=True)\n")
    return [sys.executable, '-u', str(path)]


def config(**kwargs):
    return PreventionConfig(num_uavs=4, destination=3, steps=2, **kwargs)


def test_defaults_and_config_validation():
    c = PreventionConfig()
    assert (c.x_max, c.y_max, c.mission_speed, c.mission_start_step, c.initial_spacing_fraction) == (1600, 1200, 3, 10, .55)
    for kwargs in [dict(mission_speed=float('nan')), dict(warmup=float('inf')),
                   dict(mission_start_step=-1), dict(mission_start_step=1.5),
                   dict(initial_spacing_fraction=0), dict(initial_spacing_fraction=1),
                   dict(num_uavs=1), dict(x_max=100), dict(steps=True)]:
        with pytest.raises(ValueError):
            PreventionConfig(**kwargs)


def test_full_horizon_after_partition_and_budget_zero_logs_every_frame(tmp_path):
    env = Ns3PreventionEnv(config(budget=0), child(tmp_path, [frame(0), frame(1, False), frame(2, False, True)]), tmp_path)
    try:
        obs, info = env.reset(options=dict(layout_seed=11, radio_seed=22))
        assert env.observation_space.contains(obs)
        for action in [np.zeros(5), np.full(6, np.nan), np.full(6, 2)]:
            with pytest.raises(ValueError): env.step(action)
        _, _, terminated, truncated, info = env.step(np.zeros(6))
        assert not terminated and not truncated and info['budget_exhausted']
        _, _, terminated, truncated, info = env.step(np.zeros(6))
        assert not terminated and truncated and info['step'] == 2
        records = [json.loads(line) for line in (env.episode_output_dir/'transitions.jsonl').read_text().splitlines()]
        assert len(records) == 3 and records[1]['proposed_action'] == [0]*6
        saved = json.loads((env.episode_output_dir/'config.json').read_text())
        assert saved['layout_seed'] == 11 and saved['radio_seed'] == 22
        assert '--scenarioMode=relay-prevention' in saved['command']
    finally: env.close()


def test_explicit_failed_seed_is_not_replaced_and_setup_error_logged(tmp_path):
    env = Ns3PreventionEnv(config(), child(tmp_path, [frame(0, False)]), tmp_path)
    with pytest.raises(RuntimeError, match='initially connected.*11.*22'):
        env.reset(options=dict(layout_seed=11, radio_seed=22))
    assert len(list(tmp_path.glob('episode_*'))) == 1 and env.bridge is None
    assert json.loads((env.episode_output_dir/'setup_error.json').read_text())['layout_seed'] == 11


@pytest.mark.parametrize('bad', [-1, True, 1.5, 2**31, '10'])
def test_bad_seeds_fail_before_launch(tmp_path, bad):
    env = Ns3PreventionEnv(config(), ['unused'], tmp_path)
    with pytest.raises(ValueError, match='seed'):
        env.reset(options=dict(layout_seed=bad))


def test_early_terminal_rejected_even_after_budget_exhaustion(tmp_path):
    env = Ns3PreventionEnv(config(budget=0), child(tmp_path, [frame(0), frame(1, False, True)]), tmp_path)
    try:
        env.reset()
        with pytest.raises(ProtocolError, match='before the configured horizon'): env.step(np.zeros(6))
        assert env.bridge is None
    finally: env.close()

"""Live prevention environment. Every reset uses one seed pair, never resampling failures.

Observations concatenate measured positions, velocities, node travel, rx_ratio,
RSSI (zero only where missing), and RSSI mask. Future mission metadata is not an
observation. transitions.jsonl retains complete v1 frames for causal features.
"""
from collections.abc import Sequence
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
from uuid import uuid4

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..bridge import ProtocolError, SimulatorBridge, simulator_output_path
from .config import PreventionConfig


class Ns3PreventionEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config: PreventionConfig, launcher: Sequence[str], output_root: Path, timeout_s=30, service_observation=False):
        super().__init__()
        if not launcher or not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('A launcher and finite positive timeout are required')
        self.config, self.launcher = config, list(launcher)
        if type(service_observation) is not bool: raise ValueError('service_observation must be boolean')
        self.service_observation = service_observation
        self.output_root, self.timeout_s = Path(output_root), timeout_s
        n = config.num_uavs
        self.action_space = spaces.Box(-1., 1., shape=(3*len(config.relay_ids),), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7*n+3*n*n,), dtype=np.float32)
        self.bridge = None
        self.last_frame = None
        self.episode_output_dir = None
        self._done = True
        self._layout_seed = self._radio_seed = 0
        self._output_reservation = ''

    @property
    def layout_seed(self): return self._layout_seed

    @property
    def radio_seed(self): return self._radio_seed

    def _command(self, output_dir):
        c = self.config
        return self.launcher + (['--enableServiceObservation=true'] if self.service_observation else []) + [
            '--scenarioMode=relay-prevention', '--recoveryPolicy=external',
            f'--numUavs={c.num_uavs}', f'--timeSteps={c.steps+1}', f'--seed={self.radio_seed}',
            f'--recoveryLayoutSeed={self.layout_seed}', f'--recoveryRandomLayout={str(c.random_layout).lower()}',
            f'--sourceId={c.source}', f'--destId={c.destination}', f'--warmup={c.warmup}',
            f'--xMax={c.x_max}', f'--yMax={c.y_max}', f'--zMin={c.z_min}', f'--zMax={c.z_max}',
            f'--commRange={c.comm_range}', f'--recoverySpeed={c.max_speed}', f'--recoveryBudget={c.budget}',
            f'--recoveryMinSeparation={c.min_separation}', f'--recoveryConfirmWindows={c.confirm_windows}',
            f'--preventionMissionSpeed={c.mission_speed}', f'--preventionMissionStartStep={c.mission_start_step}',
            f'--preventionSpacingFraction={c.initial_spacing_fraction}', '--routingStrategy=olsr',
            f'--preventionOutputReservation={self._output_reservation}',
            '--enableDataFlow=true', '--enableAnim=false',
            f'--outputDir={simulator_output_path(output_dir, self.launcher)}',
        ]

    def _observation(self, frame):
        mask = np.asarray(frame['rssi_mask'], dtype=bool)
        fields = [np.asarray(frame[k], dtype=np.float32).ravel() for k in ('positions', 'velocities', 'travel', 'rx_ratio')]
        fields += [np.where(mask, frame['rssi'], 0).astype(np.float32).ravel(), mask.astype(np.float32).ravel()]
        result = np.concatenate(fields)
        if not np.isfinite(result).all(): raise ProtocolError('Nonfinite prevention observation')
        return result

    def _info(self):
        f = self.last_frame
        return dict(layout_seed=self.layout_seed, radio_seed=self.radio_seed,
                    **{k: f[k] for k in ('step', 'components', 'source_dest_connected', 'olsr_route_usable', 'safety_rejected', 'travel')},
                    budget_exhausted=all(f['travel'][i] >= self.config.budget-1e-6 for i in self.config.relay_ids),
                    terminal_reason='horizon' if self._done else None)

    def _write(self, name, data):
        (self.episode_output_dir/name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')

    def _record(self, data):
        with (self.episode_output_dir/'transitions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(data, allow_nan=False, separators=(',', ':'))+'\n')

    def reset(self, *, seed=None, options=None):
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**31):
            raise ValueError('Reset seed must be an integer in [0,2**31)')
        super().reset(seed=seed)
        self.close()
        self.last_frame, self._done = None, True
        options = options or {}
        if set(options)-{'layout_seed', 'radio_seed'}:
            raise ValueError('Only layout_seed and radio_seed reset options are supported')
        values = [options[k] if k in options else int(self.np_random.integers(1, 2**31-1)) for k in ('layout_seed', 'radio_seed')]
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v < 2**31 for v in values):
            raise ValueError('Layout and radio seeds must be integers in [0,2**31)')
        self._layout_seed, self._radio_seed = map(int, values)
        self.episode_output_dir = self.output_root/f'episode_{uuid4().hex}'
        self.episode_output_dir.mkdir(parents=True, exist_ok=False)
        self._output_reservation = uuid4().hex
        (self.episode_output_dir/'.prevention-reservation').write_bytes((self._output_reservation+'\n').encode('ascii'))
        command = self._command(self.episode_output_dir)
        self._write('config.json', dict(config=asdict(self.config), simulator_snapshots=self.config.steps+1,
                                      layout_seed=self.layout_seed, radio_seed=self.radio_seed, command=command))
        try:
            extra = dict(stdout_path=self.episode_output_dir/'simulator.stdout.log') if self.service_observation else {}
            self.bridge = SimulatorBridge(command, self.config, self.episode_output_dir/'simulator.stderr.log', self.timeout_s, **extra)
            frame = self.bridge.next_frame(0)
            if self.service_observation:
                from ..path_service.observation import validate_service_observation
                validate_service_observation(frame, self.config)
            initial_info = dict(layout_seed=self.layout_seed, radio_seed=self.radio_seed, step=frame['step'])
            self._record(dict(event='initial', frame=frame, info=initial_info))
            if frame['terminal']: raise ProtocolError('Initial prevention frame was terminal')
            if frame['components'] != 1 or not frame['source_dest_connected']:
                raise RuntimeError(f'Prevention setup requires an initially connected measured graph: layout seed {self.layout_seed}, radio seed {self.radio_seed}')
            self.last_frame, self._done = frame, False
            return self._observation(frame), self._info()
        except BaseException as exc:
            self._write('setup_error.json', dict(error=str(exc), layout_seed=self.layout_seed, radio_seed=self.radio_seed))
            self.close()
            raise

    def step(self, action):
        if self.bridge is None or self.last_frame is None or self._done:
            raise RuntimeError('Reset the environment before stepping')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all() or (np.abs(action)>1).any():
            raise ValueError(f'Action must be finite, in [-1,1], with shape {self.action_space.shape}')
        previous = self.last_frame
        try:
            self.bridge.send_action(previous['step'], action.tolist())
            frame = self.bridge.next_frame(previous['step']+1)
            if self.service_observation:
                from ..path_service.observation import validate_service_observation
                validate_service_observation(frame, self.config, previous)
            if frame['terminal'] and frame['step'] < self.config.steps:
                raise ProtocolError('Simulator ended before the configured horizon')
        except BaseException:
            self.close()
            raise
        self.last_frame = frame
        self._done = frame['step'] >= self.config.steps
        # A measured connectivity score; prevention policies do not train an RL reward.
        reward = float(frame['components'] == 1)
        info = self._info()
        observation = self._observation(frame)
        self._record(dict(event='transition', previous_step=previous['step'], proposed_action=action.tolist(),
                          executed_velocities=frame['velocities'], frame=frame, reward=reward,
                          terminated=False, truncated=self._done, info=info))
        if self._done: self.close()
        return observation, reward, False, self._done, info

    def close(self):
        if self.bridge is not None:
            bridge, self.bridge = self.bridge, None
            bridge.close()

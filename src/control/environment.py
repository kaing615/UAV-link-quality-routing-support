"""Gymnasium environment backed by a continuously running ns-3 process."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any
from uuid import uuid4

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bridge import ProtocolError, SimulatorBridge, simulator_output_path
from .config import RecoveryConfig
from .reward import is_success, reward_transition


class Ns3RecoveryEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: RecoveryConfig = RecoveryConfig(),
        launcher: Sequence[str] | None = None,
        output_root: Path = Path("reports/recovery_runs"),
        training: bool = True,
        timeout_s: float = 30.0,
        max_reset_attempts: int = 5,
    ) -> None:
        super().__init__()
        if not launcher:
            raise ValueError("A simulator launcher is required")
        if timeout_s <= 0 or max_reset_attempts < 1:
            raise ValueError("Invalid timeout or reset attempt limit")
        self.config = config
        self.launcher = list(launcher)
        self.output_root = Path(output_root)
        self.training = training
        self.timeout_s = timeout_s
        self.max_reset_attempts = max_reset_attempts
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3 * len(config.relay_ids),), dtype=np.float32)
        n = config.num_uavs
        # Per node: position, velocity, remaining travel, fixed flag. Per edge: rx,
        # observed link, RSSI, RSSI mask. Two global scalars: time and streak.
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(8 * n + 4 * n * n + 2,), dtype=np.float32)
        self.bridge: SimulatorBridge | None = None
        self.last_frame: dict[str, Any] | None = None
        self.episode_output_dir: Path | None = None
        self._episode_number = 0
        self._session_id = uuid4().hex[:10]
        self._layout_seed = 0
        self._radio_seed = 0
        self._succeeded = False
        self._recovery_time_s: int | None = None
        self._done = False
        self._group_separation = 0.0
        self._curriculum_stage = "nominal"

    @property
    def layout_seed(self) -> int:
        return self._layout_seed

    @property
    def radio_seed(self) -> int:
        return self._radio_seed

    def _command(self, output_dir: Path, layout_seed: int, radio_seed: int) -> list[str]:
        c = self.config
        return self.launcher + [
            "--scenarioMode=relay-recovery",
            "--recoveryPolicy=external",
            f"--numUavs={c.num_uavs}",
            f"--timeSteps={c.steps + 1}",
            f"--seed={radio_seed}",
            f"--recoveryLayoutSeed={layout_seed}",
            f"--recoveryRandomLayout={'true' if c.random_layout else 'false'}",
            f"--sourceId={c.source}",
            f"--destId={c.destination}",
            f"--warmup={c.warmup}",
            f"--xMax={c.x_max}",
            f"--yMax={c.y_max}",
            f"--zMin={c.z_min}",
            f"--zMax={c.z_max}",
            f"--commRange={c.comm_range}",
            f"--groupSeparation={self._group_separation}",
            f"--recoverySpeed={c.max_speed}",
            f"--recoveryBudget={c.budget}",
            f"--recoveryMinSeparation={c.min_separation}",
            f"--recoveryConfirmWindows={c.confirm_windows}",
            "--routingStrategy=olsr",
            "--enableDataFlow=true",
            "--enableAnim=false",
            f"--outputDir={simulator_output_path(output_dir, self.launcher)}",
        ]

    def _observation(self, frame: dict[str, Any]) -> np.ndarray:
        c = self.config
        positions = np.asarray(frame["positions"], dtype=np.float32)
        positions = (positions - np.asarray([0, 0, c.z_min], dtype=np.float32)) / np.asarray(
            [c.x_max, c.y_max, c.z_max - c.z_min], dtype=np.float32
        )
        velocities = np.asarray(frame["velocities"], dtype=np.float32) / c.max_speed
        remaining_travel = (
            1.0 - np.asarray(frame["travel"], dtype=np.float32) / c.budget
            if c.budget > 0
            else np.zeros(c.num_uavs, dtype=np.float32)
        )
        fixed_mask = np.zeros(c.num_uavs, dtype=np.float32)
        fixed_mask[[c.source, c.destination]] = 1.0
        rx = np.asarray(frame["rx_ratio"], dtype=np.float32)
        measured_link = (rx >= 0.5).astype(np.float32)
        rssi_mask = np.asarray(frame["rssi_mask"], dtype=np.float32)
        rssi = np.clip((np.asarray(frame["rssi"], dtype=np.float32) + 120.0) / 120.0, 0.0, 1.0) * rssi_mask
        remaining_time = 1.0 - frame["step"] / c.steps
        streak = min(1.0, frame["connected_streak"] / c.confirm_windows)
        observation = np.concatenate(
            (
                positions.ravel(),
                velocities.ravel(),
                remaining_travel,
                fixed_mask,
                rx.ravel(),
                measured_link.ravel(),
                rssi.ravel(),
                rssi_mask.ravel(),
                [remaining_time, streak],
            )
        ).astype(np.float32)
        if not np.all(np.isfinite(observation)):
            raise ProtocolError("Observation contains nonfinite values")
        return np.clip(observation, -1.0, 1.0)

    def _info(self, *, reason: str | None = None, reward_components: dict[str, float] | None = None) -> dict[str, Any]:
        frame = self.last_frame
        assert frame is not None
        return {
            "success": self._succeeded,
            "terminal_reason": reason,
            "recovery_time_s": self._recovery_time_s,
            "censored": not self._succeeded,
            "layout_seed": self._layout_seed,
            "radio_seed": self._radio_seed,
            "group_separation": self._group_separation,
            "curriculum_stage": self._curriculum_stage,
            "step": frame["step"],
            "components": frame["components"],
            "source_dest_connected": frame["source_dest_connected"],
            "olsr_route_usable": frame["olsr_route_usable"],
            "connected_streak": frame["connected_streak"],
            "safety_rejected": frame["safety_rejected"],
            "travel": list(frame["travel"]),
            "budget_exhausted": all(frame["travel"][i] >= self.config.budget - 1e-6 for i in self.config.relay_ids),
            "reward_components": reward_components or {},
            "reward_weights": asdict(self.config.reward),
        }

    def _append_record(self, record: dict[str, Any]) -> None:
        assert self.episode_output_dir is not None
        with (self.episode_output_dir / "transitions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.close()
        self._episode_number += 1
        options = options or {}
        separation = options.get("group_separation", 0.0)
        if isinstance(separation, bool) or not isinstance(separation, Real) or not isfinite(separation):
            raise ValueError("Group separation must be a finite number in meters")
        separation = float(separation)
        stage = options.get("curriculum_stage", "nominal")
        if stage not in ("easy", "intermediate", "nominal"):
            raise ValueError("Invalid curriculum_stage")
        if not self.training and (separation != 0.0 or stage != "nominal"):
            raise ValueError("Curriculum overrides are not allowed during evaluation")
        c = self.config
        if separation != 0.0:
            # Mirror ConfigureRecoveryMobility's compact-group geometry checks.
            spacing = max(12.0, 2.0 * c.min_separation)
            largest_group = (c.num_uavs - 1) // 2
            if (
                separation <= 1.5 * c.comm_range
                or separation > c.x_max
                or spacing * largest_group >= separation / 2.0
                or spacing * largest_group >= 0.3 * c.comm_range
            ):
                raise ValueError("Group separation does not fit the simulator recovery geometry")
        self._group_separation = separation
        self._curriculum_stage = stage
        layout_value = options["layout_seed"] if "layout_seed" in options else self.np_random.integers(1, 2**31 - 1)
        radio_value = options["radio_seed"] if "radio_seed" in options else self.np_random.integers(1, 2**31 - 1)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (layout_value, radio_value)
        ):
            raise ValueError("Layout and radio seeds must be integers")
        initial_layout = int(layout_value)
        initial_radio = int(radio_value)
        if initial_layout < 0 or initial_radio < 0:
            raise ValueError("Seeds must be nonnegative")
        if "layout_seed" not in options and "radio_seed" not in options and initial_layout == initial_radio:
            initial_radio += 1
        last_error = "initial frame was not fragmented"
        attempts = 1 if "layout_seed" in options or "radio_seed" in options else self.max_reset_attempts
        for attempt in range(attempts):
            self._layout_seed = initial_layout + attempt
            self._radio_seed = initial_radio + attempt
            output_dir = (
                self.output_root / f"episode_{self._session_id}_{self._episode_number:06d}_attempt_{attempt + 1:02d}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            self.episode_output_dir = output_dir
            try:
                self.bridge = SimulatorBridge(
                    self._command(output_dir, self._layout_seed, self._radio_seed),
                    self.config,
                    output_dir / "simulator.stderr.log",
                    self.timeout_s,
                )
                frame = self.bridge.next_frame(0)
                if frame["terminal"]:
                    raise ProtocolError("Initial recovery frame was terminal")
            except BaseException:
                self.close()
                raise
            if frame["components"] > 1:
                self.last_frame = frame
                self._succeeded = False
                self._recovery_time_s = None
                self._done = False
                (output_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "config": asdict(self.config),
                            "simulator_snapshots": self.config.steps + 1,
                            "layout_seed": self._layout_seed,
                            "radio_seed": self._radio_seed,
                            "group_separation": self._group_separation,
                            "curriculum_stage": self._curriculum_stage,
                            "command": self._command(output_dir, self._layout_seed, self._radio_seed),
                        },
                        indent=2,
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                info = self._info()
                self._append_record({"event": "initial", "frame": frame, "info": info})
                return self._observation(frame), info
            last_error = f"initial frame connected for layout seed {self._layout_seed}, radio seed {self._radio_seed}"
            self.close()
        raise RuntimeError(f"Could not start a fragmented recovery layout after {attempts} attempts: {last_error}")

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.bridge is None or self.last_frame is None or self._done:
            raise RuntimeError("Reset the environment before stepping")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1):
            raise ValueError(f"Action must be finite, in [-1,1], with shape {self.action_space.shape}")
        previous = self.last_frame
        try:
            self.bridge.send_action(previous["step"], action.tolist())
            current = self.bridge.next_frame(previous["step"] + 1)
            if current["terminal"] and current["step"] < self.config.steps:
                raise ProtocolError("Simulator ended before the configured horizon")
        except BaseException:
            self.close()
            raise
        self.last_frame = current
        reward, components = reward_transition(previous, current, self.config, self._succeeded)
        if is_success(current, self.config) and not self._succeeded:
            self._succeeded = True
            self._recovery_time_s = int(current["step"])
        exhausted = all(current["travel"][i] >= self.config.budget - 1e-6 for i in self.config.relay_ids)
        horizon = current["step"] >= self.config.steps
        terminated = bool(self.training and self._succeeded)
        truncated = horizon and not terminated
        reason = (
            "success"
            if self.training and self._succeeded
            else "budget_exhausted"
            if exhausted and not self._succeeded and horizon
            else "horizon"
            if horizon
            else None
        )
        self._done = terminated or truncated
        info = self._info(reason=reason, reward_components=components)
        observation = self._observation(current)
        self._append_record(
            {
                "event": "transition",
                "previous_step": previous["step"],
                "proposed_action": action.tolist(),
                "executed_velocities": current["velocities"],
                "frame": current,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )
        if self._done:
            self.close()
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None

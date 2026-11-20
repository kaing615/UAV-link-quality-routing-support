"""Train centralized PPO against the live ns-3 recovery simulator."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict
from numbers import Integral
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from src.control.bridge import make_launcher
from src.control.config import RecoveryConfig
from src.control.curriculum import CurriculumWrapper
from src.control.environment import Ns3RecoveryEnv
from src.control.evaluation import (
    EpisodeRecord,
    evaluate_episode,
    git_dirty,
    recovery_source_hashes,
    runtime_versions,
    scenario_seeds,
    sha256_file,
    validate_seed_partitions,
    verify_build_manifest,
)


class TrainingSeedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, policy_seed: int):
        super().__init__(env)
        self.scenarios = scenario_seeds("training", 49_999, policy_seed)
        self.index = 0

    def reset(self, *, seed=None, options=None):
        if self.index >= len(self.scenarios):
            raise RuntimeError("training scenario seed range exhausted")
        scenario = self.scenarios[self.index]
        self.index += 1
        episode_options = dict(options or {})
        episode_options.update(layout_seed=scenario.layout_seed, radio_seed=scenario.radio_seed)
        return self.env.reset(seed=seed, options=episode_options)


class EpisodeDiagnostics(gym.Wrapper):
    """Stream completed training episodes and record curriculum changes at reset."""

    def __init__(self, env: gym.Env, *, output: Path):
        super().__init__(env)
        self.output = output
        self.timesteps = 0
        self.episode_number = 0
        self.stage_transitions: list[dict] = []
        self._stream = (output / "training_episodes.csv").open("w", newline="", encoding="utf-8")
        self._writer = None
        self._episode = None

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.episode_number += 1
        stage = info.get("curriculum_stage", "nominal")
        if not self.stage_transitions or self.stage_transitions[-1]["stage"] != stage:
            self.stage_transitions.append({
                "timesteps": self.timesteps, "episode": self.episode_number,
                "stage": stage, "group_separation": info.get("group_separation", 0.0),
            })
            (self.output / "curriculum_transitions.json").write_text(
                json.dumps(self.stage_transitions, indent=2), encoding="utf-8"
            )
        self._episode = {
            "episode": self.episode_number, "curriculum_stage": stage,
            "group_separation": info.get("group_separation", 0.0),
            "layout_seed": info["layout_seed"], "radio_seed": info["radio_seed"],
            "initial_components": info.get("components"),
            "initial_travel": np.asarray(info.get("travel", []), dtype=float),
            "elapsed_steps": 0, "return": 0.0, "rejected_actions": 0,
        }
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.timesteps += 1
        episode = self._episode
        if episode is None:
            raise RuntimeError("reset diagnostics before stepping")
        episode["elapsed_steps"] += 1
        episode["return"] += float(reward)
        episode["rejected_actions"] += int(bool(info.get("safety_rejected", False)))
        if terminated or truncated:
            travel = np.asarray(info.get("travel", []), dtype=float) - episode.pop("initial_travel")
            episode.update(
                timesteps=self.timesteps, success=bool(info.get("success", False)),
                recovery_time_s=info.get("recovery_time_s"), terminal_reason=info.get("terminal_reason"),
                capped_recovery_time_s=(info.get("recovery_time_s") if info.get("success")
                                        else self.env.unwrapped.config.steps),
                travel_total_m=float(travel.sum()), travel_max_m=float(travel.max()) if travel.size else 0.0,
                budget_exhausted=bool(info.get("budget_exhausted", False)),
                rejection_rate=episode["rejected_actions"] / episode["elapsed_steps"],
                final_components=info.get("components"),
            )
            if self._writer is None:
                self._writer = csv.DictWriter(self._stream, fieldnames=list(episode))
                self._writer.writeheader()
            self._writer.writerow(episode)
            self._stream.flush()
            self._episode = None
        return observation, reward, terminated, truncated, info

    def close(self):
        try:
            self.env.close()
        finally:
            self._stream.close()


def validation_metrics(records: list[EpisodeRecord], *, steps: int) -> dict:
    """Success first; failures are charged the full nominal evaluation horizon."""
    if not records or any(record.status == "technical_error" for record in records):
        raise ValueError("validation requires nonempty technically valid episodes")
    success_rate = float(np.mean([r.status == "success" for r in records]))
    capped_time = float(np.mean([r.recovery_time_s if r.status == "success" else steps for r in records]))
    mean_travel = float(np.mean([r.travel_total_m for r in records]))
    mean_return = float(np.mean([r.return_ for r in records]))
    components = [r.final_components for r in records if r.final_components is not None]
    transitions = sum(r.elapsed_steps for r in records)
    return {
        "recovery_rate": success_rate,
        "mean_capped_recovery_time_s": capped_time,
        "mean_travel_total_m": mean_travel,
        "mean_validation_return": mean_return,
        "budget_exhaustion_rate": float(np.mean([r.budget_exhausted for r in records])),
        "rejection_rate": sum(r.rejected_actions for r in records) / transitions if transitions else 0.0,
        "mean_final_components": float(np.mean(components)) if components else None,
        "selection_rank": (success_rate, -capped_time, -mean_travel, mean_return),
    }


class ValidationCheckpoint(BaseCallback):
    def __init__(self, *, config, launcher, output: Path, scenarios, every_steps: int):
        super().__init__()
        self.config = config
        self.launcher = launcher
        self.output = output
        self.scenarios = scenarios
        self.every_steps = every_steps
        self.last_evaluation = 0
        self.best: tuple[float, float, float, float] | None = None
        self.best_metrics: dict | None = None
        self.validation_wall_time_s = 0.0
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_evaluation < self.every_steps:
            return True
        self.last_evaluation = self.num_timesteps
        self._evaluate_checkpoint("rollout")
        return True

    def _on_training_end(self) -> None:
        # SB3 calls _on_step during collection, before optimizing that rollout.
        # The final update must have its own validation chance.
        self._evaluate_checkpoint("final_updated")

    def _evaluate_checkpoint(self, phase: str) -> None:
        started = time.perf_counter()
        records = []
        evaluation_output = self.output / "validation" / f"step_{self.num_timesteps}_{phase}"
        evaluation_output.mkdir(parents=True, exist_ok=True)
        for index, scenario in enumerate(self.scenarios):
            env = Ns3RecoveryEnv(
                config=self.config,
                launcher=self.launcher,
                output_root=evaluation_output / f"scenario_{index}",
                training=False,
            )
            record = evaluate_episode(
                env,
                lambda observation, frame: self.model.predict(observation, deterministic=True)[0],
                scenario,
                "ppo",
                self.model.seed,
            )
            records.append(record)
            # Flush evidence even when a later simulator case fails.
            with (evaluation_output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(record.to_dict()))
                writer.writeheader()
                writer.writerows(row.to_dict() for row in records)
            if record.status == "technical_error":
                raise RuntimeError(f"validation simulator error: {record.error}")
        metrics = validation_metrics(records, steps=self.config.steps)
        rank = metrics["selection_rank"]
        elapsed = time.perf_counter() - started
        self.validation_wall_time_s += elapsed
        self.history.append(
            {
                "timesteps": self.num_timesteps,
                "phase": phase,
                **{key: value for key, value in metrics.items() if key != "selection_rank"},
                "scenarios": len(records),
                "validation_wall_time_s": elapsed,
            }
        )
        with (self.output / "validation_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)
        if self.best is None or rank > self.best:
            self.best = rank
            self.best_metrics = metrics
            self.model.save(self.output / "best_model")
            (self.output / "best_checkpoint.json").write_text(
                json.dumps({
                    "timesteps": self.num_timesteps, "phase": phase,
                    "selection_order": ["recovery_rate", "negative_capped_recovery_time_s",
                                        "negative_mean_travel_total_m", "mean_validation_return"],
                    "selection_rank": rank, "metrics": metrics,
                    "scenario_records": str((evaluation_output / "episodes.csv").relative_to(self.output)),
                }, indent=2, allow_nan=False), encoding="utf-8",
            )


def validate_training_args(args: argparse.Namespace) -> None:
    """Check all sizes and reserved ranges before creating or launching anything."""
    for name in ("total_timesteps", "steps", "n_steps", "batch_size", "n_epochs", "validation_every",
                 "validation_scenarios", "test_scenarios", "torch_threads"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be a positive integer")
    if args.n_steps < 2 or args.batch_size < 2:
        raise ValueError("n-steps and batch-size must be at least 2")
    if args.batch_size > args.n_steps or args.n_steps % args.batch_size:
        raise ValueError("batch-size must divide n-steps")
    if isinstance(args.seed, bool) or not isinstance(args.seed, Integral) or not 0 <= args.seed < 999:
        raise ValueError("seed must be an integer in [0, 998]")
    if args.validation_scenarios >= 50_000 or args.test_scenarios >= 50_000:
        raise ValueError("scenario counts must be less than 50000 to fit reserved seed ranges")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("training output must be an empty directory or a new path")


def _git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def train(args: argparse.Namespace) -> dict:
    validate_training_args(args)
    binary = args.binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    build_hashes = verify_build_manifest(binary)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = RecoveryConfig(steps=args.steps)
    launcher = make_launcher(binary, args.wsl)
    validation = scenario_seeds("validation", args.validation_scenarios)
    test = scenario_seeds("test", args.test_scenarios)
    train_preview = scenario_seeds("training", 49_999, args.seed)
    validate_seed_partitions(train_preview, validation, test)
    dependencies = runtime_versions()
    source_hashes = recovery_source_hashes()
    torch.set_num_threads(args.torch_threads)
    rollout_timesteps = ((args.total_timesteps + args.n_steps - 1) // args.n_steps) * args.n_steps
    manifest = {
        "scope": "short_smoke" if args.total_timesteps < 100_000 else "training_run",
        "config": asdict(config),
        "training": {
            "policy_seed": args.seed,
            "total_timesteps_requested": args.total_timesteps,
            "total_timesteps_expected": rollout_timesteps,
            "torch_threads": torch.get_num_threads(),
            "n_envs": 1,
            "curriculum": {
                "enabled": args.curriculum,
                "schedule_basis": "total_timesteps_requested",
                "schedule_basis_timesteps": args.total_timesteps,
                "stage_selection": "on_episode_reset_using_completed_transitions",
                "stages": [
                    {"stage": "easy", "start_fraction": 0.0, "end_fraction": 0.25,
                     "group_separation_m": 1.65 * config.comm_range},
                    {"stage": "intermediate", "start_fraction": 0.25, "end_fraction": 0.5,
                     "group_separation_m": 2.0 * config.comm_range},
                    {"stage": "nominal", "start_fraction": 0.5, "group_separation_m": 0.0},
                ] if args.curriculum else [{"stage": "nominal", "start_fraction": 0.0,
                                           "group_separation_m": 0.0}],
            },
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "learning_rate": 0.0003,
            "architecture": [128, 128],
            "device": "cpu",
        },
        "seed_partitions": {
            "training_first": asdict(train_preview[0]),
            "training_last_reserved": asdict(train_preview[-1]),
            "validation": [asdict(s) for s in validation],
            "test_reserved": [asdict(s) for s in test],
        },
        "provenance": {
            "binary": str(binary),
            "binary_sha256": sha256_file(binary),
            "build_manifest": build_hashes,
            "git_revision": _git_revision(),
            "git_dirty": git_dirty(),
            "python_source_sha256": source_hashes,
            "dependency_versions": dependencies,
            **dependencies,
        },
    }
    (output / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    diagnostics = EpisodeDiagnostics(
        TrainingSeedWrapper(
            CurriculumWrapper(
                Ns3RecoveryEnv(config=config, launcher=launcher, output_root=output / "training", training=True),
                total_timesteps=args.total_timesteps, enabled=args.curriculum,
            ),
            args.seed,
        ),
        output=output,
    )
    env = Monitor(
        diagnostics,
        str(output / "monitor.csv"),
    )
    try:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [128, 128]},
            learning_rate=0.0003,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            seed=args.seed,
            device="cpu",
            verbose=1,
        )
        model.set_logger(configure(str(output), ["stdout", "csv"]))
        before = {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()}
        callback = ValidationCheckpoint(
            config=config,
            launcher=launcher,
            output=output,
            scenarios=validation,
            every_steps=args.validation_every,
        )
        started = time.perf_counter()
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
        # SB3 normally dumps before each update; include the last update as well.
        model.logger.record("time/total_timesteps", model.num_timesteps)
        model.logger.dump(step=model.num_timesteps)
        elapsed = time.perf_counter() - started
        model.save(output / "model")
        changed = sum(int(torch.count_nonzero(value - before[key])) for key, value in model.policy.state_dict().items())
        if changed == 0:
            raise RuntimeError("PPO policy weights did not change")
        if not (output / "best_model.zip").is_file():
            raise RuntimeError("validation never selected a checkpoint")
        manifest["result"] = {
            "total_timesteps_actual": model.num_timesteps,
            "training_wall_time_s": elapsed,
            "changed_weight_elements": changed,
            "best_validation_return": callback.best_metrics["mean_validation_return"],
            "best_validation_selection_rank": callback.best,
            "best_validation_metrics": callback.best_metrics,
            "validation_evaluations": len(callback.history),
            "validation_wall_time_s": callback.validation_wall_time_s,
            "transitions_per_wall_second": model.num_timesteps / elapsed,
            "transitions_per_training_second": model.num_timesteps / max(elapsed - callback.validation_wall_time_s, 1e-9),
            "curriculum_stage_transitions": diagnostics.stage_transitions,
            "completed_training_episodes": diagnostics.episode_number - int(diagnostics._episode is not None),
            "final_model_sha256": sha256_file(output / "model.zip"),
            "best_model_sha256": sha256_file(output / "best_model.zip"),
            "python_source_sha256_at_completion": recovery_source_hashes(),
        }
        manifest["result"]["source_changed_during_run"] = (
            manifest["result"]["python_source_sha256_at_completion"] != source_hashes
        )
        (output / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--wsl", default=None, help="WSL distribution, e.g. Ubuntu; omit on Linux")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--validation-every", type=int, default=1024)
    parser.add_argument("--validation-scenarios", type=int, default=8)
    parser.add_argument("--test-scenarios", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false",
                        help="Use nominal training layouts throughout")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_training_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    result = train(args)
    print(json.dumps({"output": str(args.output.resolve()), "result": result["result"]}, indent=2))


if __name__ == "__main__":
    main()

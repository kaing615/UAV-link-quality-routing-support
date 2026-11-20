"""Training bookkeeping tests; simulator behavior is covered by live integration runs."""

import csv
import json
from argparse import Namespace

import gymnasium as gym
import numpy as np
import pytest
import scripts.train.train_recovery_ppo as training

from src.control.config import RecoveryConfig
from src.control.evaluation import EpisodeRecord, ScenarioSeeds


def episode(status="success", recovery=10, travel=20.0, return_=1.0, **kwargs):
    return EpisodeRecord("ppo", 1, 10, 11, status, recovery, 120, return_, travel_total_m=travel, **kwargs)


def test_validation_selection_prioritizes_success_then_capped_time_travel_return():
    def rank(rows):
        return training.validation_metrics(rows, steps=120)["selection_rank"]

    assert rank([episode()]) > rank([episode("horizon", None, 0, 1000)])
    assert rank([episode(recovery=9, travel=100)]) > rank([episode(recovery=10, travel=0, return_=1000)])
    assert rank([episode(travel=19, return_=-100)]) > rank([episode(travel=20, return_=1000)])
    assert rank([episode(return_=2)]) > rank([episode(return_=1)])
    assert rank([episode(), episode("horizon", None, 0)])[1] == -65


def test_validation_metrics_include_failures_budget_rejections_components():
    metrics = training.validation_metrics(
        [episode(rejected_actions=6, final_components=1),
         episode("budget_exhausted", None, 60, budget_exhausted=True, rejected_actions=18, final_components=3)],
        steps=120,
    )
    assert metrics["recovery_rate"] == 0.5
    assert metrics["mean_capped_recovery_time_s"] == 65
    assert metrics["mean_travel_total_m"] == 40
    assert metrics["budget_exhaustion_rate"] == 0.5
    assert metrics["rejection_rate"] == 0.1
    assert metrics["mean_final_components"] == 2


class RecordingEnv(gym.Env):
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    training = True
    config = RecoveryConfig(steps=2)

    def reset(self, *, seed=None, options=None):
        self.options = dict(options or {})
        self.calls = 0
        return np.zeros(1, dtype=np.float32), {
            **self.options, "travel": [0, 0], "components": 3,
        }

    def step(self, action):
        self.calls += 1
        done = self.calls == 2
        return np.zeros(1, dtype=np.float32), 1.0, False, done, {
            **self.options, "travel": [self.calls, self.calls], "components": 2,
            "success": False, "recovery_time_s": None, "terminal_reason": "horizon" if done else None,
            "budget_exhausted": False, "safety_rejected": True,
        }


def test_seed_wrapper_preserves_options_and_never_reuses_reserved_seeds():
    original = {"group_separation": 396, "curriculum_stage": "easy", "layout_seed": 1, "radio_seed": 2}
    first, second = RecordingEnv(), RecordingEnv()
    wrappers = [training.TrainingSeedWrapper(env, 7) for env in [first, second]]
    seen = []
    for _ in range(3):
        infos = [wrapper.reset(seed=100, options=original)[1] for wrapper in wrappers]
        assert infos[0] == infos[1]
        assert infos[0]["group_separation"] == 396
        assert infos[0]["curriculum_stage"] == "easy"
        seen.append((infos[0]["layout_seed"], infos[0]["radio_seed"]))
    assert len(set(seen)) == 3
    assert original["layout_seed"] == 1
    assert seen[0] == (800_000, 800_001)


def test_episode_csv_and_stage_transitions_stream_without_log_scanning(tmp_path):
    curriculum = training.CurriculumWrapper(RecordingEnv(), total_timesteps=8)
    env = training.EpisodeDiagnostics(training.TrainingSeedWrapper(curriculum, 1), output=tmp_path)
    try:
        for _ in range(3):
            env.reset()
            env.step(np.zeros(1))
            env.step(np.zeros(1))
        with (tmp_path / "training_episodes.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["curriculum_stage"] for row in rows] == ["easy", "intermediate", "nominal"]
        assert [float(row["travel_total_m"]) for row in rows] == [4, 4, 4]
        assert [float(row["rejection_rate"]) for row in rows] == [1, 1, 1]
        assert [int(row["timesteps"]) for row in rows] == [2, 4, 6]
        transitions = env.stage_transitions
        assert [row["timesteps"] for row in transitions] == [0, 2, 4]
        assert json.loads((tmp_path / "curriculum_transitions.json").read_text()) == transitions
    finally:
        env.close()


def valid_args(tmp_path):
    return Namespace(binary=tmp_path / "missing", output=tmp_path / "out", wsl=None,
                     total_timesteps=8, seed=1, steps=120, n_steps=8, batch_size=4, n_epochs=1,
                     validation_every=8, validation_scenarios=8, test_scenarios=20, torch_threads=2,
                     curriculum=True)


@pytest.mark.parametrize(("key", "value"), [
    ("total_timesteps", 0), ("steps", 0), ("n_steps", 1), ("batch_size", 1),
    ("batch_size", 3), ("n_epochs", 0), ("validation_every", 0), ("validation_scenarios", 0),
    ("test_scenarios", 0), ("validation_scenarios", 50_000), ("test_scenarios", 50_000),
    ("seed", -1), ("seed", 999), ("torch_threads", 0),
])
def test_invalid_training_inputs_rejected_before_binary_or_output_access(tmp_path, key, value):
    args = valid_args(tmp_path)
    setattr(args, key, value)
    with pytest.raises(ValueError):
        training.train(args)
    assert not args.output.exists()


def test_existing_output_rejected_before_launch(tmp_path):
    args = valid_args(tmp_path)
    args.output.mkdir()
    (args.output / "evidence.txt").write_text("preserve")
    with pytest.raises(ValueError, match="empty"):
        training.validate_training_args(args)
    assert (args.output / "evidence.txt").read_text() == "preserve"


def test_cli_defaults_and_curriculum_opt_out():
    parser = training.build_parser()
    args = parser.parse_args(["--binary", "binary", "--output", "output"])
    assert args.validation_scenarios == 8 and args.torch_threads == 2 and args.curriculum
    assert not parser.parse_args(["--binary", "binary", "--output", "output", "--no-curriculum"]).curriculum


def test_checkpoint_records_and_metadata_preserve_selection_reasons(tmp_path, monkeypatch):
    class Model:
        seed = 1

        def save(self, path):
            pass

    records = [episode(), episode("budget_exhausted", None, 60, budget_exhausted=True)]
    monkeypatch.setattr(training, "Ns3RecoveryEnv", lambda **kwargs: object())
    monkeypatch.setattr(training, "evaluate_episode", lambda *args, **kwargs: records.pop(0))
    callback = training.ValidationCheckpoint(config=RecoveryConfig(), launcher=["sim"], output=tmp_path,
                                           scenarios=[ScenarioSeeds(10, 11), ScenarioSeeds(12, 13)], every_steps=8)
    callback.model = Model()
    callback.num_timesteps = 8
    callback._evaluate_checkpoint("final_updated")
    metadata = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert metadata["phase"] == "final_updated"
    assert metadata["metrics"]["recovery_rate"] == 0.5
    assert metadata["selection_rank"][:2] == [0.5, -65]
    with (tmp_path / "validation" / "step_8_final_updated" / "episodes.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2


def test_one_rollout_logs_optimization_and_final_updated_nominal_validation(tmp_path, monkeypatch):
    """Exercise real PPO bookkeeping on a toy Gym env, without asserting radio outcomes."""
    args = valid_args(tmp_path)
    args.binary.write_bytes(b"test binary placeholder; never executed")
    validation_configs = []
    envs = []

    def make_env(**kwargs):
        if not kwargs["training"]:
            validation_configs.append(kwargs["config"])
        env = RecordingEnv()
        envs.append(env)
        return env

    monkeypatch.setattr(training, "Ns3RecoveryEnv", make_env)
    monkeypatch.setattr(training, "make_launcher", lambda *args: ["test binary never executed"])
    monkeypatch.setattr(training, "evaluate_episode", lambda *args, **kwargs: episode())
    manifest = training.train(args)
    assert manifest["result"]["total_timesteps_actual"] == 8
    assert manifest["result"]["changed_weight_elements"] > 0
    assert manifest["training"]["n_envs"] == 1
    assert manifest["training"]["curriculum"]["schedule_basis_timesteps"] == 8
    assert all(config == RecoveryConfig(steps=args.steps) for config in validation_configs)
    assert manifest["result"]["transitions_per_wall_second"] > 0
    with (args.output / "validation_history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    assert history[-1]["phase"] == "final_updated"
    with (args.output / "progress.csv").open(newline="") as stream:
        progress = list(csv.DictReader(stream))
    assert progress[-1]["train/loss"]
    assert progress[-1]["train/entropy_loss"]
    assert progress[-1]["train/value_loss"]
    assert int(progress[-1]["time/total_timesteps"]) == 8

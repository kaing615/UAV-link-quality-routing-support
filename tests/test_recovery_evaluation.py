"""Episode-level accounting for recovery experiments (no radio surrogate)."""

from pathlib import Path

import numpy as np
import pytest

from src.control.evaluation import (
    EpisodeRecord,
    ScenarioSeeds,
    baseline_action,
    evaluate_episode,
    file_hashes,
    hierarchical_bootstrap_recovery,
    read_final_metrics,
    runtime_versions,
    scenario_seeds,
    summarize_episodes,
    validate_paired_grid,
    validate_seed_partitions,
)


def test_failures_are_censored_and_technical_errors_are_separate():
    rows = [
        EpisodeRecord("ppo", 1, 10, 11, "success", 13, 120, 12.0),
        EpisodeRecord("ppo", 1, 12, 13, "horizon", None, 120, 14.0),
        EpisodeRecord("ppo", 1, 14, 15, "technical_error", None, 3, 0.0),
    ]
    summary = summarize_episodes(rows)["ppo"]
    assert summary["episodes"] == 2
    assert summary["technical_errors"] == 1
    assert summary["successes"] == 1
    assert summary["censored"] == 1
    assert summary["recovery_rate"] == 0.5
    assert summary["time_to_recovery_success_only_mean_s"] == 13
    assert rows[1].recovery_time_s is None


def test_summary_uses_policy_seed_means_before_group_mean():
    rows = [
        EpisodeRecord("ppo", 1, 10, 11, "success", 5, 120, 1.0),
        EpisodeRecord("ppo", 1, 12, 13, "success", 6, 120, 1.0),
        EpisodeRecord("ppo", 1, 14, 15, "success", 7, 120, 1.0),
        EpisodeRecord("ppo", 2, 16, 17, "horizon", None, 120, 1.0),
    ]
    summary = summarize_episodes(rows)["ppo"]
    assert summary["episodes"] == 4
    assert summary["policy_seeds"] == 2
    assert summary["recovery_rate"] == 0.5
    assert summary["pooled_episode_recovery_rate"] == 0.75


def test_hierarchical_bootstrap_resamples_shared_scenarios():
    rows = [
        EpisodeRecord(
            "ppo",
            policy_seed,
            200_000_000 + 2 * case,
            200_000_001 + 2 * case,
            "success" if case < 10 else "horizon",
            1 if case < 10 else None,
            120,
            1.0,
        )
        for policy_seed in range(5)
        for case in range(20)
    ]
    low, high = hierarchical_bootstrap_recovery(rows, replicates=2000)
    assert low < 0.4 and high > 0.6


def test_hierarchical_bootstrap_requires_complete_unique_grid():
    rows = [
        EpisodeRecord("ppo", 1, 10, 11, "success", 1, 2, 1.0),
        EpisodeRecord("ppo", 2, 10, 11, "success", 1, 2, 1.0),
        EpisodeRecord("ppo", 2, 12, 13, "horizon", None, 2, 0.0),
    ]
    with pytest.raises(ValueError, match="unequal paired"):
        hierarchical_bootstrap_recovery(rows)
    with pytest.raises(ValueError, match="duplicate scenario"):
        hierarchical_bootstrap_recovery([rows[0], rows[0], rows[1]])


def test_comparative_grid_rejects_missing_policy_case():
    rows = [
        EpisodeRecord("zero", None, 10, 11, "horizon", None, 2, 0),
        EpisodeRecord("zero", None, 12, 13, "horizon", None, 2, 0),
        EpisodeRecord("ppo", 1, 10, 11, "success", 1, 2, 1),
    ]
    with pytest.raises(ValueError, match="unequal paired"):
        validate_paired_grid(rows)


def test_rejects_layout_or_radio_overlap_across_splits():
    train = [ScenarioSeeds(10, 11)]
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_partitions(train, [ScenarioSeeds(10, 13)], [])
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_partitions(train, [ScenarioSeeds(12, 11)], [])
    validate_seed_partitions(
        scenario_seeds("training", 2, 998), scenario_seeds("validation", 2), scenario_seeds("test", 2)
    )
    with pytest.raises(ValueError, match="range"):
        scenario_seeds("training", 1, 999)


def test_baselines_have_correct_shape_and_relay_targets():
    frame = {"positions": [[0, 0, 100], [0, 10, 100], [0, 20, 100], [300, 0, 100]]}
    config = type("Config", (), {"num_uavs": 4, "source": 0, "destination": 3, "max_speed": 8})()
    zero = baseline_action("zero", frame, config, np.random.default_rng(1))
    chain = baseline_action("relay_chain", frame, config, np.random.default_rng(1))
    assert zero.shape == chain.shape == (6,)
    assert np.all(zero == 0)
    assert chain[0] > 0 and chain[3] > 0
    assert np.all(np.linalg.norm(chain.reshape(-1, 3), axis=1) <= 1)


def test_metrics_read_only_after_normal_completion(tmp_path: Path):
    metrics = tmp_path / "closed_loop_metrics.csv"
    metrics.write_text("tx_packets,rx_packets,offered_packets,pdr,mean_delay_ms\n10,4,10,0.4,12.5\n")
    assert read_final_metrics(metrics, normally_completed=True)["pdr"] == 0.4
    with pytest.raises(ValueError, match="normal"):
        read_final_metrics(metrics, normally_completed=False)


def test_source_provenance_detects_uncommitted_file_change(tmp_path: Path):
    source = tmp_path / "control.py"
    source.write_text("value = 1\n", encoding="utf-8")
    before = file_hashes([source], tmp_path)
    source.write_text("value = 2\n", encoding="utf-8")
    after = file_hashes([source], tmp_path)
    assert list(before) == ["control.py"]
    assert before["control.py"] != after["control.py"]
    with pytest.raises(ValueError, match="outside root"):
        file_hashes([source], tmp_path / "other")
    versions = runtime_versions()
    assert versions["python"] and versions["numpy"]


def test_episode_continues_after_recovery_and_reads_final_metrics(tmp_path: Path):
    class AccountingEnv:
        config = type("Config", (), {"steps": 2})()
        episode_output_dir = tmp_path

        def reset(self, *, options):
            self.last_frame = {
                "components": 2,
                "source_dest_connected": False,
                "olsr_route_usable": False,
                "travel": [0, 0],
                "safety_rejected": False,
            }
            return np.zeros(2), options

        def step(self, action):
            current = getattr(self, "calls", 0) + 1
            self.calls = current
            self.last_frame = {
                "components": 1,
                "source_dest_connected": True,
                "olsr_route_usable": True,
                "travel": [current, 0],
                "safety_rejected": False,
            }
            return (
                np.zeros(2),
                1.0,
                False,
                current == 2,
                {"success": True, "recovery_time_s": 1, "terminal_reason": "horizon" if current == 2 else None,
                 "budget_exhausted": current == 2},
            )

        def close(self):
            (tmp_path / "closed_loop_metrics.csv").write_text(
                "tx_packets,rx_packets,offered_packets,pdr,mean_delay_ms\n10,4,10,0.4,12.5\n"
            )

    env = AccountingEnv()
    result = evaluate_episode(env, lambda obs, frame: np.zeros(2), ScenarioSeeds(10, 11), "zero", None)
    assert result.status == "success"
    assert result.elapsed_steps == 2
    assert result.recovery_time_s == 1
    assert result.pdr == 0.4
    assert result.final_components == 1
    assert result.budget_exhausted


def test_episode_marks_bridge_failure_as_technical(tmp_path: Path):
    class FailingEnv:
        config = type("Config", (), {"steps": 3})()
        episode_output_dir = tmp_path

        def reset(self, *, options):
            self.last_frame = {"components": 2, "travel": [0, 0]}
            return np.zeros(2), options

        def step(self, action):
            raise TimeoutError("simulator stalled")

        def close(self):
            pass

    result = evaluate_episode(FailingEnv(), lambda obs, frame: np.zeros(2), ScenarioSeeds(10, 11), "zero", None)
    assert result.status == "technical_error"
    assert "TimeoutError" in result.error


def test_episode_rejects_changed_reset_seeds(tmp_path: Path):
    class WrongSeedEnv:
        config = type("Config", (), {"steps": 2})()
        episode_output_dir = tmp_path

        def reset(self, *, options):
            self.last_frame = {"travel": [0, 0]}
            return np.zeros(2), {"layout_seed": options["layout_seed"] + 1, "radio_seed": options["radio_seed"]}

        def close(self):
            pass

    result = evaluate_episode(WrongSeedEnv(), lambda obs, frame: np.zeros(2), ScenarioSeeds(10, 11), "zero", None)
    assert result.status == "technical_error"
    assert result.layout_seed == 11
    assert result.requested_layout_seed == 10


def test_final_updated_weights_can_win_validation(tmp_path: Path, monkeypatch):
    import scripts.train.train_recovery_ppo as training

    class FakeModel:
        seed = 1
        value = 1.0

        def predict(self, observation, deterministic):
            return np.zeros(2), None

        def save(self, path):
            Path(str(path) + ".zip").write_text(str(self.value))

    model = FakeModel()
    monkeypatch.setattr(training, "Ns3RecoveryEnv", lambda **kwargs: object())
    monkeypatch.setattr(
        training, "evaluate_episode", lambda *args, **kwargs: EpisodeRecord(
            "ppo", 1, 10, 11, "success", 10, 120, model.value, travel_total_m=20, final_components=1
        )
    )
    callback = training.ValidationCheckpoint(
        config=type("Config", (), {"steps": 120})(),
        launcher=["sim"], output=tmp_path, scenarios=[ScenarioSeeds(10, 11)], every_steps=64
    )
    callback.model = model
    callback.num_timesteps = 64
    callback._on_step()
    model.value = 2.0  # PPO's update after the rollout callback.
    callback._on_training_end()
    assert (tmp_path / "best_model.zip").read_text() == "2.0"
    assert [row["phase"] for row in callback.history] == ["rollout", "final_updated"]

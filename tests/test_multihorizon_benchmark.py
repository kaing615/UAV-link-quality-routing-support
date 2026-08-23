"""Public artifact checks for the multi-horizon benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from scripts.train.run_multihorizon_benchmark import (
    aggregate_metrics,
    collect_metrics,
    discover_datasets,
    select_representative_runs,
)
from scripts.train.run_temporal_benchmark import run_temporal_benchmark

from src.training.baselines.loro_baselines import load_run_rows


def test_discovery_and_collection_preserve_benchmark_coordinates(tmp_path: Path):
    data_root = tmp_path / "data"
    run_root = data_root / "qos" / "k2" / "run-a"
    (run_root / "graph_dataset").mkdir(parents=True)
    for split in ("train", "val", "test"):
        (run_root / "graph_dataset" / f"{split}.pt").touch()

    assert discover_datasets(data_root) == [("qos", 2, run_root)]

    metrics_path = tmp_path / "outputs" / "logreg" / "qos" / "k2" / "run-a" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    pd.DataFrame([{"split": "test", "macro_f1": 0.75}]).to_csv(metrics_path, index=False)

    result = collect_metrics(tmp_path / "outputs")
    row = result.iloc[0]
    assert (row["model_id"], row["target"], row["horizon"], row["run_name"]) == (
        "logreg",
        "qos",
        2,
        "run-a",
    )


def test_representative_runs_are_deterministic_and_balanced_by_controlled_scenario():
    runs = [
        "stress_sparse_rwp_s60003",
        "stress_sparse_rwp_s60001",
        "stress_fast_rwp_s60003",
        "stress_fast_rwp_s60001",
        "stress_baseline_gm_s60002",
        "stress_sparse_gm_s60002",
        "stress_fast_gm_s60002",
        "stress_baseline_rwp_s60001",
    ]

    assert select_representative_runs(runs, 2) == [
        "stress_baseline_rwp_s60001",
        "stress_baseline_gm_s60002",
        "stress_fast_rwp_s60001",
        "stress_fast_gm_s60002",
        "stress_sparse_rwp_s60001",
        "stress_sparse_gm_s60002",
    ]


def test_aggregate_metrics_keeps_run_as_the_independent_unit():
    detailed = pd.DataFrame(
        [
            {"model_id": "xgb", "target": "qos", "horizon": 1, "split": "test", "run_name": "a", "macro_f1": 0.6},
            {"model_id": "xgb", "target": "qos", "horizon": 1, "split": "test", "run_name": "b", "macro_f1": 0.8},
        ]
    )

    row = aggregate_metrics(detailed).iloc[0]
    assert row["n_runs"] == 2
    assert row["macro_f1_mean"] == 0.7


def test_aggregate_metrics_reports_run_level_bootstrap_interval():
    detailed = pd.DataFrame(
        [
            {"model_id": "xgb", "target": "qos", "horizon": 1, "split": "test", "run_name": run, "macro_f1": 0.7}
            for run in ("a", "b", "c")
        ]
    )

    row = aggregate_metrics(detailed).iloc[0]
    assert row["macro_f1_ci95_low"] == pytest.approx(0.7)
    assert row["macro_f1_ci95_high"] == pytest.approx(0.7)


def test_loro_baseline_loader_accepts_multihorizon_data_root(tmp_path: Path):
    run_root = tmp_path / "run-a"
    (run_root / "features").mkdir(parents=True)
    (run_root / "splits").mkdir()
    row = {
        "time": 0,
        "distance": 100.0,
        "rssi": -60.0,
        "snr": 30.0,
        "delay": 2.0,
        "packet_loss": 0.0,
        "relative_speed": 1.0,
        "throughput": 10.0,
        "label": 1,
    }
    pd.DataFrame([row]).to_csv(run_root / "features" / "edges_labeled.csv", index=False)
    pd.DataFrame([{"time": 0, "split": "train"}]).to_csv(run_root / "splits" / "time_splits.csv", index=False)

    result = load_run_rows(tmp_path, "run-a", ["train"])
    assert len(result) == 1
    assert result.iloc[0]["label"] == 1


def test_temporal_benchmark_dispatches_selected_runs_and_resumes_complete_metrics(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data"
    run_root = data_root / "survival" / "k3" / "stress_fast_rwp_s60001"
    (run_root / "graph_dataset").mkdir(parents=True)
    for split in ("train", "val", "test"):
        (run_root / "graph_dataset" / f"{split}.pt").touch()
    output_root = tmp_path / "outputs"
    stale_metrics = output_root / "edge-sage-gru" / "survival" / "k3" / "unselected" / "metrics.csv"
    stale_metrics.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"model_id": "edge-sage-gru", "split": "test", "macro_f1": 0.1, "pr_auc": 0.1}]
    ).to_csv(stale_metrics, index=False)
    commands = []

    def fake_run(command):
        commands.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"model_id": "edge-sage-gru", "split": "val", "macro_f1": 0.7, "pr_auc": 0.8},
                {"model_id": "edge-sage-gru", "split": "test", "macro_f1": 0.8, "pr_auc": 0.9},
            ]
        ).to_csv(output_dir / "metrics.csv", index=False)
        (output_dir / "best_model.pt").touch()
        (output_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "model_id": "edge-sage-gru",
                    "run_name": command[command.index("--run-name") + 1],
                    "window_size": int(command[command.index("--window-size") + 1]),
                }
            ),
            encoding="utf-8",
        )
        for split in ("val", "test"):
            pd.DataFrame(
                [{"time": 1, "src": 1, "dst": 2, "y_true": 1, "y_pred": 1, "pred_score": 0.9}]
            ).to_csv(output_dir / f"{split}_predictions.csv", index=False)

    monkeypatch.setattr("scripts.train.run_temporal_benchmark._run", fake_run)
    summary = run_temporal_benchmark(data_root, output_root, tmp_path / "summary.csv")

    assert len(commands) == 1
    assert commands[0][1:3] == ["-m", "src.training.gnn.train_temporal_gnn"]
    assert commands[0][commands[0].index("--epochs") + 1] == "100"
    assert commands[0][commands[0].index("--patience") + 1] == "10"
    assert set(summary["split"]) == {"val", "test"}
    assert set(summary["run_name"]) == {run_root.name}

    monkeypatch.setattr(
        "scripts.train.run_temporal_benchmark._run",
        lambda command: pytest.fail(f"complete metrics should resume, got {command}"),
    )
    run_temporal_benchmark(data_root, output_root, tmp_path / "resumed.csv")

    output_dir = output_root / "edge-sage-gru" / "survival" / "k3" / run_root.name
    (output_dir / "test_predictions.csv").unlink()
    monkeypatch.setattr("scripts.train.run_temporal_benchmark._run", fake_run)
    run_temporal_benchmark(data_root, output_root, tmp_path / "incomplete.csv")
    assert len(commands) == 2

    marker = output_dir / "temporal_benchmark_config.json"
    marker.write_text(
        json.dumps({"model_id": "edge-sage-gru", "window_size": 9, "epochs": 100, "patience": 10}),
        encoding="utf-8",
    )
    run_temporal_benchmark(data_root, output_root, tmp_path / "stale.csv")
    assert len(commands) == 3

    (output_dir / "metadata.json").write_text(
        json.dumps({"model_id": "edge-sage-gru", "run_name": "foreign", "window_size": 4}),
        encoding="utf-8",
    )
    run_temporal_benchmark(data_root, output_root, tmp_path / "foreign.csv")
    assert len(commands) == 4

    metrics = pd.read_csv(output_dir / "metrics.csv")
    pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True).to_csv(output_dir / "metrics.csv", index=False)
    run_temporal_benchmark(data_root, output_root, tmp_path / "duplicate.csv")
    assert len(commands) == 5

    malformed_cases = [
        ("blank-macro", "", 0.9),
        ("nan-pr", 0.8, float("nan")),
        ("inf-macro", float("inf"), 0.9),
    ]
    for suffix, macro_f1, pr_auc in malformed_cases:
        pd.DataFrame(
            [
                {"model_id": "edge-sage-gru", "split": "val", "macro_f1": 0.7, "pr_auc": 0.8},
                {"model_id": "edge-sage-gru", "split": "test", "macro_f1": macro_f1, "pr_auc": pr_auc},
            ]
        ).to_csv(output_dir / "metrics.csv", index=False)
        run_temporal_benchmark(data_root, output_root, tmp_path / f"{suffix}.csv")

    assert len(commands) == 8

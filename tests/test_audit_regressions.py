"""Regression checks for the October research audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from scripts.analysis import benchmark_inference_resources as inference
from scripts.train import run_stage6_benchmark as stage6
from scripts.train.run_multihorizon_benchmark import MODELS, aggregate_metrics

from src.evaluation.paired_statistics import paired_comparisons
from src.routing.replay_eval import evaluate_run
from src.serving.app import app
from src.utils.experiment import (
    cache_matches,
    fingerprint,
    save_manifest,
    validate_ns3_binary,
    validate_test_predictions,
)


def test_controlled_statistics_average_scenarios_before_resampling():
    runs = ["stress_baseline_rwp_s60001", "stress_dense_rwp_s60001", "stress_baseline_gm_s60002"]
    detailed = pd.DataFrame(
        [
            {"model_id": "xgb", "target": "qos", "horizon": 1, "split": "test", "run_name": run, "macro_f1": value}
            for run, value in zip(runs, [0.0, 0.0, 1.0], strict=True)
        ]
    )
    row = aggregate_metrics(detailed).iloc[0]
    assert row["n_runs"] == 3 and row["n_independent_runs"] == 2
    assert row["macro_f1_mean"] == 0.5
    pairs = pd.DataFrame(
        [
            {"run_name": run, "strategy": strategy, "score": value}
            for run in runs
            for strategy, value in [("hop", 0.0), ("xgb", 1.0)]
        ]
    )
    effect = paired_comparisons(pairs, [("xgb", "hop")], {"score": True}).iloc[0]
    assert effect["n_pairs"] == 3 and effect["n_independent_pairs"] == 2
    assert effect["p_raw"] == 0.5


def test_aggregation_rejects_unequal_model_sample_counts():
    rows = [
        {
            "model_id": model,
            "target": "qos",
            "horizon": 1,
            "split": "test",
            "run_name": "run-a",
            "n_samples": count,
            "macro_f1": 0.7,
        }
        for model, count in [("xgb", 100), ("edge-sage", 99)]
    ]
    with pytest.raises(ValueError, match="different sample counts"):
        aggregate_metrics(pd.DataFrame(rows))
    assert "persistence" in MODELS


def test_manifest_requires_matching_inputs_config_and_complete_outputs(tmp_path):
    source, output, manifest_path = [tmp_path / name for name in ("source", "output", "experiment.json")]
    source.write_text("first")
    output.write_text("result")
    expected = fingerprint({"horizon": 1}, [source])
    save_manifest(manifest_path, expected)
    assert cache_matches(manifest_path, expected, [output])
    assert not cache_matches(manifest_path, fingerprint({"horizon": 5}, [source]), [output])
    source.write_text("second")
    assert not cache_matches(manifest_path, fingerprint({"horizon": 1}, [source]), [output])
    output.write_text("")
    assert not cache_matches(manifest_path, expected, [output])


def test_stage6_holds_out_seed_and_reports_only_invocation(tmp_path, monkeypatch):
    data, output, reports = [tmp_path / name for name in ("data", "output", "reports")]
    runs = [f"stress_{scenario}_rwp_s{seed}" for seed in (60001, 60003) for scenario in ("baseline", "dense")]
    for run in runs:
        root = data / "qos" / "k1" / run
        for part in ("graph_dataset", "features", "splits"):
            (root / part).mkdir(parents=True)
        for split in ("train", "val", "test"):
            (root / "graph_dataset" / f"{split}.pt").touch()
        pd.DataFrame([{"time": 0, "src": 0, "dst": 1, "label": 1}]).to_csv(
            root / "features" / "edges_labeled.csv", index=False
        )
        pd.DataFrame([{"time": 0, "split": "test"}]).to_csv(root / "splits" / "time_splits.csv", index=False)
    stale = output / "loro" / "xgb" / "survival" / "k5" / "old" / "metrics.csv"
    stale.parent.mkdir(parents=True)
    pd.DataFrame([{"split": "test", "macro_f1": 0.9}]).to_csv(stale, index=False)
    commands = []

    def fake_run(command):
        commands.append(command)
        tested = command[command.index("--test-runs") + 1]
        trained = command[command.index("--train-runs") + 1].split(",")
        assert tested.endswith("s60001")
        assert all(run.endswith("s60003") for run in trained)
        destination = Path(command[command.index("--output-dir") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"run_name": tested, "split": "test", "n_samples": 1, "macro_f1": 0.5}]).to_csv(
            destination / "metrics.csv", index=False
        )
        pd.DataFrame([{"time": 0, "src": 0, "dst": 1, "label": 1, "pred_score": 0.9}]).to_csv(
            destination / f"test_predictions_{tested}.csv", index=False
        )

    monkeypatch.setattr(stage6, "_run", fake_run)
    kwargs = dict(
        data_root=data,
        raw_root=tmp_path / "raw",
        output_root=output,
        reports_root=reports,
        protocols={"loro"},
        models=["logreg"],
        targets={"qos"},
        horizons={1},
        force=False,
        gnn_epochs=2,
        gnn_patience=1,
        benchmark_summary=tmp_path / "unused.csv",
        loro_runs_per_scenario=1,
        ablation_runs_per_scenario=None,
    )
    stage6.run_stage6(**kwargs)
    stage6.run_stage6(**kwargs)
    assert len(commands) == 2
    result = pd.read_csv(reports / "loro_summary.csv")
    assert set(result["model_id"]) == {"logreg"} and set(result["target"]) == {"qos"}
    (data / "qos" / "k1" / runs[-1] / "features" / "new.csv").write_text("changed")
    stage6.run_stage6(**kwargs)
    assert len(commands) == 4


def test_inference_cache_invalidates_measurement_and_artifact(tmp_path, monkeypatch):
    args = argparse.Namespace(
        runs=["run-a"],
        models=["logreg"],
        target="qos",
        horizon=1,
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
        report_dir=tmp_path / "reports",
        force=False,
        warmups=1,
        repeats=2,
        threads=1,
    )
    model = args.output_root / "logreg" / "qos" / "k1" / "run-a" / "model.pkl"
    test_csv = args.data_root / "qos" / "k1" / "run-a" / "baseline_standardized" / "test_scaled.csv"
    for path in (model, test_csv):
        path.parent.mkdir(parents=True)
        path.write_text("first")
    calls = []
    collected = []

    def fake_worker(command, **kwargs):
        calls.append(command)
        result = {
            "model_id": "logreg",
            "run_name": "run-a",
            "target": args.target,
            "horizon": args.horizon,
            "threads": args.threads,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(result))

    monkeypatch.setattr(inference.subprocess, "run", fake_worker)
    monkeypatch.setattr(inference, "_write_outputs", lambda frame, _: collected.append(frame))
    inference.run_benchmark(args)
    inference.run_benchmark(args)
    assert len(calls) == 1
    args.threads = 4
    inference.run_benchmark(args)
    assert len(calls) == 2 and collected[-1].iloc[0]["threads"] == 4
    model.write_text("second")
    inference.run_benchmark(args)
    assert len(calls) == 3
    args.target, args.horizon = "survival", 5
    for path in (
        args.output_root / "logreg" / "survival" / "k5" / "run-a" / "model.pkl",
        args.data_root / "survival" / "k5" / "run-a" / "baseline_standardized" / "test_scaled.csv",
    ):
        path.parent.mkdir(parents=True)
        path.write_text("new-coordinate")
    inference.run_benchmark(args)
    assert len(calls) == 4 and collected[-1].iloc[0]["horizon"] == 5


def test_replay_rejects_censored_horizon(tmp_path):
    raw = tmp_path / "raw" / "run-a"
    raw.mkdir(parents=True)
    pd.DataFrame(
        [{"time": t, "src": 0, "dst": 1, "connected": 1, "snr": 20, "packet_loss": 0, "delay": 1} for t in (0, 1)]
    ).to_csv(raw / "edges.csv", index=False)
    split = tmp_path / "splits.csv"
    pd.DataFrame({"time": [0], "split": ["test"]}).to_csv(split, index=False)
    with pytest.raises(ValueError, match="No complete test windows"):
        evaluate_run("run-a", None, horizon=5, split_csv=split, raw_root=tmp_path / "raw")


def test_http_rejects_overflow_and_invalid_physical_features():
    payload = {
        "nodes": [
            {"node_id": i, "x": 0, "y": 0, "z": 10, "vx": 0, "vy": 0, "vz": 0, "degree": 1, "load": 0} for i in (0, 1)
        ],
        "edges": [
            {
                "src": 0,
                "dst": 1,
                "distance": 10,
                "rssi": -60,
                "snr": 30,
                "delay": 1,
                "packet_loss": 0.1,
                "relative_speed": 1,
                "throughput": 10,
            }
        ],
    }
    client = TestClient(app)
    for field, invalid in [
        ("distance", 1e100),
        ("distance", -1),
        ("delay", -1),
        ("packet_loss", 1.1),
        ("relative_speed", -1),
        ("throughput", -1),
    ]:
        changed = json.loads(json.dumps(payload))
        changed["edges"][0][field] = invalid
        assert client.post("/predict", json=changed).status_code == 422
    payload["nodes"][0]["vx"] = 1e100
    assert client.post("/predict", json=payload).status_code == 422


def test_prediction_cohort_checks_identities_and_refreshes_gnn_scores(tmp_path):
    root = tmp_path / "run-a"
    for part in ("features", "splits", "graph_dataset"):
        (root / part).mkdir(parents=True)
    pd.DataFrame([{"time": 0, "src": 0, "dst": 1, "label": 1}]).to_csv(
        root / "features" / "edges_labeled.csv", index=False
    )
    pd.DataFrame([{"time": 0, "split": "test"}]).to_csv(root / "splits" / "time_splits.csv", index=False)
    torch.save(
        [
            {
                "time": 0,
                "node_ids": [0, 1],
                "edge_label_index": torch.tensor([[0], [1]]),
                "edge_label": torch.tensor([1.0]),
            }
        ],
        root / "graph_dataset" / "test.pt",
    )
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame([{"y_true": 1.0, "pred_score": 0.9}]).to_csv(predictions, index=False)
    validate_test_predictions(root, predictions)
    pd.DataFrame([{"y_true": 1.0, "pred_score": 0.8}]).to_csv(predictions, index=False)
    validate_test_predictions(root, predictions)
    assert pd.read_csv(tmp_path / "predictions_identified.csv").iloc[0]["pred_score"] == 0.8
    pd.DataFrame([{"time": 0, "src": 0, "dst": 2, "label": 1}]).to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="cohort differs"):
        validate_test_predictions(root, predictions)


def test_routing_runner_filters_cohort_and_invalidates_changed_predictions(tmp_path, monkeypatch):
    from src.routing import multihorizon_eval as routing

    data, models, raw, output, reports = [tmp_path / name for name in ("data", "models", "raw", "output", "reports")]
    run = "run-a"
    run_root = data / "qos" / "k1" / run
    (run_root / "graph_dataset").mkdir(parents=True)
    (run_root / "splits").mkdir()
    (run_root / "splits" / "time_splits.csv").write_text("test")
    (raw / run).mkdir(parents=True)
    (raw / run / "edges.csv").write_text("raw")
    prediction = models / "logreg" / "qos" / "k1" / run / "test_predictions.csv"
    prediction.parent.mkdir(parents=True)
    prediction.write_text("first")
    stale = output / "survival" / "k5" / "one-minus" / "old" / "summary.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("unrelated historical output")
    calls = []

    def fake_replay(run_name, _, **kwargs):
        calls.append(run_name)
        destination = kwargs["output_dir"]
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "run_name": run_name,
                    "strategy": strategy,
                    "n_sessions": 1,
                    "target": "qos",
                    "prediction_horizon": 1,
                    "weight_mode": "neglog",
                    **dict.fromkeys(routing.ROUTING_METRICS, 1.0),
                }
                for strategy in ("hop", "delay", "logreg")
            ]
        ).to_csv(destination / "summary.csv", index=False)
        (destination / "details.csv").write_text("complete")

    monkeypatch.setattr(routing, "evaluate_run", fake_replay)
    kwargs = dict(
        data_root=data,
        model_root=models,
        raw_root=raw,
        routing_root=output,
        reports_root=reports,
        models=["logreg"],
        targets={"qos"},
        horizons={1},
        cost_modes=["neglog"],
    )
    detailed, _, _ = routing.run_multihorizon_routing(**kwargs)
    routing.run_multihorizon_routing(**kwargs)
    assert len(calls) == 1
    assert len(detailed) == 3 and set(detailed["run_name"]) == {run}
    prediction.write_text("changed")
    routing.run_multihorizon_routing(**kwargs)
    assert len(calls) == 2


def test_serving_loads_promoted_architectures_from_metadata(tmp_path, monkeypatch):
    from src.serving import app as serving

    monkeypatch.delenv("MODEL_ID", raising=False)
    for name in ("_model", "_model_id"):
        monkeypatch.setattr(serving, name, None)
    monkeypatch.setattr(serving, "_threshold", 0.5)
    for model_id in ("gat", "graphsage", "edge-sage-noedge", "edge-sage-decoder-only"):
        directory = tmp_path / model_id
        directory.mkdir()
        base_id = "edge-sage" if model_id.startswith("edge-sage-") else model_id
        flags = (
            {"use_message_edge_features": False, "use_decoder_edge_features": model_id.endswith("decoder-only")}
            if base_id == "edge-sage"
            else {}
        )
        model = serving._MODELS[base_id](node_in_channels=8, edge_in_channels=7, hidden_channels=32, **flags)
        torch.save(model.state_dict(), directory / "best_model.pt")
        (directory / "metadata.json").write_text(json.dumps({"model_id": model_id, "hidden_channels": 32, **flags}))
        monkeypatch.setenv("MODEL_DIR", str(directory))
        serving.load_model()
        assert serving._model_id == model_id and serving._model is not None


def test_ns3_binary_must_match_recorded_source_and_executable(tmp_path):
    source = tmp_path / "simulation" / "ns3"
    binary = source / "build" / "uav-olsr-dataset"
    binary.parent.mkdir(parents=True)
    inputs = {
        "uav-olsr-dataset.cc": source / "uav-olsr-dataset.cc",
        "CMakeLists.txt": source / "CMakeLists.txt",
        "build/uav-olsr-dataset": binary,
    }
    for path in inputs.values():
        path.write_text("first")
    hashes = fingerprint({}, list(inputs.values()))["files"]
    binary.with_name("source.sha256").write_text(
        "".join(f"{hashes[str(path.resolve())]}  {name}\n" for name, path in inputs.items())
    )
    validate_ns3_binary(binary, source)
    (source / "uav-olsr-dataset.cc").write_text("second")
    with pytest.raises(ValueError, match="binary/source mismatch"):
        validate_ns3_binary(binary, source)

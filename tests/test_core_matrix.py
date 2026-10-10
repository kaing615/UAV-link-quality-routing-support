"""Core matrix coverage, held-seed isolation and metric definitions."""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scripts.analysis.analyze_core_matrix import (
    aggregate_runs,
    event_metrics,
    manipulation_report,
    prediction_metrics,
    routing_metrics,
)
from scripts.dataset.run_core_matrix import build_core_jobs, held_seed_groups, ns3_command

from src.utils.experiment import validate_test_predictions


def _write_held_seed_run(root):
    for part in ("features", "splits", "graph_dataset"):
        (root / part).mkdir(parents=True, exist_ok=True)
    times = {"train": 0, "val": 2, "test": 4, "purged": 1}
    labels = pd.DataFrame(
        [
            {
                "time": time,
                "src": 10,
                "dst": 20,
                "label": time % 2,
                "distance": 100.0,
                "rssi": -60.0,
                "snr": 30.0,
                "delay": 2.0,
                "packet_loss": 0.0,
                "relative_speed": 1.0,
                "throughput": 10.0,
            }
            for time in sorted(times.values())
        ]
    )
    labels.to_csv(root / "features/edges_labeled.csv", index=False)
    pd.DataFrame([{"time": time, "split": split} for split, time in times.items()]).to_csv(
        root / "splits/time_splits.csv", index=False
    )
    for split, time in times.items():
        torch.save(
            [
                {
                    "time": time,
                    "node_ids": [10, 20],
                    "edge_label_index": torch.tensor([[0], [1]]),
                    "edge_label": torch.tensor([time % 2]),
                }
            ],
            root / "graph_dataset" / f"{split}.pt",
        )
    return labels, times


def test_loro_baseline_predictions_keep_full_run_edge_identity(tmp_path):
    from src.training.baselines.common import evaluate_split
    from src.training.baselines.loro_baselines import load_run_rows
    from src.training.baselines.persistence_baseline import PersistenceModel

    root = tmp_path / "run"
    labels, times = _write_held_seed_run(root)
    rows = load_run_rows(tmp_path, "run", list(times))
    _, predictions = evaluate_split(PersistenceModel("qos", 20, 0.1, 5), "persistence", "Persistence", rows, "test")
    assert predictions[["time", "src", "dst", "label"]].equals(labels[["time", "src", "dst", "label"]])
    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)
    assert validate_test_predictions(root, path, test_splits=tuple(times)) == path


def test_full_run_gnn_identity_uses_all_splits_in_loader_order(tmp_path):
    root = tmp_path / "run"
    _, times = _write_held_seed_run(root)
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"y_true": [time % 2 for time in times.values()], "y_pred": [0, 0, 0, 1], "pred_score": [0.1, 0.2, 0.3, 0.9]}
    ).to_csv(path, index=False)
    identified = validate_test_predictions(root, path, test_splits=tuple(times))
    result = pd.read_csv(identified)
    assert result["time"].tolist() == list(times.values())
    assert result["pred_score"].tolist() == [0.1, 0.2, 0.3, 0.9]
    # Row counts alone cannot detect a changed graph/CSV order.
    bad = pd.read_csv(path)
    bad.loc[0, "y_true"] = 1
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="edge order"):
        validate_test_predictions(root, path, test_splits=tuple(times))


def test_core_report_handles_baseline_and_unidentified_gnn_exports(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.analysis import analyze_core_matrix as analysis

    job = SimpleNamespace(run_name="stress_test_gm_s70001", seed=70001, family="density", condition="control")
    raw, data, outputs, reports = [tmp_path / name for name in ("raw", "data", "outputs", "reports")]
    directory = raw / job.run_name
    directory.mkdir(parents=True)
    reports.mkdir()
    (directory / "scenario.json").write_text(
        json.dumps({"num_uavs": 2, "source_id": 10, "dest_id": 20, "scenario_mode": "standard"})
    )
    pd.DataFrame([{"time": time, "src": 10, "dst": 20, "connected": 1} for time in range(10)]).to_csv(
        directory / "edges.csv", index=False
    )
    monkeypatch.setattr(
        analysis,
        "manipulation_report",
        lambda *args: pd.DataFrame(
            [{"run_name": job.run_name, "seed": job.seed, "family": job.family, "condition": job.condition}]
        ),
    )
    monkeypatch.setattr(
        analysis,
        "load_raw_edges",
        lambda *args: {time: {(10, 20): {"delay": 2, "packet_loss": 0, "valid": True}} for time in range(10)},
    )
    for target in ("qos", "survival"):
        for horizon in (1, 2, 3, 5):
            labels, times = _write_held_seed_run(data / target / f"k{horizon}" / job.run_name)
            for model in ("logreg", "edge-sage"):
                fold = outputs / model / target / f"k{horizon}" / f"seed{job.seed}"
                fold.mkdir(parents=True)
                (fold / "metadata.json").write_text(json.dumps({"train_runs": ["stress_train_gm_s79999"]}))
                if model == "logreg":
                    predictions = labels.assign(pred_label=labels["label"], pred_score=labels["label"] * 0.8 + 0.1)
                else:
                    truth = [time % 2 for time in times.values()]
                    predictions = pd.DataFrame(
                        {"y_true": truth, "y_pred": truth, "pred_score": [value * 0.8 + 0.1 for value in truth]}
                    )
                predictions.to_csv(fold / f"test_predictions_{job.run_name}.csv", index=False)
    # One seed intentionally falls short of the final coverage gate, after scoring every export.
    with pytest.raises(ValueError, match="Incomplete experimental seed coverage"):
        analysis.generate_report([job], raw, data, outputs, reports, ["logreg", "edge-sage"])
    result = pd.read_csv(reports / "metrics_runs.csv")
    assert len(result) == 16
    assert result["n_samples"].eq(4).all()
    assert result["macro_f1"].eq(1).all()


@pytest.mark.parametrize(
    "case, expected_pass", [("no_delivery", True), ("missing_trace", False), ("all_missing", False)]
)
def test_load_checks_allow_no_delivery_but_require_queue_evidence(tmp_path, case, expected_pass):
    jobs = [job for job in build_core_jobs() if job.level == "f1_cbr_p20"]
    raw = tmp_path / "raw"
    for index, job in enumerate(jobs):
        directory = raw / job.run_name
        directory.mkdir(parents=True)
        metadata = {
            "core_matrix": asdict(job),
            "num_uavs": 2,
            "time_steps": 1,
            "comm_range": job.comm_range,
            "gauss_markov_alpha": job.alpha,
            "rwp_speed_range": [job.speed_min, job.speed_max],
            "z_limit": [job.z_min, job.z_max],
        }
        (directory / "scenario.json").write_text(json.dumps(metadata), encoding="utf-8")
        pd.DataFrame({"time": [0], "x": [0], "y": [0], "z": [100], "speed": [8]}).to_csv(
            directory / "nodes.csv", index=False
        )
        pd.DataFrame({"time": [0], "src": [0], "dst": [1], "connected": [0], "packet_loss": [1]}).to_csv(
            directory / "edges.csv", index=False
        )
        pd.DataFrame({"reachable": [0]}).to_csv(directory / "traffic_log.csv", index=False)
        samples = 1 if index < 9 and case != "all_missing" else 0
        received = int(samples > 0 or case == "missing_trace")
        pd.DataFrame(
            [
                {
                    "flow_index": 0,
                    "observed_offered_rate_kbps": 1200,
                    "tx_packets": 10,
                    "rx_packets": received,
                    "lost_packets": 10 - received,
                    "throughput_mbps": 0,
                    "jain_fairness": 0,
                    "queue_samples": samples,
                    "queue_delay_ms": 1 if samples else np.nan,
                    "traffic_pattern": "cbr",
                }
            ]
        ).to_csv(directory / "background_flow_metrics.csv", index=False)
    frame = manipulation_report(jobs, raw, tmp_path / "report", require_complete=False)
    checks = pd.read_csv(tmp_path / "report/manipulation_checks.csv")
    assert bool(checks["pass"].iloc[0]) == expected_pass
    if case == "no_delivery":
        assert frame["scenario_check"].all()
        assert np.isnan(frame["queue_delay_ms"].iloc[-1])
        assert frame["network_loss"].iloc[-1] == 1
    assert checks["queue_observed_runs"].iloc[0] == (0 if case == "all_missing" else 9)


def test_core_matrix_covers_every_level_with_ten_paired_seeds():
    jobs = build_core_jobs()
    counts = Counter(job.condition for job in jobs)
    assert len(jobs) == 210 and len(counts) == 21 and set(counts.values()) == {10}
    assert len({job.run_name for job in jobs}) == 210
    seed_sets = [{job.seed for job in jobs if job.condition == condition} for condition in counts]
    assert all(seeds == seed_sets[0] for seeds in seed_sets)
    assert {(job.speed_min, job.speed_max) for job in jobs if job.factor == "speed"} == {(2, 5), (6, 10), (12, 20)}
    assert {job.alpha for job in jobs if job.factor == "alpha"} == {0.50, 0.85, 0.95}
    assert {job.mobility for job in jobs if job.factor == "alpha"} == {"gauss-markov"}
    assert {(job.z_min, job.z_max) for job in jobs if job.factor == "altitude"} == {(50, 150), (30, 300)}
    assert {job.comm_range for job in jobs if job.factor == "range"} == {150, 180, 240, 300}
    assert {job.scenario_mode for job in jobs if job.family == "partition"} == {
        "group-control",
        "partition-reconnect",
        "crossing-swarms",
    }
    load = [job for job in jobs if job.family == "load"]
    assert {job.flows for job in load} == {1, 4, 8}
    assert {job.traffic_pattern for job in load} == {"cbr", "bursty"}
    assert {job.offered_fraction for job in load} == {0.2, 0.5, 0.8}
    baseline = next(job for job in load if job.level == "f4_cbr_p20")
    for job in load:
        changing = sum(
            getattr(job, name) != getattr(baseline, name) for name in ("flows", "traffic_pattern", "offered_fraction")
        )
        assert changing <= 1
    with pytest.raises(ValueError, match="10 paired seeds"):
        build_core_jobs(9)


def test_fold_excludes_all_conditions_of_test_seed_and_cli_keeps_total_load():
    jobs = build_core_jobs()
    assert len(held_seed_groups(jobs)) == 10
    for seed, train, test in held_seed_groups(jobs):
        assert len(train) == 189 and len(test) == 21
        assert all(not name.endswith(f"_s{seed}") for name in train)
        assert all(name.endswith(f"_s{seed}") for name in test)
    for job in jobs:
        if job.family == "load":
            command = ns3_command(job, Path("binary"), Path("output"))
            rate = float(next(arg.split("=")[1] for arg in command if arg.startswith("--backgroundRateKbps=")))
            assert rate * job.flows == pytest.approx(6000 * job.offered_fraction)
    command = ns3_command(next(job for job in jobs if job.level == "partition"), Path("binary"), Path("output"))
    assert "--groupSeparationStep=40" in command and "--groupReconnectStep=80" in command


def test_event_metrics_use_observed_cut_edges_and_report_undefined_cases():
    predictions = pd.DataFrame(
        [
            {"time": 0, "src": 0, "dst": 1, "y_true": 0, "y_pred": 0, "pred_score": 0.2, "horizon": 2},
            {"time": 0, "src": 1, "dst": 2, "y_true": 1, "y_pred": 1, "pred_score": 0.9, "horizon": 2},
        ]
    )
    links = {0: {(0, 1), (1, 2)}, 1: {(0, 1), (1, 2)}, 2: {(1, 2)}}
    metadata = {
        "scenario_mode": "partition-reconnect",
        "partition_start_step": 1,
        "group_separation_step": 2,
        "partition_end_step": 3,
        "reconnect_step": 4,
        "partition_groups": {"group_a": [0], "group_b": [1, 2]},
    }
    result = event_metrics(predictions, metadata, links)
    assert result["transition_f1"] == 1 and result["bridge_recall"] == 1
    assert result["bridge_failure_recall"] == 1 and result["bridge_lead_time_steps"] == 2
    bad = predictions.copy()
    bad.loc[0, "y_pred"] = 1
    assert prediction_metrics(bad)["false_stable"] == 1
    one_class = predictions.iloc[[1]]
    assert np.isnan(prediction_metrics(one_class)["pr_auc"])
    assert not prediction_metrics(one_class)["pr_auc_defined"]


def test_routing_includes_unreachable_sessions_in_found_rate_and_lifetime():
    predictions = pd.DataFrame(
        [
            {"time": 0, "src": 0, "dst": 1, "pred_score": 0.9, "horizon": 1},
            {"time": 1, "src": 2, "dst": 3, "pred_score": 0.9, "horizon": 1},
        ]
    )
    raw = {0: {(0, 1): {"delay": 1, "packet_loss": 0}}, 1: {(2, 3): {"delay": 1, "packet_loss": 0}}, 2: {}}
    result = routing_metrics(predictions, {"source_id": 0, "dest_id": 1, "scenario_mode": "standard"}, raw)
    assert result["route_sessions"] == 2 and result["found_rate"] == 0.5
    assert result["route_lifetime"] == 0


def test_full_held_seed_cohort_includes_purged_only_when_requested(tmp_path):
    root = tmp_path / "run"
    (root / "features").mkdir(parents=True)
    (root / "splits").mkdir()
    labels = pd.DataFrame({"time": [0, 1], "src": [0, 0], "dst": [1, 1], "label": [0, 1]})
    labels.to_csv(root / "features/edges_labeled.csv", index=False)
    pd.DataFrame({"time": [0, 1], "split": ["purged", "test"]}).to_csv(root / "splits/time_splits.csv", index=False)
    prediction = tmp_path / "prediction.csv"
    labels.rename(columns={"label": "y_true"}).to_csv(prediction, index=False)
    validate_test_predictions(root, prediction, test_splits=("purged", "test"))
    with pytest.raises(ValueError, match="cohort differs"):
        validate_test_predictions(root, prediction)
    frame = pd.DataFrame(
        [
            {
                "family": "density",
                "condition": "range_240",
                "model_id": "logreg",
                "target": "survival",
                "horizon": 1,
                "seed": seed,
                "macro_f1": 0.7,
            }
            for seed in range(10)
        ]
    )
    row = aggregate_runs(frame).iloc[0]
    assert row["n_seeds"] == 10 and row["macro_f1_ci95_low"] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_runs(pd.concat([frame, frame.iloc[[0]]]))

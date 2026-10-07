"""Generalization benchmark runner contracts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import torch
from scripts.train.run_stage6_benchmark import (
    _mobility_groups,
    collect_protocol_metrics,
    discover_coordinates,
    with_full_ablation_reference,
)

from src.training.gnn.train_gnn_loro import load_run_split


def test_controlled_launcher_dry_run_states_extension_count_and_temporal_selection():
    """Catches a launcher that omits the accepted extension or temporal cohort."""
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            "bash",
            "-c",
            "{ printf 'set -- --dry-run\\n'; tr -d '\\r' < scripts/run_controlled_paper_pipeline.sh; } | bash -s",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "400 core + 20 extension" in completed.stdout
    assert "420 accepted runs" in completed.stdout
    assert "temporal survival coordinates k=3,5" in completed.stdout
    assert completed.stdout.index("temporal survival coordinates") < completed.stdout.index("LORO")


def test_controlled_launcher_runs_temporal_artifacts_before_stage6(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    launcher = project_root / "scripts" / "run_controlled_paper_pipeline.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(Path("scripts/run_controlled_paper_pipeline.sh").read_bytes().replace(b"\r", b""))
    launcher.chmod(0o755)

    ns3_binary = project_root / "simulation" / "ns3" / "build" / "uav-olsr-dataset"
    ns3_binary.parent.mkdir(parents=True)
    ns3_binary.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    ns3_binary.chmod(0o755)
    helper = project_root / "scripts" / "utils" / "ensure_ns3_binary.sh"
    helper.parent.mkdir()
    helper.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    for seed in range(60001, 60011):
        (project_root / f"data/multihorizon_controlled/survival/k3/stress_baseline_rwp_s{seed}").mkdir(parents=True)

    command_log = project_root / "python-commands.log"
    fake_bin = project_root / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_bytes(b'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> python-commands.log\n')
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "CLOSED_LOOP_RUNS": "1",
        "WITHIN_RUNS_PER_SCENARIO": "1",
        "LORO_FOLDS_PER_SCENARIO": "1",
        "ABLATION_RUNS_PER_SCENARIO": "1",
        "EXTENSION_RUNS": "1",
        "TEMPORAL_RUNS_PER_SCENARIO": "1",
        "TEMPORAL_EPOCHS": "1",
        "TEMPORAL_PATIENCE": "1",
    }

    completed = subprocess.run(
        ["bash", "scripts/run_controlled_paper_pipeline.sh"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    modules = [line.split()[1] for line in command_log.read_text(encoding="utf-8").splitlines()]
    ordered = [
        "scripts.analysis.summarize_extension_scenarios",
        "scripts.train.run_multihorizon_benchmark",
        "scripts.train.run_temporal_benchmark",
        "scripts.analysis.generate_temporal_artifacts",
        "scripts.analysis.analyze_multihorizon_stage6",
        "scripts.train.run_stage6_benchmark",
        "src.routing.multihorizon_eval",
        "scripts.analysis.generate_stage7_artifacts",
    ]
    assert [modules.index(module) for module in ordered] == sorted(modules.index(module) for module in ordered)


def test_discover_coordinates_groups_runs_by_target_and_horizon(tmp_path: Path):
    for run in ("run-a", "run-b"):
        graph_dir = tmp_path / "qos" / "k5" / run / "graph_dataset"
        graph_dir.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (graph_dir / f"{split}.pt").touch()

    assert discover_coordinates(tmp_path, {"qos"}, {5}) == [("qos", 5, ["run-a", "run-b"])]


def test_collect_protocol_metrics_preserves_test_run_from_file(tmp_path: Path):
    path = tmp_path / "loro" / "logreg" / "qos" / "k1" / "fold-a" / "metrics.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([{"split": "test", "run_name": "run-a", "model_id": "logreg", "macro_f1": 0.8}]).to_csv(
        path, index=False
    )

    row = collect_protocol_metrics(tmp_path / "loro", "loro").iloc[0]
    assert (row["protocol"], row["target"], row["horizon"], row["run_name"]) == (
        "loro",
        "qos",
        1,
        "run-a",
    )


def test_collect_protocol_metrics_uses_output_directory_when_metric_has_no_run_name(tmp_path: Path):
    path = tmp_path / "ablation" / "edge-sage-noedge" / "qos" / "k1" / "run-a" / "metrics.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([{"split": "test", "model_id": "edge-sage-noedge", "macro_f1": 0.4}]).to_csv(path, index=False)

    row = collect_protocol_metrics(tmp_path / "ablation", "ablation").iloc[0]
    assert row["run_name"] == "run-a"


def test_loro_gnn_loader_accepts_multihorizon_data_root(tmp_path: Path):
    graph_dir = tmp_path / "run-a" / "graph_dataset"
    graph_dir.mkdir(parents=True)
    graph = {
        "x": torch.zeros((2, 8)),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_attr": torch.zeros((2, 7)),
        "edge_label_index": torch.tensor([[0], [1]]),
        "edge_label": torch.tensor([1]),
    }
    torch.save([graph], graph_dir / "train.pt")

    result = load_run_split(tmp_path, "run-a", "train")
    assert len(result) == 1
    assert result[0].labeled_edge_attr.shape == (1, 7)


def test_ablation_report_includes_existing_full_edge_sage_reference():
    ablation = pd.DataFrame(
        [{"model_id": "edge-sage-noedge", "target": "qos", "horizon": 1, "split": "test", "run_name": "r1"}]
    )
    benchmark = pd.DataFrame(
        [
            {"model_id": "edge-sage", "target": "qos", "horizon": 1, "split": "test", "run_name": "r1"},
            {"model_id": "edge-sage", "target": "qos", "horizon": 1, "split": "test", "run_name": "r2"},
            {"model_id": "logreg", "target": "qos", "horizon": 1, "split": "test", "run_name": "r1"},
        ]
    )

    result = with_full_ablation_reference(ablation, benchmark)
    assert set(result["model_id"]) == {"edge-sage", "edge-sage-noedge"}
    assert set(result["run_name"]) == {"r1"}


def test_cross_mobility_excludes_partition_and_load_run_identities(tmp_path: Path):
    mobility_by_run = {
        "stress_baseline_rwp_s60001": "random-waypoint",
        "stress_fast_gm_s60002": "gauss-markov",
        "stress_sparse_rwp_s60001": "random-waypoint",
        "stress_dense_gm_s60002": "gauss-markov",
        "stress_partition_ctrl_s60001": "event-controlled-waypoint",
        "stress_load_rwp_s60001": "random-waypoint",
    }
    for run_name, mobility in mobility_by_run.items():
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        (run_dir / "scenario.json").write_text('{"mobility_model": "' + mobility + '"}', encoding="utf-8")

    assert _mobility_groups(tmp_path, list(mobility_by_run)) == {
        "gauss-markov": ["stress_fast_gm_s60002", "stress_dense_gm_s60002"],
        "random-waypoint": ["stress_baseline_rwp_s60001", "stress_sparse_rwp_s60001"],
    }

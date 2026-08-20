from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression


def test_closed_loop_generator_writes_figures_and_tables(tmp_path: Path) -> None:
    from scripts.analysis.generate_closed_loop_artifacts import generate_artifacts

    summary = pd.DataFrame(
        [
            {
                "strategy": "olsr",
                "n_runs": 10,
                "pdr_mean": 0.30,
                "pdr_ci95_low": 0.20,
                "pdr_ci95_high": 0.40,
                "mean_delay_ms_mean": 120.0,
                "mean_delay_ms_ci95_low": 100.0,
                "mean_delay_ms_ci95_high": 140.0,
                "throughput_mbps_mean": 0.08,
                "throughput_mbps_ci95_low": 0.05,
                "throughput_mbps_ci95_high": 0.11,
                "route_changes_mean": 2.0,
                "route_changes_ci95_low": 1.0,
                "route_changes_ci95_high": 3.0,
            },
            {
                "strategy": "edge-sage",
                "n_runs": 10,
                "pdr_mean": 0.55,
                "pdr_ci95_low": 0.45,
                "pdr_ci95_high": 0.65,
                "mean_delay_ms_mean": 100.0,
                "mean_delay_ms_ci95_low": 90.0,
                "mean_delay_ms_ci95_high": 110.0,
                "throughput_mbps_mean": 0.15,
                "throughput_mbps_ci95_low": 0.12,
                "throughput_mbps_ci95_high": 0.18,
                "route_changes_mean": 4.0,
                "route_changes_ci95_low": 3.0,
                "route_changes_ci95_high": 5.0,
            },
        ]
    )
    paired = pd.DataFrame(
        [
            {
                "metric": "pdr",
                "reference_strategy": "olsr",
                "comparator_strategy": "edge-sage",
                "n_pairs": 10,
                "mean_delta": 0.25,
                "ci95_low": 0.10,
                "ci95_high": 0.40,
                "p_raw": 0.03,
                "p_holm": 0.20,
            }
        ]
    )
    summary_csv = tmp_path / "summary.csv"
    paired_csv = tmp_path / "paired.csv"
    summary.to_csv(summary_csv, index=False)
    paired.to_csv(paired_csv, index=False)

    generated = generate_artifacts(summary_csv, paired_csv, tmp_path / "artifacts")

    expected = {
        "figures/closed_loop_comparison.png",
        "figures/closed_loop_comparison.pdf",
        "tables/closed_loop_summary.csv",
        "tables/closed_loop_summary.tex",
        "tables/closed_loop_paired.csv",
        "tables/closed_loop_paired.tex",
    }
    assert expected.issubset({str(path.relative_to(tmp_path / "artifacts")).replace("\\", "/") for path in generated})
    table = pd.read_csv(tmp_path / "artifacts/tables/closed_loop_summary.csv")
    assert table.loc[table["strategy"] == "edge-sage", "pdr_mean"].item() == 0.55


def test_model_complexity_counts_trainable_values_and_serialized_size(tmp_path: Path) -> None:
    from scripts.analysis.benchmark_inference_resources import artifact_size_mb, model_complexity

    torch_model = torch.nn.Linear(3, 2)
    count, definition = model_complexity(torch_model, "edge-sage")
    assert count == 8
    assert definition == "trainable_parameters"

    logreg = LogisticRegression().fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    count, definition = model_complexity(logreg, "logreg")
    assert count == 2
    assert definition == "coefficients_plus_intercept"

    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"x" * 1_048_576)
    assert artifact_size_mb(artifact) == 1.0


def test_resource_aggregation_uses_runs_as_independent_units() -> None:
    from scripts.analysis.benchmark_inference_resources import aggregate_benchmarks

    detailed = pd.DataFrame(
        [
            {
                "model_id": "edge-sage",
                "run_name": "run-a",
                "latency_ms_per_sample_median": 0.10,
                "latency_ms_per_snapshot_median": 1.0,
                "latency_ms_per_sample_p95": 0.12,
                "peak_rss_mb": 100.0,
                "checkpoint_mb": 0.30,
                "complexity_count": 1000,
            },
            {
                "model_id": "edge-sage",
                "run_name": "run-b",
                "latency_ms_per_sample_median": 0.20,
                "latency_ms_per_snapshot_median": 2.0,
                "latency_ms_per_sample_p95": 0.24,
                "peak_rss_mb": 120.0,
                "checkpoint_mb": 0.32,
                "complexity_count": 1000,
            },
        ]
    )

    summary = aggregate_benchmarks(detailed)

    assert summary.loc[0, "n_runs"] == 2
    assert summary.loc[0, "latency_ms_per_sample_median_mean"] == 0.15
    assert summary.loc[0, "peak_rss_mb_mean"] == 110.0


def test_resource_outputs_accept_single_run_confidence_intervals(tmp_path: Path) -> None:
    from scripts.analysis.benchmark_inference_resources import _write_outputs

    detailed = pd.DataFrame(
        [
            {
                "model_id": "logreg",
                "run_name": "run-a",
                "latency_ms_per_sample_median": 0.0292701234564,
                "latency_ms_per_snapshot_median": 0.040244,
                "latency_ms_per_sample_p95": 0.001314,
                "peak_rss_mb": 424.1875,
                "checkpoint_mb": 0.00089,
                "complexity_count": 8,
            }
        ]
    )

    _write_outputs(detailed, tmp_path)

    assert (tmp_path / "figures/inference_resources.png").exists()


def test_resource_benchmark_does_not_load_gnn_runtime_for_tabular_import() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import sys; import scripts.analysis.benchmark_inference_resources; print(int('torch' in sys.modules))"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "0"

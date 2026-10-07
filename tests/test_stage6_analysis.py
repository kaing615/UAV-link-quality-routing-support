"""Run-level generalization analysis contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scripts.analysis.analyze_multihorizon_stage6 import (
    load_run_metadata,
    paired_model_comparisons,
    worst_group_metrics,
)


def test_load_run_metadata_derives_spatial_density(tmp_path: Path):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "scenario.json").write_text(
        json.dumps(
            {
                "run_name": "run-a",
                "mobility_model": "random-waypoint",
                "num_uavs": 20,
                "x_limit": [0, 1000],
                "y_limit": [0, 500],
                "comm_range": 200,
            }
        ),
        encoding="utf-8",
    )

    row = load_run_metadata(tmp_path, ["run-a"]).iloc[0]
    assert row["mobility"] == "random-waypoint"
    assert row["spatial_density_per_km2"] == 40.0


def test_load_run_metadata_keeps_controlled_sparse_and_dense_separate(tmp_path: Path):
    for run_name, comm_range in (
        ("stress_sparse_rwp_s60001", 160),
        ("stress_dense_rwp_s60001", 300),
    ):
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        (run_dir / "scenario.json").write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "mobility_model": "random-waypoint",
                    "num_uavs": 20,
                    "x_limit": [0, 800],
                    "y_limit": [0, 800],
                    "comm_range": comm_range,
                }
            ),
            encoding="utf-8",
        )

    metadata = load_run_metadata(tmp_path, ["stress_sparse_rwp_s60001", "stress_dense_rwp_s60001"])

    assert metadata.set_index("run_name")["density_group"].to_dict() == {
        "stress_sparse_rwp_s60001": "sparse",
        "stress_dense_rwp_s60001": "dense",
    }


def test_paired_model_comparisons_stays_within_target_and_horizon():
    detailed = pd.DataFrame(
        [
            {
                "model_id": model,
                "target": "qos",
                "horizon": horizon,
                "split": "test",
                "run_name": run,
                "macro_f1": value,
            }
            for horizon in (1, 5)
            for run, base in (("r1", 0.5), ("r2", 0.6))
            for model, value in (("logreg", base), ("edge-sage", base + 0.1))
        ]
    )

    result = paired_model_comparisons(
        detailed,
        comparisons=[("edge-sage", "logreg")],
        metrics={"macro_f1": True},
        n_resamples=100,
    )

    assert len(result) == 2
    assert set(result["horizon"]) == {1, 5}
    assert set(result["n_pairs"]) == {2}
    assert set(result["mean_delta"].round(8)) == {0.1}


def test_worst_group_metrics_marks_lower_mobility_group():
    detailed = pd.DataFrame(
        [
            {
                "model_id": "edge-sage",
                "target": "qos",
                "horizon": 1,
                "split": "test",
                "run_name": "rwp",
                "macro_f1": 0.8,
            },
            {
                "model_id": "edge-sage",
                "target": "qos",
                "horizon": 1,
                "split": "test",
                "run_name": "gm",
                "macro_f1": 0.6,
            },
        ]
    )
    metadata = pd.DataFrame(
        [
            {"run_name": "rwp", "mobility": "random-waypoint", "density_group": "dense"},
            {"run_name": "gm", "mobility": "gauss-markov", "density_group": "sparse"},
        ]
    )

    result = worst_group_metrics(detailed, metadata, metrics=("macro_f1",))
    mobility = result[result["group_type"] == "mobility"]
    assert mobility.loc[mobility["is_worst_group"], "group_value"].tolist() == ["gauss-markov"]

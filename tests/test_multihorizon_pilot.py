"""Tests for the isolated multi-horizon preprocessing pilot."""

from __future__ import annotations

import json

import pandas as pd
from scripts.dataset.build_multihorizon_pilot import run_pilot, select_pilot_runs


def _write_raw_run(raw_root, run_name: str, mobility: str, time_steps: int = 50) -> None:
    run_dir = raw_root / run_name
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "time": time,
                "node_id": node,
                "x": float(node),
                "y": 0.0,
                "z": 50.0,
                "vx": 1.0,
                "vy": 0.0,
                "vz": 0.0,
                "speed": 1.0,
                "degree": 1,
            }
            for time in range(time_steps)
            for node in (0, 1)
        ]
    ).to_csv(run_dir / "nodes.csv", index=False)
    pd.DataFrame(
        [
            {
                "time": time,
                "src": 0,
                "dst": 1,
                "connected": 1,
                "distance": 100.0,
                "rssi": -60.0,
                "snr": 25.0,
                "delay": 2.0,
                "packet_loss": 0.01,
                "relative_speed": 0.0,
                "throughput": 10.0,
                "p_stable": 0.9,
                "weight": 0.1,
            }
            for time in range(time_steps)
        ]
    ).to_csv(run_dir / "edges.csv", index=False)
    (run_dir / "scenario.json").write_text(
        json.dumps({"run_name": run_name, "mobility_model": mobility}), encoding="utf-8"
    )


def test_select_pilot_runs_alternates_mobility(tmp_path):
    _write_raw_run(tmp_path, "run_001_rwp", "random-waypoint")
    _write_raw_run(tmp_path, "run_002_rwp", "random-waypoint")
    _write_raw_run(tmp_path, "run_003_gm", "gauss-markov")

    assert select_pilot_runs(tmp_path, "run_*", limit=3) == [
        "run_003_gm",
        "run_001_rwp",
        "run_002_rwp",
    ]


def test_run_pilot_writes_all_targets_horizons_and_summary(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_run(raw_root, "run_001_rwp", "random-waypoint")
    summary_csv = tmp_path / "reports" / "summary.csv"

    summary = run_pilot(
        raw_root=raw_root,
        run_names=["run_001_rwp"],
        output_root=tmp_path / "multihorizon",
        summary_csv=summary_csv,
    )

    assert len(summary) == 8
    assert set(zip(summary["target"], summary["horizon"])) == {
        (target, horizon) for target in ("qos", "survival") for horizon in (1, 2, 3, 5)
    }
    assert set(summary["support_horizon"]) == {5}
    assert {
        "num_train_samples",
        "train_positive_ratio",
        "num_val_samples",
        "val_positive_ratio",
        "num_test_samples",
        "test_positive_ratio",
    }.issubset(summary.columns)
    assert summary_csv.exists()
    for row in summary.itertuples(index=False):
        assert (tmp_path / "multihorizon" / row.target / f"k{row.horizon}" / row.run_name).is_dir()

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


def test_controlled_matrix_creates_one_hundred_paired_runs_per_scenario() -> None:
    from scripts.dataset.run_controlled_scenarios import build_jobs

    jobs = build_jobs(1, 100)

    assert len(jobs) == 400
    assert Counter(job.scenario for job in jobs) == {
        "baseline": 100,
        "fast": 100,
        "sparse": 100,
        "dense": 100,
    }
    first_seed = [job for job in jobs if job.seed == 60001]
    assert len(first_seed) == 4
    assert {job.mobility for job in first_seed} == {"random-waypoint"}
    assert {(job.scenario, job.comm_range, job.speed_min, job.speed_max) for job in first_seed} == {
        ("baseline", 240, 3, 8),
        ("fast", 240, 12, 20),
        ("sparse", 160, 3, 8),
        ("dense", 300, 3, 8),
    }


def _write_synthetic_run(
    raw_root: Path,
    graph_root: Path,
    scenario: str,
    *,
    speed: float,
    degree: int,
    connected: list[int],
) -> None:
    run_name = f"stress_{scenario}_rwp_s60001"
    raw_dir = raw_root / run_name
    graph_dir = graph_root / run_name
    raw_dir.mkdir(parents=True)
    (graph_dir / "graph_dataset").mkdir(parents=True)
    (graph_dir / "features").mkdir(parents=True)
    (raw_dir / "scenario.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "seed": 60001,
                "mobility_model": "random-waypoint",
                "num_uavs": 20,
                "time_steps": 2,
                "comm_range": {"baseline": 240, "fast": 240, "sparse": 160, "dense": 300}[scenario],
                "rwp_speed_range": [12, 20] if scenario == "fast" else [3, 8],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"time": time, "node_id": node, "speed": speed, "degree": degree}
            for time in (0, 1)
            for node in (0, 1)
        ]
    ).to_csv(raw_dir / "nodes.csv", index=False)
    pd.DataFrame(
        [
            {"time": time, "src": 0, "dst": 1, "connected": value}
            for time, value in enumerate(connected)
        ]
    ).to_csv(raw_dir / "edges.csv", index=False)
    pd.DataFrame({"label": [0, 1]}).to_csv(
        graph_dir / "features" / "edges_labeled.csv", index=False
    )
    for split in ("train", "val", "test"):
        (graph_dir / "graph_dataset" / f"{split}.pt").write_bytes(b"complete")


def test_controlled_report_checks_completeness_and_manipulations(tmp_path: Path) -> None:
    from scripts.analysis.summarize_controlled_scenarios import generate_report

    raw_root = tmp_path / "raw"
    graph_root = tmp_path / "graph"
    _write_synthetic_run(raw_root, graph_root, "baseline", speed=5, degree=4, connected=[1, 1])
    _write_synthetic_run(raw_root, graph_root, "fast", speed=15, degree=4, connected=[1, 0])
    _write_synthetic_run(raw_root, graph_root, "sparse", speed=5, degree=2, connected=[0, 0])
    _write_synthetic_run(raw_root, graph_root, "dense", speed=5, degree=6, connected=[1, 1])

    inventory, summary, checks = generate_report(
        raw_root,
        graph_root,
        tmp_path / "reports",
        expected_per_scenario=1,
    )

    assert len(inventory) == 4
    assert set(summary["scenario"]) == {"baseline", "fast", "sparse", "dense"}
    assert set(checks["status"]) == {"PASS"}
    assert (tmp_path / "reports/run_inventory.csv").exists()
    assert (tmp_path / "reports/scenario_summary.csv").exists()
    assert (tmp_path / "reports/manipulation_checks.csv").exists()
    assert (tmp_path / "reports/REPORT.md").exists()

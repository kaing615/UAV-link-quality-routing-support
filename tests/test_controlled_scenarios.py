from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def test_ns3_pipeline_forwards_extension_environment_to_binary(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    script = project_root / "scripts" / "dataset" / "run_one_dataset_ns3.sh"
    binary = project_root / "simulation" / "ns3" / "build" / "uav-olsr-dataset"
    python = project_root / "simulation" / ".venv" / "bin" / "python"
    script.parent.mkdir(parents=True)
    binary.parent.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    script.write_bytes(Path("scripts/dataset/run_one_dataset_ns3.sh").read_bytes().replace(b"\r", b""))
    binary.write_bytes(b'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > captured_args.txt\n')
    python.write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
    script.chmod(0o755)
    binary.chmod(0o755)
    python.chmod(0o755)
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            "SIM_SCENARIO_MODE=partition-reconnect SIM_BACKGROUND_FLOWS=8 "
            "SIM_BACKGROUND_RATE_KBPS=600 SIM_SOURCE_ID=2 SIM_DEST_ID=15 "
            "scripts/dataset/run_one_dataset_ns3.sh test_run 7 random-waypoint",
        ],
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    arguments = (project_root / "captured_args.txt").read_text(encoding="utf-8").splitlines()
    assert "--scenarioMode=partition-reconnect" in arguments
    assert "--backgroundFlows=8" in arguments
    assert "--backgroundRateKbps=600" in arguments
    assert "--sourceId=2" in arguments
    assert "--destId=15" in arguments


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


def test_extension_jobs_add_ten_ctrl_load_pairs_by_default() -> None:
    from scripts.dataset.run_controlled_scenarios import build_extension_jobs

    jobs = build_extension_jobs()

    assert len(jobs) == 20
    assert len({job.run_name for job in jobs}) == 20
    ctrl_jobs = [job for job in jobs if job.scenario == "ctrl"]
    load_jobs = [job for job in jobs if job.scenario == "load"]
    assert len(ctrl_jobs) == 10
    assert len(load_jobs) == 10
    assert {job.run_name for job in ctrl_jobs} == {
        f"stress_ctrl_{'gm' if seed % 2 == 0 else 'rwp'}_s{seed}" for seed in range(60001, 60011)
    }
    assert {(job.scenario_mode, job.background_flows, job.background_rate_kbps) for job in ctrl_jobs} == {
        ("partition-reconnect", 0, 0.0)
    }
    assert {(job.seed, job.mobility) for job in load_jobs} == {
        (seed, "gauss-markov" if seed % 2 == 0 else "random-waypoint") for seed in range(60001, 60011)
    }
    assert {(job.scenario_mode, job.background_flows, job.background_rate_kbps) for job in load_jobs} == {
        ("high-load", 8, 600.0)
    }


def test_dockerignore_excludes_generated_ns3_build_dirs_after_ns3_reinclude() -> None:
    lines = Path(".dockerignore").read_text(encoding="utf-8").splitlines()
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "!requirements-serve.txt" in lines
    assert "!simulation/ns3/**" in lines
    assert "simulation/ns3/build/" in lines
    assert "simulation/ns3/build-current/" in lines
    assert lines.index("simulation/ns3/build/") > lines.index("!simulation/ns3/**")
    assert "COPY requirements.txt requirements-serve.txt ./" in dockerfile


def test_generated_controlled_scenario_outputs_are_git_ignored() -> None:
    completed = subprocess.run(
        [
            "git",
            "check-ignore",
            "reports/controlled_scenarios/extensions_seed60001/REPORT.md",
            "reports/controlled_scenarios/extensions_seed60001/figures/extension_manipulation_checks.pdf",
            "reports/controlled_scenarios/workers/worker_001_001.csv",
            "simulation/ns3/build-current/CMakeCache.txt",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_run_jobs_passes_extension_env_fields(monkeypatch, tmp_path: Path) -> None:
    from scripts.dataset import run_controlled_scenarios as module

    seen: dict[str, Any] = {}
    complete_checks = iter([False, True])

    def fake_complete(*_args: Any, **_kwargs: Any) -> bool:
        return next(complete_checks)

    def fake_run(command: list[str], *, check: bool, cwd: Path, env: dict[str, str]) -> None:
        seen["command"] = command
        seen["check"] = check
        seen["cwd"] = cwd
        seen["env"] = env

    monkeypatch.setattr(module, "_complete", fake_complete)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    job = module.ScenarioJob(
        scenario="load",
        index=1,
        seed=60001,
        mobility="random-waypoint",
        mobility_tag="rwp",
        comm_range=240,
        speed_min=3,
        speed_max=8,
        run_name="stress_load_rwp_s60001",
        scenario_mode="load",
        background_flows=3,
        background_rate_kbps=512.0,
        source_id=2,
        dest_id=5,
    )

    result = module.run_jobs(
        [job],
        raw_root=tmp_path / "raw",
        graph_root=tmp_path / "graph",
        manifest_path=tmp_path / "manifest.csv",
    )

    assert result.loc[0, "status"] == "COMPLETED"
    assert seen["command"] == ["bash", str(module.PIPELINE_SCRIPT), job.run_name, str(job.seed), job.mobility]
    assert seen["env"]["SIM_SCENARIO_MODE"] == "load"
    assert seen["env"]["SIM_BACKGROUND_FLOWS"] == "3"
    assert seen["env"]["SIM_BACKGROUND_RATE_KBPS"] == "512.0"
    assert seen["env"]["SIM_SOURCE_ID"] == "2"
    assert seen["env"]["SIM_DEST_ID"] == "5"


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
        [{"time": time, "node_id": node, "speed": speed, "degree": degree} for time in (0, 1) for node in (0, 1)]
    ).to_csv(raw_dir / "nodes.csv", index=False)
    pd.DataFrame(
        [{"time": time, "src": 0, "dst": 1, "connected": value} for time, value in enumerate(connected)]
    ).to_csv(raw_dir / "edges.csv", index=False)
    pd.DataFrame({"label": [0, 1]}).to_csv(graph_dir / "features" / "edges_labeled.csv", index=False)
    for split in ("train", "val", "test"):
        (graph_dir / "graph_dataset" / f"{split}.pt").write_bytes(b"complete")


def _write_extension_run(
    raw_root: Path,
    run_name: str,
    *,
    scenario_mode: str,
    seed: int,
    mobility_model: str,
    edges: list[dict[str, object]],
    traffic: list[dict[str, object]],
    partition_start_step: int | None = None,
    partition_end_step: int | None = None,
    reconnect_step: int | None = None,
    partition_groups: dict[str, list[int]] | None = None,
    background_flows: int = 0,
    background_rate_kbps: float = 0.0,
    background_rows: list[dict[str, object]] | None = None,
) -> None:
    run_dir = raw_root / run_name
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"time": time, "node_id": node_id, "speed": 5.0, "degree": 2}
            for time in range(12)
            for node_id in range(4)
        ]
    ).to_csv(run_dir / "nodes.csv", index=False)
    pd.DataFrame(edges).to_csv(run_dir / "edges.csv", index=False)
    pd.DataFrame(traffic).to_csv(run_dir / "traffic_log.csv", index=False)
    if background_rows is not None:
        pd.DataFrame(background_rows).to_csv(run_dir / "background_flow_metrics.csv", index=False)
    (run_dir / "scenario.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "simulator": "ns-3",
                "seed": seed,
                "mobility_model": mobility_model,
                "num_uavs": 4,
                "time_steps": 12,
                "comm_range": 240,
                "source_id": 0,
                "dest_id": 3,
                "scenario_mode": scenario_mode,
                "partition_start_step": partition_start_step,
                "partition_end_step": partition_end_step,
                "reconnect_step": reconnect_step,
                "partition_groups": partition_groups or {"group_a": [], "group_b": []},
                "background_flows": background_flows,
                "background_rate_kbps": background_rate_kbps,
                "configured_total_offered_load_kbps": background_flows * background_rate_kbps,
                "rwp_speed_range": [3, 8],
            }
        ),
        encoding="utf-8",
    )


def test_extension_report_validates_partition_and_load_interventions(tmp_path: Path) -> None:
    from scripts.analysis.summarize_extension_scenarios import generate_extension_report

    raw_root = tmp_path / "raw"
    partition_groups = {"group_a": [0, 1], "group_b": [2, 3]}
    pre_edges = [(0, 1), (1, 2), (2, 3), (0, 2)]
    mid_edges = [(0, 1), (2, 3)]
    post_edges = pre_edges

    def edge_rows(time: int, links: list[tuple[int, int]], *, delay: float, loss: float) -> list[dict[str, object]]:
        return [
            {
                "time": time,
                "src": src,
                "dst": dst,
                "distance": 100.0,
                "connected": 1,
                "relative_speed": 0.0,
                "rssi": -55.0,
                "snr": 20.0,
                "delay": delay,
                "packet_loss": loss,
                "throughput": 6.0,
                "p_stable": 0.95,
                "weight": 0.05,
            }
            for src, dst in links
        ]

    ctrl_edges: list[dict[str, object]] = []
    for time in range(12):
        if time < 4:
            ctrl_edges.extend(edge_rows(time, pre_edges, delay=5.0, loss=0.02))
        elif time <= 8:
            ctrl_edges.extend(edge_rows(time, mid_edges, delay=7.0, loss=0.10))
        else:
            ctrl_edges.extend(edge_rows(time, post_edges, delay=5.0, loss=0.02))

    baseline_edges = [
        row
        for time in range(12)
        for row in edge_rows(time, pre_edges, delay=5.0, loss=0.02)
    ]
    load_edges = [
        row
        for time in range(12)
        for row in edge_rows(time, pre_edges, delay=8.0, loss=0.07)
    ]
    traffic_rows = [
        {
            "time": time,
            "source": 0,
            "destination": 3,
            "reachable": int(time < 4 or time >= 11),
            "route_path": "0-1-2-3" if time < 4 or time >= 11 else "",
            "num_edges": 3 if time < 4 or time >= 11 else 0,
            "routing_protocol": "OLSR",
            "olsr_mpr_nodes": 2,
            "olsr_avg_rt_size": 3.0,
        }
        for time in range(12)
    ]

    def background_rows(offered: tuple[float, float], throughput: tuple[float, float]) -> list[dict[str, object]]:
        return [
            {
                "flow_index": 0,
                "source": 0,
                "destination": 2,
                "port": 9000,
                "configured_offered_rate_kbps": 600.0,
                "observed_offered_rate_kbps": offered[0],
                "tx_packets": 120,
                "rx_packets": 100,
                "lost_packets": 20,
                "pdr": 100 / 120,
                "mean_delay_ms": 8.0,
                "throughput_mbps": throughput[0],
                "aggregate_throughput_mbps": sum(throughput),
                "jain_fairness": 0.98,
            },
            {
                "flow_index": 1,
                "source": 1,
                "destination": 3,
                "port": 9001,
                "configured_offered_rate_kbps": 600.0,
                "observed_offered_rate_kbps": offered[1],
                "tx_packets": 120,
                "rx_packets": 92,
                "lost_packets": 28,
                "pdr": 92 / 120,
                "mean_delay_ms": 9.0,
                "throughput_mbps": throughput[1],
                "aggregate_throughput_mbps": sum(throughput),
                "jain_fairness": 0.98,
            },
        ]

    for seed, mobility_model, tag, offered, throughput in (
        (60001, "random-waypoint", "rwp", (540.0, 510.0), (0.55, 0.45)),
        (60002, "gauss-markov", "gm", (520.0, 500.0), (0.52, 0.44)),
    ):
        _write_extension_run(
            raw_root,
            f"stress_baseline_{tag}_s{seed}",
            scenario_mode="standard",
            seed=seed,
            mobility_model=mobility_model,
            edges=baseline_edges,
            traffic=traffic_rows,
        )
        _write_extension_run(
            raw_root,
            f"stress_ctrl_{tag}_s{seed}",
            scenario_mode="partition-reconnect",
            seed=seed,
            mobility_model=mobility_model,
            edges=ctrl_edges,
            traffic=traffic_rows,
            partition_start_step=4,
            partition_end_step=8,
            reconnect_step=10,
            partition_groups=partition_groups,
        )
        _write_extension_run(
            raw_root,
            f"stress_load_{tag}_s{seed}",
            scenario_mode="high-load",
            seed=seed,
            mobility_model=mobility_model,
            edges=load_edges,
            traffic=traffic_rows,
            background_flows=2,
            background_rate_kbps=600.0,
            background_rows=background_rows(offered, throughput),
        )

    inventory, summary, checks = generate_extension_report(
        raw_root,
        tmp_path / "reports",
        expected_per_scenario=2,
    )

    assert set(inventory["scenario"]) == {"baseline", "ctrl", "load"}
    assert set(inventory[inventory["scenario"] == "load"]["seed"]) == {60001, 60002}
    assert set(summary["scenario"]) == {"baseline", "ctrl", "load"}
    expected_checks = {
        "ctrl_complete_runs",
        "load_complete_runs",
        "paired_baseline_load_seed_sets",
        "partition_sequence",
        "partition_recovery",
        "load_multiple_flows",
        "load_channel_effect",
    }
    assert expected_checks.issubset(set(checks["check"]))
    status_by_check = checks.set_index("check")["status"].to_dict()
    observed_by_check = checks.set_index("check")["observed"].to_dict()
    assert observed_by_check["ctrl_complete_runs"] == 2
    assert observed_by_check["load_complete_runs"] == 2
    assert observed_by_check["paired_baseline_load_seed_sets"] == 2
    assert status_by_check["partition_sequence"] == "PASS"
    assert status_by_check["partition_recovery"] == "PASS"
    assert status_by_check["load_multiple_flows"] == "PASS"
    assert status_by_check["load_channel_effect"] == "PASS"
    recovery = checks.set_index("check").loc["partition_recovery", "observed"]
    assert float(recovery) == 1.0
    assert (tmp_path / "reports/run_inventory.csv").exists()
    assert (tmp_path / "reports/scenario_summary.csv").exists()
    assert (tmp_path / "reports/manipulation_checks.csv").exists()
    assert (tmp_path / "reports/REPORT.md").exists()
    assert (tmp_path / "reports/figures/extension_manipulation_checks.png").exists()
    assert (tmp_path / "reports/figures/extension_manipulation_checks.pdf").exists()


def test_extension_report_fails_ctrl_run_with_missing_partition_groups(tmp_path: Path) -> None:
    from scripts.analysis.summarize_extension_scenarios import generate_extension_report

    raw_root = tmp_path / "raw"
    _write_extension_run(
        raw_root,
        "stress_ctrl_rwp_s60001",
        scenario_mode="partition-reconnect",
        seed=60001,
        mobility_model="random-waypoint",
        edges=[
            {
                "time": time,
                "src": 0,
                "dst": 1,
                "distance": 100.0,
                "connected": 1,
                "relative_speed": 0.0,
                "rssi": -55.0,
                "snr": 20.0,
                "delay": 5.0,
                "packet_loss": 0.02,
                "throughput": 6.0,
                "p_stable": 0.95,
                "weight": 0.05,
            }
            for time in range(12)
        ],
        traffic=[
            {
                "time": time,
                "source": 0,
                "destination": 3,
                "reachable": 0,
                "route_path": "",
                "num_edges": 0,
                "routing_protocol": "OLSR",
                "olsr_mpr_nodes": 2,
                "olsr_avg_rt_size": 3.0,
            }
            for time in range(12)
        ],
        partition_start_step=4,
        partition_end_step=8,
        reconnect_step=10,
        partition_groups=None,
    )

    inventory, _, checks = generate_extension_report(raw_root, tmp_path / "reports", expected_per_scenario=1)

    row = inventory.set_index("run_name").loc["stress_ctrl_rwp_s60001"]
    assert bool(row["raw_complete"]) is True
    assert bool(row["partition_metadata_complete"]) is False
    assert bool(row["complete"]) is False
    assert str(row["partition_metadata_issue"]) == "missing_or_empty_partition_groups"
    status_by_check = checks.set_index("check")["status"].to_dict()
    assert status_by_check["ctrl_complete_runs"] == "FAIL"
    assert status_by_check["partition_sequence"] == "FAIL"
    assert status_by_check["partition_recovery"] == "FAIL"


def test_extension_report_fails_ctrl_run_with_missing_phase_metadata(tmp_path: Path) -> None:
    from scripts.analysis.summarize_extension_scenarios import generate_extension_report

    raw_root = tmp_path / "raw"
    _write_extension_run(
        raw_root,
        "stress_ctrl_rwp_s60001",
        scenario_mode="partition-reconnect",
        seed=60001,
        mobility_model="random-waypoint",
        edges=[
            {
                "time": time,
                "src": 0,
                "dst": 1,
                "distance": 100.0,
                "connected": 1,
                "relative_speed": 0.0,
                "rssi": -55.0,
                "snr": 20.0,
                "delay": 5.0,
                "packet_loss": 0.02,
                "throughput": 6.0,
                "p_stable": 0.95,
                "weight": 0.05,
            }
            for time in range(12)
        ],
        traffic=[
            {
                "time": time,
                "source": 0,
                "destination": 3,
                "reachable": 0,
                "route_path": "",
                "num_edges": 0,
                "routing_protocol": "OLSR",
                "olsr_mpr_nodes": 2,
                "olsr_avg_rt_size": 3.0,
            }
            for time in range(12)
        ],
        partition_groups={"group_a": [0, 1], "group_b": [2, 3]},
    )

    inventory, _, checks = generate_extension_report(raw_root, tmp_path / "reports", expected_per_scenario=1)

    row = inventory.set_index("run_name").loc["stress_ctrl_rwp_s60001"]
    assert bool(row["raw_complete"]) is True
    assert bool(row["partition_metadata_complete"]) is False
    assert bool(row["complete"]) is False
    assert str(row["partition_metadata_issue"]) == "missing_phase_steps"
    status_by_check = checks.set_index("check")["status"].to_dict()
    assert status_by_check["ctrl_complete_runs"] == "FAIL"
    assert status_by_check["partition_sequence"] == "FAIL"
    assert status_by_check["partition_recovery"] == "FAIL"


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

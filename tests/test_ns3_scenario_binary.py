from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NS3_BINARY = PROJECT_ROOT / "simulation" / "ns3" / "build" / "uav-olsr-dataset"

pytestmark = pytest.mark.skipif(not NS3_BINARY.exists(), reason="ns-3 binary is not built")


def _run_ns3(output_dir: Path, *arguments: str) -> None:
    if os.name == "nt":
        command = [
            "docker",
            "run",
            "--rm",
            "-e",
            "LD_LIBRARY_PATH=/opt/ns3/lib",
            "-v",
            f"{PROJECT_ROOT}:/workspace",
            "-v",
            f"{output_dir}:/output",
            "-w",
            "/workspace",
            "uav-ns3-closed-loop-build",
            "simulation/ns3/build/uav-olsr-dataset",
        ]
        output_argument = "--outputDir=/output"
    else:
        command = [str(NS3_BINARY)]
        output_argument = f"--outputDir={output_dir}"
    subprocess.run(
        [*command, *arguments, output_argument, "--enableAnim=false"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_background_throughput_uses_one_observation_duration(tmp_path: Path) -> None:
    _run_ns3(
        tmp_path,
        "--runName=throughput_duration",
        "--numUavs=20",
        "--timeSteps=12",
        "--seed=7",
        "--mobility=random-waypoint",
        "--commRange=240",
        "--xMax=800",
        "--yMax=800",
        "--scenarioMode=high-load",
        "--backgroundFlows=8",
        "--backgroundRateKbps=600",
    )

    with (tmp_path / "background_flow_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    receiving = [row for row in rows if int(row["rx_packets"]) > 0]
    throughput_per_packet = [float(row["throughput_mbps"]) / int(row["rx_packets"]) for row in receiving]
    assert len(receiving) >= 2
    assert max(throughput_per_packet) - min(throughput_per_packet) < 1e-9
    metadata = json.loads((tmp_path / "scenario.json").read_text(encoding="utf-8"))
    assert metadata["mobility_model"] == "random-waypoint"


def test_zero_background_rerun_removes_only_stale_background_metrics(tmp_path: Path) -> None:
    stale = tmp_path / "background_flow_metrics.csv"
    unrelated = tmp_path / "keep.txt"
    stale.write_text("stale\n", encoding="utf-8")
    unrelated.write_text("keep\n", encoding="utf-8")

    _run_ns3(
        tmp_path,
        "--runName=zero_background",
        "--numUavs=2",
        "--timeSteps=2",
        "--seed=7",
        "--mobility=random-waypoint",
        "--commRange=80",
        "--xMax=100",
        "--yMax=100",
        "--scenarioMode=standard",
        "--backgroundFlows=0",
        "--backgroundRateKbps=0",
    )

    assert not stale.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep\n"


def test_partition_metadata_marks_divergence_window_start(tmp_path: Path) -> None:
    _run_ns3(
        tmp_path,
        "--runName=partition_metadata",
        "--numUavs=20",
        "--timeSteps=12",
        "--seed=7",
        "--mobility=random-waypoint",
        "--commRange=240",
        "--xMax=800",
        "--yMax=800",
        "--scenarioMode=partition-reconnect",
    )

    metadata = json.loads((tmp_path / "scenario.json").read_text(encoding="utf-8"))
    assert metadata["mobility_model"] == "event-controlled-waypoint"
    assert metadata["partition_start_step"] == 3
    assert metadata["partition_end_step"] == 8
    assert metadata["reconnect_step"] == 10

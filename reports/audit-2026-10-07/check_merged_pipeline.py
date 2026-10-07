"""Run merged shell and ns-3 contracts on Linux using only the standard library."""

import ast
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path


def run_cases(path: str, names: list[str]) -> None:
    tree = ast.parse(Path(path).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in [*names, "_run_ns3"]]
    namespace = {
        "Path": Path,
        "csv": csv,
        "json": json,
        "os": os,
        "subprocess": subprocess,
        "__file__": str(Path(path).resolve()),
        "PROJECT_ROOT": Path.cwd(),
        "NS3_BINARY": Path("/scenario/build/uav-olsr-dataset"),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), namespace)
    for name in names:
        with tempfile.TemporaryDirectory(prefix="uav-merged-") as directory:
            function = namespace[name]
            function(Path(directory)) if function.__code__.co_argcount else function()
        print(f"PASS {name}")


if __name__ == "__main__":
    run_cases(
        "tests/test_controlled_scenarios.py",
        ["test_ns3_pipeline_forwards_extension_environment_to_binary"],
    )
    run_cases(
        "tests/test_stage6_runner.py",
        [
            "test_controlled_launcher_dry_run_states_extension_count_and_temporal_selection",
            "test_controlled_launcher_runs_temporal_artifacts_before_stage6",
        ],
    )
    run_cases(
        "tests/test_ns3_scenario_binary.py",
        [
            "test_background_throughput_uses_one_observation_duration",
            "test_zero_background_rerun_removes_only_stale_background_metrics",
            "test_partition_metadata_marks_divergence_window_start",
        ],
    )

"""Generate paired baseline/fast/sparse/dense ns-3 datasets with resume support."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "dataset" / "run_one_dataset_ns3.sh"
SCENARIOS = {
    "baseline": (240, 3, 8),
    "fast": (240, 12, 20),
    "sparse": (160, 3, 8),
    "dense": (300, 3, 8),
}


@dataclass(frozen=True)
class ScenarioJob:
    scenario: str
    index: int
    seed: int
    mobility: str
    mobility_tag: str
    comm_range: int
    speed_min: int
    speed_max: int
    run_name: str
    scenario_mode: str = "standard"
    background_flows: int = 0
    background_rate_kbps: float = 0.0
    source_id: int = 0
    dest_id: int = 4


def _mobility_for_index(index: int) -> tuple[str, str]:
    mobility = "gauss-markov" if index % 2 == 0 else "random-waypoint"
    return mobility, ("gm" if mobility == "gauss-markov" else "rwp")


def build_jobs(start_index: int, end_index: int, seed_base: int = 60_000) -> list[ScenarioJob]:
    if not 1 <= start_index <= end_index <= 100:
        raise ValueError("Expected 1 <= start_index <= end_index <= 100")
    jobs = []
    for index in range(start_index, end_index + 1):
        seed = seed_base + index
        mobility, tag = _mobility_for_index(index)
        for scenario, (comm_range, speed_min, speed_max) in SCENARIOS.items():
            jobs.append(
                ScenarioJob(
                    scenario=scenario,
                    index=index,
                    seed=seed,
                    mobility=mobility,
                    mobility_tag=tag,
                    comm_range=comm_range,
                    speed_min=speed_min,
                    speed_max=speed_max,
                    run_name=f"stress_{scenario}_{tag}_s{seed}",
                )
            )
    return jobs


def build_extension_jobs(count: int = 10, seed_base: int = 60_000) -> list[ScenarioJob]:
    if not 1 <= count <= 100:
        raise ValueError("Expected 1 <= count <= 100")
    jobs = []
    comm_range, speed_min, speed_max = SCENARIOS["baseline"]
    for index in range(1, count + 1):
        seed = seed_base + index
        mobility, tag = _mobility_for_index(index)
        jobs.append(
            ScenarioJob(
                scenario="ctrl",
                index=index,
                seed=seed,
                mobility="event-controlled-waypoint",
                mobility_tag="ctrl",
                comm_range=comm_range,
                speed_min=speed_min,
                speed_max=speed_max,
                run_name=f"stress_partition_ctrl_s{seed}",
                scenario_mode="partition-reconnect",
            )
        )
        jobs.append(
            ScenarioJob(
                scenario="load",
                index=index,
                seed=seed,
                mobility=mobility,
                mobility_tag=tag,
                comm_range=comm_range,
                speed_min=speed_min,
                speed_max=speed_max,
                run_name=f"stress_load_{tag}_s{seed}",
                scenario_mode="high-load",
                background_flows=8,
                background_rate_kbps=600.0,
            )
        )
    return jobs


def _complete(job: ScenarioJob, raw_root: Path, graph_root: Path) -> bool:
    required = [
        raw_root / job.run_name / "scenario.json",
        raw_root / job.run_name / "nodes.csv",
        raw_root / job.run_name / "edges.csv",
        raw_root / job.run_name / "traffic_log.csv",
        graph_root / job.run_name / "graph_dataset" / "train.pt",
        graph_root / job.run_name / "graph_dataset" / "val.pt",
        graph_root / job.run_name / "graph_dataset" / "test.pt",
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False

    raw_dir = raw_root / job.run_name
    if job.scenario_mode == "high-load":
        metrics_path = raw_dir / "background_flow_metrics.csv"
        if not metrics_path.exists() or metrics_path.stat().st_size == 0:
            return False
        try:
            metrics = pd.read_csv(metrics_path, usecols=["flow_index"])
            metadata = json.loads((raw_dir / "scenario.json").read_text(encoding="utf-8"))
            configured_flows = int(metadata["background_flows"])
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, pd.errors.EmptyDataError):
            return False
        if (
            metadata.get("scenario_mode") != job.scenario_mode
            or configured_flows != job.background_flows
            or len(metrics) != configured_flows
            or metrics["flow_index"].nunique() != configured_flows
        ):
            return False

    if job.scenario_mode == "partition-reconnect":
        try:
            metadata = json.loads((raw_dir / "scenario.json").read_text(encoding="utf-8"))
            start = metadata["partition_start_step"]
            end = metadata["partition_end_step"]
            reconnect = metadata["reconnect_step"]
            num_uavs = int(metadata["num_uavs"])
            time_steps = int(metadata["time_steps"])
            groups = metadata["partition_groups"]
            group_a = {int(node) for node in groups["group_a"]}
            group_b = {int(node) for node in groups["group_b"]}
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if (
            metadata.get("scenario_mode") != job.scenario_mode
            or not all(isinstance(step, int) and not isinstance(step, bool) for step in (start, end, reconnect))
            or not 0 <= start < end < reconnect < time_steps
            or not group_a
            or not group_b
            or group_a & group_b
            or group_a | group_b != set(range(num_uavs))
        ):
            return False
    return True


def run_jobs(
    jobs: list[ScenarioJob],
    *,
    raw_root: Path,
    graph_root: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for number, job in enumerate(jobs, start=1):
        started = time.perf_counter()
        status = "SKIPPED" if _complete(job, raw_root, graph_root) else "RUNNING"
        print(f"[{number}/{len(jobs)}] {status} {job.run_name}", flush=True)
        if status == "RUNNING":
            env = {
                **os.environ,
                "SIM_NUM_UAVS": "20",
                "SIM_COMM_RANGE": str(job.comm_range),
                "SIM_TIME_STEPS": "120",
                "SIM_RWP_SPEED_MIN": str(job.speed_min),
                "SIM_RWP_SPEED_MAX": str(job.speed_max),
                "SIM_X_MAX": "800",
                "SIM_Y_MAX": "800",
                "SIM_SCENARIO_MODE": job.scenario_mode,
                "SIM_BACKGROUND_FLOWS": str(job.background_flows),
                "SIM_BACKGROUND_RATE_KBPS": str(job.background_rate_kbps),
                "SIM_SOURCE_ID": str(job.source_id),
                "SIM_DEST_ID": str(job.dest_id),
            }
            try:
                subprocess.run(
                    ["bash", str(PIPELINE_SCRIPT), job.run_name, str(job.seed), job.mobility],
                    check=True,
                    cwd=PROJECT_ROOT,
                    env=env,
                )
                status = "COMPLETED" if _complete(job, raw_root, graph_root) else "INCOMPLETE"
            except subprocess.CalledProcessError:
                status = "FAILED"
        row = {
            **asdict(job),
            "status": status,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        if status in {"FAILED", "INCOMPLETE"}:
            raise RuntimeError(f"{status}: {job.run_name}; rerun the same range after fixing the error")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=60_000)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_snapshots"))
    parser.add_argument("--graph-root", type=Path, default=Path("data/graph_dataset"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-extensions", action="store_true")
    parser.add_argument("--extension-count", type=int, default=10)
    args = parser.parse_args()
    jobs = build_jobs(args.start_index, args.end_index, args.seed_base)
    if args.include_extensions:
        jobs.extend(build_extension_jobs(args.extension_count, args.seed_base))
    counts = pd.Series([job.scenario for job in jobs]).value_counts().sort_index()
    print(f"[PLAN] indices={args.start_index}..{args.end_index}; total_jobs={len(jobs)}")
    print(counts.to_string())
    if args.dry_run:
        return
    manifest = args.manifest or Path(
        f"reports/controlled_scenarios/workers/worker_{args.start_index:03d}_{args.end_index:03d}.csv"
    )
    result = run_jobs(
        jobs,
        raw_root=args.raw_root,
        graph_root=args.graph_root,
        manifest_path=manifest,
    )
    print(f"[OK] {len(result)} jobs completed or resumed; manifest: {manifest}")


if __name__ == "__main__":
    main()

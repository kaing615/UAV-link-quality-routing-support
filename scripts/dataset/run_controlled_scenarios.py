"""Generate paired baseline/fast/sparse/dense ns-3 datasets with resume support."""

from __future__ import annotations

import argparse
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


def build_jobs(start_index: int, end_index: int, seed_base: int = 60_000) -> list[ScenarioJob]:
    if not 1 <= start_index <= end_index <= 100:
        raise ValueError("Expected 1 <= start_index <= end_index <= 100")
    jobs = []
    for index in range(start_index, end_index + 1):
        seed = seed_base + index
        mobility = "gauss-markov" if index % 2 == 0 else "random-waypoint"
        tag = "gm" if mobility == "gauss-markov" else "rwp"
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


def _complete(job: ScenarioJob, raw_root: Path, graph_root: Path) -> bool:
    required = [
        raw_root / job.run_name / "scenario.json",
        raw_root / job.run_name / "nodes.csv",
        raw_root / job.run_name / "edges.csv",
        graph_root / job.run_name / "graph_dataset" / "train.pt",
        graph_root / job.run_name / "graph_dataset" / "val.pt",
        graph_root / job.run_name / "graph_dataset" / "test.pt",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


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
    args = parser.parse_args()
    jobs = build_jobs(args.start_index, args.end_index, args.seed_base)
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

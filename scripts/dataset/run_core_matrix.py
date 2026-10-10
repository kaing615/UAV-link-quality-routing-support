"""Run the preregistered core scenario levels with paired seeds and held-seed evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from src.utils.experiment import cache_matches, fingerprint, save_manifest, seed_group, validate_ns3_binary

PHY_RATE_KBPS = 6000.0  # ErpOfdmRate6Mbps; offered percentages refer to aggregate payload rate.


@dataclass(frozen=True)
class CoreJob:
    family: str
    condition: str
    factor: str
    level: str
    seed: int
    mobility: str
    run_name: str
    speed_min: float = 6
    speed_max: float = 10
    alpha: float = 0.85
    z_min: float = 50
    z_max: float = 150
    comm_range: float = 240
    scenario_mode: str = "standard"
    flows: int = 0
    offered_fraction: float = 0
    traffic_pattern: str = "cbr"


def build_core_jobs(seeds: int = 10, seed_base: int = 70000) -> list[CoreJob]:
    if seeds < 10:
        raise ValueError("Core matrix requires at least 10 paired seeds per condition")
    jobs = []
    for index in range(1, seeds + 1):
        seed = seed_base + index
        mobility = "gauss-markov" if index % 2 == 0 else "random-waypoint"
        base = CoreJob("", "", "", "", seed, mobility, "")

        def add(family: str, factor: str, level: str, **changes) -> None:
            job = replace(base, family=family, factor=factor, level=level,
                          condition=f"{factor}_{level}", **changes)
            tag = "ctrl" if job.scenario_mode in ("partition-reconnect", "crossing-swarms", "group-control") else (
                "gm" if job.mobility == "gauss-markov" else "rwp")
            jobs.append(replace(job, run_name=f"stress_core_{job.condition}_{tag}_s{seed}"))

        for low, high in ((2, 5), (6, 10), (12, 20)):
            add("fast3d", "speed", f"{low}to{high}", speed_min=low, speed_max=high)
        for alpha in (0.50, 0.85, 0.95):
            add("fast3d", "alpha", f"{int(alpha * 100)}", mobility="gauss-markov", alpha=alpha)
        for low, high in ((50, 150), (30, 300)):
            add("fast3d", "altitude", f"{low}to{high}", z_min=low, z_max=high)
        for distance in (150, 180, 240, 300):
            add("density", "range", str(distance), comm_range=distance)
        for level, mode in (("control", "group-control"), ("partition", "partition-reconnect"),
                            ("crossing", "crossing-swarms")):
            add("partition", "event", level, mobility="event-controlled-waypoint", scenario_mode=mode)
        # One factor at a time: every requested level is covered without a Cartesian product.
        for flows, fraction, pattern in ((1, .2, "cbr"), (4, .2, "cbr"), (8, .2, "cbr"),
                                         (4, .2, "bursty"), (4, .5, "cbr"), (4, .8, "cbr")):
            level = f"f{flows}_{pattern}_p{int(fraction * 100)}"
            add("load", "traffic", level, scenario_mode="high-load", flows=flows,
                offered_fraction=fraction, traffic_pattern=pattern)
    return jobs


def ns3_command(job: CoreJob, binary: Path, output: Path, steps: int = 120) -> list[str]:
    if steps < 120:
        raise ValueError("At least 120 snapshots are required for the core evaluation")
    arguments = {
        "runName": job.run_name, "seed": job.seed, "numUavs": 20, "timeSteps": steps,
        "mobility": "random-waypoint" if job.mobility == "event-controlled-waypoint" else job.mobility,
        "xMax": 800, "yMax": 800, "zMin": job.z_min, "zMax": job.z_max,
        "commRange": job.comm_range, "rwpSpeedMin": job.speed_min, "rwpSpeedMax": job.speed_max,
        "gmAlpha": job.alpha, "scenarioMode": job.scenario_mode,
        "backgroundFlows": job.flows,
        "backgroundRateKbps": PHY_RATE_KBPS * job.offered_fraction / job.flows if job.flows else 0,
        "backgroundPattern": job.traffic_pattern,
        "sourceId": 0, "destId": 4, "enableAnim": "false", "outputDir": output,
    }
    if job.family == "partition":
        arguments.update(groupSeparation=360, groupDepartureStep=steps // 6,
                         groupSeparationStep=steps // 3, groupReturnStep=steps // 2,
                         groupReconnectStep=2 * steps // 3)
    return [str(binary), *(f"--{key}={value}" for key, value in arguments.items())]


def save_plan(jobs: list[CoreJob], report: Path) -> None:
    report.mkdir(parents=True, exist_ok=True)
    with (report / "plan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(jobs[0])))
        writer.writeheader()
        writer.writerows(asdict(job) for job in jobs)


def simulate(jobs: list[CoreJob], binary: Path, raw: Path, report: Path, workers: int, steps: int) -> None:
    validate_ns3_binary(binary, binary.parent.parent)
    # Validate source mounted in the workspace against the binary too.
    validate_ns3_binary(binary)
    if workers < 1:
        raise ValueError("workers must be positive")
    save_plan(jobs, report)

    def run(job: CoreJob) -> dict:
        output = raw / job.run_name
        output.mkdir(parents=True, exist_ok=True)
        expected = fingerprint({**asdict(job), "steps": steps},
                               [binary, binary.with_name("source.sha256"), Path(__file__)])
        required = [output / name for name in ("scenario.json", "nodes.csv", "edges.csv", "traffic_log.csv")]
        if job.flows:
            required.append(output / "background_flow_metrics.csv")
        if cache_matches(output / "experiment.json", expected, required):
            status = "SKIPPED"
        else:
            with (output / "simulation.log").open("w", encoding="utf-8") as log:
                subprocess.run(ns3_command(job, binary, output, steps), check=True, stdout=log, stderr=log)
            if not all(path.is_file() and path.stat().st_size for path in required):
                raise RuntimeError(f"Incomplete simulation: {job.run_name}")
            metadata = json.loads((output / "scenario.json").read_text())
            metadata["core_matrix"] = asdict(job)
            (output / "scenario.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            save_manifest(output / "experiment.json", expected)
            status = "COMPLETED"
        print(f"[{status}] {job.run_name}", flush=True)
        return {"run_name": job.run_name, "status": status}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        with (report / "simulation_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["run_name", "status"])
            writer.writeheader()
            for row in pool.map(run, jobs):
                writer.writerow(row)
                stream.flush()


def held_seed_groups(jobs: list[CoreJob]) -> list[tuple[int, list[str], list[str]]]:
    folds = []
    for seed in sorted({job.seed for job in jobs}):
        train = [job.run_name for job in jobs if job.seed != seed]
        test = [job.run_name for job in jobs if job.seed == seed]
        assert not {seed_group(name) for name in train} & {seed_group(name) for name in test}
        folds.append((seed, train, test))
    return folds


def train(jobs: list[CoreJob], data: Path, output: Path, models: list[str], epochs: int, patience: int) -> None:
    from scripts.train.run_stage6_benchmark import _training_command

    from src.utils.experiment import validate_test_predictions

    for target in ("qos", "survival"):
        for horizon in (1, 2, 3, 5):
            coordinate = data / target / f"k{horizon}"
            for seed, train_runs, test_runs in held_seed_groups(jobs):
                for model in models:
                    if model == "persistence":
                        continue
                    destination = output / model / target / f"k{horizon}" / f"seed{seed}"
                    command = _training_command(model, coordinate, train_runs, test_runs, destination, epochs, patience)
                    command.append("--include-purged-test")
                    inputs = [coordinate / run / part for run in [*train_runs, *test_runs]
                              for part in ("features", "splits", "graph_dataset")]
                    manifest = fingerprint({"command": command, "train_seed_holdout": seed},
                                           [*inputs, Path("src/training"), Path("src/models"), Path("src/utils")])
                    predictions = [destination / f"test_predictions_{run}.csv" for run in test_runs]
                    if not cache_matches(destination / "experiment.json", manifest,
                                         [destination / "metadata.json", destination / "metrics.csv", *predictions]):
                        destination.mkdir(parents=True, exist_ok=True)
                        with (destination / "training.log").open("w", encoding="utf-8") as log:
                            subprocess.run(command, check=True, stdout=log, stderr=log)
                        for run, prediction in zip(test_runs, predictions, strict=True):
                            validate_test_predictions(coordinate / run, prediction,
                                                      test_splits=("train", "val", "test", "purged"))
                        save_manifest(destination / "experiment.json", manifest)
                    print(f"[TRAINED] {model}/{target}/k{horizon}/seed{seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=70000)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--binary", type=Path, default=Path("simulation/ns3/build/uav-olsr-dataset"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/core_matrix/raw"))
    parser.add_argument("--data-root", type=Path, default=Path("data/core_matrix/multihorizon"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/core_matrix"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/core_matrix"))
    parser.add_argument("--stages", nargs="+", choices=("simulate", "prepare", "train", "report"),
                        default=["simulate", "prepare", "train", "report"])
    parser.add_argument("--models", nargs="+", choices=("persistence", "logreg", "xgb", "edge-sage"),
                        default=["persistence", "logreg", "xgb", "edge-sage"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = build_core_jobs(args.seeds, args.seed_base)
    save_plan(jobs, args.report_root)
    print(f"[PLAN] {len(jobs)} runs; {len(jobs) // args.seeds} conditions; {args.seeds} paired seeds/condition", flush=True)
    if args.dry_run:
        return
    if "simulate" in args.stages:
        simulate(jobs, args.binary.resolve(), args.raw_root.resolve(), args.report_root, args.workers, args.steps)
    if "prepare" in args.stages:
        from scripts.analysis.analyze_core_matrix import manipulation_report
        from scripts.dataset.build_multihorizon_pilot import run_pilot
        manipulation_report(jobs, args.raw_root, args.report_root, require_complete=True)
        run_pilot(args.raw_root, [job.run_name for job in jobs], args.data_root, args.report_root / "dataset_summary.csv")
    if "train" in args.stages:
        train(jobs, args.data_root, args.output_root, args.models, args.epochs, args.patience)
    if "report" in args.stages:
        from scripts.analysis.analyze_core_matrix import generate_report
        generate_report(jobs, args.raw_root, args.data_root, args.output_root, args.report_root, args.models)
    print("[OK] requested core-matrix stages completed", flush=True)


if __name__ == "__main__":
    main()

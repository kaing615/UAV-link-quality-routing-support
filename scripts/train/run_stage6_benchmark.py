"""Run LORO, cross-mobility, and Edge-SAGE ablation benchmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scripts.train.run_multihorizon_benchmark import (
    aggregate_metrics,
    discover_datasets,
    select_representative_runs,
)

from src.utils.experiment import cache_matches, fingerprint, save_manifest, seed_group, validate_test_predictions

MODELS = ("logreg", "xgb", "edge-sage")
ABLATION_MODES = ("decoder-only", "message-only", "noedge")


def discover_coordinates(
    data_root: Path,
    targets: set[str] | None = None,
    horizons: set[int] | None = None,
) -> list[tuple[str, int, list[str]]]:
    grouped: dict[tuple[str, int], list[str]] = {}
    for target, horizon, run_root in discover_datasets(data_root):
        if (targets is None or target in targets) and (horizons is None or horizon in horizons):
            grouped.setdefault((target, horizon), []).append(run_root.name)
    return [(target, horizon, sorted(runs)) for (target, horizon), runs in sorted(grouped.items())]


def collect_protocol_metrics(protocol_root: Path, protocol: str, paths: list[Path] | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted(paths if paths is not None else protocol_root.glob("*/*/k*/*/metrics.csv")):
        frame = pd.read_csv(path)
        frame["protocol"] = protocol
        frame["model_id"] = frame.get("model_id", path.parents[3].name)
        frame["target"] = path.parents[2].name
        frame["horizon"] = int(path.parents[1].name.removeprefix("k"))
        if "run_name" not in frame:
            frame["run_name"] = path.parent.name
        else:
            frame["run_name"] = frame["run_name"].fillna(path.parent.name)
        frame["source_metrics"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def with_full_ablation_reference(ablation: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    reference = benchmark[(benchmark["model_id"] == "edge-sage") & (benchmark["split"] == "test")].copy()
    if not ablation.empty:
        coordinates = set(zip(ablation["target"], ablation["horizon"], ablation["run_name"], strict=True))
        reference = reference[
            [
                coordinate in coordinates
                for coordinate in zip(reference["target"], reference["horizon"], reference["run_name"], strict=True)
            ]
        ]
    reference["protocol"] = "ablation"
    return pd.concat([reference, ablation], ignore_index=True)


def _run(command: list[str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)


def _run_cached(command: list[str], data_paths: list[Path], output_dir: Path, force: bool) -> None:
    cross_run = "--test-runs" in command
    test_runs = command[command.index("--test-runs" if cross_run else "--run-name") + 1].split(",")
    train_runs = command[command.index("--train-runs") + 1].split(",") if cross_run else test_runs
    predictions = [
        output_dir / (f"test_predictions_{run}.csv" if cross_run else "test_predictions.csv") for run in test_runs
    ]
    manifest = fingerprint(
        {
            "command": command,
            "train_runs": train_runs,
            "test_runs": test_runs,
            "train_seed_groups": sorted({seed_group(run) for run in train_runs}),
            "test_seed_groups": sorted({seed_group(run) for run in test_runs}),
        },
        [
            *data_paths,
            Path(__file__),
            Path("src/training"),
            Path("src/models"),
            Path("src/preprocessing"),
            Path("src/utils"),
        ],
    )
    manifest_path = output_dir / "experiment.json"
    if force or not cache_matches(manifest_path, manifest, [output_dir / "metrics.csv", *predictions]):
        _run(command)
    data_root = Path(command[command.index("--data-root") + 1])
    for run, prediction in zip(test_runs, predictions, strict=True):
        validate_test_predictions(data_root / run, prediction)
    save_manifest(manifest_path, manifest)


def _mobility_groups(raw_root: Path, runs: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for run in runs:
        scenario = json.loads((raw_root / run / "scenario.json").read_text(encoding="utf-8"))
        groups.setdefault(str(scenario["mobility_model"]), []).append(run)
    return groups


def _training_command(
    model: str,
    data_root: Path,
    train_runs: list[str],
    test_runs: list[str],
    output_dir: Path,
    gnn_epochs: int,
    gnn_patience: int,
) -> list[str]:
    common = [
        "--train-runs",
        ",".join(train_runs),
        "--test-runs",
        ",".join(test_runs),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
    ]
    if model == "edge-sage":
        return [
            sys.executable,
            "-m",
            "src.training.gnn.train_gnn_loro",
            *common,
            "--model",
            model,
            "--hidden",
            "128",
            "--lr",
            "5e-4",
            "--lr-scheduler",
            "--epochs",
            str(gnn_epochs),
            "--patience",
            str(gnn_patience),
        ]
    return [
        sys.executable,
        "-m",
        "src.training.baselines.loro_baselines",
        *common,
        "--model",
        model,
    ]


def run_stage6(
    data_root: Path,
    raw_root: Path,
    output_root: Path,
    reports_root: Path,
    protocols: set[str],
    models: list[str],
    targets: set[str] | None,
    horizons: set[int] | None,
    force: bool,
    gnn_epochs: int,
    gnn_patience: int,
    benchmark_summary: Path,
    loro_runs_per_scenario: int | None,
    ablation_runs_per_scenario: int | None,
) -> None:
    coordinates = discover_coordinates(data_root, targets, horizons)
    if not coordinates:
        raise FileNotFoundError(f"No multi-horizon datasets found under {data_root}")
    selected_paths = {protocol: [] for protocol in protocols}

    for target, horizon, runs in coordinates:
        coordinate_root = data_root / target / f"k{horizon}"
        if "ablation" in protocols:
            ablation_runs = (
                select_representative_runs(runs, ablation_runs_per_scenario)
                if ablation_runs_per_scenario is not None
                else runs
            )
            for edge_mode in ABLATION_MODES:
                model_id = f"edge-sage-{edge_mode}"
                for run in ablation_runs:
                    output_dir = output_root / "ablation" / model_id / target / f"k{horizon}" / run
                    selected_paths["ablation"].append(output_dir / "metrics.csv")
                    _run_cached(
                        [
                            sys.executable,
                            "-m",
                            "src.training.gnn.train_gnn",
                            "--run-name",
                            run,
                            "--data-root",
                            str(coordinate_root),
                            "--model",
                            "edge-sage",
                            "--edge-mode",
                            edge_mode,
                            "--hidden",
                            "128",
                            "--lr",
                            "5e-4",
                            "--lr-scheduler",
                            "--output-dir",
                            str(output_dir),
                            "--epochs",
                            str(gnn_epochs),
                            "--patience",
                            str(gnn_patience),
                        ],
                        [
                            coordinate_root / run / "features",
                            coordinate_root / run / "splits",
                            coordinate_root / run / "graph_dataset",
                        ],
                        output_dir,
                        force,
                    )

        if "loro" in protocols:
            test_runs = (
                select_representative_runs(runs, loro_runs_per_scenario) if loro_runs_per_scenario is not None else runs
            )
            for test_run in test_runs:
                train_runs = [run for run in runs if seed_group(run) != seed_group(test_run)]
                if not train_runs:
                    raise ValueError(f"No independent training seeds remain for {test_run}")
                for model in models:
                    output_dir = output_root / "loro" / model / target / f"k{horizon}" / test_run
                    selected_paths["loro"].append(output_dir / "metrics.csv")
                    _run_cached(
                        _training_command(
                            model,
                            coordinate_root,
                            train_runs,
                            [test_run],
                            output_dir,
                            gnn_epochs,
                            gnn_patience,
                        ),
                        [
                            coordinate_root / run / part
                            for run in [*train_runs, test_run]
                            for part in ("features", "splits", "graph_dataset")
                        ],
                        output_dir,
                        force,
                    )

        if "cross-mobility" in protocols:
            mobility = _mobility_groups(raw_root, runs)
            for source, train_runs in sorted(mobility.items()):
                test_runs = [run for name, group in mobility.items() if name != source for run in group]
                test_seeds = {seed_group(run) for run in test_runs}
                train_runs = [run for run in train_runs if seed_group(run) not in test_seeds]
                if not train_runs or not test_runs:
                    raise ValueError(f"Cross-mobility {source} requires independent training and test seeds")
                direction = f"{source}-to-other"
                for model in models:
                    output_dir = output_root / "cross-mobility" / model / target / f"k{horizon}" / direction
                    selected_paths["cross-mobility"].append(output_dir / "metrics.csv")
                    _run_cached(
                        _training_command(
                            model,
                            coordinate_root,
                            train_runs,
                            test_runs,
                            output_dir,
                            gnn_epochs,
                            gnn_patience,
                        ),
                        [
                            coordinate_root / run / part
                            for run in [*train_runs, *test_runs]
                            for part in ("features", "splits", "graph_dataset")
                        ],
                        output_dir,
                        force,
                    )

    reports_root.mkdir(parents=True, exist_ok=True)
    for protocol in protocols:
        detailed = collect_protocol_metrics(output_root / protocol, protocol, selected_paths[protocol])
        if protocol == "ablation":
            benchmark = pd.read_csv(benchmark_summary)
            detailed = with_full_ablation_reference(detailed, benchmark)
        detailed.to_csv(reports_root / f"{protocol}_summary.csv", index=False)
        if not detailed.empty:
            aggregate_metrics(detailed[detailed["split"] == "test"]).to_csv(
                reports_root / f"{protocol}_summary_aggregate.csv", index=False
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/multihorizon"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_snapshots"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stage6"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/stage6"))
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("loro", "cross-mobility", "ablation"),
        default=["loro", "cross-mobility", "ablation"],
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--targets", nargs="+", choices=("qos", "survival"), default=None)
    parser.add_argument("--horizons", nargs="+", type=int, choices=(1, 2, 3, 5), default=[1, 5])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gnn-epochs", type=int, default=200)
    parser.add_argument("--gnn-patience", type=int, default=20)
    parser.add_argument("--loro-runs-per-scenario", type=int, default=None)
    parser.add_argument("--ablation-runs-per-scenario", type=int, default=None)
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        default=Path("reports/multihorizon_benchmark_summary.csv"),
    )
    args = parser.parse_args()
    run_stage6(
        args.data_root,
        args.raw_root,
        args.output_root,
        args.reports_root,
        set(args.protocols),
        args.models,
        set(args.targets) if args.targets else None,
        set(args.horizons) if args.horizons else None,
        args.force,
        args.gnn_epochs,
        args.gnn_patience,
        args.benchmark_summary,
        args.loro_runs_per_scenario,
        args.ablation_runs_per_scenario,
    )


if __name__ == "__main__":
    main()

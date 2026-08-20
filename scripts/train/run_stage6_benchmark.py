"""Run LORO, cross-mobility, and Edge-SAGE ablation benchmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scripts.train.run_multihorizon_benchmark import aggregate_metrics, discover_datasets

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


def collect_protocol_metrics(protocol_root: Path, protocol: str) -> pd.DataFrame:
    frames = []
    for path in sorted(protocol_root.glob("*/*/k*/*/metrics.csv")):
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
        coordinates = set(zip(ablation["target"], ablation["horizon"], strict=True))
        reference = reference[
            [coordinate in coordinates for coordinate in zip(reference["target"], reference["horizon"], strict=True)]
        ]
    reference["protocol"] = "ablation"
    return pd.concat([reference, ablation], ignore_index=True)


def _run(command: list[str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)


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
) -> None:
    coordinates = discover_coordinates(data_root, targets, horizons)
    if not coordinates:
        raise FileNotFoundError(f"No multi-horizon datasets found under {data_root}")

    for target, horizon, runs in coordinates:
        coordinate_root = data_root / target / f"k{horizon}"
        if "ablation" in protocols:
            for edge_mode in ABLATION_MODES:
                model_id = f"edge-sage-{edge_mode}"
                for run in runs:
                    output_dir = output_root / "ablation" / model_id / target / f"k{horizon}" / run
                    if (output_dir / "metrics.csv").exists() and not force:
                        continue
                    _run(
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
                        ]
                    )

        if "loro" in protocols:
            for test_run in runs:
                train_runs = [run for run in runs if run != test_run]
                for model in models:
                    output_dir = output_root / "loro" / model / target / f"k{horizon}" / test_run
                    if (output_dir / "metrics.csv").exists() and not force:
                        continue
                    _run(
                        _training_command(
                            model,
                            coordinate_root,
                            train_runs,
                            [test_run],
                            output_dir,
                            gnn_epochs,
                            gnn_patience,
                        )
                    )

        if "cross-mobility" in protocols:
            mobility = _mobility_groups(raw_root, runs)
            for source, train_runs in sorted(mobility.items()):
                test_runs = [run for name, group in mobility.items() if name != source for run in group]
                direction = f"{source}-to-other"
                for model in models:
                    output_dir = output_root / "cross-mobility" / model / target / f"k{horizon}" / direction
                    if (output_dir / "metrics.csv").exists() and not force:
                        continue
                    _run(
                        _training_command(
                            model,
                            coordinate_root,
                            train_runs,
                            test_runs,
                            output_dir,
                            gnn_epochs,
                            gnn_patience,
                        )
                    )

    reports_root.mkdir(parents=True, exist_ok=True)
    for protocol in protocols:
        detailed = collect_protocol_metrics(output_root / protocol, protocol)
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
    )


if __name__ == "__main__":
    main()

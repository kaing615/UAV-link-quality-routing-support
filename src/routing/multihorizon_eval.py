"""Replay and aggregate routing with predictors trained for each future horizon."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.paired_statistics import bootstrap_mean_ci, paired_comparisons
from src.routing.predict_edges import attach_prediction_identity
from src.routing.replay_eval import evaluate_run

MODELS = ("persistence", "logreg", "xgb", "edge-sage")
TARGETS = ("qos", "survival")
HORIZONS = (1, 2, 3, 5)
COST_MODES = ("neglog", "one-minus")

ROUTING_METRICS = {
    "route_found_rate": True,
    "mean_hops": False,
    "mean_e2e_delay_ms": False,
    "mean_est_pdr": True,
    "mean_route_lifetime": True,
    "survival_at_1": True,
    "survival_at_horizon": True,
    "mean_realized_pdr_t1": True,
    "mean_realized_pdr_at_horizon": True,
    "mean_route_changes": False,
    "disconnected_rate": False,
}

COMPARISONS = (
    ("delay", "hop"),
    ("persistence", "hop"),
    ("logreg", "hop"),
    ("xgb", "hop"),
    ("edge-sage", "hop"),
    ("edge-sage", "xgb"),
    ("edge-sage", "logreg"),
)


def discover_coordinates(
    data_root: Path,
    targets: set[str] | None = None,
    horizons: set[int] | None = None,
) -> list[tuple[str, int, Path]]:
    coordinates = []
    for graph_dir in sorted(data_root.glob("*/k*/*/graph_dataset")):
        target = graph_dir.parents[2].name
        horizon = int(graph_dir.parents[1].name.removeprefix("k"))
        if (targets is None or target in targets) and (horizons is None or horizon in horizons):
            coordinates.append((target, horizon, graph_dir.parent))
    return coordinates


def _prediction_paths(
    model_root: Path,
    run_root: Path,
    routing_run_dir: Path,
    models: list[str],
    force: bool,
) -> dict[str, Path]:
    target = run_root.parents[1].name
    horizon_dir = run_root.parent.name
    paths = {}
    for model in models:
        source = model_root / model / target / horizon_dir / run_root.name / "test_predictions.csv"
        if not source.exists():
            raise FileNotFoundError(f"Missing {model} predictions: {source}")
        if model == "edge-sage":
            identified = routing_run_dir / "predictions_edge-sage.csv"
            if force or not identified.exists():
                attach_prediction_identity(source, run_root / "graph_dataset" / "test.pt", identified)
            paths[model] = identified
        else:
            paths[model] = source
    return paths


def _completed(summary_csv: Path, expected_strategies: set[str]) -> bool:
    if not summary_csv.exists():
        return False
    try:
        summary = pd.read_csv(summary_csv)
    except (pd.errors.EmptyDataError, OSError):
        return False
    return set(summary.get("strategy", [])) == expected_strategies


def run_multihorizon_routing(
    data_root: Path,
    model_root: Path,
    raw_root: Path,
    routing_root: Path,
    reports_root: Path,
    models: list[str],
    targets: set[str] | None = None,
    horizons: set[int] | None = None,
    cost_modes: list[str] | None = None,
    limit: int | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coordinates = discover_coordinates(data_root, targets, horizons)
    if limit is not None:
        selected_runs = sorted({run_root.name for _, _, run_root in coordinates})[:limit]
        coordinates = [item for item in coordinates if item[2].name in selected_runs]
    if not coordinates:
        raise FileNotFoundError(f"No multi-horizon datasets found under {data_root}")

    expected_strategies = {"hop", "delay", *models}
    for target, horizon, run_root in coordinates:
        for cost_mode in cost_modes or list(COST_MODES):
            routing_run_dir = routing_root / target / f"k{horizon}" / cost_mode / run_root.name
            summary_csv = routing_run_dir / "summary.csv"
            if not force and _completed(summary_csv, expected_strategies):
                print(f"[SKIP] routing {target}/k{horizon}/{cost_mode}/{run_root.name}")
                continue
            predictions = _prediction_paths(model_root, run_root, routing_run_dir, models, force)
            print(f"[RUN] routing {target}/k{horizon}/{cost_mode}/{run_root.name}")
            evaluate_run(
                run_root.name,
                None,
                horizon=horizon,
                output_dir=routing_run_dir,
                prediction_csvs=predictions,
                split_csv=run_root / "splits" / "time_splits.csv",
                raw_root=raw_root,
                weight_mode=cost_mode,
                include_olsr=False,
                target=target,
                prediction_horizon=horizon,
            )

    return aggregate_multihorizon_results(routing_root, reports_root)


def aggregate_multihorizon_results(
    routing_root: Path,
    reports_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for summary_csv in sorted(routing_root.glob("*/k*/*/*/summary.csv")):
        try:
            frame = pd.read_csv(summary_csv)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No routing summaries found under {routing_root}")

    detailed = pd.concat(frames, ignore_index=True)
    coordinates = ["target", "prediction_horizon", "weight_mode"]
    required = {"run_name", "strategy", "n_sessions", *coordinates, *ROUTING_METRICS}
    missing = required.difference(detailed.columns)
    if missing:
        raise ValueError(f"Routing summaries are missing columns: {sorted(missing)}")
    session_counts = detailed.groupby([*coordinates, "run_name"])["n_sessions"].nunique()
    if (session_counts > 1).any():
        raise ValueError("Strategies were not evaluated on the same number of sessions")

    aggregate_rows = []
    paired_frames = []
    for coordinate, group in detailed.groupby(coordinates, sort=True):
        context = dict(zip(coordinates, coordinate, strict=True))
        for strategy, strategy_group in group.groupby("strategy", sort=True):
            row = {
                **context,
                "strategy": strategy,
                "n_runs": int(strategy_group["run_name"].nunique()),
                "n_sessions": int(strategy_group["n_sessions"].sum()),
            }
            for metric in ROUTING_METRICS:
                values = strategy_group[metric].to_numpy(dtype=float)
                ci_low, ci_high = bootstrap_mean_ci(values, seed=42)
                row[f"{metric}_mean"] = float(strategy_group[metric].mean())
                row[f"{metric}_std"] = float(strategy_group[metric].std())
                row[f"{metric}_ci95_low"] = ci_low
                row[f"{metric}_ci95_high"] = ci_high
            aggregate_rows.append(row)

        strategies = set(group["strategy"])
        comparisons = [pair for pair in COMPARISONS if pair[0] in strategies and pair[1] in strategies]
        paired = pd.concat(
            [
                paired_comparisons(group, comparisons, {metric: higher_is_better}, seed=42)
                for metric, higher_is_better in ROUTING_METRICS.items()
            ],
            ignore_index=True,
        )
        for key, value in context.items():
            paired[key] = value
        paired_frames.append(paired)

    aggregate = pd.DataFrame(aggregate_rows).sort_values([*coordinates, "strategy"])
    paired = pd.concat(paired_frames, ignore_index=True)
    detailed = detailed.sort_values([*coordinates, "run_name", "strategy"])
    reports_root.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(reports_root / "detailed_by_run.csv", index=False)
    aggregate.to_csv(reports_root / "summary_by_strategy.csv", index=False)
    paired.to_csv(reports_root / "paired_comparisons.csv", index=False)
    return detailed, aggregate, paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/multihorizon"))
    parser.add_argument("--model-root", type=Path, default=Path("outputs/multihorizon"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_snapshots"))
    parser.add_argument("--routing-root", type=Path, default=Path("outputs/routing_multihorizon"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/routing_multihorizon"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=None)
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=None)
    parser.add_argument("--cost-modes", nargs="+", choices=COST_MODES, default=list(COST_MODES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detailed, aggregate, paired = run_multihorizon_routing(
        args.data_root,
        args.model_root,
        args.raw_root,
        args.routing_root,
        args.reports_root,
        args.models,
        set(args.targets) if args.targets else None,
        set(args.horizons) if args.horizons else None,
        args.cost_modes,
        args.limit,
        args.force,
    )
    print(f"[OK] {len(detailed)} run/strategy rows")
    print(f"[OK] {len(aggregate)} aggregate rows")
    print(f"[OK] {len(paired)} paired-comparison rows")
    print(f"[OK] reports: {args.reports_root}")


if __name__ == "__main__":
    main()

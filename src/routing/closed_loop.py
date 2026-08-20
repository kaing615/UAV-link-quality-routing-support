"""Build route plans and run trace-driven closed-loop routing in ns-3."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from src.evaluation.paired_statistics import bootstrap_mean_ci, paired_comparisons
from src.routing.predict_edges import attach_prediction_identity
from src.routing.replay_eval import (
    build_strategy_graph,
    load_prediction_scores,
    load_raw_edges,
    load_test_times,
    path_edges,
    shortest_path,
)

CLOSED_LOOP_METRICS = {
    "pdr": True,
    "mean_delay_ms": False,
    "throughput_mbps": True,
    "route_changes": False,
    "route_found_rate": True,
}

CLOSED_LOOP_COMPARISONS = (
    ("hop", "olsr"),
    ("delay", "olsr"),
    ("persistence", "olsr"),
    ("logreg", "olsr"),
    ("xgb", "olsr"),
    ("edge-sage", "olsr"),
    ("logreg", "hop"),
    ("xgb", "hop"),
    ("edge-sage", "hop"),
    ("edge-sage", "xgb"),
    ("edge-sage", "logreg"),
)

STRATEGIES = ("olsr", "hop", "delay", "persistence", "logreg", "xgb", "edge-sage")


def build_ns3_command(
    scenario: dict,
    *,
    binary: Path,
    route_plan: Path,
    output_dir: Path,
    strategy: str,
    target: str,
    horizon: int,
    cost_mode: str,
    source: int,
    destination: int,
    app_rate_kbps: float,
    packet_size: int,
) -> list[str]:
    """Translate a recorded scenario into one comparable ns-3 experiment."""
    speed_range = scenario.get("rwp_speed_range", [3.0, 8.0])
    return [
        str(binary),
        f"--runName={scenario['run_name']}_closed_{strategy}",
        f"--numUavs={scenario['num_uavs']}",
        f"--timeSteps={scenario['time_steps']}",
        f"--seed={scenario['seed']}",
        f"--mobility={scenario['mobility_model']}",
        f"--xMax={scenario['x_limit'][1]}",
        f"--yMax={scenario['y_limit'][1]}",
        f"--zMin={scenario['z_limit'][0]}",
        f"--zMax={scenario['z_limit'][1]}",
        f"--commRange={scenario['comm_range']}",
        f"--txPower={scenario.get('tx_power_dbm', 20.0)}",
        f"--refLoss={scenario.get('reference_path_loss_db', 40.0)}",
        f"--pathLossExp={scenario.get('path_loss_exponent', 2.2)}",
        f"--noiseFloor={scenario.get('noise_floor_dbm', -90.0)}",
        f"--gmAlpha={scenario.get('gauss_markov_alpha', 0.85)}",
        f"--rwpSpeedMin={speed_range[0]}",
        f"--rwpSpeedMax={speed_range[1]}",
        f"--warmup={scenario.get('warmup_s', 10.0)}",
        f"--sourceId={source}",
        f"--destId={destination}",
        f"--outputDir={output_dir}",
        "--enableAnim=false",
        "--enableDataFlow=true",
        f"--routePlan={route_plan}",
        f"--routingStrategy={strategy}",
        f"--predictionTarget={target}",
        f"--predictionHorizon={horizon}",
        f"--costMode={cost_mode}",
        f"--appRateKbps={app_rate_kbps}",
        f"--appPacketSize={packet_size}",
    ]


def build_route_plan(
    run_name: str,
    strategy: str,
    *,
    source: int,
    destination: int,
    horizon: int,
    weight_mode: str,
    raw_root: Path,
    split_csv: Path,
    output_csv: Path,
    predictions_csv: Path | None = None,
) -> Path:
    """Write the per-snapshot path that ns-3 will install as static routes."""
    if strategy not in {"hop", "delay"} and predictions_csv is None:
        raise ValueError(f"predictions_csv is required for strategy {strategy!r}")
    raw = load_raw_edges(run_name, raw_root)
    if not raw:
        raise ValueError(f"No connected raw edges found for {run_name}")
    max_time = max(raw)
    scores = {strategy: load_prediction_scores(predictions_csv)} if predictions_csv is not None else {}

    rows = []
    for time in load_test_times(run_name, split_csv):
        if time + horizon > max_time:
            continue
        graph = build_strategy_graph(raw.get(time, {}), strategy, time, scores, 0.0, weight_mode)
        path = shortest_path(graph, source, destination)
        rows.append(
            {
                "time": time,
                "source": source,
                "destination": destination,
                "route_found": int(path is not None),
                "route_path": "" if path is None else "->".join(map(str, path)),
                "route_cost": float("nan")
                if path is None
                else sum(float(graph.edges[edge]["weight"]) for edge in path_edges(path)),
                "strategy": strategy,
                "prediction_horizon": horizon,
                "weight_mode": weight_mode,
            }
        )
    if not rows:
        raise ValueError(f"No eligible test snapshots for {run_name} at k={horizon}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def aggregate_closed_loop_results(
    output_root: Path,
    reports_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate measured ns-3 flow metrics and pair strategies by seed/run."""
    frames = []
    for metrics_csv in sorted(output_root.glob("*/*/k*/*/*/closed_loop_metrics.csv")):
        frame = pd.read_csv(metrics_csv)
        if frame.empty:
            continue
        frame["baseline_run"] = metrics_csv.parents[4].name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No closed-loop metrics found under {output_root}")

    detailed = pd.concat(frames, ignore_index=True)
    coordinates = [
        "target",
        "horizon",
        "cost_mode",
        "source",
        "destination",
        "app_rate_kbps",
        "packet_size",
    ]
    required = {"baseline_run", "strategy", *coordinates, *CLOSED_LOOP_METRICS}
    missing = required.difference(detailed.columns)
    if missing:
        raise ValueError(f"Closed-loop metrics are missing columns: {sorted(missing)}")

    summary_rows = []
    paired_frames = []
    for coordinate, group in detailed.groupby(coordinates, sort=True):
        context = dict(zip(coordinates, coordinate, strict=True))
        for strategy, strategy_group in group.groupby("strategy", sort=True):
            row = {
                **context,
                "strategy": strategy,
                "n_runs": int(strategy_group["baseline_run"].nunique()),
            }
            for metric in CLOSED_LOOP_METRICS:
                values = strategy_group[metric].to_numpy(dtype=float)
                ci_low, ci_high = bootstrap_mean_ci(values, seed=42)
                row[f"{metric}_mean"] = float(strategy_group[metric].mean())
                row[f"{metric}_std"] = float(strategy_group[metric].std())
                row[f"{metric}_ci95_low"] = ci_low
                row[f"{metric}_ci95_high"] = ci_high
            summary_rows.append(row)

        paired_input = group.rename(columns={"baseline_run": "run_name"})
        strategies = set(group["strategy"])
        comparisons = [pair for pair in CLOSED_LOOP_COMPARISONS if pair[0] in strategies and pair[1] in strategies]
        if comparisons:
            paired = pd.concat(
                [
                    paired_comparisons(
                        paired_input,
                        comparisons,
                        {metric: higher_is_better},
                        seed=42,
                    )
                    for metric, higher_is_better in CLOSED_LOOP_METRICS.items()
                ],
                ignore_index=True,
            )
            for key, value in context.items():
                paired[key] = value
            paired_frames.append(paired)

    summary = pd.DataFrame(summary_rows).sort_values([*coordinates, "strategy"])
    paired = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    detailed = detailed.sort_values([*coordinates, "baseline_run", "strategy"])
    reports_root.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(reports_root / "detailed.csv", index=False)
    summary.to_csv(reports_root / "summary.csv", index=False)
    paired.to_csv(reports_root / "paired_comparisons.csv", index=False)
    return detailed, summary, paired


def _prediction_csv(
    strategy: str,
    *,
    model_root: Path,
    run_root: Path,
    output_dir: Path,
    target: str,
    horizon: int,
    force: bool,
) -> Path | None:
    if strategy in {"olsr", "hop", "delay"}:
        return None
    source = model_root / strategy / target / f"k{horizon}" / run_root.name / "test_predictions.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing {strategy} predictions: {source}")
    if strategy != "edge-sage":
        return source
    identified = output_dir / "predictions_edge-sage.csv"
    if force or not identified.exists():
        attach_prediction_identity(source, run_root / "graph_dataset" / "test.pt", identified)
    return identified


def run_closed_loop_experiment(
    baseline_run: str,
    *,
    binary: Path,
    data_root: Path,
    model_root: Path,
    raw_root: Path,
    output_root: Path,
    reports_root: Path,
    strategies: list[str],
    target: str,
    horizon: int,
    cost_mode: str,
    source: int | None,
    destination: int | None,
    app_rate_kbps: float,
    packet_size: int,
    plan_only: bool,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    scenario_path = raw_root / baseline_run / "scenario.json"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario metadata: {scenario_path}")
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    source = int(scenario.get("source_id", 0) if source is None else source)
    destination = int(scenario.get("dest_id", 1) if destination is None else destination)
    run_root = data_root / target / f"k{horizon}" / baseline_run
    split_csv = run_root / "splits" / "time_splits.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing multi-horizon split: {split_csv}")
    if not plan_only and not binary.exists():
        raise FileNotFoundError(f"ns-3 binary not found: {binary}")

    for strategy in strategies:
        output_dir = output_root / baseline_run / target / f"k{horizon}" / cost_mode / strategy
        metrics_csv = output_dir / "closed_loop_metrics.csv"
        if metrics_csv.exists() and not force and not plan_only:
            print(f"[SKIP] closed loop {baseline_run}/{target}/k{horizon}/{cost_mode}/{strategy}")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_csv = _prediction_csv(
            strategy,
            model_root=model_root,
            run_root=run_root,
            output_dir=output_dir,
            target=target,
            horizon=horizon,
            force=force,
        )
        plan_strategy = "hop" if strategy == "olsr" else strategy
        plan_csv = build_route_plan(
            baseline_run,
            plan_strategy,
            source=source,
            destination=destination,
            horizon=horizon,
            weight_mode=cost_mode,
            raw_root=raw_root,
            split_csv=split_csv,
            predictions_csv=predictions_csv,
            output_csv=output_dir / "route_plan.csv",
        )
        if plan_only:
            print(f"[PLAN] {strategy}: {plan_csv}")
            continue
        command = build_ns3_command(
            scenario,
            binary=binary,
            route_plan=plan_csv,
            output_dir=output_dir,
            strategy=strategy,
            target=target,
            horizon=horizon,
            cost_mode=cost_mode,
            source=source,
            destination=destination,
            app_rate_kbps=app_rate_kbps,
            packet_size=packet_size,
        )
        print(f"[RUN] closed loop {baseline_run}/{target}/k{horizon}/{cost_mode}/{strategy}")
        subprocess.run(command, check=True)
        if not metrics_csv.exists():
            raise RuntimeError(f"ns-3 did not write expected metrics: {metrics_csv}")

    if plan_only:
        return None
    return aggregate_closed_loop_results(output_root, reports_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("simulation/ns3/build/uav-olsr-dataset"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/multihorizon"))
    parser.add_argument("--model-root", type=Path, default=Path("outputs/multihorizon"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_snapshots"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/closed_loop"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/closed_loop"))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--target", choices=("qos", "survival"), default="survival")
    parser.add_argument("--horizon", type=int, choices=(1, 2, 3, 5), default=3)
    parser.add_argument("--cost-mode", choices=("neglog", "one-minus"), default="neglog")
    parser.add_argument("--source", type=int, default=None)
    parser.add_argument("--destination", type=int, default=None)
    parser.add_argument("--app-rate-kbps", type=float, default=256.0)
    parser.add_argument("--packet-size", type=int, default=512)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_closed_loop_experiment(
        args.baseline_run,
        binary=args.binary,
        data_root=args.data_root,
        model_root=args.model_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
        reports_root=args.reports_root,
        strategies=args.strategies,
        target=args.target,
        horizon=args.horizon,
        cost_mode=args.cost_mode,
        source=args.source,
        destination=args.destination,
        app_rate_kbps=args.app_rate_kbps,
        packet_size=args.packet_size,
        plan_only=args.plan_only,
        force=args.force,
    )
    if result is not None:
        detailed, summary, paired = result
        print(f"[OK] {len(detailed)} measured strategy/run rows")
        print(f"[OK] {len(summary)} aggregate rows")
        print(f"[OK] {len(paired)} paired-comparison rows")


if __name__ == "__main__":
    main()

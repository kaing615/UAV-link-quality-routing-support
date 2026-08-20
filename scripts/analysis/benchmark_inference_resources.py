"""Benchmark CPU inference latency, memory, artifact size, and model complexity."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import psutil

from src.evaluation.paired_statistics import bootstrap_mean_ci
from src.training.baselines.common import FEATURE_COLUMNS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ("logreg", "xgb", "edge-sage")
MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "xgb": "XGBoost",
    "edge-sage": "Edge-SAGE",
}
MODEL_COLORS = {"logreg": "#2ca02c", "xgb": "#ff7f0e", "edge-sage": "#286aaa"}
SUMMARY_METRICS = (
    "latency_ms_per_sample_median",
    "latency_ms_per_snapshot_median",
    "latency_ms_per_sample_p95",
    "peak_rss_mb",
    "checkpoint_mb",
    "complexity_count",
)


def artifact_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 6)


def model_complexity(model: object, model_id: str) -> tuple[int, str]:
    if model_id == "edge-sage":
        import torch

        if not isinstance(model, torch.nn.Module):
            raise TypeError("edge-sage complexity requires a torch module")
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "trainable_parameters"
    if model_id == "logreg" and hasattr(model, "coef_"):
        return int(model.coef_.size + model.intercept_.size), "coefficients_plus_intercept"
    if model_id == "xgb" and hasattr(model, "get_booster"):
        return int(len(model.get_booster().trees_to_dataframe())), "tree_nodes"
    raise ValueError(f"Unsupported model complexity definition: {model_id}")


def aggregate_benchmarks(detailed: pd.DataFrame) -> pd.DataFrame:
    required = {"model_id", "run_name", *SUMMARY_METRICS}
    missing = required.difference(detailed.columns)
    if missing:
        raise ValueError(f"Benchmark details missing columns: {sorted(missing)}")
    rows = []
    for model_id, group in detailed.groupby("model_id", sort=True):
        row: dict[str, object] = {
            "model_id": model_id,
            "n_runs": int(group["run_name"].nunique()),
        }
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, seed=42)
            row[f"{metric}_mean"] = round(float(values.mean()), 12)
            row[f"{metric}_std"] = round(float(values.std(ddof=1)), 12) if len(values) > 1 else 0.0
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model_id").reset_index(drop=True)


def _monitor_peak_rss(stop: threading.Event, peak: list[int]) -> None:
    process = psutil.Process()
    while not stop.wait(0.002):
        peak[0] = max(peak[0], process.memory_info().rss)


def _load_model_and_predictor(
    model_id: str,
    run_name: str,
    target: str,
    horizon: int,
    data_root: Path,
    output_root: Path,
    threads: int,
) -> tuple[object, Path, int, int, Callable[[], None]]:
    model_dir = output_root / model_id / target / f"k{horizon}" / run_name
    run_root = data_root / target / f"k{horizon}" / run_name
    if model_id in {"logreg", "xgb"}:
        artifact = model_dir / "model.pkl"
        with artifact.open("rb") as handle:
            model = pickle.load(handle)
        if model_id == "xgb":
            model.set_params(n_jobs=threads)
        frame = pd.read_csv(run_root / "baseline_standardized" / "test_scaled.csv")
        features = frame[FEATURE_COLUMNS]
        n_samples = len(frame)
        n_snapshots = int(frame["time"].nunique()) if "time" in frame else 1

        def predict() -> None:
            model.predict_proba(features)

        return model, artifact, n_samples, n_snapshots, predict

    import torch

    from src.models.gnn.edge_gnn import EdgeAwareSAGEEdgeClassifier
    from src.training.gnn.common import load_graphs

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    artifact = model_dir / "best_model.pt"
    model = EdgeAwareSAGEEdgeClassifier(
        node_in_channels=8,
        edge_in_channels=7,
        hidden_channels=int(metadata["hidden_channels"]),
        num_layers=int(metadata["num_layers"]),
        dropout=float(metadata["dropout"]),
        use_edge_features=bool(metadata["use_edge_features"]),
    )
    model.load_state_dict(torch.load(artifact, map_location="cpu", weights_only=True))
    model.eval()
    graphs = load_graphs(run_root / "graph_dataset" / "test.pt")
    n_samples = sum(int(graph.edge_label.numel()) for graph in graphs)
    n_snapshots = len(graphs)

    def predict() -> None:
        with torch.no_grad():
            for graph in graphs:
                model(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    graph.edge_label_index,
                    graph.labeled_edge_attr,
                )

    return model, artifact, n_samples, n_snapshots, predict


def benchmark_one(
    model_id: str,
    run_name: str,
    *,
    target: str,
    horizon: int,
    data_root: Path,
    output_root: Path,
    warmups: int,
    repeats: int,
    threads: int,
) -> dict[str, object]:
    if model_id == "edge-sage":
        import torch

        torch.set_num_threads(threads)
    model, artifact, n_samples, n_snapshots, predict = _load_model_and_predictor(
        model_id, run_name, target, horizon, data_root, output_root, threads
    )
    process = psutil.Process()
    rss_loaded = process.memory_info().rss
    stop = threading.Event()
    peak = [rss_loaded]
    monitor = threading.Thread(target=_monitor_peak_rss, args=(stop, peak), daemon=True)
    monitor.start()
    try:
        for _ in range(warmups):
            predict()
        durations_ms = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            predict()
            durations_ms.append((time.perf_counter_ns() - start) / 1_000_000)
    finally:
        stop.set()
        monitor.join()
    complexity, definition = model_complexity(model, model_id)
    per_sample = np.asarray(durations_ms, dtype=float) / max(n_samples, 1)
    per_snapshot = np.asarray(durations_ms, dtype=float) / max(n_snapshots, 1)
    return {
        "model_id": model_id,
        "run_name": run_name,
        "target": target,
        "horizon": horizon,
        "threads": threads,
        "warmups": warmups,
        "repeats": repeats,
        "n_samples": n_samples,
        "n_snapshots": n_snapshots,
        "latency_ms_per_sample_median": float(np.median(per_sample)),
        "latency_ms_per_sample_p95": float(np.percentile(per_sample, 95)),
        "latency_ms_per_snapshot_median": float(np.median(per_snapshot)),
        "peak_rss_mb": peak[0] / (1024 * 1024),
        "rss_loaded_mb": rss_loaded / (1024 * 1024),
        "inference_rss_delta_mb": max(0.0, (peak[0] - rss_loaded) / (1024 * 1024)),
        "checkpoint_mb": artifact_size_mb(artifact),
        "complexity_count": complexity,
        "complexity_definition": definition,
    }


def _write_outputs(detailed: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(output_dir / "detailed.csv", index=False)
    summary = aggregate_benchmarks(detailed)
    summary.to_csv(output_dir / "summary.csv", index=False)
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    summary.to_csv(tables / "inference_resources.csv", index=False)
    printable = summary[[
        "model_id",
        "n_runs",
        "latency_ms_per_sample_median_mean",
        "latency_ms_per_snapshot_median_mean",
        "peak_rss_mb_mean",
        "checkpoint_mb_mean",
        "complexity_count_mean",
    ]].copy()
    printable["Model"] = printable["model_id"].map(MODEL_LABELS)
    printable = printable.drop(columns="model_id")
    latex = printable.to_latex(
        index=False,
        escape=True,
        caption="CPU inference resource benchmark on ten runs (one thread).",
        label="tab:inference-resources",
        position="tb",
    )
    (tables / "inference_resources.tex").write_text(
        latex.replace("\\centering\n", "\\centering\n\\scriptsize\n", 1), encoding="utf-8"
    )

    ordered = summary.set_index("model_id").reindex(MODELS).dropna(how="all").reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    specs = [
        ("latency_ms_per_sample_median", "Median latency (ms/edge)"),
        ("peak_rss_mb", "Peak RSS (MB)"),
        ("checkpoint_mb", "Artifact size (MB)"),
    ]
    for ax, (metric, ylabel) in zip(axes, specs, strict=True):
        means = ordered[f"{metric}_mean"].to_numpy(dtype=float)
        low = ordered[f"{metric}_ci95_low"].to_numpy(dtype=float)
        high = ordered[f"{metric}_ci95_high"].to_numpy(dtype=float)
        labels = [MODEL_LABELS[model] for model in ordered["model_id"]]
        colors = [MODEL_COLORS[model] for model in ordered["model_id"]]
        errors = np.maximum(0.0, np.vstack([means - low, high - means]))
        ax.bar(labels, means, color=colors, yerr=errors, capsize=3)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("CPU inference cost (survival, k=3, one thread)")
    fig.tight_layout()
    fig.savefig(figures / "inference_resources.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "inference_resources.pdf", bbox_inches="tight")
    plt.close(fig)


def run_benchmark(args: argparse.Namespace) -> None:
    runs = args.runs or sorted(
        path.name for path in (args.output_root / "edge-sage" / args.target / f"k{args.horizon}").iterdir() if path.is_dir()
    )
    detail_dir = args.report_dir / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    for model_id in args.models:
        for run_name in runs:
            result_json = detail_dir / f"{model_id}__{run_name}.json"
            if result_json.exists() and not args.force:
                print(f"[SKIP] {model_id}/{run_name}")
                continue
            command = [
                sys.executable,
                "-m",
                "scripts.analysis.benchmark_inference_resources",
                "--worker",
                "--model",
                model_id,
                "--run",
                run_name,
                "--target",
                args.target,
                "--horizon",
                str(args.horizon),
                "--data-root",
                str(args.data_root),
                "--output-root",
                str(args.output_root),
                "--warmups",
                str(args.warmups),
                "--repeats",
                str(args.repeats),
                "--threads",
                str(args.threads),
            ]
            print(f"[RUN] {model_id}/{run_name}")
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            result_json.write_text(completed.stdout.strip().splitlines()[-1], encoding="utf-8")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(detail_dir.glob("*.json"))]
    _write_outputs(pd.DataFrame(rows), args.report_dir)
    print(f"[OK] wrote {len(rows)} run/model resource measurements to {args.report_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--run")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--target", choices=("qos", "survival"), default="survival")
    parser.add_argument("--horizon", type=int, choices=(1, 2, 3, 5), default=3)
    parser.add_argument("--data-root", type=Path, default=Path("data/multihorizon"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/multihorizon"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/inference_benchmark"))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if not args.model or not args.run:
            raise SystemExit("--worker requires --model and --run")
        result = benchmark_one(
            args.model,
            args.run,
            target=args.target,
            horizon=args.horizon,
            data_root=args.data_root,
            output_root=args.output_root,
            warmups=args.warmups,
            repeats=args.repeats,
            threads=args.threads,
        )
        print(json.dumps(result))
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()

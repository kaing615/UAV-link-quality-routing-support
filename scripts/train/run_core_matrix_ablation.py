"""Run a held-seed Edge-SAGE ablation on the verified core matrix."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.experiment import cache_matches, fingerprint, save_manifest, seed_group, validate_test_predictions


MODES = ("decoder-only", "message-only", "noedge")
TARGETS = ("qos", "survival")
HORIZONS = (1, 2, 3, 5)


def make_command(
    data_root: Path,
    train_runs: list[str],
    test_runs: list[str],
    output_dir: Path,
    edge_mode: str,
    epochs: int,
    patience: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.training.gnn.train_gnn_loro",
        "--train-runs",
        ",".join(train_runs),
        "--test-runs",
        ",".join(test_runs),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--model",
        "edge-sage",
        "--edge-mode",
        edge_mode,
        "--hidden",
        "128",
        "--num-layers",
        "2",
        "--dropout",
        "0.3",
        "--lr",
        "5e-4",
        "--weight-decay",
        "1e-4",
        "--lr-scheduler",
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--include-purged-test",
    ]


def run_fold(
    *,
    target: str,
    horizon: int,
    seed: int,
    train_runs: list[str],
    test_runs: list[str],
    data_root: Path,
    output_root: Path,
    plan_path: Path,
    modes: list[str],
    epochs: int,
    patience: int,
    force: bool,
    dry_run: bool,
) -> None:
    coordinate = data_root / target / f"k{horizon}"
    data_paths = [
        coordinate / run / part
        for run in [*train_runs, *test_runs]
        for part in ("features", "splits", "graph_dataset")
    ]
    for mode in modes:
        output_dir = output_root / mode / target / f"k{horizon}" / f"seed{seed}"
        command = make_command(coordinate, train_runs, test_runs, output_dir, mode, epochs, patience)
        if dry_run:
            print(f"[PLAN] {mode}/{target}/k{horizon}/seed{seed}: {len(train_runs)} train, {len(test_runs)} test")
            continue
        manifest = fingerprint(
            {
                "command": command,
                "protocol": "held-seed-core-matrix-ablation",
                "held_seed": seed,
                "plan_sha256": fingerprint({"plan": "core-matrix"}, [plan_path])["fingerprint"],
            },
            [*data_paths, Path(__file__), Path("src/training"), Path("src/models"), Path("src/utils")],
        )
        required = [
            output_dir / "metadata.json",
            output_dir / "best_model.pt",
            output_dir / "metrics.csv",
            *(output_dir / f"test_predictions_{run}.csv" for run in test_runs),
        ]
        manifest_path = output_dir / "experiment.json"
        if force or not cache_matches(manifest_path, manifest, required):
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "training.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, check=True, stdout=log, stderr=log)
            for run in test_runs:
                validate_test_predictions(
                    coordinate / run,
                    output_dir / f"test_predictions_{run}.csv",
                    test_splits=("train", "val", "test", "purged"),
                )
            save_manifest(manifest_path, manifest)
        print(f"[DONE] {mode}/{target}/k{horizon}/seed{seed}", flush=True)


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.maximum.accumulate(p_values[order] * np.arange(len(p_values), 0, -1))
    result = np.empty_like(p_values)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def summarize(
    *,
    plan: pd.DataFrame,
    reference_path: Path,
    output_root: Path,
    report_root: Path,
    modes: list[str],
    targets: list[str],
    horizons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(reference_path)
    full = full[full.model_id == "edge-sage"]
    plan_columns = plan[["run_name", "seed", "condition"]]
    full = full.merge(plan_columns, on=["run_name", "seed"], how="left", validate="many_to_one")
    if full.condition.isna().any():
        raise ValueError("Core-matrix reference rows do not map to the saved plan")

    frames = [full.assign(model_id="edge-sage-full")]
    for mode in modes:
        for target in targets:
            for horizon in horizons:
                for seed in sorted(plan.seed.unique()):
                    fold = output_root / mode / target / f"k{horizon}" / f"seed{seed}"
                    metrics = pd.read_csv(fold / "metrics.csv")
                    metrics = metrics[(metrics.split == "test") & (metrics.run_name != "__validation__")].copy()
                    expected = set(plan.loc[plan.seed == seed, "run_name"])
                    if set(metrics.run_name) != expected or metrics.run_name.duplicated().any():
                        raise ValueError(f"Incomplete test-run metrics for {mode}/{target}/k{horizon}/seed{seed}")
                    metrics["false_stable"] = metrics.fp / (metrics.tn + metrics.fp).replace(0, np.nan)
                    metrics["target"] = target
                    metrics["horizon"] = horizon
                    metrics["seed"] = seed
                    metrics["model_id"] = f"edge-sage-{mode}"
                    metrics["condition"] = metrics.run_name.map(plan.set_index("run_name").condition)
                    frames.append(metrics)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    selected = combined[
        combined.target.isin(targets)
        & combined.horizon.isin(horizons)
        & combined.model_id.isin(["edge-sage-full", *(f"edge-sage-{m}" for m in modes)])
    ].copy()
    selected.to_csv(report_root / "ablation_metrics_runs.csv", index=False)

    rng = np.random.default_rng(42)
    signs = np.array(list(itertools.product((-1, 1), repeat=10)))
    comparisons = []
    metrics_to_compare = [metric for metric in ("macro_f1", "false_stable", "pr_auc") if metric in selected]
    for metric in metrics_to_compare:
        for target in targets:
            for horizon in horizons:
                block = selected[(selected.target == target) & (selected.horizon == horizon)]
                pivot = block.pivot(index=["seed", "condition"], columns="model_id", values=metric)
                for mode in modes:
                    model = f"edge-sage-{mode}"
                    delta = (pivot["edge-sage-full"] - pivot[model]).dropna()
                    by_seed = delta.groupby("seed").mean().reindex(sorted(plan.seed.unique())).to_numpy()
                    if len(by_seed) != 10 or not np.isfinite(by_seed).all():
                        raise ValueError(f"Expected 10 complete paired seeds for {metric}/{target}/k{horizon}/{mode}")
                    samples = rng.choice(by_seed, (20000, len(by_seed)), replace=True).mean(axis=1)
                    low, high = np.quantile(samples, [0.025, 0.975])
                    p_value = float((np.abs((signs * by_seed).mean(axis=1)) >= abs(by_seed.mean()) - 1e-15).mean())
                    comparisons.append(
                        {
                            "metric": metric,
                            "target": target,
                            "horizon": horizon,
                            "ablation": mode,
                            "delta_full_minus_ablation": float(by_seed.mean()),
                            "ci95_low": float(low),
                            "ci95_high": float(high),
                            "n_seed_units": len(by_seed),
                            "exact_two_sided_sign_flip_p": p_value,
                            "conditions_won": int((delta.groupby("condition").mean() > 1e-12).sum()),
                        }
                    )
    effects = pd.DataFrame(comparisons)
    effects["holm_p_within_metric"] = np.nan
    for _, indexes in effects.groupby("metric").groups.items():
        effects.loc[indexes, "holm_p_within_metric"] = holm(
            effects.loc[indexes, "exact_two_sided_sign_flip_p"].to_numpy()
        )
    effects.to_csv(report_root / "ablation_paired_effects.csv", index=False)
    return selected, effects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("reports/core_matrix_bursty_fixed/plan.csv"))
    parser.add_argument("--reference-metrics", type=Path, default=Path("reports/core_matrix_bursty_fixed/metrics_runs.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/core_matrix_bursty_fixed/multihorizon"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/core_matrix_bursty_fixed/ablation"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/dacn_verification_2026-11-10"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = pd.read_csv(args.plan)
    if len(plan) != 210 or plan.run_name.duplicated().any() or plan.seed.nunique() != 10:
        raise ValueError("Expected the verified 210-run / 10-seed core-matrix plan")
    args.report_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for target in args.targets:
        for horizon in args.horizons:
            for seed in sorted(plan.seed.unique()):
                test_runs = sorted(plan.loc[plan.seed == seed, "run_name"].tolist())
                train_runs = sorted(plan.loc[plan.seed != seed, "run_name"].tolist())
                if len(train_runs) != 189 or len(test_runs) != 21:
                    raise ValueError("Held-seed split must use 189 train and 21 test runs")
                if {seed_group(run) for run in train_runs} & {seed_group(run) for run in test_runs}:
                    raise ValueError(f"Seed leakage in held-out fold {seed}")
                run_fold(
                    target=target,
                    horizon=horizon,
                    seed=int(seed),
                    train_runs=train_runs,
                    test_runs=test_runs,
                    data_root=args.data_root,
                    output_root=args.output_root,
                    plan_path=args.plan,
                    modes=args.modes,
                    epochs=args.epochs,
                    patience=args.patience,
                    force=args.force,
                    dry_run=args.dry_run,
                )
                total += len(args.modes)
    if args.dry_run:
        print(f"[PLAN] {total} fold-model runs; no training started")
        return
    selected, effects = summarize(
        plan=plan,
        reference_path=args.reference_metrics,
        output_root=args.output_root,
        report_root=args.report_root,
        modes=args.modes,
        targets=args.targets,
        horizons=args.horizons,
    )
    print(f"[SUMMARY] {len(selected)} run-model metric rows; {len(effects)} paired effects")
    print(effects[effects.metric == "macro_f1"].to_string(index=False))


if __name__ == "__main__":
    main()

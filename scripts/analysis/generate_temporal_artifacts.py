"""Generate strict run-paired temporal-versus-static benchmark artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from src.evaluation.paired_statistics import paired_comparisons

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COORDINATES = ["target", "horizon", "run_name", "split"]
METRICS = {"macro_f1": True, "pr_auc": True}


def _require(frame: pd.DataFrame, name: str) -> None:
    missing = set(COORDINATES).union(METRICS).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _coordinate_index(frame: pd.DataFrame, name: str) -> pd.MultiIndex:
    _require(frame, name)
    duplicates = frame.duplicated(COORDINATES, keep=False)
    if duplicates.any():
        values = frame.loc[duplicates, COORDINATES].to_dict("records")
        raise ValueError(f"{name} contains duplicate coordinates: {values}")
    return pd.MultiIndex.from_frame(frame[COORDINATES])


def _model_rows(frame: pd.DataFrame, model_id: str, name: str) -> pd.DataFrame:
    if "model_id" not in frame:
        return frame
    result = frame[frame["model_id"] == model_id].copy()
    if result.empty:
        raise ValueError(f"{name} has no {model_id} rows")
    return result


def paired_detail(static: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    """Return one static/temporal row per validated run coordinate."""
    static_index = _coordinate_index(static, "static metrics")
    temporal_index = _coordinate_index(temporal, "temporal metrics")
    missing_temporal = static_index.difference(temporal_index)
    if len(missing_temporal):
        raise ValueError(f"missing temporal coordinates: {missing_temporal.tolist()}")
    missing_static = temporal_index.difference(static_index)
    if len(missing_static):
        raise ValueError(f"missing static coordinates: {missing_static.tolist()}")
    return static[COORDINATES + list(METRICS)].merge(
        temporal[COORDINATES + list(METRICS)],
        on=COORDINATES,
        suffixes=("_static", "_temporal"),
        validate="one_to_one",
    ).assign(
        macro_f1_delta=lambda frame: frame["macro_f1_temporal"] - frame["macro_f1_static"],
        pr_auc_delta=lambda frame: frame["pr_auc_temporal"] - frame["pr_auc_static"],
    )


def temporal_effects(
    static: pd.DataFrame,
    temporal: pd.DataFrame,
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute temporal-minus-static effects after exact coordinate validation."""
    paired_detail(static, temporal)
    rows = []
    combined = pd.concat(
        [
            static.assign(strategy="static"),
            temporal.assign(strategy="temporal"),
        ],
        ignore_index=True,
    )
    for coordinates, group in combined.groupby(["target", "horizon", "split"], sort=True):
        result = paired_comparisons(
            group,
            comparisons=[("temporal", "static")],
            metrics=METRICS,
            n_resamples=n_resamples,
            seed=seed,
        )
        for column, value in zip(("target", "horizon", "split"), coordinates, strict=True):
            result[column] = value
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_latex(effects: pd.DataFrame, path: Path) -> None:
    printable = effects[effects["split"] == "test"].copy()
    printable["Delta [95% CI]"] = printable.apply(
        lambda row: f"{row['mean_delta']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]", axis=1
    )
    printable = printable.rename(columns={"target": "Target", "horizon": "k", "metric": "Metric", "n_pairs": "Runs", "p_holm": "Holm p"})[
        ["Target", "k", "Metric", "Runs", "Delta [95% CI]", "Holm p"]
    ]
    latex = printable.to_latex(index=False, escape=True, caption="Temporal minus static run-paired effects.", label="tab:temporal-effects", position="tb")
    path.write_text(latex.replace("\\centering\n", "\\centering\n\\scriptsize\n", 1), encoding="utf-8")


def _plot(effects: pd.DataFrame, output_stem: Path) -> None:
    test = effects[effects["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for axis, metric in zip(axes, METRICS, strict=True):
        subset = test[test["metric"] == metric].sort_values(["target", "horizon"])
        labels = [f"{row.target}\nk={row.horizon}" for row in subset.itertuples()]
        values = subset["mean_delta"].to_numpy(dtype=float)
        errors = np.vstack((values - subset["ci95_low"], subset["ci95_high"] - values)) if len(subset) else None
        axis.bar(range(len(subset)), values, yerr=errors, capsize=3, color="#286aaa")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(subset)), labels)
        axis.set_title(f"Temporal − static {metric.replace('_', '-').upper()}")
        axis.set_ylabel("Paired mean delta (95% CI)")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def generate_temporal_artifacts(
    static: pd.DataFrame,
    temporal: pd.DataFrame,
    output_dir: Path,
    n_resamples: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write detailed, aggregate, table, and figure artifacts for the paired comparison."""
    static = _model_rows(static, "edge-sage", "static metrics")
    temporal = _model_rows(temporal, "edge-sage-gru", "temporal metrics")
    detailed = paired_detail(static, temporal)
    effects = temporal_effects(static, temporal, n_resamples=n_resamples)
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(output_dir / "temporal_static_detailed.csv", index=False)
    effects.to_csv(output_dir / "temporal_static_aggregate.csv", index=False)
    tables = output_dir / "tables"
    tables.mkdir(exist_ok=True)
    _write_latex(effects, tables / "temporal_static_effects.tex")
    _plot(effects, output_dir / "figures" / "temporal_static_comparison")
    return detailed, effects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", type=Path, default=Path("reports/multihorizon_benchmark_summary.csv"))
    parser.add_argument("--temporal", type=Path, default=Path("reports/temporal_benchmark_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/temporal"))
    args = parser.parse_args()
    detailed, effects = generate_temporal_artifacts(pd.read_csv(args.static), pd.read_csv(args.temporal), args.output_dir)
    print(f"[OK] wrote {len(detailed)} paired rows and {len(effects)} effects to {args.output_dir}")


if __name__ == "__main__":
    main()

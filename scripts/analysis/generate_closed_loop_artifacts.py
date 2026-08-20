"""Generate figures and compact tables from closed-loop ns-3 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STRATEGY_ORDER = ["olsr", "hop", "delay", "persistence", "logreg", "xgb", "edge-sage"]
STRATEGY_LABELS = {
    "olsr": "OLSR",
    "hop": "Shortest hop",
    "delay": "Delay weighted",
    "persistence": "Persistence",
    "logreg": "Logistic Regression",
    "xgb": "XGBoost",
    "edge-sage": "Edge-SAGE",
}
STRATEGY_COLORS = {
    "olsr": "#7f7f7f",
    "hop": "#4c78a8",
    "delay": "#72b7b2",
    "persistence": "#b0b0b0",
    "logreg": "#2ca02c",
    "xgb": "#ff7f0e",
    "edge-sage": "#286aaa",
}
METRICS = [
    ("pdr", "PDR", "higher is better"),
    ("mean_delay_ms", "Mean delay (ms)", "lower is better"),
    ("throughput_mbps", "Throughput (Mbps)", "higher is better"),
    ("route_changes", "Route changes", "lower is better"),
]


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _ordered(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary[summary["strategy"].isin(STRATEGY_ORDER)].copy()
    result["strategy"] = pd.Categorical(result["strategy"], STRATEGY_ORDER, ordered=True)
    return result.sort_values("strategy").reset_index(drop=True)


def _write_latex(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    latex = frame.to_latex(index=False, escape=True, caption=caption, label=label, position="tb")
    path.write_text(latex.replace("\\centering\n", "\\centering\n\\scriptsize\n", 1), encoding="utf-8")


def generate_artifacts(summary_csv: Path, paired_csv: Path, output_dir: Path) -> list[Path]:
    """Write the publication figure and source/LaTeX tables."""
    summary = pd.read_csv(summary_csv)
    paired = pd.read_csv(paired_csv)
    required = {"strategy", "n_runs"}
    for metric, _, _ in METRICS:
        required.update(
            {f"{metric}_mean", f"{metric}_ci95_low", f"{metric}_ci95_high"}
        )
    _require(summary, required, "closed-loop summary")
    _require(
        paired,
        {
            "metric",
            "reference_strategy",
            "comparator_strategy",
            "n_pairs",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "p_raw",
            "p_holm",
        },
        "closed-loop paired comparisons",
    )

    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    ordered = _ordered(summary)
    x = np.arange(len(ordered))
    colors = [STRATEGY_COLORS[str(strategy)] for strategy in ordered["strategy"]]
    labels = [STRATEGY_LABELS[str(strategy)] for strategy in ordered["strategy"]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (metric, ylabel, direction) in zip(axes.ravel(), METRICS, strict=True):
        mean = ordered[f"{metric}_mean"].to_numpy(dtype=float)
        low = ordered[f"{metric}_ci95_low"].to_numpy(dtype=float)
        high = ordered[f"{metric}_ci95_high"].to_numpy(dtype=float)
        ax.bar(x, mean, color=colors, yerr=np.vstack([mean - low, high - mean]), capsize=3)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} ({direction})")
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Trace-driven closed-loop ns-3 routing (10 paired runs)")
    fig.tight_layout()
    figure_png = figures / "closed_loop_comparison.png"
    figure_pdf = figures / "closed_loop_comparison.pdf"
    fig.savefig(figure_png, dpi=220, bbox_inches="tight")
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    summary_out = tables / "closed_loop_summary.csv"
    ordered.to_csv(summary_out, index=False)
    summary_print = ordered[["strategy", "n_runs"]].copy()
    summary_print["Strategy"] = summary_print["strategy"].astype(str).map(STRATEGY_LABELS)
    for metric, label, _ in METRICS:
        summary_print[label] = ordered.apply(
            lambda row, name=metric: (
                f"{row[f'{name}_mean']:.3f} "
                f"[{row[f'{name}_ci95_low']:.3f}, {row[f'{name}_ci95_high']:.3f}]"
            ),
            axis=1,
        )
    summary_print = summary_print[["Strategy", "n_runs", *[item[1] for item in METRICS]]]
    summary_tex = tables / "closed_loop_summary.tex"
    _write_latex(
        summary_print,
        summary_tex,
        "Trace-driven closed-loop ns-3 results with run-level bootstrap 95% confidence intervals.",
        "tab:closed-loop-summary",
    )

    paired_out = tables / "closed_loop_paired.csv"
    paired.to_csv(paired_out, index=False)
    paired_print = paired.copy()
    paired_print["Comparison"] = (
        paired_print["reference_strategy"].map(STRATEGY_LABELS)
        + " -> "
        + paired_print["comparator_strategy"].map(STRATEGY_LABELS)
    )
    paired_print["Effect [95% CI]"] = paired_print.apply(
        lambda row: f"{row['mean_delta']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]",
        axis=1,
    )
    paired_print = paired_print.rename(
        columns={"metric": "Metric", "n_pairs": "Runs", "p_holm": "Holm p"}
    )[["Metric", "Comparison", "Runs", "Effect [95% CI]", "Holm p"]]
    paired_tex = tables / "closed_loop_paired.tex"
    _write_latex(
        paired_print,
        paired_tex,
        "Run-paired closed-loop effects (comparator minus reference) and Holm-adjusted p-values.",
        "tab:closed-loop-paired",
    )
    return [figure_png, figure_pdf, summary_out, summary_tex, paired_out, paired_tex]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("reports/closed_loop/summary.csv"))
    parser.add_argument(
        "--paired", type=Path, default=Path("reports/closed_loop/paired_comparisons.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/closed_loop/artifacts")
    )
    args = parser.parse_args()
    paths = generate_artifacts(args.summary, args.paired, args.output_dir)
    print(f"[OK] wrote {len(paths)} closed-loop artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Generate publication-ready figures, tables, and a LaTeX snippet."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scripts.analysis.analyze_multihorizon_stage6 import paired_model_comparisons

from src.evaluation.paired_statistics import paired_comparisons

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_LABELS = {
    "persistence": "Persistence",
    "logreg": "Logistic Regression",
    "xgb": "XGBoost",
    "edge-sage": "Edge-SAGE (full)",
    "edge-sage-decoder-only": "Decoder-only",
    "edge-sage-message-only": "Message-only",
    "edge-sage-noedge": "No-edge",
}
MODEL_COLORS = {
    "persistence": "#7f7f7f",
    "logreg": "#2ca02c",
    "xgb": "#ff7f0e",
    "edge-sage": "#286aaa",
    "edge-sage-decoder-only": "#9467bd",
    "edge-sage-message-only": "#d62728",
    "edge-sage-noedge": "#8c564b",
}
TARGET_LABELS = {"qos": "QoS stability", "survival": "Link survival"}
PROTOCOL_LABELS = {
    "within-run": "Within-run",
    "loro": "LORO",
    "cross-mobility": "Cross-mobility",
}


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def combine_protocol_aggregates(
    within: pd.DataFrame,
    loro: pd.DataFrame,
    cross_mobility: pd.DataFrame,
) -> pd.DataFrame:
    """Combine aggregate results while retaining only held-out test rows."""
    frames = []
    for protocol, frame in (
        ("Within-run", within),
        ("LORO", loro),
        ("Cross-mobility", cross_mobility),
    ):
        _require(frame, {"model_id", "target", "horizon", "split", "macro_f1_mean"}, protocol)
        test = frame[frame["split"] == "test"].copy()
        test.insert(0, "protocol", protocol)
        frames.append(test)
    return pd.concat(frames, ignore_index=True)


def paired_ablation_effects(detailed: pd.DataFrame, n_resamples: int = 2000) -> pd.DataFrame:
    """Compare each Edge-SAGE ablation against full Edge-SAGE on shared runs."""
    _require(
        detailed,
        {"model_id", "target", "horizon", "split", "run_name", "macro_f1"},
        "ablation detail",
    )
    rows = []
    test = detailed[detailed["split"] == "test"]
    for (target, horizon), group in test.groupby(["target", "horizon"], sort=True):
        variants = sorted(set(group["model_id"]) - {"edge-sage"})
        result = paired_comparisons(
            group.rename(columns={"model_id": "strategy"}),
            comparisons=[(variant, "edge-sage") for variant in variants],
            metrics={"macro_f1": True},
            n_resamples=n_resamples,
        )
        result["target"] = target
        result["horizon"] = horizon
        result["split"] = "test"
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def combine_paired_effects(
    within: pd.DataFrame,
    loro: pd.DataFrame,
    cross_mobility: pd.DataFrame,
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute all paired effects from complete detailed run-level results."""
    frames = []
    for protocol, detailed in (
        ("Within-run", within),
        ("LORO", loro),
        ("Cross-mobility", cross_mobility),
    ):
        result = paired_model_comparisons(detailed[detailed["split"] == "test"])
        result = result[(result["metric"] == "macro_f1") & (result["n_pairs"] > 0)].copy()
        result.insert(0, "protocol", protocol)
        frames.append(result)
    if not ablation.empty:
        result = paired_ablation_effects(ablation)
        result.insert(0, "protocol", "Ablation")
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_lines(
    axes: np.ndarray,
    data: pd.DataFrame,
    models: list[str],
    horizons: list[int],
) -> None:
    for ax, target in zip(axes, ("qos", "survival"), strict=True):
        subset = data[data["target"] == target]
        for model in models:
            line = subset[subset["model_id"] == model].sort_values("horizon")
            if line.empty:
                continue
            x = line["horizon"].to_numpy(dtype=float)
            y = line["macro_f1_mean"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
            if {"macro_f1_ci95_low", "macro_f1_ci95_high"}.issubset(line.columns):
                ax.fill_between(
                    x,
                    line["macro_f1_ci95_low"].to_numpy(dtype=float),
                    line["macro_f1_ci95_high"].to_numpy(dtype=float),
                    color=MODEL_COLORS[model],
                    alpha=0.12,
                )
        ax.set_title(TARGET_LABELS[target])
        ax.set_xticks(horizons)
        ax.set_xlabel("Prediction horizon k")
        ax.set_ylabel("Macro-F1 (higher is better)")
        ax.grid(alpha=0.25)
        ax.set_ylim(0.35, 1.0)


def plot_multihorizon(within: pd.DataFrame, output_stem: Path) -> None:
    test = within[within["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _plot_lines(
        axes,
        test,
        ["persistence", "logreg", "xgb", "edge-sage"],
        [1, 2, 3, 5],
    )
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Within-run multi-horizon prediction")
    fig.tight_layout()
    _save_figure(fig, output_stem)


def plot_ood(protocols: pd.DataFrame, output_stem: Path) -> None:
    models = ["logreg", "xgb", "edge-sage"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=False, sharey=True)
    for row, target in enumerate(("qos", "survival")):
        for col, protocol in enumerate(("Within-run", "LORO", "Cross-mobility")):
            ax = axes[row, col]
            subset = protocols[(protocols["target"] == target) & (protocols["protocol"] == protocol)]
            for model in models:
                line = subset[subset["model_id"] == model].sort_values("horizon")
                ax.plot(
                    line["horizon"],
                    line["macro_f1_mean"],
                    marker="o",
                    linewidth=2,
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                )
            if row == 0:
                ax.set_title(protocol)
            if col == 0:
                ax.set_ylabel(f"{TARGET_LABELS[target]}\nMacro-F1")
            if row == 1:
                ax.set_xlabel("Prediction horizon k")
            ax.set_xticks(sorted(subset["horizon"].unique()))
            ax.set_ylim(0.5, 1.0)
            ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Generalization across evaluation protocols")
    fig.tight_layout()
    _save_figure(fig, output_stem)


def plot_ablation(ablation: pd.DataFrame, output_stem: Path) -> None:
    test = ablation[ablation["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _plot_lines(
        axes,
        test,
        ["edge-sage", "edge-sage-decoder-only", "edge-sage-message-only", "edge-sage-noedge"],
        [1, 5],
    )
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Edge-SAGE edge-feature ablation")
    fig.tight_layout()
    _save_figure(fig, output_stem)


def worst_group_summary(worst_group: pd.DataFrame) -> pd.DataFrame:
    _require(
        worst_group,
        {"model_id", "target", "horizon", "split", "group_type", "group_value", "metric", "metric_mean", "is_worst_group"},
        "worst-group report",
    )
    result = worst_group[
        (worst_group["split"] == "test")
        & (worst_group["metric"] == "macro_f1")
        & worst_group["is_worst_group"].astype(bool)
    ].copy()
    return result.sort_values(["group_type", "target", "horizon", "model_id"])


def plot_worst_group(worst_group: pd.DataFrame, output_stem: Path) -> None:
    worst = worst_group_summary(worst_group)
    mobility = worst[worst["group_type"] == "mobility"].copy()
    mobility = mobility.rename(columns={"metric_mean": "macro_f1_mean"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _plot_lines(
        axes,
        mobility,
        ["persistence", "logreg", "xgb", "edge-sage"],
        sorted(mobility["horizon"].unique()),
    )
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Worst mobility-group performance")
    fig.tight_layout()
    _save_figure(fig, output_stem)


def _ci_text(row: pd.Series) -> str:
    return (
        f"{row['macro_f1_mean']:.3f} "
        f"[{row['macro_f1_ci95_low']:.3f}, {row['macro_f1_ci95_high']:.3f}]"
    )


def _write_latex(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latex = frame.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
        position="tb",
    )
    path.write_text(latex.replace("\\centering\n", "\\centering\n\\scriptsize\n", 1), encoding="utf-8")


def write_tables(
    protocols: pd.DataFrame,
    ablation: pd.DataFrame,
    paired: pd.DataFrame,
    worst_group: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "protocol",
        "model_id",
        "target",
        "horizon",
        "n_runs",
        "macro_f1_mean",
        "macro_f1_ci95_low",
        "macro_f1_ci95_high",
        "pr_auc_mean",
    ]
    protocol_table = protocols[[column for column in columns if column in protocols]].copy()
    protocol_table.to_csv(output_dir / "protocol_summary.csv", index=False)
    printable = protocol_table[
        protocol_table["model_id"].isin(["logreg", "xgb", "edge-sage"])
        & protocol_table["horizon"].isin([1, 5])
    ].copy()
    printable["Model"] = printable["model_id"].map(MODEL_LABELS)
    printable["value"] = printable.apply(_ci_text, axis=1)
    printable = printable.pivot(
        index=["protocol", "target", "Model"], columns="horizon", values="value"
    ).reset_index()
    printable = printable.rename(columns={"protocol": "Protocol", "target": "Target", 1: "k=1", 5: "k=5"})
    _write_latex(
        printable,
        output_dir / "protocol_summary.tex",
        "Prediction performance across evaluation protocols; Macro-F1 is higher-is-better.",
        "tab:stage7-protocols",
    )

    ablation.to_csv(output_dir / "ablation_summary.csv", index=False)
    printable = ablation.copy()
    printable["Model"] = printable["model_id"].map(MODEL_LABELS)
    printable["value"] = printable.apply(_ci_text, axis=1)
    printable = printable.pivot(index=["target", "Model"], columns="horizon", values="value").reset_index()
    printable = printable.rename(columns={"target": "Target", 1: "k=1", 5: "k=5"})
    _write_latex(
        printable,
        output_dir / "ablation_summary.tex",
        "Edge-SAGE ablation on ten runs; Macro-F1 is higher-is-better.",
        "tab:stage7-ablation",
    )

    paired.to_csv(output_dir / "paired_effects.csv", index=False)
    paired_print = paired[
        ((paired["protocol"] != "Ablation") & (paired["reference_strategy"] == "logreg") & (paired["comparator_strategy"] == "edge-sage"))
        | (paired["protocol"] == "Ablation")
    ].copy()
    paired_print["Comparison"] = (
        paired_print["reference_strategy"].map(MODEL_LABELS)
        + " -> "
        + paired_print["comparator_strategy"].map(MODEL_LABELS)
    )
    paired_print["Delta Macro-F1 [95% CI]"] = paired_print.apply(
        lambda row: f"{row['mean_delta']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]",
        axis=1,
    )
    paired_print = paired_print.rename(
        columns={"protocol": "Protocol", "target": "Target", "horizon": "k", "n_pairs": "Runs", "p_holm": "Holm p"}
    )[
        ["Protocol", "Target", "k", "Comparison", "Runs", "Delta Macro-F1 [95% CI]", "Holm p"]
    ]
    _write_latex(
        paired_print,
        output_dir / "paired_effects.tex",
        "Run-paired Macro-F1 effects (comparator minus reference), bootstrap 95% CI and Holm-adjusted p-value.",
        "tab:stage7-paired",
    )

    worst_group.to_csv(output_dir / "worst_group_summary.csv", index=False)
    worst_print = worst_group.copy()
    worst_print["Model"] = worst_print["model_id"].map(MODEL_LABELS)
    worst_print["value"] = worst_print.apply(
        lambda row: f"{row['metric_mean']:.3f} ({row['group_value']})", axis=1
    )
    worst_print = worst_print.pivot(
        index=["group_type", "target", "Model"], columns="horizon", values="value"
    ).reset_index()
    worst_print = worst_print.rename(
        columns={"group_type": "Group", "target": "Target", 1: "k=1", 2: "k=2", 3: "k=3", 5: "k=5"}
    )
    _write_latex(
        worst_print,
        output_dir / "worst_group_summary.tex",
        "Worst-group Macro-F1 by mobility and density group.",
        "tab:stage7-worst-group",
    )


def write_latex_snippet(output_dir: Path) -> None:
    snippet = r"""% Set this path relative to the paper's main.tex when necessary.
\providecommand{\StageSevenRoot}{reports/stage7}

\subsection{Multi-horizon and generalization results}
Figure~\ref{fig:stage7-multihorizon} reports the within-run trend across prediction horizons.
Figure~\ref{fig:stage7-ood} compares within-run, LORO, and cross-mobility protocols.
\begin{figure}[tb]
  \centering
  \includegraphics[width=\linewidth]{\StageSevenRoot/figures/multihorizon_macro_f1.pdf}
  \caption{Within-run Macro-F1 across prediction horizons with run-level bootstrap 95\% confidence intervals.}
  \label{fig:stage7-multihorizon}
\end{figure}
\begin{figure}[tb]
  \centering
  \includegraphics[width=\linewidth]{\StageSevenRoot/figures/ood_macro_f1.pdf}
  \caption{Macro-F1 under within-run, leave-one-run-out, and cross-mobility evaluation.}
  \label{fig:stage7-ood}
\end{figure}
\input{\StageSevenRoot/tables/protocol_summary.tex}

\subsection{Edge-feature ablation and worst-group analysis}
Figure~\ref{fig:stage7-ablation} separates edge features used in message passing from those used in the decoder.
Figure~\ref{fig:stage7-worst-group} reports the weakest mobility group for each configuration.
\begin{figure}[tb]
  \centering
  \includegraphics[width=\linewidth]{\StageSevenRoot/figures/edge_sage_ablation.pdf}
  \caption{Edge-SAGE ablation comparing full, decoder-only, message-only, and no-edge variants.}
  \label{fig:stage7-ablation}
\end{figure}
\begin{figure}[tb]
  \centering
  \includegraphics[width=\linewidth]{\StageSevenRoot/figures/worst_group_macro_f1.pdf}
  \caption{Macro-F1 of the worst mobility group, exposing variation hidden by the overall mean.}
  \label{fig:stage7-worst-group}
\end{figure}
\input{\StageSevenRoot/tables/ablation_summary.tex}
\input{\StageSevenRoot/tables/paired_effects.tex}
\input{\StageSevenRoot/tables/worst_group_summary.tex}
"""
    (output_dir / "stage7_results.tex").write_text(snippet, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--within",
        type=Path,
        default=Path("reports/multihorizon_benchmark_summary_aggregate.csv"),
    )
    parser.add_argument(
        "--within-detail",
        type=Path,
        default=Path("reports/multihorizon_benchmark_summary.csv"),
    )
    parser.add_argument("--stage6-root", type=Path, default=Path("reports/stage6"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/stage7"))
    args = parser.parse_args()

    within = pd.read_csv(args.within)
    within_detail = pd.read_csv(args.within_detail)
    loro = pd.read_csv(args.stage6_root / "loro_summary_aggregate.csv")
    cross = pd.read_csv(args.stage6_root / "cross-mobility_summary_aggregate.csv")
    loro_detail = pd.read_csv(args.stage6_root / "loro_summary.csv")
    cross_detail = pd.read_csv(args.stage6_root / "cross-mobility_summary.csv")
    ablation = pd.read_csv(args.stage6_root / "ablation_summary_aggregate.csv")
    ablation_detail = pd.read_csv(args.stage6_root / "ablation_summary.csv")
    worst = pd.read_csv(args.stage6_root / "worst_group_metrics.csv")

    protocols = combine_protocol_aggregates(within, loro, cross)
    paired = combine_paired_effects(within_detail, loro_detail, cross_detail, ablation_detail)
    worst_summary = worst_group_summary(worst)

    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    plot_multihorizon(within, figures / "multihorizon_macro_f1")
    plot_ood(protocols, figures / "ood_macro_f1")
    plot_ablation(ablation, figures / "edge_sage_ablation")
    plot_worst_group(worst, figures / "worst_group_macro_f1")
    write_tables(protocols, ablation, paired, worst_summary, tables)
    write_latex_snippet(args.output_dir)
    print(f"[OK] wrote publication figures, tables, and LaTeX to {args.output_dir}")


if __name__ == "__main__":
    main()

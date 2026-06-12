"""Aggregate per-run routing replay results and plot a strategy comparison.

Reads outputs/routing/*/summary.csv, writes:
  - outputs/aggregates/routing/summary_by_strategy.csv
  - outputs/aggregates/routing/routing_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STRATEGY_LABELS = {
    "olsr": "OLSR (actual, ns-3)",
    "hop": "Shortest Hop",
    "delay": "Delay-Weighted",
    "xgb": "XGBoost-Assisted",
    "gnn": "GNN-Assisted (Edge-SAGE)",
}
STRATEGY_ORDER = ["olsr", "hop", "delay", "xgb", "gnn"]
COLORS = {
    "olsr": "#d62728",
    "hop": "#9e9e9e",
    "delay": "#1f77b4",
    "xgb": "#ff7f0e",
    "gnn": "#2ca02c",
}

PANELS = [
    ("mean_route_lifetime", "Route Lifetime (steps)", False),
    ("mean_route_changes", "Route Changes / session", True),
    ("mean_realized_pdr_t1", "Realized PDR @ t+1", False),
    ("mean_e2e_delay_ms", "E2E Delay (ms)", True),
]


def aggregate(
    routing_root: Path,
    output_dir: Path,
    pattern: str = "*",
    filename: str = "summary.csv",
    out_suffix: str = "",
) -> pd.DataFrame:
    files = sorted(routing_root.glob(f"{pattern}/{filename}"))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No usable {filename} under {routing_root}/{pattern}/")
    detailed = pd.concat(frames, ignore_index=True)

    grouped = detailed.groupby("strategy").agg(
        n_runs=("run_name", "nunique"),
        n_sessions=("n_sessions", "sum"),
        **{
            f"{col}_{stat}": (col, stat)
            for col in [
                "route_found_rate",
                "mean_hops",
                "mean_e2e_delay_ms",
                "mean_est_pdr",
                "mean_route_lifetime",
                "survival_at_1",
                "mean_realized_pdr_t1",
                "mean_route_changes",
                "disconnected_rate",
            ]
            for stat in ["mean", "std"]
        },
    ).reset_index()

    output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(output_dir / f"detailed_by_run{out_suffix}.csv", index=False)
    grouped.to_csv(output_dir / f"summary_by_strategy{out_suffix}.csv", index=False)
    return grouped


def plot(
    grouped: pd.DataFrame,
    output_dir: Path,
    title: str,
    filename: str = "routing_comparison.png",
) -> Path:
    strategies = [s for s in STRATEGY_ORDER if s in grouped["strategy"].tolist()]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 4.6))

    for ax, (col, label, lower_better) in zip(axes, PANELS):
        means = [float(grouped.loc[grouped["strategy"] == s, f"{col}_mean"].iloc[0]) for s in strategies]
        stds = [float(grouped.loc[grouped["strategy"] == s, f"{col}_std"].iloc[0]) for s in strategies]
        xs = range(len(strategies))
        bars = ax.bar(
            xs,
            means,
            yerr=stds,
            capsize=4,
            color=[COLORS[s] for s in strategies],
            edgecolor="white",
        )
        for x, bar, m in zip(xs, bars, means):
            ax.text(x, bar.get_height(), f"{m:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(f"{label}\n({'lower' if lower_better else 'higher'} is better)", fontsize=10)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = output_dir / filename
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate routing replay results across runs.")
    p.add_argument("--routing-root", type=Path, default=Path("outputs/routing"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/aggregates/routing"))
    p.add_argument("--pattern", type=str, default="*")
    p.add_argument("--title", type=str, default="Routing Strategy Comparison (Replay over Test Snapshots)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cols = ["strategy", "n_runs", "mean_route_lifetime_mean", "mean_route_changes_mean",
            "mean_realized_pdr_t1_mean", "mean_e2e_delay_ms_mean"]

    grouped = aggregate(args.routing_root, args.output_dir, pattern=args.pattern)
    out_png = plot(grouped, args.output_dir, args.title)
    print(f"[OK] summary : {args.output_dir / 'summary_by_strategy.csv'}")
    print(f"[OK] chart   : {out_png}")
    print(grouped[cols].to_string(index=False))

    # Same-pair comparison against the routes ns-3's OLSR actually chose
    # (only the recorded (src, dst) flow of each run).
    try:
        grouped_olsr = aggregate(
            args.routing_root, args.output_dir, pattern=args.pattern,
            filename="summary_olsr_pair.csv", out_suffix="_olsr_pair",
        )
        out_png_olsr = plot(
            grouped_olsr, args.output_dir,
            "Routing Strategy Comparison — Same Pair as OLSR Recorded Flow",
            filename="routing_comparison_olsr_pair.png",
        )
        print(f"[OK] olsr-pair chart: {out_png_olsr}")
        print(grouped_olsr[cols].to_string(index=False))
    except FileNotFoundError:
        print("[INFO] no summary_olsr_pair.csv found — skip olsr-pair comparison")

"""Plot the p_th trade-off study (design doc §13.5): route safety vs
network connectivity as risky links are excluded from route candidates.

Reads outputs/routing/*/summary*.csv (every p_th value found), aggregates
across runs, and draws metric-vs-p_th curves for the prediction strategies
with the shortest-hop result as a reference line.

Output:
  - outputs/aggregates/routing/pth_sweep.csv
  - outputs/aggregates/routing/pth_tradeoff.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STRATEGY_LABELS = {"xgb": "XGBoost-Assisted", "gnn": "GNN-Assisted (Edge-SAGE)"}
COLORS = {"xgb": "#ff7f0e", "gnn": "#2ca02c"}
REF_COLOR = "#9e9e9e"

PANELS = [
    ("route_found_rate", "Route Found Rate", "Khả năng duy trì liên thông"),
    ("mean_route_lifetime", "Route Lifetime (steps)", "Độ an toàn của tuyến"),
    ("mean_realized_pdr_t1", "Realized PDR @ t+1", "Độ an toàn của tuyến"),
    ("mean_route_changes", "Route Changes / session", "Chi phí duy trì tuyến"),
]


def collect(routing_root: Path, pattern: str) -> pd.DataFrame:
    files = sorted(routing_root.glob(f"{pattern}/summary*.csv"))
    if not files:
        raise FileNotFoundError(f"No summary*.csv under {routing_root}/{pattern}/")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def plot(df: pd.DataFrame, output_dir: Path, title: str) -> Path:
    pred = df[df["strategy"].isin(STRATEGY_LABELS)]
    p_values = sorted(pred["p_th"].unique())
    hop = df[(df["strategy"] == "hop") & (df["p_th"] == 0.0)]

    agg = (
        pred.groupby(["strategy", "p_th"])
        .agg(**{
            f"{col}_{stat}": (col, stat)
            for col, _, _ in PANELS
            for stat in ["mean", "std"]
        })
        .reset_index()
    )

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 4.4))
    for ax, (col, label, aspect) in zip(axes, PANELS):
        for st in [s for s in ["xgb", "gnn"] if s in pred["strategy"].unique()]:
            sub = agg[agg["strategy"] == st].sort_values("p_th")
            ax.errorbar(
                sub["p_th"],
                sub[f"{col}_mean"],
                yerr=sub[f"{col}_std"],
                marker="o",
                capsize=3,
                label=STRATEGY_LABELS[st],
                color=COLORS[st],
            )
        if not hop.empty and col in hop.columns:
            ax.axhline(
                float(hop[col].mean()),
                color=REF_COLOR,
                linestyle="--",
                linewidth=1.2,
                label="Shortest Hop (ref)",
            )
        ax.set_xlabel("p_th (ngưỡng loại link)")
        ax.set_title(f"{label}\n{aspect}", fontsize=10)
        ax.set_xticks(p_values)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_dir / "pth_sweep.csv", index=False)
    out_png = output_dir / "pth_tradeoff.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the p_th safety/connectivity trade-off.")
    p.add_argument("--routing-root", type=Path, default=Path("outputs/routing"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/aggregates/routing"))
    p.add_argument("--pattern", type=str, default="*")
    p.add_argument("--title", type=str,
                   default="Trade-off giữa an toàn tuyến và duy trì liên thông theo p_th")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = collect(args.routing_root, args.pattern)
    out_png = plot(df, args.output_dir, args.title)
    print(f"[OK] sweep table: {args.output_dir / 'pth_sweep.csv'}")
    print(f"[OK] chart      : {out_png}")

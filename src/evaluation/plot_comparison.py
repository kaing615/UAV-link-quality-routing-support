from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "threshold": "#7f7f7f",
    "logreg": "#c5b0d5",
    "rf": "#9467bd",
    "xgb": "#1f77b4",
    "mlp": "#aec7e8",
    "gat": "#ff7f0e",
    "graphsage": "#ffbb78",
    "edge-sage": "#2ca02c",
    "gat-noedge": "#c49c94",
    "graphsage-noedge": "#dbb8ab",
    "edge-sage-noedge": "#98df8a",
}
MODEL_NAMES = {
    "threshold": "RSSI/SNR Threshold",
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "mlp": "MLP (Baseline)",
    "gat": "GAT (GNN)",
    "graphsage": "GraphSAGE (GNN)",
    "edge-sage": "Edge-Aware GraphSAGE (Proposed)",
    "gat-noedge": "GAT (no edge feats)",
    "graphsage-noedge": "GraphSAGE (no edge feats)",
    "edge-sage-noedge": "Edge-Aware SAGE (no edge feats)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model comparison charts.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/aggregates/all_models/summary_by_model_split.csv"),
        help="Path to summary_by_model_split.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/aggregates/all_models"), help="Directory to save the plots."
    )
    parser.add_argument(
        "--title", type=str, default="Performance Comparison of All Models (Test Split)", help="Chart title."
    )
    parser.add_argument("--filename", type=str, default="model_comparison.png", help="Output image filename.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary_csv.exists():
        print(f"[ERROR] Summary file not found at {args.summary_csv}")
        print("Please run scripts/train/aggregate_all.sh first.")
        return
    df = pd.read_csv(args.summary_csv)
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        print("[ERROR] No 'test' split data found in the summary CSV.")
        return
    order = [
        "threshold",
        "logreg",
        "rf",
        "xgb",
        "mlp",
        "gat",
        "graphsage",
        "edge-sage",
        "gat-noedge",
        "graphsage-noedge",
        "edge-sage-noedge",
    ]
    test_df["sort_idx"] = test_df["model_id"].map(lambda x: order.index(x) if x in order else 99)
    test_df = test_df.sort_values("sort_idx")
    present = [m for m in order if m in set(test_df["model_id"])]
    n_models = len(present)
    metrics = ["accuracy_mean", "f1_mean", "macro_f1_mean", "recall_mean"]
    metric_labels = ["Accuracy", "F1-Score", "Macro F1-Score", "Recall"]
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fcfcfc")
    x_indices = range(len(metric_labels))
    width = 0.8 / max(n_models, 1)
    for idx, order_id in enumerate(present):
        model_row = test_df[test_df["model_id"] == order_id]
        if model_row.empty:
            continue
        means = [model_row[m].values[0] for m in metrics]
        stds = [model_row[m.replace("_mean", "_std")].values[0] for m in metrics]
        positions = [x + (idx - (n_models - 1) / 2) * width for x in x_indices]
        ax.bar(
            positions,
            means,
            width,
            label=MODEL_NAMES.get(order_id, order_id),
            color=COLORS.get(order_id, "#7f7f7f"),
            edgecolor="#ffffff",
            linewidth=1,
            yerr=stds,
            error_kw={"ecolor": "#555555", "capsize": 3, "elinewidth": 0.8},
        )
        for pos, val in zip(positions, means):
            ax.text(
                pos,
                val + 0.015,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
                fontweight="semibold",
            )
    ax.set_title(args.title, fontsize=14, fontweight="bold", pad=50, color="#2c3e50")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="semibold", color="#2c3e50")
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="semibold", color="#2c3e50")
    ax.set_ylim(0, 1.1)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#dddddd",
        fontsize=9,
    )
    ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    plt.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.filename
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[OK] Generated comparison chart saved to: {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "gnn"))
sys.path.insert(0, str(THIS_DIR / "common"))

from build_features import build_feature_tables
from build_graph_dataset import build_graph_records
from build_labels import build_labeled_edges
from split_dataset import build_time_split

DEFAULT_NODES = Path("data/raw/nodes.csv")
DEFAULT_EDGES = Path("data/raw/edges.csv")
DEFAULT_OUTPUT_ROOT = Path("data")


def run_pipeline(
    nodes_csv: Path,
    edges_csv: Path,
    output_root: Path,
    tau_snr: float = 18.0,
    tau_loss: float = 0.10,
    tau_delay: float = 10.0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, Path]:
    processed_dir = output_root / "processed"
    splits_dir = output_root / "splits"
    graph_dir = output_root / "graph"

    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building feature tables...")
    nodes_features_csv, edges_features_csv = build_feature_tables(
        nodes_csv=nodes_csv,
        edges_csv=edges_csv,
        output_dir=processed_dir,
    )

    print("[2/4] Building t+1 edge labels...")
    edges_labeled_csv = processed_dir / "edges_labeled.csv"
    build_labeled_edges(
        edges_features_csv=edges_features_csv,
        output_csv=edges_labeled_csv,
        tau_snr=tau_snr,
        tau_loss=tau_loss,
        tau_delay=tau_delay,
    )

    print("[3/4] Splitting dataset by time...")
    splits_csv = build_time_split(
        edges_labeled_csv=edges_labeled_csv,
        output_dir=splits_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    print("[4/4] Building graph dataset...")
    summary_json = build_graph_records(
        nodes_csv=nodes_features_csv,
        edges_labeled_csv=edges_labeled_csv,
        split_csv=splits_csv,
        output_dir=graph_dir,
    )

    return {
        "nodes_features": nodes_features_csv,
        "edges_features": edges_features_csv,
        "edges_labeled": edges_labeled_csv,
        "splits": splits_csv,
        "train_pt": graph_dir / "train.pt",
        "val_pt": graph_dir / "val.pt",
        "test_pt": graph_dir / "test.pt",
        "summary": summary_json,
    }


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    nodes_csv = args.nodes
    edges_csv = args.edges
    output_root = args.output_root

    if args.run_name:
        raw_run_dir = Path("data/raw_runs") / args.run_name

        if nodes_csv == DEFAULT_NODES:
            nodes_csv = raw_run_dir / "nodes.csv"
        if edges_csv == DEFAULT_EDGES:
            edges_csv = raw_run_dir / "edges.csv"
        if output_root == DEFAULT_OUTPUT_ROOT:
            output_root = Path("data/preprocessed_runs") / args.run_name

    return nodes_csv, edges_csv, output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full graph dataset preprocessing pipeline."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Simulation run name under data/raw_runs/<RUN_NAME>. "
        "If provided, default nodes/edges/output paths are resolved from that run.",
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=DEFAULT_NODES,
        help="Path to raw nodes.csv",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=DEFAULT_EDGES,
        help="Path to raw edges.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for processed, splits, and graph outputs",
    )
    parser.add_argument("--tau-snr", type=float, default=18.0)
    parser.add_argument("--tau-loss", type=float, default=0.10)
    parser.add_argument("--tau-delay", type=float, default=10.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def validate_args(
    nodes_csv: Path,
    edges_csv: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    if not nodes_csv.exists():
        raise FileNotFoundError(f"Raw nodes file not found: {nodes_csv}")
    if not edges_csv.exists():
        raise FileNotFoundError(f"Raw edges file not found: {edges_csv}")

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    output_root.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    args = parse_args()
    nodes_csv, edges_csv, output_root = resolve_paths(args)
    validate_args(nodes_csv, edges_csv, output_root, args)

    print(f"[RUN] run_name={args.run_name or 'default'}")
    print(f"- nodes      : {nodes_csv}")
    print(f"- edges      : {edges_csv}")
    print(f"- output_root: {output_root}")

    outputs = run_pipeline(
        nodes_csv=nodes_csv,
        edges_csv=edges_csv,
        output_root=output_root,
        tau_snr=args.tau_snr,
        tau_loss=args.tau_loss,
        tau_delay=args.tau_delay,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print("\n[OK] Preprocessing completed.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")

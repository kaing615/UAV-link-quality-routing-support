from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_load_proxy(nodes_df: pd.DataFrame) -> pd.DataFrame:
    df = nodes_df.copy()
    node_counts = df.groupby("time")["node_id"].transform("nunique")
    denom = (node_counts - 1).clip(lower=1)
    df["load"] = (df["degree"] / denom).round(6)
    return df


def build_feature_tables(nodes_csv: Path, edges_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    nodes = compute_load_proxy(nodes)
    nodes = nodes[["time", "node_id", "x", "y", "z", "vx", "vy", "vz", "speed", "degree", "load"]]

    edge_cols = [
        "time", "src", "dst", "connected", "distance", "rssi", "snr",
        "delay", "packet_loss", "relative_speed", "throughput", "p_stable", "weight",
    ]
    edges = edges[edge_cols].copy()

    nodes_out = output_dir / "nodes_features.csv"
    edges_out = output_dir / "edges_features.csv"
    nodes.to_csv(nodes_out, index=False)
    edges.to_csv(edges_out, index=False)
    return nodes_out, edges_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build node and edge feature tables.")
    parser.add_argument("--nodes", required=True, type=Path, help="Path to raw nodes.csv")
    parser.add_argument("--edges", required=True, type=Path, help="Path to raw edges.csv")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for processed CSVs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    nodes_out, edges_out = build_feature_tables(args.nodes, args.edges, args.output_dir)
    print(f"[OK] wrote {nodes_out}")
    print(f"[OK] wrote {edges_out}")

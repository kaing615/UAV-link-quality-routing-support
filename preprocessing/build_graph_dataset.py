from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

NODE_FEATURES = ["x", "y", "z", "vx", "vy", "vz", "degree", "load"]
EDGE_FEATURES = ["distance", "rssi", "delay", "packet_loss", "relative_speed"]


def duplicate_undirected_edges(edge_pairs: list[tuple[int, int]], edge_attrs: list[list[float]]):
    edge_index = []
    edge_attr = []
    for (src, dst), attr in zip(edge_pairs, edge_attrs):
        edge_index.append([src, dst])
        edge_index.append([dst, src])
        edge_attr.append(attr)
        edge_attr.append(attr)
    return edge_index, edge_attr


def build_graph_records(
    nodes_csv: Path,
    edges_labeled_csv: Path,
    split_csv: Path,
    output_dir: Path,
) -> Path:
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_labeled_csv)
    splits = pd.read_csv(split_csv)

    output_dir.mkdir(parents=True, exist_ok=True)

    graphs_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    summary = {"feature_names": {"node": NODE_FEATURES, "edge": EDGE_FEATURES}, "splits": {}}

    split_map = dict(zip(splits["time"], splits["split"]))

    for time in sorted(edges["time"].unique().tolist()):
        split = split_map[int(time)]

        node_t = nodes[nodes["time"] == time].sort_values("node_id").copy()
        edge_t = edges[edges["time"] == time].sort_values(["src", "dst"]).copy()

        node_ids = node_t["node_id"].tolist()
        id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}

        x = torch.tensor(node_t[NODE_FEATURES].to_numpy(), dtype=torch.float32)

        edge_pairs = [(id_map[s], id_map[d]) for s, d in zip(edge_t["src"], edge_t["dst"])]
        edge_label_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_label = torch.tensor(edge_t["label"].to_numpy(), dtype=torch.long)
        edge_label_name = edge_t["label_name"].tolist()

        edge_attrs_raw = edge_t[EDGE_FEATURES].to_numpy().tolist()
        edge_index_raw, edge_attr_raw = duplicate_undirected_edges(edge_pairs, edge_attrs_raw)
        edge_index = torch.tensor(edge_index_raw, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float32)

        graph = {
            "time": int(time),
            "split": split,
            "node_ids": node_ids,
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_label_index": edge_label_index,
            "edge_label": edge_label,
            "edge_label_name": edge_label_name,
            "node_feature_names": NODE_FEATURES,
            "edge_feature_names": EDGE_FEATURES,
        }
        graphs_by_split[split].append(graph)

    for split, graphs in graphs_by_split.items():
        out_path = output_dir / f"{split}.pt"
        torch.save(graphs, out_path)

        num_labels_1 = sum(int(g["edge_label"].sum().item()) for g in graphs)
        num_labels_0 = sum(int(g["edge_label"].numel() - g["edge_label"].sum().item()) for g in graphs)

        summary["splits"][split] = {
            "num_graphs": len(graphs),
            "num_edge_labels": num_labels_0 + num_labels_1,
            "label_1_count": num_labels_1,
            "label_0_count": num_labels_0,
            "output": str(out_path),
            "times": [g["time"] for g in graphs],
        }

    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph dataset tensors.")
    parser.add_argument("--nodes-features", required=True, type=Path)
    parser.add_argument("--edges-labeled", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_graph_records(args.nodes_features, args.edges_labeled, args.splits, args.output_dir)
    print(f"[OK] wrote {out}")

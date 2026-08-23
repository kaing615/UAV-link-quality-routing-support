from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TemporalWindow:
    history: list[dict]
    endpoint: dict
    split: str


def build_temporal_windows(run_root: Path, split: str, window_size: int = 4) -> list[TemporalWindow]:
    if window_size < 1:
        raise ValueError("window_size must be positive")

    graph_dir = run_root / "graph_dataset"
    records_by_split = {
        name: torch.load(graph_dir / f"{name}.pt", weights_only=False)
        for name in ("train", "val", "test", "purged")
    }
    records = [(name, record) for name, values in records_by_split.items() for record in values]
    for _, record in records:
        if "time" not in record or record["time"] is None:
            raise ValueError("Each graph record must include time")
        if "node_ids" not in record:
            raise ValueError("Each graph record must include node_ids")

    records.sort(key=lambda item: item[1]["time"])
    times = [record["time"] for _, record in records]
    if len(times) != len(set(times)):
        raise ValueError("Each graph record time must be unique")

    windows = []
    for index, (record_split, endpoint) in enumerate(records):
        if record_split != split or index + 1 < window_size:
            continue
        graphs = [record for _, record in records[index - window_size + 1 : index + 1]]
        if any(graph["time"] != previous["time"] + 1 for previous, graph in zip(graphs, graphs[1:])):
            raise ValueError("time values must be consecutive within a temporal window")
        node_ids = graphs[0]["node_ids"]
        if any(graph["node_ids"] != node_ids for graph in graphs[1:]):
            raise ValueError("node_ids must match across a temporal window")
        windows.append(TemporalWindow(history=graphs[:-1], endpoint=endpoint, split=split))
    return windows

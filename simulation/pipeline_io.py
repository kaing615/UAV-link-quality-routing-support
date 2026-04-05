from __future__ import annotations

import config
from io_utils import write_csv, write_parquet


def ensure_output_dir() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if getattr(config, "SAVE_PLOTS", False):
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_all_outputs(
    all_node_rows: list[dict],
    all_edge_rows: list[dict],
    traffic_rows: list[dict],
) -> None:
    write_csv(
        config.NODES_CSV,
        all_node_rows,
        ["time", "node_id", "x", "y", "z", "vx", "vy", "vz", "speed", "degree"],
    )

    write_csv(
        config.EDGES_CSV,
        all_edge_rows,
        [
            "time",
            "src",
            "dst",
            "distance",
            "connected",
            "relative_speed",
            "rssi",
            "snr",
            "delay",
            "packet_loss",
            "throughput",
            "p_stable",
            "weight",
        ],
    )

    write_csv(
        config.TRAFFIC_CSV,
        traffic_rows,
        ["time", "source", "destination", "reachable", "route_path", "num_edges"],
    )

    write_parquet(config.NODES_PARQUET, all_node_rows)
    write_parquet(config.EDGES_PARQUET, all_edge_rows)
    write_parquet(config.TRAFFIC_PARQUET, traffic_rows)

    print("\nDone.")
    print(f"- Nodes CSV      : {config.NODES_CSV}")
    print(f"- Edges CSV      : {config.EDGES_CSV}")
    print(f"- Traffic CSV    : {config.TRAFFIC_CSV}")
    print(f"- Nodes Parquet  : {config.NODES_PARQUET}")
    print(f"- Edges Parquet  : {config.EDGES_PARQUET}")
    print(f"- Traffic Parquet: {config.TRAFFIC_PARQUET}")

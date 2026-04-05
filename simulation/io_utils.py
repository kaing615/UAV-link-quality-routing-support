from __future__ import annotations

import csv
import pandas as pd


def make_node_rows(time_step: int, uavs, degree_map: dict[int, int]) -> list[dict]:
    rows: list[dict] = []

    for uav in uavs:
        rows.append(
            {
                "time": time_step,
                "node_id": uav.node_id,
                "x": round(uav.x, 4),
                "y": round(uav.y, 4),
                "z": round(uav.z, 4),
                "vx": round(uav.vx, 4),
                "vy": round(uav.vy, 4),
                "vz": round(uav.vz, 4),
                "speed": round(uav.speed, 4),
                "degree": degree_map[uav.node_id],
            }
        )

    return rows


def make_edge_rows(time_step: int, edges: list[dict]) -> list[dict]:
    rows: list[dict] = []

    for edge in edges:
        rows.append(
            {
                "time": time_step,
                "src": edge["src"],
                "dst": edge["dst"],
                "distance": edge["distance"],
                "connected": edge["connected"],
                "relative_speed": edge["relative_speed"],
                "rssi": edge["rssi"],
                "snr": edge["snr"],
                "delay": edge["delay"],
                "packet_loss": edge["packet_loss"],
                "throughput": edge["throughput"],
                "p_stable": edge["p_stable"],
                "weight": edge["weight"],
            }
        )

    return rows


def write_csv(path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)

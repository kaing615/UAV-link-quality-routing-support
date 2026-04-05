from __future__ import annotations

import math

import config
from entities import UAV
from metrics import (
    estimate_delay,
    estimate_p_stable,
    estimate_packet_loss,
    estimate_rssi,
    estimate_snr,
    estimate_throughput,
)


def euclidean_distance_3d(a: UAV, b: UAV) -> float:
    return math.sqrt(
        (a.x - b.x) ** 2
        + (a.y - b.y) ** 2
        + (a.z - b.z) ** 2
    )


def compute_relative_speed(a: UAV, b: UAV) -> float:
    dx = b.x - a.x
    dy = b.y - a.y
    dz = b.z - a.z
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

    if distance <= 1e-9:
        return 0.0

    ux = dx / distance
    uy = dy / distance
    uz = dz / distance

    dvx = b.vx - a.vx
    dvy = b.vy - a.vy
    dvz = b.vz - a.vz

    radial_speed = dvx * ux + dvy * uy + dvz * uz
    return abs(radial_speed)


def build_topology(
    uavs: list[UAV],
) -> tuple[list[dict], dict[int, int], dict[int, list[int]], dict[int, list[tuple[int, float]]]]:
    edges: list[dict] = []
    degree_map = {uav.node_id: 0 for uav in uavs}
    adjacency = {uav.node_id: [] for uav in uavs}
    weighted_adjacency = {uav.node_id: [] for uav in uavs}

    for i in range(len(uavs)):
        for j in range(i + 1, len(uavs)):
            a = uavs[i]
            b = uavs[j]

            distance = euclidean_distance_3d(a, b)
            connected = int(distance <= config.COMM_RANGE)

            if connected:
                degree_map[a.node_id] += 1
                degree_map[b.node_id] += 1
                adjacency[a.node_id].append(b.node_id)
                adjacency[b.node_id].append(a.node_id)

            edges.append(
                {
                    "src": a.node_id,
                    "dst": b.node_id,
                    "distance": distance,
                    "connected": connected,
                }
            )

    max_degree = max(config.NUM_UAVS - 1, 1)
    uav_map = {uav.node_id: uav for uav in uavs}

    for edge in edges:
        a = uav_map[edge["src"]]
        b = uav_map[edge["dst"]]

        relative_speed = compute_relative_speed(a, b)
        rssi = estimate_rssi(edge["distance"])
        snr = estimate_snr(rssi)

        load_factor = (degree_map[a.node_id] + degree_map[b.node_id]) / (2.0 * max_degree)

        delay = estimate_delay(edge["distance"], relative_speed, edge["connected"], load_factor)
        packet_loss = estimate_packet_loss(snr, edge["connected"], load_factor)
        throughput = estimate_throughput(snr, packet_loss, edge["connected"], load_factor)

        p_stable = estimate_p_stable(snr, packet_loss, delay, edge["connected"])
        weight = round(1.0 - p_stable, 4) if edge["connected"] else float("inf")

        edge["distance"] = round(edge["distance"], 4)
        edge["relative_speed"] = round(relative_speed, 4)
        edge["rssi"] = round(rssi, 4)
        edge["snr"] = round(snr, 4)
        edge["delay"] = delay
        edge["packet_loss"] = packet_loss
        edge["throughput"] = throughput
        edge["p_stable"] = p_stable
        edge["weight"] = weight

        if edge["connected"]:
            weighted_adjacency[edge["src"]].append((edge["dst"], weight))
            weighted_adjacency[edge["dst"]].append((edge["src"], weight))

    return edges, degree_map, adjacency, weighted_adjacency

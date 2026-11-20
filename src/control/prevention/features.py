"""Immutable prevention-link-v1: only measured frame t and t-1 are accepted.

Edges are undirected probe delivery >= .5 (the simulator's graph contract).
RSSI is an average of available measured directions, zero plus a mask when
absent. A delta is present only when both windows contain a measurement.
No scenario metadata, reconstructed RSSI, or future frame enters this module.
"""
import numpy as np

FEATURE_VERSION = 'prevention-link-v1'
FEATURE_NAMES = ('distance_range', 'radial_speed', 'rx_ratio', 'rx_delta',
                 'rssi', 'rssi_delta', 'rssi_present', 'rssi_delta_present', 'horizon_s')
EDGE_THRESHOLD = .5


def measured_edges(frame):
    rx = np.asarray(frame['rx_ratio'], dtype=float)
    return [(i, j) for i in range(len(rx)) for j in range(i+1, len(rx))
            if min(rx[i, j], rx[j, i]) >= EDGE_THRESHOLD]


def bridge_edges(frame):
    """Tarjan bridges of the measured undirected graph, including components."""
    n = len(frame['positions'])
    neighbors = [[] for _ in range(n)]
    for i, j in measured_edges(frame):
        neighbors[i].append(j); neighbors[j].append(i)
    discovery, low, result = {}, {}, set()

    def visit(node, parent):
        discovery[node] = low[node] = len(discovery)
        for neighbor in neighbors[node]:
            if neighbor == parent: continue
            if neighbor not in discovery:
                visit(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]: result.add(tuple(sorted((node, neighbor))))
            else:
                low[node] = min(low[node], discovery[neighbor])

    for node in range(n):
        if node not in discovery: visit(node, None)
    return result


def _rssi(frame, i, j):
    values = [frame['rssi'][a][b] for a, b in ((i, j), (j, i)) if frame['rssi_mask'][a][b]]
    return (float(np.mean(values)), 1.) if values else (0., 0.)


def edge_features(current, previous, edge, comm_range, horizon):
    """Return FEATURE_NAMES order; t=0 has zero deltas and no delta mask."""
    if not np.isfinite(comm_range) or comm_range <= 0 or horizon not in (3, 5, 10, 20):
        raise ValueError('Positive finite communication range and supported horizon required')
    if previous is not None and previous['step'] != current['step']-1:
        raise ValueError('Features require consecutive t and t-1 frames')
    i, j = edge
    delta = np.asarray(current['positions'][j], float)-current['positions'][i]
    distance = np.linalg.norm(delta)
    relative_velocity = np.asarray(current['velocities'][j], float)-current['velocities'][i]
    radial_speed = float(delta @ relative_velocity / distance) if distance > 0 else 0.
    rx = min(current['rx_ratio'][i][j], current['rx_ratio'][j][i])
    rssi, mask = _rssi(current, i, j)
    old_rx = min(previous['rx_ratio'][i][j], previous['rx_ratio'][j][i]) if previous is not None else rx
    old_rssi, old_mask = _rssi(previous, i, j) if previous is not None else (0., 0.)
    delta_mask = mask*old_mask
    features = np.array([distance/comm_range, radial_speed, rx, rx-old_rx, rssi,
                         (rssi-old_rssi) if delta_mask else 0., mask, delta_mask, horizon], float)
    if not np.isfinite(features).all(): raise ValueError('Nonfinite live features')
    return features

"""The exact 25-feature contract; only measured frames up to the origin."""
import numpy as np

from .config import ServiceV2Config
from .observation import full_window, service_observation

FEATURE_VERSION = 'service-decline-features-v2'
METHOD = ServiceV2Config()
FEATURE_NAMES = (
    'arrival_offer_1', 'arrival_offer_5', 'arrival_offer_5_delta', 'source_offer_5', 'zero_arrival_fraction_5',
    'route_success', 'route_hops', 'route_failure_fraction_5', 'route_change_fraction_5',
    'endpoint_distance', 'endpoint_radial_speed', 'ordered_chain_max_gap',
    'route_max_gap', 'route_max_radial_speed', 'route_min_bidirectional_probe', 'route_min_rssi', 'route_min_rssi_delta',
    'relay_count', 'minimum_remaining_budget', 'mean_relay_speed', 'probe_source_destination_connected',
    'route_features_present', 'route_rssi_complete', 'route_rssi_delta_present', 'arrival_delta_present')
FEATURE_TRANSFORMS = (
    'clip(A1,0,2)', 'clip(A5,0,2)', 'clip(raw_A5-current_minus_previous,-1,1); absent=0',
    'clip(X5,0,2)', 'zero_R_windows/5', 'successful_traversal_bool', 'successful_hops/(n-1); absent=0',
    'failed_traversals/5', 'changes_of_tuple(nodes,reason)/4', '3D_endpoint_distance/240',
    'endpoint_relative_radial_velocity/8; zero_distance=0', 'projection_ordered_chain_max_3D_gap/240; tie=node_id',
    'successful_route_max_3D_gap/240; absent=0', 'successful_route_max_radial_velocity/8; absent=0',
    'minimum_both_probe_directions_across_hops; absent=0',
    'minimum_hop_mean_available_measured_RSSI_clip[-120,-20]/100; incomplete=0',
    'same_ordered_successful_route_min_RSSI_current_minus_previous/10; incomplete=0',
    'relay_count/3', 'minimum_max(0,400-measured_travel)/400', 'mean_measured_relay_speed/8',
    'source_destination_connectivity_bidirectional_probe>=0.5',
    'successful_route_bool', 'every_route_hop_has_measured_RSSI_direction_bool',
    'same_ordered_successful_route_and_complete_RSSI_at_both_frames_bool', 'prior_valid_A5_bool')


def _array(frame, name, shape, *, boolean=False):
    try:
        raw = frame[name]
        raw_values = np.asarray(raw, dtype=object)
        allowed = (bool, int, float) if boolean else (int, float)
        if any(type(v) not in allowed for v in raw_values.flat):
            raise ValueError(f'Non-numeric mandatory geometry/measurement: {name}')
        a = np.asarray(raw, dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Missing mandatory geometry/measurement: {name}') from exc
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'Invalid finite geometry/measurement: {name}')
    if boolean and not np.isin(a, (0, 1)).all():
        raise ValueError('Invalid RSSI mask')
    return a.astype(bool) if boolean else a


def _rssi(frame, nodes, n):
    rssi = _array(frame, 'rssi', (n, n))
    mask = _array(frame, 'rssi_mask', (n, n), boolean=True)
    hops, clipped = [], []
    for a, b in zip(nodes, nodes[1:]):
        measured = [rssi[i, j] for i, j in ((a, b), (b, a)) if mask[i, j]]
        if not measured:
            return None, clipped
        clipped.extend([float(v) for v in measured if v < METHOD.rssi_clip_min or v > METHOD.rssi_clip_max])
        hops.append(float(np.mean(np.clip(measured, METHOD.rssi_clip_min, METHOD.rssi_clip_max))))
    return min(hops) if hops else None, clipped


def _measure(history, config):
    method = METHOD
    if config.comm_range != method.comm_range:
        raise ValueError('Unsupported communication range; requires 240')
    if not history or len(history) < 5:
        raise ValueError('Five full consecutive service windows required')
    recent = list(history[-6:])
    last = recent[-5:]
    for i, f in enumerate(last):
        previous = last[i - 1] if i else None
        if previous is not None and f['step'] != previous['step'] + 1:
            raise ValueError('Nonconsecutive service feature history')
        if not full_window(f, config, method, previous):
            raise ValueError('Invalid full service feature history')
    current = last[-1]
    s = current['service_observation']
    n = config.num_uavs
    p = _array(current, 'positions', (n, 3))
    v = _array(current, 'velocities', (n, 3))
    travel = _array(current, 'travel', (n,))
    probes = _array(current, 'rx_ratio', (n, n))
    if np.any(travel < 0) or np.any((probes < 0) | (probes > 1)):
        raise ValueError('Invalid measured travel/probes')
    # Validate even masked RSSI; the existing bridge emits finite placeholders.
    _array(current, 'rssi', (n, n))
    _array(current, 'rssi_mask', (n, n), boolean=True)
    windows = [f['service_observation']['window'] for f in last]
    offers = sum(w['offered_packets'] for w in windows)
    a1 = windows[-1]['sink_rx_packets'] / windows[-1]['offered_packets']
    a5 = sum(w['sink_rx_packets'] for w in windows) / offers
    x5 = sum(w['source_tx_packets'] for w in windows) / offers
    delta, delta_present = 0., False
    if len(recent) == 6 and all(recent[i]['step'] + 1 == recent[i + 1]['step'] for i in range(5)):
        try:
            prior_valid = full_window(recent[0], config, method)
        except (ValueError, KeyError, TypeError):
            prior_valid = False
        if prior_valid:
            old = [f['service_observation']['window'] for f in recent[:-1]]
            delta = a5 - sum(w['sink_rx_packets'] for w in old) / sum(w['offered_packets'] for w in old)
            delta_present = True
    routes = [f['service_observation']['route'] for f in last]
    route = routes[-1]
    success = route['success']
    route_keys = [(tuple(r['nodes']), r['reason']) for r in routes]

    def distance(a, b):
        return float(np.linalg.norm(p[b] - p[a]))

    def radial(a, b):
        d = distance(a, b)
        return float(np.dot(p[b] - p[a], v[b] - v[a]) / d) if d else 0.

    source, destination = config.source, config.destination
    axis = p[destination] - p[source]
    relays = sorted(config.relay_ids, key=lambda i: (float(np.dot(p[i] - p[source], axis)), i))
    chain = [source, *relays, destination]
    max_chain = max(distance(a, b) for a, b in zip(chain, chain[1:]))
    route_gap = route_speed = route_probe = route_rssi = rssi_delta = 0.
    rssi_complete = rssi_delta_present = False
    clipped_rssi = []
    if success:
        nodes = route['nodes']
        route_gap = max(distance(a, b) for a, b in zip(nodes, nodes[1:])) / method.comm_range
        route_speed = max(radial(a, b) for a, b in zip(nodes, nodes[1:])) / method.speed_normalizer
        route_probe = min(min(h['rx_ratio_forward'], h['rx_ratio_reverse']) for h in s['route_probe_hops'])
        minimum, clipped_rssi = _rssi(current, nodes, n)
        rssi_complete = minimum is not None
        if rssi_complete:
            route_rssi = minimum / method.rssi_normalizer
            old_route = routes[-2]
            if old_route['success'] and old_route['nodes'] == nodes:
                old_minimum, _ = _rssi(last[-2], nodes, n)
                if old_minimum is not None:
                    rssi_delta, rssi_delta_present = (minimum - old_minimum) / method.rssi_delta_normalizer, True
    reached, pending = {source}, [source]
    while pending:
        a = pending.pop()
        for b in range(n):
            if b not in reached and min(probes[a, b], probes[b, a]) >= method.probe_threshold:
                reached.add(b)
                pending.append(b)
    x = np.asarray([
        np.clip(a1, method.arrival_clip_min, method.arrival_clip_max),
        np.clip(a5, method.arrival_clip_min, method.arrival_clip_max),
        np.clip(delta, method.arrival_delta_clip_min, method.arrival_delta_clip_max),
        np.clip(x5, method.arrival_clip_min, method.arrival_clip_max),
        sum(w['sink_rx_packets'] == 0 for w in windows) / 5., float(success),
        (len(route['nodes']) - 1) / (n - 1) if success else 0.,
        sum(not r['success'] for r in routes) / 5., sum(a != b for a, b in zip(route_keys, route_keys[1:])) / 4.,
        distance(source, destination) / method.comm_range, radial(source, destination) / method.speed_normalizer, max_chain / method.comm_range,
        route_gap, route_speed, route_probe, route_rssi, rssi_delta,
        len(relays) / method.relay_count_normalizer,
        min(max(0., method.budget_normalizer - travel[i]) / method.budget_normalizer for i in relays),
        float(np.mean([np.linalg.norm(v[i]) for i in relays])) / method.speed_normalizer, float(destination in reached),
        float(success), float(rssi_complete), float(rssi_delta_present), float(delta_present)], dtype=float)
    if x.shape != (25,) or not np.isfinite(x).all():
        raise ValueError('Nonfinite service features')
    diagnostics = dict(raw_ratios=dict(A1=a1, A5=a5, X5=x5, arrival_delta=delta),
        clipped={k: bool(raw != x[i]) for k, raw, i in [('A1', a1, 0), ('A5', a5, 1), ('arrival_delta', delta, 2), ('X5', x5, 3)]},
        clipped_rssi_values=clipped_rssi, masks=dict(zip(FEATURE_NAMES[21:], x[21:].tolist())))
    return x, diagnostics


def service_features(history, config):
    return _measure(history, config)[0]


def service_feature_diagnostics(history, config):
    """Serializable clipping/mask evidence for decision trace writers."""
    return _measure(history, config)[1]

"""Finite same-chain geometry hypotheses and exact one-second motion encoding."""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .features import _array

V2_POLICIES = ('service-v2', 'service-v2-no-forecast',
               'service-v2-no-online-service-state', 'service-v2-no-motion-gate')


@dataclass(frozen=True)
class PlanAssessment:
    candidates: list[dict]
    chosen: dict | None
    rejection_counts: dict[str, int]


def encode_displacement(displacement, remaining_budget, simulator_speed):
    d = np.asarray(displacement, dtype=float)
    if (d.shape != (3,) or not np.isfinite(d).all()
            or not np.isfinite(remaining_budget) or remaining_budget < 0
            or not np.isfinite(simulator_speed) or simulator_speed <= 0):
        raise ValueError('Invalid finite displacement/budget/speed')
    scale = min(simulator_speed, remaining_budget)
    distance = float(np.linalg.norm(d))
    if distance > min(2., scale) + 1e-8:
        raise ValueError('Infeasible displacement')
    if distance > min(2., scale):
        d = d * (min(2., scale) / distance)
    return d / scale if scale else np.zeros(3)


def benefit_accepts(g_hold, g_move, total_motion, tolerance=1e-8):
    if not np.isfinite([g_hold, g_move, total_motion, tolerance]).all() or tolerance < 0:
        raise ValueError('Invalid finite benefit inputs')
    gain = g_hold - g_move
    return bool(total_motion > 0 and gain + tolerance >= 2.
                and gain / total_motion + tolerance >= .10)


def _measure(frame, config):
    p = _array(frame, 'positions', (config.num_uavs, 3))
    v = _array(frame, 'velocities', (config.num_uavs, 3))
    travel = _array(frame, 'travel', (config.num_uavs,))
    if np.any(travel < 0) or type(frame.get('step')) is not int or not 0 <= frame['step'] <= config.steps:
        raise ValueError('Invalid measured travel/step')
    if np.any(np.abs(_clip(p, config) - p) > 1e-8):
        raise ValueError('Measured positions outside bounds')
    return p, v, np.maximum(0., config.budget - travel)


def _clip(p, config):
    return np.clip(p, [0., 0., config.z_min], [config.x_max, config.y_max, config.z_max])


def _separated(p, separation, tolerance):
    return all(np.linalg.norm(p[a] - p[b]) + tolerance >= separation
               for a, b in combinations(range(len(p)), 2))


def _swept_safe(p, q, separation, tolerance):
    for a, b in combinations(range(len(p)), 2):
        relative = p[a] - p[b]
        velocity = (q[a] - p[a]) - (q[b] - p[b])
        speed2 = float(velocity @ velocity)
        t = float(np.clip(-(relative @ velocity) / speed2, 0., 1.)) if speed2 else 0.
        if np.linalg.norm(relative + t * velocity) + tolerance < separation:
            return False
    return True


def _step(p, v, remaining, targets, config, method, stop_distance):
    """One assumed interval: clipped endpoints, bounded compensated relay motion."""
    q, actions = p.copy(), np.zeros((len(config.relay_ids), 3))
    for anchor in (config.source, config.destination):
        q[anchor] = _clip(p[anchor] + v[anchor], config)
    for index, relay in enumerate(config.relay_ids):
        if str(relay) not in targets:
            continue
        delta = np.asarray(targets[str(relay)]) - p[relay]
        distance = float(np.linalg.norm(delta))
        if distance <= stop_distance:
            continue
        length = min(method.requested_speed, config.max_speed, remaining[relay], distance)
        displacement = delta * (length / distance)
        q[relay] = _clip(p[relay] + displacement, config)
        actions[index] = encode_displacement(q[relay] - p[relay], remaining[relay], config.max_speed)
    safe = _swept_safe(p, q, config.min_separation, method.numerical_tolerance)
    distances = np.linalg.norm(q - p, axis=1)
    new_budget = remaining.copy()
    for relay in config.relay_ids:
        new_budget[relay] = max(0., remaining[relay] - distances[relay])
    return q, new_budget, actions.ravel(), safe


def _hops(hold, moved, chain, tolerance):
    ids = [[a, b] for a, b in zip(chain, chain[1:])]
    h = [float(np.linalg.norm(hold[b] - hold[a])) for a, b in ids]
    m = [float(np.linalg.norm(moved[b] - moved[a])) for a, b in ids]
    gains = [a - b for a, b in zip(h, m)]
    return dict(hop_ids=ids, hold_hops=h, move_hops=m, hop_gains=gains,
        lengthening_hops=[pair for pair, gain in zip(ids, gains) if gain < -tolerance],
        largest_lengthening=max(0., -min(gains)), g_hold=max(h), g_move=max(m),
        hold_max_hops=[pair for pair, x in zip(ids, h) if abs(x - max(h)) <= tolerance],
        move_max_hops=[pair for pair, x in zip(ids, m) if abs(x - max(m)) <= tolerance])


def enumerate_plans(frame, config, method, mode):
    if mode not in V2_POLICIES or len(config.relay_ids) > 3:
        raise ValueError('Unsupported v2 role or relay count')
    p, v, remaining = _measure(frame, config)
    tolerance = method.numerical_tolerance
    ungated = mode == 'service-v2-no-motion-gate'
    remaining_seconds = config.steps - frame['step']
    axis = p[config.destination] - p[config.source]
    relays = sorted(config.relay_ids, key=lambda i: (float((p[i] - p[config.source]) @ axis), i))
    orders = [list(s) for length in range(1, len(relays) + 1) for s in combinations(relays, length)]
    observation = frame.get('service_observation')
    if not isinstance(observation, dict) or not isinstance(observation.get('route'), dict):
        raise ValueError('Invalid observed service/route')
    route = observation['route']
    nodes = route.get('nodes', [])
    if not isinstance(nodes, list) or any(type(i) is not int for i in nodes):
        raise ValueError('Invalid observed route nodes')
    internal = nodes[1:-1]
    if (route.get('success') is True and len(nodes) >= 2 and nodes[0] == config.source
            and nodes[-1] == config.destination and internal and len(set(internal)) == len(internal)
            and all(i in config.relay_ids for i in internal) and internal not in orders):
        orders.append(internal)
    candidates = [dict(kind='hold', chain=[config.source, config.destination], accepted=False, reason='hold')]
    rejections = {}
    for order in orders:
        chain = [config.source, *order, config.destination]
        fractions = np.arange(1, len(order) + 1)[:, None] / (len(order) + 1)
        initial = p[config.source] + fractions * (p[config.destination] - p[config.source])
        tau0 = float(max(np.linalg.norm(initial - p[order], axis=1)) / method.requested_speed)
        lookahead = min(method.extrapolation_cap, tau0 + method.routing_allowance, remaining_seconds)
        raw_endpoints = p[[config.source, config.destination]] + lookahead * v[[config.source, config.destination]]
        endpoints = _clip(raw_endpoints, config)
        raw_targets = endpoints[0] + fractions * (endpoints[1] - endpoints[0])
        target_values = _clip(raw_targets, config)
        targets = {str(i): x.tolist() for i, x in zip(order, target_values)}
        distances = np.linalg.norm(target_values - p[order], axis=1)
        tau = float(max(distances) / method.requested_speed)
        arrangement = p.copy()
        arrangement[[config.source, config.destination]] = endpoints
        arrangement[order] = target_values
        plan = dict(kind='move', chain=chain, targets=targets, tau0=tau0, tau=tau,
            tau_interpretation='nominal_geometry_only_excludes_cooldown_and_repeated_bursts',
            endpoint_lookahead=lookahead, endpoint_estimates=endpoints.tolist(),
            endpoint_velocities=v[[config.source, config.destination]].tolist(),
            endpoint_clipped=bool(np.any(raw_endpoints != endpoints)),
            target_clipped=bool(np.any(raw_targets != target_values)),
            full_target_distances=distances.tolist(), remaining_budgets=remaining[order].tolist(),
            remaining_episode_seconds=remaining_seconds, accepted=False, reason=None, rollout=[])
        reason = None
        if np.any(distances > remaining[order] + tolerance):
            reason = 'budget'
        elif tau > remaining_seconds + tolerance:
            reason = 'episode_time'
        elif not _separated(arrangement, config.min_separation, tolerance):
            reason = 'target_separation'
        elif max(distances) < (tolerance if ungated else method.start_deadband):
            reason = 'deadband'
        moved, held, budgets = p.copy(), p.copy(), remaining.copy()
        total_motion = 0.
        if reason is None:
            for index in range(min(method.burst_commands, remaining_seconds)):
                q, budgets, _, safe = _step(moved, v, budgets, targets, config, method,
                    tolerance if ungated else method.stop_deadband)
                total_motion += float(sum(np.linalg.norm(q[i] - moved[i]) for i in config.relay_ids))
                moved = q
                for anchor in (config.source, config.destination):
                    held[anchor] = _clip(held[anchor] + v[anchor], config)
                plan['rollout'].append(dict(interval=index + 1,
                    **_hops(held, moved, chain, tolerance), swept_safe=safe))
                if not safe:
                    reason = 'swept_separation'
                    break
            if not plan['rollout']:
                reason = 'episode_time'
            else:
                plan.update(_hops(held, moved, chain, tolerance))
                gain = plan['g_hold'] - plan['g_move']
                moving_count = sum(int(np.linalg.norm(moved[i] - p[i]) > tolerance) for i in order)
                plan.update(total_motion=total_motion, gain=gain,
                    efficiency=gain / total_motion if total_motion > 0 else None,
                    rank=[plan['g_move'], total_motion, moving_count, chain])
                if reason is None and total_motion <= 0:
                    reason = 'no_motion'
                if reason is None and not ungated and not benefit_accepts(plan['g_hold'], plan['g_move'], total_motion, tolerance):
                    reason = 'benefit'
        plan.update(reason=reason, accepted=reason is None)
        if reason:
            rejections[reason] = rejections.get(reason, 0) + 1
        candidates.append(plan)
    passed = [c for c in candidates if c['accepted']]
    return PlanAssessment(candidates, min(passed, key=lambda c: c['rank']) if passed else None, rejections)

"""Strict causal service observations; recent arrivals/offers is NOT cohort PDR."""
from collections import deque
from math import isfinite

COUNTS = ('offered_packets', 'source_tx_packets', 'sink_rx_packets')
REASONS = ('success', 'missing_protocol', 'missing_entry', 'unknown_next', 'loop', 'hop_limit')


def validate_service_observation(frame, config, previous=None):
    s = frame.get('service_observation')
    if not isinstance(s, dict) or s.get('schema') != 'service-observation-v1':
        raise ValueError('Required service-observation-v1 extension missing/incompatible')
    for key in ('sim_time_s', 'window_start_s', 'window_end_s'):
        if type(s.get(key)) not in (int, float) or not isfinite(s[key]): raise ValueError('Invalid service time')
    if s['window_end_s'] != s['sim_time_s'] or s['window_start_s'] > s['window_end_s']:
        raise ValueError('Invalid service window')
    expected = config.warmup + frame['step'] + 1
    if abs(s['sim_time_s']-expected) > 1e-7: raise ValueError('Service/frame timestamp mismatch')
    flow = s.get('flow', {})
    if flow != dict(source=config.source, destination=config.destination, destination_port=9000, protocol='udp'):
        raise ValueError('Wrong observed service flow')
    for group in ('cumulative', 'window'):
        if set(s.get(group, {})) != set(COUNTS): raise ValueError('Missing service counters')
        if any(type(s[group][k]) is not int or s[group][k] < 0 for k in COUNTS): raise ValueError('Invalid service count')
    old = previous['service_observation'] if previous else None
    if previous and frame['step'] != previous['step']+1: raise ValueError('Nonconsecutive service frames')
    if s['window_start_s'] != (old['window_end_s'] if old else 0): raise ValueError('Service count origin/window mismatch')
    if any(s['window'][k] != s['cumulative'][k]-(old['cumulative'][k] if old else 0) for k in COUNTS):
        raise ValueError('Service window/cumulative disagreement')
    route = s.get('route', {})
    nodes = route.get('nodes')
    if (not isinstance(nodes, list) or not nodes or len(nodes) > config.num_uavs+1 or
        any(type(n) is not int or not 0 <= n < config.num_uavs for n in nodes) or nodes[0] != config.source):
        raise ValueError('Invalid observed route')
    if route.get('reason') not in REASONS or type(route.get('success')) is not bool:
        raise ValueError('Invalid traversal status')
    if route['success'] != (route['reason'] == 'success') or (route['success'] and (nodes[-1] != config.destination or len(set(nodes)) != len(nodes))):
        raise ValueError('Traversal success mismatch')
    if route.get('failing_node') != (None if route['success'] else nodes[-1]): raise ValueError('Traversal failing node mismatch')
    hops = s.get('route_probe_hops')
    if not isinstance(hops, list) or len(hops) != len(nodes)-1: raise ValueError('Missing directional probe hops')
    for hop, a, b in zip(hops, nodes, nodes[1:]):
        if hop.get('from') != a or hop.get('to') != b: raise ValueError('Probe hop identity mismatch')
        if any(type(hop.get(k)) not in (int,float) or not isfinite(hop[k]) or not 0 <= hop[k] <= 1
               for k in ('rx_ratio_forward', 'rx_ratio_reverse')): raise ValueError('Invalid directional probe ratio')
    return s


class ServiceHistory:
    def __init__(self, method):
        self.method, self.windows, self.previous = method, deque(maxlen=method.service_windows), None
        self.restoration = False

    def update(self, frame, config):
        s = validate_service_observation(frame, config, self.previous)
        w = s['window']
        if w['offered_packets'] > 0: self.windows.append(w)
        else: self.windows.clear()
        enough = len(self.windows) == self.method.service_windows
        arrival_ratio = sum(w['sink_rx_packets'] for w in self.windows)/sum(w['offered_packets'] for w in self.windows) if self.windows else None
        route_degraded = not s['route']['success'] or any(min(h['rx_ratio_forward'], h['rx_ratio_reverse']) < .5 for h in s['route_probe_hops'])
        delivery_degraded = enough and arrival_ratio < self.method.service_ratio
        recent_good = enough and arrival_ratio >= self.method.service_ratio and not route_degraded
        # Three consecutive windows must also have successful verified routes.
        current_good = not route_degraded and w['offered_packets'] > 0
        self.good_streak = getattr(self, 'good_streak', 0)+1 if current_good else 0
        recent_good = recent_good and self.good_streak >= self.method.service_windows
        if (route_degraded and s['cumulative']['offered_packets'] > 0) or delivery_degraded: self.restoration = True
        elif recent_good: self.restoration = False
        self.previous = frame
        return dict(service_unknown=not enough, recent_arrival_offer_ratio=arrival_ratio,
                    ratio_semantics='arrival/offer throughput ratio; may exceed 1; not cohort window PDR',
                    route_degraded=route_degraded, delivery_degraded=delivery_degraded,
                    restoration=self.restoration, recent_service_evidence=recent_good,
                    zero_recent_arrivals=bool(self.windows) and sum(w['sink_rx_packets'] for w in self.windows)==0,
                    partial_delivery_proxy=arrival_ratio is not None and 0 < arrival_ratio < self.method.service_ratio,
                    route_missing=not s['route']['success'], probe_inconsistent=s['route']['success'] and route_degraded,
                    cumulative_delivery_to_date=s['cumulative']['sink_rx_packets']/s['cumulative']['offered_packets'] if s['cumulative']['offered_packets'] else None)

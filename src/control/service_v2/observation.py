"""Causal arrival/offer state. Window ratios are not packet-cohort PDR."""
from collections import deque
from dataclasses import dataclass
from math import isfinite

from ..path_service.observation import COUNTS, REASONS


def service_observation(frame, config, previous=None):
    """Validate current measurement; reconcile adjacent observed counter streams.

    Gaps cannot be reconciled as one-second windows. They are UNKNOWN to the
    state machine; complete-episode validation remains the collector's job.
    """
    s = frame.get('service_observation')
    if not isinstance(s, dict) or s.get('schema') != 'service-observation-v1':
        raise ValueError('Missing service-observation-v1')
    step = frame.get('step')
    if type(step) is not int or step < 0:
        raise ValueError('Invalid service step')
    for k in ('sim_time_s', 'window_start_s', 'window_end_s'):
        if type(s.get(k)) not in (int, float) or not isfinite(s[k]):
            raise ValueError('Invalid service time')
    if (abs(s['sim_time_s'] - (config.warmup + step + 1)) > 1e-7
            or s['sim_time_s'] != s['window_end_s'] or s['window_start_s'] > s['window_end_s']):
        raise ValueError('Invalid service timestamp')
    if s.get('flow') != dict(source=config.source, destination=config.destination, destination_port=9000, protocol='udp'):
        raise ValueError('Invalid service flow')
    for group in ('window', 'cumulative'):
        if not isinstance(s.get(group), dict) or set(s[group]) != set(COUNTS):
            raise ValueError('Missing service counters')
        if any(type(s[group][k]) is not int or s[group][k] < 0 for k in COUNTS):
            raise ValueError('Invalid service counters')
    if any(s['window'][k] > s['cumulative'][k] for k in COUNTS):
        raise ValueError('Invalid cumulative/window counters')
    if step == 0 and (s['window_start_s'] != 0 or s['window'] != s['cumulative']):
        raise ValueError('Invalid initial counter origin')
    if previous is not None and previous.get('step') == step - 1 and isinstance(previous.get('service_observation'), dict):
        old = previous['service_observation']
        if 'cumulative' in old and set(old['cumulative']) == set(COUNTS):
            if any(s['window'][k] != s['cumulative'][k] - old['cumulative'][k] for k in COUNTS):
                raise ValueError('Service window/cumulative disagreement')
    route = s.get('route', {})
    nodes = route.get('nodes')
    if (not isinstance(nodes, list) or not nodes or len(nodes) > config.num_uavs + 1
            or any(type(n) is not int or not 0 <= n < config.num_uavs for n in nodes)
            or nodes[0] != config.source or route.get('reason') not in REASONS
            or type(route.get('success')) is not bool):
        raise ValueError('Invalid observed route')
    if (route['success'] != (route['reason'] == 'success')
            or route.get('failing_node') != (None if route['success'] else nodes[-1])
            or (route['success'] and (len(nodes) < 2 or nodes[-1] != config.destination or len(set(nodes)) != len(nodes)))):
        raise ValueError('Invalid route traversal status')
    hops = s.get('route_probe_hops')
    if not isinstance(hops, list) or len(hops) != len(nodes) - 1:
        raise ValueError('Missing directional route probes')
    for hop, a, b in zip(hops, nodes, nodes[1:]):
        if hop.get('from') != a or hop.get('to') != b:
            raise ValueError('Route probe identity mismatch')
        if any(type(hop.get(k)) not in (int, float) or not isfinite(hop[k]) or not 0 <= hop[k] <= 1
               for k in ('rx_ratio_forward', 'rx_ratio_reverse')):
            raise ValueError('Invalid route probe ratio')
    return s


def full_window(frame, config, method, previous=None):
    s = service_observation(frame, config, previous)
    return (frame['step'] > 0
            and abs(s['window_end_s'] - s['window_start_s'] - method.window_seconds) <= method.window_tolerance
            and s['window']['offered_packets'] >= method.offered_floor)


@dataclass(frozen=True)
class ServiceState:
    step: int
    mode: str
    ever_armed: bool
    first_armed_step: int | None
    forecast_eligible: bool
    healthy_evidence: bool
    recovered: bool
    degraded_confirmed: bool
    severe_confirmed: bool
    outage_confirmed: bool
    route_failure_confirmed: bool
    activation_context: str
    valid_history: bool
    ratios: dict[str, float | None]
    event_onsets: dict[str, tuple[int, ...]]
    event_confirmations: dict[str, tuple[int, ...]]


class ServiceHistory:
    def __init__(self, config, method):
        self.config, self.method = config, method
        self.windows = deque(maxlen=method.service_windows)
        self.previous = None
        self.ever_armed = False
        self.first_armed_step = None
        self.restoration = False
        self.restoration_context = None
        self.rebuilding = False
        self.streaks = {k: 0 for k in ('degraded', 'severe', 'outage', 'route_failure', 'recovered')}
        self.onsets = {k: [] for k in self.streaks}
        self.confirmations = {k: [] for k in self.streaks}

    def _reset(self):
        self.windows.clear()
        self.streaks = dict.fromkeys(self.streaks, 0)

    def update(self, frame):
        step = frame.get('step')
        if type(step) is not int or step < 0:
            raise ValueError('Invalid service step')
        initializing = step == 0 and self.previous is None
        consecutive = self.previous is None or step == self.previous['step'] + 1
        try:
            valid = full_window(frame, self.config, self.method, self.previous)
        except (ValueError, KeyError, TypeError):
            valid = False
        if not consecutive or not valid:
            self._reset()
            self.rebuilding = not initializing
        # A gap discards the current decision as well; its next window starts
        # a fresh history. A long/low-offer window never enters a ratio.
        valid = valid and consecutive
        if valid:
            self.windows.append(frame)
        enough = len(self.windows) == self.method.service_windows
        if enough:
            self.rebuilding = False
        ratios = dict(A1=None, A5=None, X5=None, Q5=None)
        healthy = False
        flags = dict.fromkeys(self.streaks, False)
        if valid:
            s = frame['service_observation']
            ratios['A1'] = s['window']['sink_rx_packets'] / s['window']['offered_packets']
            if enough:
                counts = {k: sum(f['service_observation']['window'][k] for f in self.windows) for k in COUNTS}
                ratios['A5'] = counts['sink_rx_packets'] / counts['offered_packets']
                ratios['X5'] = counts['source_tx_packets'] / counts['offered_packets']
                ratios['Q5'] = counts['sink_rx_packets'] / counts['source_tx_packets'] if counts['source_tx_packets'] else None
                healthy = (ratios['A5'] >= self.method.healthy_ratio and ratios['X5'] >= self.method.healthy_ratio
                           and s['route']['success'])
            flags = dict(degraded=enough and ratios['A5'] < self.method.decline_ratio,
                         severe=enough and ratios['A5'] < self.method.severe_ratio,
                         outage=s['window']['sink_rx_packets'] == 0,
                         route_failure=not s['route']['success'], recovered=healthy)
            for event, flag in flags.items():
                self.streaks[event] = self.streaks[event] + 1 if flag else 0
                if self.streaks[event] == self.method.confirmation_frames:
                    self.onsets[event].append(step - self.method.confirmation_frames + 1)
                    self.confirmations[event].append(step)
        confirmed = {k: self.streaks[k] >= self.method.confirmation_frames for k in self.streaks}
        if confirmed['recovered']:
            if not self.ever_armed:
                self.first_armed_step = step
            self.ever_armed = True
            self.restoration = False
            self.restoration_context = None
        elif any(confirmed[k] for k in ('degraded', 'severe', 'outage', 'route_failure')):
            self.restoration = True
            self.restoration_context = ('startup_restoration' if not self.ever_armed else
                'delivery_restoration' if confirmed['degraded'] or confirmed['outage'] else 'routing_restoration')
        if not valid or self.rebuilding:
            mode, context = ('UNARMED', 'initialization') if initializing else ('UNKNOWN', 'unknown_service')
        elif self.restoration:
            mode, context = 'RESTORE', self.restoration_context
        else:
            mode = 'READY' if self.ever_armed else 'UNARMED'
            context = ('incipient_delivery_decline' if self.streaks['degraded'] else
                       'marginal_service' if enough and .90 <= ratios['A5'] < .95 else
                       'healthy_service' if healthy else 'insufficient_service_history' if not enough else
                       'route_unconfirmed' if not frame['service_observation']['route']['success'] else 'source_injection_decline')
        eligible = mode == 'READY' and healthy and enough and self.streaks['degraded'] == 0
        self.previous = frame
        return ServiceState(step, mode, self.ever_armed, self.first_armed_step, eligible, healthy,
            confirmed['recovered'], confirmed['degraded'], confirmed['severe'], confirmed['outage'],
            confirmed['route_failure'], context, enough, ratios,
            {k: tuple(v) for k, v in self.onsets.items()}, {k: tuple(v) for k, v in self.confirmations.items()})

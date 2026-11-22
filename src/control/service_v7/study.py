"""v7: v6 arms in a broader setting (n5 and n7 chains, two background UDP flows) plus an
information-balanced OLSR-ETX (fast HELLO, interval tuned for the baseline) and a 2 s control
delay for the centralised route pinning (docs/service_v7_protocol.md)."""
from collections import deque
import csv
from functools import partial
import json
from pathlib import Path

import numpy as np

from ..service_v2.config import ServiceV2Config
from ..service_v2.protocol import sha256, strict_json, write_json
from ..service_v2.study import collect_case, make_controller_factory
from ..service_v3 import protocol as p3
from ..service_v4.env import RouteControlEnv, RoutedAction
from ..service_v4.study import RouteController
from ..service_v5.study import ETX_PATCH, V4_SELECTION_SHA256
from ..service_v6.study import ARP_FLAG

ARMS = ('hold', 'olsr-etx', 'olsr-etx-fast', 'lq-route', 'lq-route-delay')
HELLO_ARMS = {'olsr-etx-h050': .5, 'olsr-etx-h025': .25, 'olsr-etx-h010': .1}
V7 = p3.Study('v7', {'tuning': (6, 6), 'validation': (2, 8), 'test': (4, 20)},
              {'tuning': tuple(HELLO_ARMS), 'validation': ARMS, 'test': ARMS},
              3_200_000, 3_300_000, 'src.control.service_v7.cli', regimes=('n5', 'n7'), predictor_scenarios=False)
REGIMES = V7.regimes
load = partial(p3.load, study=V7)
MARGIN = 1.0           # v4 tuning selection, unchanged
TC_PER_HELLO = 2.5     # ns-3 default ratio (TC 5 s / HELLO 2 s)
DELAY_S = 2            # lq-route-delay: a route decided at t takes effect at t + 2
BACKGROUND = ('--backgroundFlows=2', '--backgroundRateKbps=128')
ETX_FLAG = '--olsrLinkQuality=etx'
ETX_ARMS = {'olsr-etx', 'olsr-etx-fast', *HELLO_ARMS}
ROUTE_ARMS = ('lq-route', 'lq-route-delay')


def hello_flags(hello):
    return [f'--ns3::olsr::RoutingProtocol::HelloInterval=+{hello:g}s',
            f'--ns3::olsr::RoutingProtocol::TcInterval=+{TC_PER_HELLO * hello:g}s']


def arm_flags(policy, hello=None):
    """Simulator flags beyond --routeControl=external, fixed per arm."""
    if policy not in ARMS and policy not in HELLO_ARMS: raise ValueError('Unknown v7 arm: '+policy)
    flags = [ARP_FLAG, *BACKGROUND]
    if policy in ETX_ARMS: flags.append(ETX_FLAG)
    if policy in HELLO_ARMS: flags += hello_flags(HELLO_ARMS[policy])
    if policy == 'olsr-etx-fast':
        if hello is None: raise ValueError('olsr-etx-fast needs the tuned HELLO interval')
        flags += hello_flags(hello)
    return flags


def env_for(policy, hello=None):
    extra = arm_flags(policy, hello)

    class ArmEnv(RouteControlEnv):
        def _command(self, output_dir):
            return super()._command(output_dir) + extra
    ArmEnv.__name__ = 'V7Env_' + policy.replace('-', '_')
    return ArmEnv


class DelayedRouteController:
    """lq-route whose decisions take effect DELAY_S steps late (control-loop latency).
    The logged route is the applied one, so the v4 integrity check stays exact."""

    def __init__(self, inner, delay):
        self.inner, self.delay, self.pending = inner, delay, []

    @property
    def decider(self):
        return self.inner.decider

    def act(self, frame):
        action, decision = self.inner.act(frame)
        self.pending.append(tuple(action.route))
        applied = self.pending.pop(0) if len(self.pending) > self.delay else ()
        control = dict(decision['route_control'], route=list(applied), decided_route=list(action.route))
        return RoutedAction(action, applied), dict(decision, route_control=control)


def make_factory(phase, new, old):
    base = make_controller_factory(new, old, ServiceV2Config(), phase['new_model'], phase['method_sha256'])

    def factory(policy, config, **kwargs):
        hold = base('hold', config, **kwargs)
        if policy == 'lq-route': return RouteController(hold, config, MARGIN)
        if policy == 'lq-route-delay': return DelayedRouteController(RouteController(hold, config, MARGIN), DELAY_S)
        if policy in ('hold', 'olsr-etx', 'olsr-etx-fast', *HELLO_ARMS): return hold  # relays never move
        raise ValueError('Unknown v7 arm: '+policy)
    return factory


def selected_hello(phase):
    if not phase['selection_sha256']: return None
    return strict_json(Path(phase['_path']).parent/'selection.json')['olsr_etx_fast']['hello_interval_s']


def run_case(protocol_path, cell_id, attempt, output):
    phase = load(protocol_path)
    cell = next(c for c in phase['cells'] if c['cell_id'] == cell_id)
    return collect_case(protocol_path, cell_id, attempt, output, loader=load, factory_builder=make_factory,
                        env_class=env_for(cell['policy'], selected_hello(phase)), validate_new=False)


# ---- per-episode evidence ----

def _episode(row, name):
    return Path(row['episode_dir'])/name


def recorded_command(row):
    return json.loads(_episode(row, 'config.json').read_text(encoding='utf-8'))['command']


def flags_ok(row, hello=None):
    """The recorded simulator command ends with exactly this arm's flags; ETX confirmed in stderr."""
    expected = arm_flags(row['policy'], hello)
    command = recorded_command(row)
    log = _episode(row, 'simulator.stderr.log').read_text(encoding='utf-8')
    hello_arm = row['policy'] in HELLO_ARMS or row['policy'] == 'olsr-etx-fast'
    return (command[-len(expected):] == expected and ('UAVOLSR link_quality=etx' in log) == (row['policy'] in ETX_ARMS)
            and any('HelloInterval' in c for c in command) == hello_arm)


def transmissions(row):
    """UAVTX line: {kind: [packets, bytes]} of IP transmissions in the episode."""
    for line in _episode(row, 'simulator.stderr.log').read_text(encoding='utf-8').splitlines():
        if line.startswith('UAVTX '):
            return {k: [int(v) for v in value.split(':')] for k, value in (p.split('=') for p in line.split()[1:])}
    return None


def background(row):
    """Pooled background-flow delivery [rx, tx] from background_flow_metrics.csv (two flows)."""
    path = _episode(row, 'background_flow_metrics.csv')
    if not path.is_file(): return None
    with path.open(newline='', encoding='utf-8') as stream: flows = list(csv.DictReader(stream))
    if len(flows) != 2: return None
    return [sum(int(f['rx_packets']) for f in flows), sum(int(f['tx_packets']) for f in flows)]


def startup_blackout(row):
    """PDR < 5% in seconds 10-60, when relays are still and every link is good (ARP-type failure)."""
    off = rx = 0
    for f in row['frames']:
        s = f['service_observation']
        if 10 < s['sim_time_s'] <= 60:
            off += s['window']['offered_packets']; rx += s['window']['sink_rx_packets']
    return off > 0 and rx / off < .05


def central_signalling_bytes(row, controller=0):
    """Estimated (not simulated) bytes a centralised controller needs: every second each node
    reports its N-1 link qualities (4 B each + 28 B UDP/IP) to the controller at the source, and each
    applied route change is sent to every node on the new route. Each message counts once per hop on
    the shortest probe-connected path (delivery >= 0.5 both ways); unreachable nodes send nothing."""
    total, previous = 0, []
    routes = {d['step']: (d.get('route_control') or {}).get('route', []) for d in row['decisions']}
    for f in row['frames'][1:]:
        rx = np.asarray(f['rx_ratio'], float); n = len(rx)
        up = np.minimum(rx, rx.T) >= .5
        hops = {controller: 0}; queue = deque([controller])
        while queue:
            a = queue.popleft()
            for b in range(n):
                if b != a and up[a, b] and b not in hops: hops[b] = hops[a] + 1; queue.append(b)
        total += sum((4 * (n - 1) + 28) * h for node, h in hops.items() if node != controller)
        route = routes.get(f['step'] - 1, [])
        if route and route != previous:
            total += sum((4 * len(route) + 28) * hops.get(node, 0) for node in route if node != controller)
        previous = route
    return total


def write_selection(output, hello, table, tuning_freeze_sha256):
    if Path(output).exists(): raise FileExistsError(str(output))
    write_json(output, dict(schema=V7.schema('tuning-selection'), predictor_sha256=p3.V2_MODEL_SHA256,
                            margin=MARGIN, margin_source=dict(study='v4', tuning_selection_sha256=V4_SELECTION_SHA256),
                            olsr_etx=dict(patch_sha256=sha256(ETX_PATCH), link_quality_aging=0.1),
                            olsr_etx_fast=dict(hello_interval_s=hello, tc_interval_s=TC_PER_HELLO * hello, table=table),
                            delay_s=DELAY_S, arp_flag=ARP_FLAG, background=list(BACKGROUND),
                            tuning_freeze_sha256=tuning_freeze_sha256))

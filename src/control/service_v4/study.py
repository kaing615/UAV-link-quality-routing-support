"""v4 arms (link-quality route override) on the unchanged v2/v3 collector, plus
the route integrity check and the tuning margin selection."""
from functools import partial
import json
from pathlib import Path

from ..service_v2.config import ServiceV2Config
from ..service_v2.protocol import sha256, strict_json, write_json
from ..service_v2.study import collect_case, make_controller_factory
from ..service_v3 import protocol as p3
from .env import RouteControlEnv, RoutedAction
from .routing import RouteDecider

V4 = p3.Study('v4', {'tuning': (8, 8), 'validation': (4, 12), 'test': (4, 20)},
              {'tuning': ('hold', 'lq-route-m0', 'lq-route-m05', 'lq-route-m10'),
               'validation': ('hold', 'service-v2-no-forecast', 'lq-route', 'lq-route-restore'),
               'test': ('hold', 'service-v2-no-forecast', 'lq-route', 'lq-route-restore')},
              2_600_000, 2_700_000, 'src.control.service_v4.cli')
load = partial(p3.load, study=V4)
MARGIN_ARMS = {'lq-route-m0': 0., 'lq-route-m05': .5, 'lq-route-m10': 1.}
ROUTE_ARMS = (*MARGIN_ARMS, 'lq-route', 'lq-route-restore')


class RouteController:
    """Inner motion controller unchanged; adds the pinned route to its action."""

    def __init__(self, inner, config, margin):
        self.inner = inner
        self.decider = RouteDecider(config.num_uavs, config.source, config.destination, margin)

    def act(self, frame):
        action, decision = self.inner.act(frame)
        route = (frame.get('service_observation') or {}).get('route') or {}
        olsr = route.get('nodes') if route.get('success') else []
        pinned, diagnostic = self.decider.decide(frame['step'], frame['rx_ratio'], olsr)
        return RoutedAction(action, pinned), dict(decision, route_control=diagnostic)


def make_factory(phase, new, old):
    base = make_controller_factory(new, old, ServiceV2Config(), phase['new_model'], phase['method_sha256'])
    selection = strict_json(Path(phase['_path']).parent/'selection.json') if phase['selection_sha256'] else None

    def factory(policy, config, **kwargs):
        if policy in MARGIN_ARMS:
            return RouteController(base('hold', config, **kwargs), config, MARGIN_ARMS[policy])
        if policy in ('lq-route', 'lq-route-restore'):
            if selection is None: raise ValueError('Route arm without a frozen tuning selection')
            inner = 'hold' if policy == 'lq-route' else 'service-v2-no-forecast'
            return RouteController(base(inner, config, **kwargs), config, selection['margin'])
        return base(policy, config, **kwargs)
    return factory


def run_case(protocol_path, cell_id, attempt, output):
    return collect_case(protocol_path, cell_id, attempt, output, loader=load, factory_builder=make_factory,
                        env_class=RouteControlEnv)


def route_integrity(row):
    """Commanded route changes (decisions) must equal the simulator's UAVROUTE log exactly."""
    commanded, previous = [], []
    for d in row['decisions']:
        route = (d.get('route_control') or {}).get('route', [])
        if route != previous: commanded.append([d['step'], route])
        previous = route
    applied, total = [], None
    for line in (Path(row['episode_dir'])/'simulator.stderr.log').read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if parts[:1] == ['UAVROUTE']:
            applied.append([int(parts[1]), [int(n) for n in parts[3:]]])
        elif parts[:1] == ['UAVROUTE_CHANGES']:
            total = int(parts[1])
    ok = commanded == applied and total == len(applied)
    return dict(ok=ok, commanded_changes=len(commanded), applied_changes=len(applied), simulator_total=total,
                override_seconds=sum(bool((d.get('route_control') or {}).get('route')) for d in row['decisions']))


def margin_table(rows, policies):
    """Pooled gradual PDR(arm) - PDR(hold) for every tuning margin arm."""
    from ..service_v3.analysis import blocks, contrast
    data = blocks(rows, policies)
    return [dict(contrast(data, arm, 'hold', 'actual_udp_pdr'), margin=margin) for arm, margin in MARGIN_ARMS.items()]


def choose_margin(table):
    return max(table, key=lambda t: (t['point'] if t['point'] is not None else -1e9, t['margin']))  # tie -> larger margin


def select(protocol_path, run_root, output):
    from ..service_v3.study import load_traces
    phase, _, rows = load_traces(protocol_path, run_root, loader=load)
    if phase['phase'] != 'tuning': raise ValueError('Selection uses tuning only')
    if Path(output).exists(): raise FileExistsError(str(output))
    integrity = {r['cell_id']: route_integrity(r) for r in rows}
    bad = sorted(k for k, v in integrity.items() if not v['ok'])
    if bad: raise ValueError('Commanded and applied routes differ (technical failure): '+', '.join(bad[:5]))
    table = margin_table(rows, phase['policies'])
    best = choose_margin(table)
    selection = dict(schema=V4.schema('tuning-selection'), tuning_freeze_sha256=phase['_freeze_sha256'],
                     predictor_sha256=p3.V2_MODEL_SHA256, rule='max pooled gradual PDR(arm) - PDR(hold); tie -> larger margin',
                     margins=table, margin=best['margin'], chosen_arm=best['arm'],
                     route_integrity=integrity)
    write_json(output, selection)
    return selection

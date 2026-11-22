"""v5: centralised ETX route override (v4 lq-route, margin fixed at 1.0) versus distributed
OLSR-ETX (patched ns-3 OLSR, --olsrLinkQuality=etx) versus plain OLSR. No parameter is
tuned; the frozen 'selection' only records the fixed settings."""
from functools import partial
from pathlib import Path

from ..service_v2.config import ServiceV2Config
from ..service_v2.protocol import sha256, strict_json, write_json
from ..service_v2.study import collect_case, make_controller_factory
from ..service_v3 import protocol as p3
from ..service_v4.env import RouteControlEnv
from ..service_v4.study import RouteController

V5 = p3.Study('v5', {'validation': (4, 12), 'test': (4, 20)},
              {'validation': ('hold', 'olsr-etx', 'lq-route'), 'test': ('hold', 'olsr-etx', 'lq-route')},
              2_800_000, 2_900_000, 'src.control.service_v5.cli')
load = partial(p3.load, study=V5)
V4_SELECTION_SHA256 = 'e498e159256dbd071c4ce2789687f220db9898da798f30c5bc77585751b9e24b'  # margin 1.0
ETX_PATCH = Path(__file__).resolve().parents[3] / 'simulation/ns3/patches/ns-3.45-olsr-etx.patch'


class EtxRouteEnv(RouteControlEnv):
    """Same stack as every v5 arm, plus distributed OLSR-ETX inside ns-3."""

    def _command(self, output_dir):
        return super()._command(output_dir) + ['--olsrLinkQuality=etx']


def fixed_selection():
    """Pre-registered settings (no tuning in v5)."""
    return dict(schema=V5.schema('tuning-selection'), predictor_sha256=p3.V2_MODEL_SHA256,
                margin=1.0, margin_source=dict(study='v4', tuning_selection_sha256=V4_SELECTION_SHA256),
                olsr_etx=dict(patch_sha256=sha256(ETX_PATCH), link_quality_aging=0.1, hello_interval_s=2.0,
                              tc_interval_s=5.0, tc_advertises='all symmetric neighbours'))


def make_factory(phase, new, old):
    base = make_controller_factory(new, old, ServiceV2Config(), phase['new_model'], phase['method_sha256'])
    selection = strict_json(Path(phase['_path']).parent/'selection.json')

    def factory(policy, config, **kwargs):
        if policy == 'lq-route':
            return RouteController(base('hold', config, **kwargs), config, selection['margin'])
        if policy in ('hold', 'olsr-etx'):  # relays never move; OLSR (or OLSR-ETX) routes
            return base('hold', config, **kwargs)
        raise ValueError('Unknown v5 arm: '+policy)
    return factory


def env_for(policy):
    return EtxRouteEnv if policy == 'olsr-etx' else RouteControlEnv


def run_case(protocol_path, cell_id, attempt, output):
    phase = load(protocol_path)
    cell = next(c for c in phase['cells'] if c['cell_id'] == cell_id)
    return collect_case(protocol_path, cell_id, attempt, output, loader=load, factory_builder=make_factory,
                        env_class=env_for(cell['policy']))


def etx_active(row):
    """The simulator must confirm OLSR-ETX exactly for the olsr-etx arm."""
    log = (Path(row['episode_dir'])/'simulator.stderr.log').read_text(encoding='utf-8')
    return ('UAVOLSR link_quality=etx' in log) == (row['policy'] == 'olsr-etx')


def write_selection(output):
    if Path(output).exists(): raise FileExistsError(str(output))
    write_json(output, fixed_selection())

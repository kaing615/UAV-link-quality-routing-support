"""v6: v5 design (hold vs distributed OLSR-ETX vs centralised lq-route, nothing tuned) with one simulator
fix found after v5: ns-3 ArpCache DeadTimeout 100 s -> 1 s in every arm (docs/service_v6_protocol.md)."""
from functools import partial
import json
from pathlib import Path

from ..service_v2.protocol import write_json
from ..service_v2.study import collect_case
from ..service_v3 import protocol as p3
from ..service_v4.env import RouteControlEnv
from ..service_v5 import study as s5

V6 = p3.Study('v6', {'validation': (4, 12), 'test': (4, 20)},
              {'validation': ('hold', 'olsr-etx', 'lq-route'), 'test': ('hold', 'olsr-etx', 'lq-route')},
              3_000_000, 3_100_000, 'src.control.service_v6.cli')
load = partial(p3.load, study=V6)
ARP_FLAG = '--ns3::ArpCache::DeadTimeout=+1s'
make_factory, etx_active = s5.make_factory, s5.etx_active  # same arms and margin as v5


class ArpRouteEnv(RouteControlEnv):
    def _command(self, output_dir):
        return super()._command(output_dir) + [ARP_FLAG]


class ArpEtxEnv(s5.EtxRouteEnv):
    def _command(self, output_dir):
        return super()._command(output_dir) + [ARP_FLAG]


def env_for(policy):
    return ArpEtxEnv if policy == 'olsr-etx' else ArpRouteEnv


def fixed_selection():
    return dict(s5.fixed_selection(), schema=V6.schema('tuning-selection'),
                arp=dict(flag=ARP_FLAG, reason='ns-3 default DeadTimeout 100 s blacks out a flow after one failed '
                                               'ARP resolution (v5 diagnosis, docs/service_v5_protocol.md 8.1)'))


def run_case(protocol_path, cell_id, attempt, output):
    phase = load(protocol_path)
    cell = next(c for c in phase['cells'] if c['cell_id'] == cell_id)
    return collect_case(protocol_path, cell_id, attempt, output, loader=load, factory_builder=make_factory,
                        env_class=env_for(cell['policy']))


def arp_active(row):
    """The recorded simulator command of every episode must carry the ARP fix."""
    return ARP_FLAG in json.loads((Path(row['episode_dir'])/'config.json').read_text(encoding='utf-8'))['command']


def startup_blackout(row):
    """PDR < 5% in seconds 10-60, when relays are still and every link is good: an ARP-type failure, not routing."""
    off = rx = 0
    for f in row['frames']:
        s = f['service_observation']
        if 10 < s['sim_time_s'] <= 60:
            off += s['window']['offered_packets']; rx += s['window']['sink_rx_packets']
    return off > 0 and rx / off < .05


def write_selection(output):
    if Path(output).exists(): raise FileExistsError(str(output))
    write_json(output, fixed_selection())

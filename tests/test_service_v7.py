from dataclasses import asdict
import json
from pathlib import Path

import pytest

from src.control.service_v3 import protocol as p3
from src.control.service_v4.env import RoutedAction
from src.control.service_v7 import study as s7
from tests.service_v2_support import fixture_config, service_frames

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'reports/uav_service_v7/seed-plan.json'


def test_seed_plan_regimes_and_audit():
    plan = p3.load_plan(PLAN, s7.V7)
    assert plan['phases'] == p3.build_phases(s7.V7)
    assert {s['regime'] for rows in plan['phases'].values() for s in rows} == {'n5', 'n7'}
    assert 'reports/uav_service_v6/seed-plan.json' in plan['historical_audit']['files']
    assert sum(len(plan['phases'][p]) * len(s7.V7.policies[p]) for p in plan['phases']) == 412
    # earlier studies keep their n4/n5 phases
    assert p3.build_phases(p3.V3) == p3.load_plan(ROOT / 'reports/uav_service_v3/seed-plan.json', p3.V3)['phases']


def test_arm_flags():
    assert s7.arm_flags('hold') == [s7.ARP_FLAG, *s7.BACKGROUND]
    assert s7.arm_flags('olsr-etx')[-1] == '--olsrLinkQuality=etx'
    assert s7.arm_flags('olsr-etx-h025')[-2:] == ['--ns3::olsr::RoutingProtocol::HelloInterval=+0.25s',
                                                 '--ns3::olsr::RoutingProtocol::TcInterval=+0.625s']
    assert s7.arm_flags('olsr-etx-fast', .1)[-1] == '--ns3::olsr::RoutingProtocol::TcInterval=+0.25s'
    with pytest.raises(ValueError): s7.arm_flags('olsr-etx-fast')
    command = s7.env_for('lq-route-delay')(fixture_config(), ['fixture'], Path('.'))._command(Path('.'))
    assert command[-4:] == ['--routeControl=external', s7.ARP_FLAG, *s7.BACKGROUND]


class _Pinned:
    """Inner route controller that pins step-dependent routes."""
    decider = 'inner-decider'

    def act(self, frame):
        route = (0, 1, 3) if frame['step'] >= 1 else ()
        return RoutedAction([0.] * 6, route), dict(step=frame['step'], route_control=dict(route=list(route)))


def test_delayed_route_controller_applies_decisions_late():
    controller = s7.DelayedRouteController(_Pinned(), 2)
    applied = [controller.act(dict(step=t))[0].route for t in range(5)]
    assert applied == [(), (), (), (0, 1, 3), (0, 1, 3)] and controller.decider == 'inner-decider'
    _, decision = controller.act(dict(step=5))
    assert decision['route_control']['route'] == [0, 1, 3] and decision['route_control']['decided_route'] == [0, 1, 3]


def test_central_signalling_estimate():
    frames = service_frames([60] * 3)  # 4 nodes fully connected: every node one hop from the controller
    decisions = [dict(step=t, route_control=dict(route=[0, 1, 3] if t >= 1 else [])) for t in range(3)]
    reports = 3 * 3 * (4 * 3 + 28)  # 3 seconds x 3 reporting nodes x 40 B x 1 hop
    command = 2 * (4 * 3 + 28)      # one route change sent to nodes 1 and 3
    assert s7.central_signalling_bytes(dict(frames=frames, decisions=decisions)) == reports + command


def _episode(tmp, policy, hello, pdr_shift, regime, i, blackout=False):
    frames = service_frames([60] * 240)
    if blackout:
        for f in frames[:62]: f['service_observation']['window']['sink_rx_packets'] = 0
    if policy != 'hold':
        for f in frames: f['service_observation']['route']['nodes'] = [0, 1, 3]
    pinned = policy in s7.ROUTE_ARMS
    decisions = [dict(step=t, route_control=dict(route=[0, 1, 3] if pinned and t >= 5 else [])) for t in range(240)]
    ep = tmp / f'{regime}-{i}-{policy}'; ep.mkdir(parents=True)
    flags = s7.arm_flags(policy, hello)
    (ep / 'config.json').write_text(json.dumps(dict(command=['sim', '--routeControl=external', *flags])))
    olsr = 40000 if policy == 'olsr-etx-fast' else 10000
    log = ('UAVOLSR link_quality=etx\n' if policy in s7.ETX_ARMS else '') + \
          ('UAVROUTE 5 3 0 1 3\nUAVROUTE_CHANGES 1\n' if pinned else 'UAVROUTE_CHANGES 0\n') + \
          f'UAVTX olsr=100:{olsr} probe=10:400 data=20:1000 background=5:500 other=0:0\n'
    (ep / 'simulator.stderr.log').write_text(log)
    (ep / 'background_flow_metrics.csv').write_text('flow_index,rx_packets,tx_packets\n0,90,100\n1,95,100\n')
    return dict(cell_id=f'x-{regime}-gradual-{i:02d}__{policy}', regime=regime, context='gradual',
                layout_seed=10 * i + (regime == 'n7'), radio_seed=10 * i, policy=policy, status='ok',
                config=asdict(fixture_config()), trace_sha256='t' * 64, actual_udp_pdr=.85 + pdr_shift + .001 * i,
                total_relay_travel_m=0., frames=frames, decisions=decisions, episode_dir=str(ep))


def _phase(tmp, phase, policies, hello, shifts, blackout_cell=None):
    rows = [_episode(tmp, p, hello, shifts.get(p, 0.), g, i, blackout=(f'{g}-{i}-{p}' == blackout_cell))
            for g in s7.REGIMES for i in range(3) for p in policies]
    meta = dict(phase=phase, policies=list(policies), _freeze_sha256='f' * 64, selection_sha256='s' * 64 if hello else None,
                _path=str(tmp / 'protocol.json'))
    return meta, dict(planned_cells=len(rows), ok_cells=len(rows)), rows


def test_select_prefers_best_then_longer_interval(tmp_path, monkeypatch):
    import src.control.service_v3.study as v3study
    from src.control.service_v7.analysis import select
    collection = _phase(tmp_path / 'runs', 'tuning', list(s7.HELLO_ARMS), None,
                        {'olsr-etx-h050': .02, 'olsr-etx-h025': .02, 'olsr-etx-h010': .01})
    monkeypatch.setattr(v3study, 'load_traces', lambda *a, **k: collection)
    best, table = select('protocol.json', 'runs', tmp_path / 'selection.json')
    assert best['hello_interval_s'] == .5 and len(table) == 3
    saved = json.loads((tmp_path / 'selection.json').read_text())
    assert saved['olsr_etx_fast']['tc_interval_s'] == 1.25 and saved['schema'] == 'service-v7-tuning-selection-v1'


@pytest.mark.parametrize('phase,blackout,go', [('validation', None, True), ('validation', 'n7-1-olsr-etx', False),
                                              ('test', None, None)])
def test_analyze_end_to_end(tmp_path, monkeypatch, phase, blackout, go):
    import src.control.service_v3.study as v3study
    from src.control.service_v7 import analysis
    collection = _phase(tmp_path / 'runs', phase, list(s7.ARMS), .25,
                        {'olsr-etx': .01, 'olsr-etx-fast': .02, 'lq-route-delay': .04, 'lq-route': .05}, blackout)
    monkeypatch.setattr(v3study, 'load_traces', lambda *a, **k: collection)
    monkeypatch.setattr(analysis, 'selected_hello', lambda phase: .25)
    result = analysis.analyze('protocol.json', 'runs', tmp_path / 'analysis')
    assert result['flags_ok'] and result['route_integrity_ok'] and result['background_ok']
    assert result['overhead']['olsr-etx-fast']['olsr_Bps'] == 4 * result['overhead']['olsr-etx']['olsr_Bps']
    assert 'central_estimate_Bps' in result['overhead']['lq-route']
    if phase == 'validation':
        saved = json.loads((tmp_path / 'analysis' / 'go.json').read_text())
        assert saved['go'] is go and saved['schema'] == 'service-v7-go-v1'
    else:
        assert [s['success'] for s in result['fixed_sequence']] == [True, True, True]
        assert set(result['fixed_sequence'][0]['contrast']['regimes']) == {'n5', 'n7'}

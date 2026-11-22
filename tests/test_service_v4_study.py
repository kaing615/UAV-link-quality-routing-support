import json
from pathlib import Path
import sys

import numpy as np
import pytest

from src.control.service_v3 import protocol as p3
from src.control.service_v4 import study as s4
from src.control.service_v4.analysis import fixed_sequence, go_decision, mechanism
from src.control.service_v4.env import RouteControlEnv, RoutedAction, routed_line, send_routed_action
from tests.service_v2_support import fixture_config, service_frames

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'reports/uav_service_v4/seed-plan.json'
V2_MODEL = ROOT / 'reports/uav_service_prevention_v2_2026-11-12/model/predictor.json'
REFERENCE = ROOT / 'reports/uav_prevention_pilot_2026-11-11/predictor.json'


def test_seed_plan_valid_and_disjoint_from_v3():
    plan = p3.load_plan(PLAN, s4.V4)
    assert plan['phases'] == p3.build_phases(s4.V4)
    v3 = {(s['layout_seed'], s['radio_seed']) for rows in p3.load_plan(ROOT / 'reports/uav_service_v3/seed-plan.json')['phases'].values() for s in rows}
    v4 = {(s['layout_seed'], s['radio_seed']) for rows in plan['phases'].values() for s in rows}
    assert not v3 & v4 and sum(len(plan['phases'][p]) * len(s4.V4.policies[p]) for p in plan['phases']) == 448


class FakeProcess:
    def __init__(self): self.lines = []; self.stdin = self
    def poll(self): return None
    def write(self, text): self.lines.append(text)
    def flush(self): pass


class FakeBridge:
    def __init__(self, config, frames): self.config, self.frames, self.process = config, frames, FakeProcess()
    def next_frame(self, step): return self.frames[step]
    def send_action(self, step, values): raise AssertionError('v1 line must not be sent by the route env')
    def close(self, force=False): pass


def test_route_env_sends_v2_lines(tmp_path):
    config = fixture_config(); frames = service_frames([60] * 240)
    env = RouteControlEnv(config, ['fixture'], tmp_path)
    assert env._command(tmp_path)[-1] == '--routeControl=external'
    env.bridge = bridge = FakeBridge(config, frames); env.last_frame, env._done, env.episode_output_dir = frames[0], False, tmp_path
    env.step(RoutedAction(np.zeros(6), (0, 1, 2, 3)))
    env.step(np.zeros(6))  # plain inner action (HOLD) -> OLSR
    assert bridge.process.lines == ['UAVACT 2 0 0 0 0 0 0 0 ROUTE 4 0 1 2 3\n', 'UAVACT 2 1 0 0 0 0 0 0 ROUTE 0\n']
    assert 'send_action' not in vars(bridge)
    with pytest.raises(ValueError, match='simple'):
        send_routed_action(bridge, 2, [0.] * 6, (0, 2, 2, 3))
    assert routed_line(3, [.5], ()) == 'UAVACT 2 3 0.5 ROUTE 0\n'


def test_route_controller_pins_relay_path_when_direct_link_is_weak():
    from src.control.prevention.controller import PreventionController
    config = fixture_config(); frames = service_frames([60] * 240)
    for f in frames:
        f['rx_ratio'][0][3] = f['rx_ratio'][3][0] = .3   # weak direct link; OLSR (fixture) routes [0, 3]
    controller = s4.RouteController(PreventionController('hold', config, predictor=None, threshold=.5, routing_delay_s=3.), config, .5)
    out = [controller.act(f) for f in frames[:4]]
    assert [a.route for a, _ in out] == [(), (), (0, 1, 3), (0, 1, 3)]  # fixture links are all 1.0: ETX 2 beats 11.1
    assert out[2][1]['route_control']['reason'] == 'engage' and out[2][1]['route_control']['olsr_route'] == [0, 3]
    assert np.array_equal(np.asarray(out[2][0]), np.zeros(6))


def test_route_integrity_matches_simulator_log(tmp_path):
    decisions = [dict(step=t, route_control=dict(route=[0, 1, 3] if 2 <= t < 9 else [])) for t in range(12)]
    row = dict(decisions=decisions, episode_dir=str(tmp_path))
    (tmp_path / 'simulator.stderr.log').write_text('UAVROUTE 2 3 0 1 3\nUAVROUTE 9 0\nUAVROUTE_CHANGES 2\n')
    assert s4.route_integrity(row) == dict(ok=True, commanded_changes=2, applied_changes=2, simulator_total=2, override_seconds=7)
    (tmp_path / 'simulator.stderr.log').write_text('UAVROUTE 2 3 0 1 3\nUAVROUTE_CHANGES 1\n')
    assert not s4.route_integrity(row)['ok']


def _rows(effect):
    rows = []
    for regime in ('n4', 'n5'):
        for i in range(6):
            frames = service_frames([60] * 240)
            for policy, pdr in (('hold', .85), ('service-v2-no-forecast', .86), ('lq-route', .85 + effect), ('lq-route-restore', .87)):
                route = [0, 1, 3] if policy.startswith('lq-route') and i % 2 == 0 else []
                decisions = [dict(step=t, route_control=dict(route=route if t > 5 else [])) for t in range(240)]
                rows.append(dict(regime=regime, context='gradual', layout_seed=i, radio_seed=i, policy=policy, status='ok',
                                 actual_udp_pdr=pdr + .001 * i, total_relay_travel_m=0., frames=frames, decisions=decisions))
    return rows


def test_mechanism_go_and_fixed_sequence():
    policies = ('hold', 'service-v2-no-forecast', 'lq-route', 'lq-route-restore')
    rows = _rows(.02); mech = mechanism(rows)
    assert mech['lq-route']['engaged_share'] == .5 and mech['hold']['engaged_share'] == 0
    assert mech['lq-route']['pinned_pdr'] == 1 and mech['lq-route']['pinned_seconds'] == 6 * 234
    assert go_decision(rows, policies, True, mech)['go']
    assert not go_decision(rows, policies, False, mech)['go']
    assert not go_decision(_rows(-.02), policies, True, mechanism(_rows(-.02)))['go']
    from src.control.service_v3.analysis import blocks
    seq = fixed_sequence(blocks(rows, policies))
    assert [s['success'] for s in seq] == [True, True, True]


@pytest.mark.skipif(not V2_MODEL.is_file(), reason='formal v2 predictor is kept outside git')
def test_freeze_load_and_factory_for_every_arm(tmp_path):
    from src.control.prevention.config import PreventionConfig
    from src.control.prevention.prediction import LearnedPredictor
    from src.control.service_v2.prediction import ServicePredictor
    from src.control.service_v2.protocol import write_json
    freeze = lambda phase, **kw: p3.freeze(phase, PLAN, tmp_path / phase, V2_MODEL, REFERENCE, [sys.executable],
                                           'ns-3 fixture', 'compiler fixture', study=s4.V4, **kw)
    tuning = s4.load(freeze('tuning'))
    assert tuning['expected_episodes'] == 128 and 'service_v4.cli' in ' '.join(tuning['run_argv'])
    new, old = ServicePredictor.load(V2_MODEL), LearnedPredictor.load(REFERENCE)
    config = PreventionConfig(**tuning['cells'][0]['config'])
    arms = {p: s4.make_factory(tuning, new, old)(p, config, predictor=None, threshold=.5, routing_delay_s=3.) for p in tuning['policies']}
    assert [arms[p].decider.margin for p in s4.MARGIN_ARMS] == [0., .5, 1.] and not hasattr(arms['hold'], 'decider')
    selection = tmp_path / 'selection.json'
    write_json(selection, dict(schema=s4.V4.schema('tuning-selection'), predictor_sha256=p3.V2_MODEL_SHA256, margin=.5))
    validation = s4.load(freeze('validation', selection=selection))
    arms = {p: s4.make_factory(validation, new, old)(p, config, predictor=None, threshold=.5, routing_delay_s=3.) for p in validation['policies']}
    assert arms['lq-route'].decider.margin == .5 and arms['lq-route-restore'].inner.controller.policy == 'service-v2-no-forecast'
    with pytest.raises(ValueError):
        p3.load(tmp_path / 'validation' / 'protocol.json')  # a v4 package is not a v3 package


def test_margin_table_and_choice():
    policies = s4.V4.policies['tuning']
    effects = {'hold': 0., 'lq-route-m0': .03, 'lq-route-m05': .03, 'lq-route-m10': .01}
    rows = [dict(regime=g, context='gradual', layout_seed=i, radio_seed=i, policy=p, status='ok',
                 actual_udp_pdr=.85 + e + .001 * i, total_relay_travel_m=0.)
            for g in ('n4', 'n5') for i in range(6) for p, e in effects.items()]
    table = s4.margin_table(rows, policies)
    assert [(t['arm'], t['margin']) for t in table] == [('lq-route-m0', 0.), ('lq-route-m05', .5), ('lq-route-m10', 1.)]
    assert [round(t['point'], 6) for t in table] == [.03, .03, .01] and all(t['comparator'] == 'hold' for t in table)
    assert s4.choose_margin(table)['margin'] == .5   # tie between 0 and 0.5 -> larger margin


def _collection(tmp_path, phase, effect=.02):
    """Synthetic complete collection rows as load_traces returns them, with simulator logs."""
    from dataclasses import asdict
    policies = list(s4.V4.policies[phase]); rows = []
    for regime in ('n4', 'n5'):
        for i in range(3):
            for policy in policies:
                frames = service_frames([60] * 240)
                pinned = policy.startswith('lq-route')
                decisions = [dict(step=t, route_control=dict(route=[0, 1, 3] if pinned and t >= 5 else [])) for t in range(240)]
                episode = tmp_path / f'{regime}-{i}-{policy}'; episode.mkdir(parents=True)
                (episode / 'simulator.stderr.log').write_text(
                    ('UAVROUTE 5 3 0 1 3\nUAVROUTE_CHANGES 1\n' if pinned else 'UAVROUTE_CHANGES 0\n'))
                bonus = effect if pinned else 0.
                rows.append(dict(cell_id=f'{phase}-{regime}-gradual-{i:02d}__{policy}', regime=regime, context='gradual',
                                 layout_seed=10 * i + (regime == 'n5'), radio_seed=10 * i, policy=policy, status='ok',
                                 config=asdict(fixture_config()), trace_sha256='t' * 64, actual_udp_pdr=.85 + bonus + .001 * i,
                                 total_relay_travel_m=0., frames=frames, decisions=decisions, episode_dir=str(episode)))
    meta = dict(phase=phase, policies=policies, _freeze_sha256='f' * 64, selection_sha256=None if phase == 'tuning' else 's' * 64)
    return meta, dict(planned_cells=len(rows), ok_cells=len(rows)), rows


@pytest.mark.parametrize('phase', ['tuning', 'validation', 'test'])
def test_select_and_analyze_end_to_end(tmp_path, monkeypatch, phase):
    import src.control.service_v3.study as v3study
    from src.control.service_v4.analysis import analyze
    collection = _collection(tmp_path / 'runs', phase)
    monkeypatch.setattr(v3study, 'load_traces', lambda *a, **k: collection)
    if phase == 'tuning':
        selection = s4.select('protocol.json', 'runs', tmp_path / 'selection.json')
        assert selection['margin'] == 1. and all(v['ok'] for v in selection['route_integrity'].values())
        assert json.loads((tmp_path / 'selection.json').read_text())['schema'] == 'service-v4-tuning-selection-v1'
        return
    result = analyze('protocol.json', 'runs', tmp_path / 'analysis')
    assert result['route_integrity_ok'] and result['mechanism']['lq-route']['engaged_share'] == 1
    if phase == 'validation':
        go = json.loads((tmp_path / 'analysis' / 'go.json').read_text())
        assert go['go'] and go['schema'] == 'service-v4-go-v1'
    else:
        assert [s['success'] for s in result['fixed_sequence']] == [True, True, True]  # both pinned arms get +0.02

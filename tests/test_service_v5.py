import json
from pathlib import Path
import sys

import pytest

from src.control.service_v3 import protocol as p3
from src.control.service_v4.env import RouteControlEnv
from src.control.service_v5 import study as s5
from tests.service_v2_support import fixture_config, service_frames

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'reports/uav_service_v5/seed-plan.json'
SELECTION = ROOT / 'reports/uav_service_v5/selection.json'
V2_MODEL = ROOT / 'reports/uav_service_prevention_v2_2026-11-12/model/predictor.json'
REFERENCE = ROOT / 'reports/uav_prevention_pilot_2026-11-11/predictor.json'


def test_seed_plan_and_fixed_selection():
    plan = p3.load_plan(PLAN, s5.V5)
    assert plan['phases'] == p3.build_phases(s5.V5)
    assert sum(len(plan['phases'][p]) * 3 for p in plan['phases']) == 240
    earlier = set()
    for old, study in ((ROOT / 'reports/uav_service_v3/seed-plan.json', p3.V3),):
        earlier |= {(s['layout_seed'], s['radio_seed']) for rows in p3.load_plan(old, study)['phases'].values() for s in rows}
    v4 = json.loads((ROOT / 'reports/uav_service_v4/seed-plan.json').read_text())
    earlier |= {(s['layout_seed'], s['radio_seed']) for rows in v4['phases'].values() for s in rows}
    assert not earlier & {(s['layout_seed'], s['radio_seed']) for rows in plan['phases'].values() for s in rows}
    selection = json.loads(SELECTION.read_text())
    assert selection == s5.fixed_selection()  # patch and settings unchanged since pre-registration


def test_env_flags(tmp_path):
    config = fixture_config()
    assert s5.env_for('olsr-etx') is s5.EtxRouteEnv and s5.env_for('lq-route') is RouteControlEnv
    command = s5.EtxRouteEnv(config, ['fixture'], tmp_path)._command(tmp_path)
    assert command[-2:] == ['--routeControl=external', '--olsrLinkQuality=etx']
    assert '--olsrLinkQuality=etx' not in RouteControlEnv(config, ['fixture'], tmp_path)._command(tmp_path)


def _collection(tmp_path, phase, etx_gain=.01, route_gain=.03):
    from dataclasses import asdict
    policies = list(s5.V5.policies[phase]); rows = []
    for regime in ('n4', 'n5'):
        for i in range(3):
            for policy in policies:
                frames = service_frames([60] * 240)
                if policy == 'hold':
                    pass  # fixture OLSR route is direct [0, 3]
                else:
                    for f in frames: f['service_observation']['route']['nodes'] = [0, 1, 3]
                pinned = policy == 'lq-route'
                decisions = [dict(step=t, route_control=dict(route=[0, 1, 3] if pinned and t >= 5 else [])) for t in range(240)]
                episode = tmp_path / f'{regime}-{i}-{policy}'; episode.mkdir(parents=True)
                log = ('UAVROUTE 5 3 0 1 3\nUAVROUTE_CHANGES 1\n' if pinned else 'UAVROUTE_CHANGES 0\n')
                if policy == 'olsr-etx': log = 'UAVOLSR link_quality=etx\n' + log
                (episode / 'simulator.stderr.log').write_text(log)
                pdr = .85 + {'hold': 0., 'olsr-etx': etx_gain, 'lq-route': etx_gain + route_gain}[policy] + .001 * i
                rows.append(dict(cell_id=f'{phase}-{regime}-gradual-{i:02d}__{policy}', regime=regime, context='gradual',
                                 layout_seed=10 * i + (regime == 'n5'), radio_seed=10 * i, policy=policy, status='ok',
                                 config=asdict(fixture_config()), trace_sha256='t' * 64, actual_udp_pdr=pdr,
                                 total_relay_travel_m=0., frames=frames, decisions=decisions, episode_dir=str(episode)))
    meta = dict(phase=phase, policies=policies, _freeze_sha256='f' * 64, selection_sha256='s' * 64)
    return meta, dict(planned_cells=len(rows), ok_cells=len(rows)), rows


@pytest.mark.parametrize('phase', ['validation', 'test'])
def test_analyze_end_to_end(tmp_path, monkeypatch, phase):
    import src.control.service_v3.study as v3study
    from src.control.service_v5.analysis import analyze
    collection = _collection(tmp_path / 'runs', phase)
    monkeypatch.setattr(v3study, 'load_traces', lambda *a, **k: collection)
    result = analyze('protocol.json', 'runs', tmp_path / 'analysis')
    assert result['route_integrity_ok'] and result['olsr_etx_flags_ok']
    assert result['route_mix']['hold']['seconds_share'] == {'direct': 1.0}
    if phase == 'validation':
        go = json.loads((tmp_path / 'analysis' / 'go.json').read_text())
        assert go['go'] and go['schema'] == 'service-v5-go-v1'
    else:
        assert [s['success'] for s in result['fixed_sequence']] == [True, True]


def test_etx_flag_must_match_arm(tmp_path):
    episode = tmp_path / 'e'; episode.mkdir()
    (episode / 'simulator.stderr.log').write_text('UAVROUTE_CHANGES 0\n')
    assert s5.etx_active(dict(policy='hold', episode_dir=str(episode)))
    assert not s5.etx_active(dict(policy='olsr-etx', episode_dir=str(episode)))


@pytest.mark.skipif(not V2_MODEL.is_file(), reason='formal v2 predictor is kept outside git')
def test_freeze_and_factory(tmp_path):
    from src.control.prevention.config import PreventionConfig
    from src.control.prevention.prediction import LearnedPredictor
    from src.control.service_v2.prediction import ServicePredictor
    validation = s5.load(p3.freeze('validation', PLAN, tmp_path / 'validation', V2_MODEL, REFERENCE, [sys.executable],
                                   'ns3 fixture', 'cc fixture', selection=SELECTION, study=s5.V5))
    assert validation['expected_episodes'] == 96 and 'service_v5.cli' in ' '.join(validation['run_argv'])
    factory = s5.make_factory(validation, ServicePredictor.load(V2_MODEL), LearnedPredictor.load(REFERENCE))
    config = PreventionConfig(**validation['cells'][0]['config'])
    arms = {p: factory(p, config, predictor=None, threshold=.5, routing_delay_s=3.) for p in validation['policies']}
    assert arms['lq-route'].decider.margin == 1.0 and not hasattr(arms['olsr-etx'], 'decider')

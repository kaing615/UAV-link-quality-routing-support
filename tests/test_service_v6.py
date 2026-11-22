import json
from pathlib import Path

import pytest

from src.control.service_v3 import protocol as p3
from src.control.service_v6 import study as s6
from tests.service_v2_support import fixture_config
from tests.test_service_v5 import _collection

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'reports/uav_service_v6/seed-plan.json'


def test_seed_plan_and_fixed_selection():
    plan = p3.load_plan(PLAN, s6.V6)
    assert plan['phases'] == p3.build_phases(s6.V6)
    assert 'reports/uav_service_v5/seed-plan.json' in plan['historical_audit']['files']
    assert json.loads((ROOT / 'reports/uav_service_v6/selection.json').read_text()) == s6.fixed_selection()


def test_every_arm_carries_arp_fix(tmp_path):
    config = fixture_config()
    for policy in s6.V6.policies['test']:
        command = s6.env_for(policy)(config, ['fixture'], tmp_path)._command(tmp_path)
        assert command[-1] == s6.ARP_FLAG and ('--olsrLinkQuality=etx' in command) == (policy == 'olsr-etx')


def _v6_collection(tmp_path, phase, blackout=None):
    meta, report, rows = _collection(tmp_path, phase)
    for r in rows:
        (Path(r['episode_dir']) / 'config.json').write_text(json.dumps(dict(command=['sim', s6.ARP_FLAG])))
        if r['cell_id'] == blackout:
            for f in r['frames']: f['service_observation']['window']['sink_rx_packets'] = 0
    return meta, report, rows


@pytest.mark.parametrize('phase,blackout,go', [('validation', None, True), ('validation', 'validation-n5-gradual-01__olsr-etx', False),
                                              ('test', None, None)])
def test_analyze_end_to_end(tmp_path, monkeypatch, phase, blackout, go):
    import src.control.service_v3.study as v3study
    from src.control.service_v6.analysis import analyze
    collection = _v6_collection(tmp_path / 'runs', phase, blackout)
    monkeypatch.setattr(v3study, 'load_traces', lambda *a, **k: collection)
    result = analyze('protocol.json', 'runs', tmp_path / 'analysis')
    assert result['arp_fix_ok'] and result['startup_blackouts'] == ([blackout] if blackout else [])
    if phase == 'validation':
        saved = json.loads((tmp_path / 'analysis' / 'go.json').read_text())
        assert saved['go'] is go and saved['schema'] == 'service-v6-go-v1'
    else:
        assert [s['success'] for s in result['fixed_sequence']] == [True, True]


def test_missing_arp_flag_detected(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps(dict(command=['sim'])))
    assert not s6.arp_active(dict(episode_dir=str(tmp_path)))

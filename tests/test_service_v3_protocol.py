import json
from pathlib import Path

import pytest

from src.control.service_v3 import protocol as v3

PLAN = Path(__file__).resolve().parents[1] / 'reports/uav_service_v3/seed-plan.json'


def test_committed_plan_is_valid_and_matches_generator():
    plan = v3.load_plan(PLAN)
    assert plan['phases'] == v3.build_phases()
    assert sum(len(plan['phases'][p]) * len(v3.POLICIES[p]) for p in v3.COUNTS) == 496


def test_audit_rejects_historical_collision(tmp_path):
    seed = v3.build_phases()['test'][0]['layout_seed']
    old = tmp_path / 'old.json'
    old.write_text(json.dumps({'phases': {'formal': [{'layout_seed': seed, 'radio_seed': 1}]}}))
    with pytest.raises(ValueError, match='Historical seed collision'):
        v3.audit(v3.build_phases(), [old])


def test_cross_phase_seed_reuse_rejected():
    phases = v3.build_phases()
    phases['test'][0] = dict(phases['test'][0], layout_seed=phases['validation'][0]['layout_seed'],
                             radio_seed=phases['validation'][0]['radio_seed'])
    with pytest.raises(ValueError, match='reused'):
        v3.validate_phases(phases)


def test_tampered_plan_rejected(tmp_path):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    plan['phases']['test'][0]['id'] = 'test-n4-static-99'
    bad = tmp_path / 'plan.json'
    bad.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='integrity'):
        v3.load_plan(bad)

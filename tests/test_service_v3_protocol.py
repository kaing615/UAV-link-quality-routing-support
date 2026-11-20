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


V2_MODEL = Path(__file__).resolve().parents[1] / 'reports/uav_service_prevention_v2_2026-11-12/model/predictor.json'
REFERENCE = Path(__file__).resolve().parents[1] / 'reports/uav_prevention_pilot_2026-11-11/predictor.json'
needs_model = pytest.mark.skipif(not V2_MODEL.is_file(), reason='formal v2 predictor is kept outside git')


def _freeze(tmp_path, phase, **extra):
    import sys
    return v3.freeze(phase, PLAN, tmp_path / phase, V2_MODEL, REFERENCE, [sys.executable], 'ns-3 fixture', 'compiler fixture', **extra)


@needs_model
def test_freeze_load_and_factory_for_every_arm(tmp_path):
    from src.control.prevention.config import PreventionConfig
    from src.control.prevention.prediction import LearnedPredictor
    from src.control.service_v2.prediction import ServicePredictor
    from src.control.service_v3.actionvalue import FEATURES, Ridge
    from src.control.service_v3.study import make_factory
    from src.control.service_v2.protocol import write_json
    tuning = v3.load(_freeze(tmp_path, 'tuning'))
    assert tuning['expected_episodes'] == 96 and tuning['selection_sha256'] is None
    with pytest.raises(ValueError, match='Selection required'):
        _freeze(tmp_path, 'validation')
    ridge = Ridge([0.] * len(FEATURES), [1.] * len(FEATURES), [0.] * len(FEATURES), .01).to_dict()
    selection = tmp_path / 'selection.json'
    write_json(selection, dict(schema='service-v3-tuning-selection-v1', predictor_sha256=v3.V2_MODEL_SHA256,
                               risk_on=.3, risk_off=.15, horizon=20, lambda_per_m=1e-4, ridge=ridge))
    validation = v3.load(_freeze(tmp_path, 'validation', selection=selection))
    assert validation['expected_episodes'] == 160
    factory = make_factory(validation, ServicePredictor.load(V2_MODEL), LearnedPredictor.load(REFERENCE))
    config = PreventionConfig(**validation['cells'][0]['config'])
    arms = {p: factory(p, config, predictor=None, threshold=.5, routing_delay_s=3.) for p in validation['policies']}
    gate = arms['service-v3-gate'].controller
    assert gate.method.risk_on == .3 and gate.action_gate is not None
    assert arms['service-v3-threshold'].controller.action_gate is None
    with pytest.raises(ValueError, match='inherit'):
        _freeze(tmp_path, 'test', selection=selection)
    (tmp_path / 'validation' / 'selection.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen file mismatch'):
        v3.load(tmp_path / 'validation' / 'protocol.json')

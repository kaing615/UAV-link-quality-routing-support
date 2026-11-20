from types import SimpleNamespace

import numpy as np
import pytest

from src.control.service_v2.controller import supported_method, threshold_method
from src.control.service_v2.config import ServiceV2Config
from src.control.service_v3 import actionvalue as av
from src.control.service_v3.analysis import blocks, contrast, fixed_sequence
from tests.service_v2_support import service_frames


def test_threshold_method_only_changes_hysteresis():
    m = threshold_method(.3, .15)
    assert (m.risk_on, m.risk_off) == (.3, .15) and supported_method(m) and supported_method(ServiceV2Config())
    object.__setattr__(m, 'burst_commands', 9)
    assert not supported_method(m)
    with pytest.raises(ValueError):
        threshold_method(.2, .2)


def _decision(step, plan=None, ratios=None, risks=None):
    return dict(step=step, start=True, context='forecast', risks=risks or {'10': .6, '20': .7},
                service_state=dict(ratios=ratios or dict(A1=1., A5=.9, X5=1., Q5=1.)),
                plan=plan or dict(total_motion=20., gain=3., tau=4.))


def test_burst_labels_use_matched_hold_window():
    hold = service_frames([60] * 60 + [30] * 60 + [60] * 120)   # hold loses half the packets in steps 61..120
    moved = service_frames([60] * 240)
    rows, excluded = av.burst_labels(moved, [_decision(60), _decision(230), _decision(10, ratios=dict(A1=None))],
                                     hold, ('n4', 1, 2))
    assert len(rows) == 1 and rows[0]['y'] == pytest.approx(.5) and rows[0]['motion'] == 20.
    assert excluded == {'window_unavailable': 1, 'missing_feature': 1}


def test_ridge_recovers_linear_gain_and_round_trips():
    rng = np.random.default_rng(0); X = rng.normal(size=(200, len(av.FEATURES)))
    y = .02 * X[:, 0] - .01 * X[:, 6] + .005
    model = av.Ridge.fit(X, y, alpha=1e-6)
    assert np.allclose(model.predict(X), y, atol=1e-6)
    again = av.Ridge.from_dict(model.to_dict())
    assert np.allclose(again.predict(X[:3]), model.predict(X[:3]))


def test_lambda_selection_prefers_larger_lambda_on_ties():
    rows = [dict(y=.02, motion=10.), dict(y=-.03, motion=300.)]
    lam, table = av.select_lambda(rows, [.01, .01])
    assert lam == 1e-3 and [t['allowed'] for t in table] == [2, 1, 1, 1, 1]


def test_horizon_and_threshold_from_tuning_scores():
    origins = [dict(label=int(i % 4 == 0), projected_risk=(.8 if i % 4 == 0 else .2) + i * 1e-4) for i in range(40)]
    scores = dict(horizons={'10': dict(pr_auc=.4, origins=[]), '20': dict(pr_auc=.9, origins=origins)})
    horizon, _, point = av.select_horizon_threshold(scores)
    assert horizon == 20 and point.recall == 1 and point.fp == 0 and .2 < point.threshold <= .81


def test_gain_gate_blocks_low_value_and_missing_features():
    ridge = av.Ridge(np.zeros(9), np.ones(9), np.zeros(9), .01).to_dict()
    selection = dict(ridge=ridge, risk_on=.4, horizon=20, lambda_per_m=1e-3)
    gate = av.GainGate(selection)
    state = SimpleNamespace(ratios=dict(A1=1., A5=1., X5=1., Q5=1.))
    risks = {10: .5, 20: .5}
    assert gate(risks=risks, plan=dict(total_motion=5., gain=3., tau=4.), frame={}, state=state)[0]
    assert not gate(risks=risks, plan=dict(total_motion=50., gain=3., tau=4.), frame={}, state=state)[0]  # .01 - .05 < 0
    assert gate(risks=risks, plan=dict(total_motion=5.), frame={}, state=state) == (False, dict(reason='missing_feature'))


def _rows(effect):
    rows = []
    for regime in ('n4', 'n5'):
        for i in range(6):
            base = .8 + .01 * i
            for policy, pdr, m in (('hold', base, 0.), ('service-v2', base - .01, 60.), ('service-v2-no-forecast', base, 20.),
                                   ('service-v3-threshold', base, 40.), ('service-v3-gate', base + effect, 10.)):
                rows.append(dict(regime=regime, context='gradual', layout_seed=i, radio_seed=i, policy=policy,
                                 status='ok', actual_udp_pdr=pdr, total_relay_travel_m=m + i))
    return rows


def test_contrast_and_fixed_sequence():
    policies = ('hold', 'service-v2', 'service-v2-no-forecast', 'service-v3-threshold', 'service-v3-gate')
    good = fixed_sequence(blocks(_rows(.02), policies))
    assert [s['success'] for s in good] == [True, True, True] and all(s['interpretable'] for s in good)
    assert good[0]['contrast']['point'] == pytest.approx(.02)
    bad = fixed_sequence(blocks(_rows(-.02), policies))
    assert not bad[0]['success'] and bad[1]['success'] and not bad[1]['interpretable']
    rows = _rows(.02); rows[-1]['status'] = 'technical_error'
    c = contrast(blocks(rows, policies), 'service-v3-gate', 'hold', 'actual_udp_pdr')
    assert c['regimes']['n5']['complete'] == 5 and c['regimes']['n5']['missing_sensitivity'][0] < c['point']


V2_MODEL = 'reports/uav_service_prevention_v2_2026-11-12/model/predictor.json'


@pytest.mark.skipif(not __import__('pathlib').Path(V2_MODEL).is_file(), reason='formal v2 predictor is kept outside git')
def test_real_controller_gate_and_logged_features_agree():
    from src.control.service_v2.controller import ServiceV2Controller
    from src.control.service_v2.prediction import ServicePredictor
    from tests.service_v2_support import fixture_config
    predictor, config = ServicePredictor.load(V2_MODEL), fixture_config()
    frames = service_frames([60] * 30 + [0] * 15 + [60] * 195, offers=[63] * 120 + [62] * 120)
    for f in frames:
        f['positions'] = [[300., 500., 100.], [336., 600., 100.], [372., 600., 100.], [408., 700., 100.]]

    def run(intercept=None):
        selection = dict(ridge=av.Ridge(np.zeros(9), np.ones(9), np.zeros(9), intercept or 0.).to_dict(),
                         risk_on=1e-3, horizon=20, lambda_per_m=1e-3)
        gate = None if intercept is None else av.GainGate(selection)
        controller = ServiceV2Controller('service-v2', config, predictor, threshold_method(1e-3, 5e-4),
                                         'a' * 64, 'b' * 64, action_gate=gate)
        return [controller.act(f)[1] for f in frames[:-1]]

    threshold, allowed, blocked = run(), run(.5), run(-.5)
    starts = [d['step'] for d in av.forecast_starts(threshold)]
    assert starts and starts == [d['step'] for d in av.forecast_starts(allowed)]
    assert not av.forecast_starts(blocked) and any(d['reason'] == 'action_value_gate' for d in blocked)
    first = av.forecast_starts(allowed)[0]
    assert av.logged_features(first) is not None and first['action_value_gate']['predicted_gain'] == .5
    rows, _ = av.burst_labels(frames, threshold, frames, ('n4', 1, 2))
    assert rows and all(r['y'] == 0 for r in rows)  # identical traces: zero matched-window gain


def test_build_selection_records_every_choice():
    origins = [dict(label=int(i % 4 == 0), projected_risk=.8 if i % 4 == 0 else .2) for i in range(40)]
    scores = dict(horizons={'10': dict(pr_auc=.3, origins=origins), '20': dict(pr_auc=.3, origins=origins)})
    rng = np.random.default_rng(1)
    labels = [dict(group=['n4', g, g], features=rng.normal(size=9).tolist(), y=float(rng.normal() / 50), motion=30.)
              for g in range(6) for _ in range(3)]
    s = av.build_selection(scores, labels, {'window_unavailable': 2}, tuning_freeze_sha256='f' * 64, predictor_sha256='p' * 64)
    assert s['horizon'] == 20 and s['risk_off'] == s['risk_on'] / 2 and s['labelled_bursts'] == 18
    assert s['lambda_per_m'] in av.LAMBDA_GRID and s['gate']['motion_cost_weight'] == s['lambda_per_m']
    assert len(s['loso']['predictions']) == 18 and av.Ridge.from_dict(s['ridge'])

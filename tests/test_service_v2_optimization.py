import numpy as np
import pytest


def test_threshold_selection_is_validation_only_and_meets_recall_floor():
    from src.control.service_v2.optimization import select_operating_point
    point = select_operating_point([0, 0, 1, 1, 1], [.1, .4, .45, .7, .9],
                                   min_recall=2 / 3, false_positive_cost=2,
                                   false_negative_cost=1)
    assert point.selection_source == 'validation-only'
    assert point.recall >= 2 / 3
    assert point.threshold in {1.0, .1, .4, .45, .7, .9, 0.0}


@pytest.mark.parametrize('labels,probabilities', [
    ([0, 1], [0.1]), ([0, 2], [.1, .2]), ([0, 1], [float('nan'), .2]),
])
def test_threshold_selection_rejects_invalid_rows(labels, probabilities):
    from src.control.service_v2.optimization import select_operating_point
    with pytest.raises(ValueError):
        select_operating_point(labels, probabilities)


def test_action_value_gate_requires_service_gain_and_rejects_costly_action():
    from src.control.service_v2.optimization import ActionValueGate
    gate = ActionValueGate(risk_threshold=.5, motion_cost_weight=.01,
                           min_predicted_gain=.02, min_net_benefit=.01)
    allowed, diagnostic = gate(risks={10: .8, 20: .7},
                                plan={'total_motion': 2., 'predicted_service_gain': .04},
                                frame={}, state=None)
    assert allowed is True
    assert diagnostic['net_benefit'] == pytest.approx(.02)
    allowed, diagnostic = gate(risks={10: .8, 20: .7},
                                plan={'total_motion': 10., 'predicted_service_gain': .04},
                                frame={}, state=None)
    assert allowed is False
    assert diagnostic['reason'] if 'reason' in diagnostic else diagnostic['net_benefit'] < .01
    assert gate(risks={20: .8}, plan={'total_motion': 1., 'predicted_service_gain': .04},
                frame={}, state=None)[0] is False


def test_advance_motion_action_gate_is_opt_in_and_does_not_change_default():
    from copy import deepcopy
    from src.control.service_v2.controller import MotionMemory, advance_motion
    from src.control.service_v2.planner import enumerate_plans
    from src.control.service_v2.config import ServiceV2Config
    from tests.test_service_v2_controller import geometry_frame, ready_state
    from tests.service_v2_support import fixture_config
    config, method, frame, state = fixture_config(), ServiceV2Config(), geometry_frame(), ready_state()
    assessment = enumerate_plans(frame, config, method, 'service-v2')
    kwargs = dict(memory=MotionMemory(), state=state, risks={10: .75, 20: .75},
                  assessment=assessment, frame=deepcopy(frame), config=config,
                  method=method, mode='service-v2')
    baseline_first = advance_motion(**kwargs)
    gated_first = advance_motion(**kwargs, action_gate=lambda **_: (False, {'reason': 'test'}))
    baseline = advance_motion(**{**kwargs, 'memory': baseline_first.next_memory})
    gated = advance_motion(**{**kwargs, 'memory': gated_first.next_memory},
                           action_gate=lambda **_: (False, {'reason': 'test'}))
    assert np.any(baseline.action)
    assert not np.any(gated.action)
    assert gated.diagnostic['reason'] == 'action_value_gate'

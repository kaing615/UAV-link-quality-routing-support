"""Public motion seams: hand geometry and causal state traces, no simulator."""
import numpy as np
import pytest

from src.control.service_v2.planner import encode_displacement
from tests.service_v2_support import fixture_config, service_frames
from src.control.service_v2.config import ServiceV2Config
from src.control.service_v2.observation import ServiceHistory

METHOD = ServiceV2Config()


def geometry_frame(moving=False):
    f = service_frames([60] * 7)[-1]
    f['positions'] = [[300., 500., 100.], [336., 600., 100.],
                      [372., 600., 100.], [408., 700., 100.]]
    if moving:
        f['velocities'][0][1], f['velocities'][3][1] = -1., 1.
    return f


def test_remaining_two_target_one_moves_one():
    action = encode_displacement(np.array([1., 0., 0.]), 2., 8.)
    np.testing.assert_allclose(action, [.5, 0., 0.], atol=1e-8)
    np.testing.assert_allclose(2. * action, [1., 0., 0.], atol=1e-8)


@pytest.mark.parametrize('budget', [0., .25, 2., 8., 20.])
def test_scaling_below_at_and_above_feasible_limit(budget):
    limit = min(2., budget)
    for distance in (0., limit / 2, limit):
        d = np.array([.6, .8, 0.]) * distance
        a = encode_displacement(d, budget, 8.)
        np.testing.assert_allclose(min(8., budget) * a, d, atol=1e-8)
        assert np.linalg.norm(a) <= 1 + 1e-8
    with pytest.raises(ValueError):
        encode_displacement(np.array([limit + .01, 0., 0.]), budget, 8.)


@pytest.mark.parametrize('displacement,budget,speed', [([np.nan, 0, 0], 2, 8),
    ([0, 0], 2, 8), ([0, 0, 0], -1, 8), ([0, 0, 0], 1, 0),
    ([0, 0, 0], np.inf, 8), ([0, 0, 0], 2, np.nan)])
def test_encoder_refuses_malformed_values(displacement, budget, speed):
    with pytest.raises(ValueError):
        encode_displacement(np.asarray(displacement), budget, speed)


@pytest.mark.parametrize('hold,move,motion,expected', [(10, 8, 20, True),
    (10, 8.01, 5, False), (10, 8, 20.1, False), (10, 8, 0, False),
    (106.282642, 108, 12, False)])
def test_independent_benefit_boundaries(hold, move, motion, expected):
    from src.control.service_v2.planner import benefit_accepts
    assert benefit_accepts(hold, move, motion) is expected


@pytest.mark.parametrize('moving,gain', [(False, 5.624805787), (True, 5.645066285)])
def test_hand_geometry_accepts_central_hop_growth(moving, gain):
    from src.control.service_v2.planner import enumerate_plans
    result = enumerate_plans(geometry_frame(moving), fixture_config(), METHOD, 'service-v2')
    assert len(result.candidates) == 4  # three subsets plus explicit hold
    plan = next(c for c in result.candidates if c['chain'] == [0, 1, 2, 3])
    assert plan['accepted']
    assert plan['gain'] == pytest.approx(gain, abs=1e-8)
    assert plan['total_motion'] == pytest.approx(12)
    assert plan['hop_gains'][1] == pytest.approx(-1.947331922, abs=1e-8)
    assert len(plan['rollout']) == 3
    assert plan['tau'] == pytest.approx(max(plan['full_target_distances']) / 2)
    assert result.chosen['rank'] == min(c['rank'] for c in result.candidates if c['accepted'])


def test_candidate_affordability_safety_bounds_and_time():
    from copy import deepcopy
    from src.control.service_v2.planner import enumerate_plans
    config = fixture_config()
    f = geometry_frame()
    f['travel'][1] = f['travel'][2] = 399.75
    result = enumerate_plans(f, config, METHOD, 'service-v2')
    assert result.chosen is None and result.rejection_counts['budget'] == 3
    f = geometry_frame()
    f['step'] = 239
    assert enumerate_plans(f, config, METHOD, 'service-v2').chosen is None
    f = geometry_frame()
    f['positions'][2] = deepcopy(f['positions'][1])
    assert enumerate_plans(f, config, METHOD, 'service-v2').chosen is None
    f = geometry_frame(True)
    f['velocities'][0][1] = -1000
    result = enumerate_plans(f, config, METHOD, 'service-v2')
    assert all(0 <= c['endpoint_estimates'][0][1] <= config.y_max
               for c in result.candidates if c['kind'] != 'hold')
    assert any(c['endpoint_clipped'] for c in result.candidates if c['kind'] != 'hold')


def test_finite_candidates_include_observed_route_order_and_rank():
    from src.control.service_v2.planner import enumerate_plans
    f = geometry_frame()
    f['service_observation']['route']['nodes'] = [0, 2, 1, 3]
    result = enumerate_plans(f, fixture_config(), METHOD, 'service-v2')
    assert [c['chain'] for c in result.candidates] == [[0, 3], [0, 1, 3], [0, 2, 3], [0, 1, 2, 3], [0, 2, 1, 3]]
    f['positions'][1][0] = np.nan
    with pytest.raises(ValueError):
        enumerate_plans(f, fixture_config(), METHOD, 'service-v2')


def test_n5_exhaustive_subset_inventory_and_additional_measured_order():
    from copy import deepcopy
    from dataclasses import replace
    from src.control.service_v2.planner import enumerate_plans
    config = replace(fixture_config(), num_uavs=5, destination=4, initial_spacing_fraction=.10)
    f = geometry_frame()
    f['positions'] = [[300., 500., 100.], [324., 600., 100.],
        [348., 600., 100.], [372., 600., 100.], [396., 700., 100.]]
    f['velocities'] = [[0., 0., 0.]] * 5
    f['travel'] = [0.] * 5
    f['service_observation']['route']['nodes'] = [0, 3, 2, 1, 4]
    result = enumerate_plans(f, config, METHOD, 'service-v2')
    assert [p['chain'] for p in result.candidates] == [[0, 4], [0, 1, 4], [0, 2, 4],
        [0, 3, 4], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [0, 1, 2, 3, 4], [0, 3, 2, 1, 4]]
    assert result.chosen['chain'] == [0, 1, 2, 3, 4]
    unchanged = deepcopy(f)
    enumerate_plans(f, config, METHOD, 'service-v2-no-motion-gate')
    assert f == unchanged


@pytest.mark.parametrize('observation', [None, {'route': None}, {'route': {'nodes': None}}])
def test_planner_refuses_malformed_observed_route(observation):
    from src.control.service_v2.planner import enumerate_plans
    f = geometry_frame()
    f['service_observation'] = observation
    with pytest.raises(ValueError):
        enumerate_plans(f, fixture_config(), METHOD, 'service-v2')


def ready_state():
    history = ServiceHistory(fixture_config(), METHOD)
    for frame in service_frames([60] * 7):
        state = history.update(frame)
    return state


def transition(memory=None, state=None, risks=None, frame=None, policy='service-v2'):
    from src.control.service_v2.planner import enumerate_plans
    from src.control.service_v2.controller import MotionMemory, advance_motion
    frame = geometry_frame() if frame is None else frame
    return advance_motion(memory or MotionMemory(), state or ready_state(),
        {10: .75, 20: .75} if risks is None else risks,
        enumerate_plans(frame, fixture_config(), METHOD, policy), frame,
        fixture_config(), METHOD, policy)


def test_prescribed_burst_and_five_full_cooldown_intervals():
    memory, moving, cooldown = None, [], []
    for i in range(13):
        decision = transition(memory)
        memory = decision.next_memory
        if np.linalg.norm(decision.action):
            moving.append(i)
        if decision.diagnostic['motion_mode'] == 'COOLDOWN':
            cooldown.append(i)
    assert moving == [1, 2, 3, 10, 11, 12]
    assert cooldown[:5] == [4, 5, 6, 7, 8]


def test_two_low_predictions_stop_and_rebuild_after_cooldown():
    memory = transition().next_memory
    memory = transition(memory).next_memory
    first = transition(memory, risks={10: .1, 20: .2})
    assert np.linalg.norm(first.action) > 0
    second = transition(first.next_memory, risks={10: .1, 20: .2})
    assert np.linalg.norm(second.action) == 0
    assert second.diagnostic['reason'] == 'low_risk'
    assert second.next_memory.remaining_cooldown_intervals == 4


@pytest.mark.parametrize('change,reason', [('missing', 'invalid_observation'),
    ('malformed_route', 'invalid_observation'),
    ('safety', 'simulator_safety_rejected'), ('budget', 'no_movable_budget'),
    ('separation', 'swept_separation'), ('targets', 'targets_reached')])
def test_active_burst_immediate_stops(change, reason):
    from copy import deepcopy
    from src.control.service_v2.planner import PlanAssessment
    from src.control.service_v2.controller import advance_motion
    memory = transition(transition().next_memory).next_memory
    f = geometry_frame()
    if change == 'missing':
        del f['service_observation']
    elif change == 'malformed_route':
        f['service_observation']['route'] = None
    elif change == 'safety':
        f['safety_rejected'] = True
    elif change == 'budget':
        f['travel'][1] = f['travel'][2] = 400.
    elif change == 'separation':
        f['positions'][2] = deepcopy(f['positions'][1])
    else:
        for key, target in memory.active_plan['targets'].items():
            f['positions'][int(key)] = target
    decision = advance_motion(memory, ready_state(), {10: .75, 20: .75},
        PlanAssessment([], None, {}), f, fixture_config(), METHOD, 'service-v2')
    assert not np.any(decision.action)
    assert decision.diagnostic['reason'] == reason


def test_actual_startup_route_only_and_forecast_to_restoration_contexts():
    from tests.service_v2_support import fail_route
    for arrivals, fail, expected in [([0] * 7, False, 'startup_restoration'),
                                    ([60] * 7 + [60] * 3, True, 'routing_restoration')]:
        fs = service_frames(arrivals)
        if fail:
            for f in fs[-3:]:
                fail_route(f)
        history = ServiceHistory(fixture_config(), METHOD)
        for f in fs:
            state = history.update(f)
        decision = transition(state=state, risks={})
        assert np.any(decision.action)
        assert decision.diagnostic['context'] == expected
    history = ServiceHistory(fixture_config(), METHOD)
    for f in service_frames([60] * 7 + [0] * 3):
        state = history.update(f)
    memory = transition(transition().next_memory).next_memory
    converted = transition(memory, state=state)
    assert converted.diagnostic['context'] == 'delivery_restoration'
    final = transition(converted.next_memory, state=state)
    assert np.any(final.action)
    assert final.next_memory.remaining_cooldown_intervals == 5


def test_marginal_service_suppresses_forecast_and_urgency_is_inclusive():
    from copy import deepcopy
    from src.control.service_v2.planner import enumerate_plans, PlanAssessment
    from src.control.service_v2.controller import MotionMemory, advance_motion
    history = ServiceHistory(fixture_config(), METHOD)
    for f in service_frames([60] * 7 + [56] * 5):
        state = history.update(f)
    assert state.activation_context == 'marginal_service'
    assert not np.any(transition(state=state).action)
    f = geometry_frame()
    assessment = enumerate_plans(f, fixture_config(), METHOD, 'service-v2')
    for tau, moves in [(5., True), (4.999, False)]:
        plan = deepcopy(assessment.chosen)
        plan['tau'] = tau
        chosen = PlanAssessment(assessment.candidates, plan, {})
        first = advance_motion(MotionMemory(), ready_state(), {10: .75, 20: .75}, chosen,
            f, fixture_config(), METHOD, 'service-v2')
        second = advance_motion(first.next_memory, ready_state(), {10: .75, 20: .75}, chosen,
            f, fixture_config(), METHOD, 'service-v2')
        assert bool(np.any(second.action)) is moves


def test_ablations_preserve_initial_arming_and_remove_only_declared_gates():
    from tests.service_v2_support import fail_route
    ready = ready_state()
    no_forecast = transition(policy='service-v2-no-forecast')
    assert not np.any(transition(no_forecast.next_memory, policy='service-v2-no-forecast').action)
    h = ServiceHistory(fixture_config(), METHOD)
    for f in service_frames([0] * 7):
        startup = h.update(f)
    assert not np.any(transition(state=startup, policy='service-v2-no-online-service-state').action)
    h = ServiceHistory(fixture_config(), METHOD)
    fs = service_frames([60] * 7 + [0] * 6)
    for f in fs:
        failed = h.update(f)
    first = transition(state=failed, policy='service-v2-no-online-service-state')
    second = transition(first.next_memory, state=failed, policy='service-v2-no-online-service-state')
    assert np.any(second.action) and second.diagnostic['context'] == 'forecast'
    assert second.diagnostic['training_eligibility_support'] is False
    active = transition(transition().next_memory)
    restored = transition(active.next_memory, state=failed, policy='service-v2-no-online-service-state')
    assert restored.diagnostic['context'] == 'forecast'
    # Ungated replans each interval and has no activation delay/cap/cooldown.
    memory = None
    for _ in range(12):
        decision = transition(memory, policy='service-v2-no-motion-gate')
        assert np.any(decision.action)
        memory = decision.next_memory
    assert not np.any(transition(memory, risks={10: .49, 20: .49}, policy='service-v2-no-motion-gate').action)
    f = service_frames([60] * 7)[-1]
    f['positions'][1][1] += 2.
    f['positions'][2][1] += 2.
    from src.control.service_v2.planner import enumerate_plans
    gated = enumerate_plans(f, fixture_config(), METHOD, 'service-v2')
    ungated = enumerate_plans(f, fixture_config(), METHOD, 'service-v2-no-motion-gate')
    full = lambda a: next(c for c in a.candidates if c['chain'] == [0, 1, 2, 3])
    assert full(gated)['reason'] == 'deadband'
    assert full(ungated)['accepted']


@pytest.fixture(scope='module')
def real_predictor(tmp_path_factory):
    import hashlib
    from src.control.service_v2.prediction import fit_service_predictor, ServicePredictor
    from tests.service_v2_support import synthetic_partition
    predictor = fit_service_predictor(synthetic_partition('train', 6), synthetic_partition('calibration', 3))
    path = tmp_path_factory.mktemp('immutable-service-v2-artifact') / 'synthetic-fitted.json'
    predictor.save_exclusive(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ServicePredictor.load(path), digest, path


def test_real_controller_inference_causal_state_hashes_and_execution(real_predictor):
    import hashlib
    import json
    from dataclasses import asdict
    from src.control.service_v2.controller import ServiceV2Controller
    predictor, digest, path = real_predictor
    method_hash = hashlib.sha256(json.dumps(asdict(METHOD), sort_keys=True).encode()).hexdigest()
    controller = ServiceV2Controller('service-v2-no-forecast', fixture_config(), predictor, METHOD, digest, method_hash)
    fs = service_frames([60] * 7 + [0] * 8)
    moving = []
    for frame in fs:
        frame['positions'] = geometry_frame()['positions']
        action, log = controller.act(frame)
        assert action.shape == (6,) and np.isfinite(action).all()
        assert log['model_hash'] == digest and log['method_hash'] == method_hash
        assert not (log['start'] and log['context'] == 'forecast')
        if frame['step'] >= 5:
            assert log['risks'] == {str(k): v for k, v in predictor.predict(fs[:frame['step'] + 1], fixture_config()).items()}
            assert len(log['features']) == 25
        if np.any(action):
            moving.append(frame['step'])
            assert log['context'] == 'delivery_restoration'
        json.dumps(log, allow_nan=False)
    assert moving == [10, 11, 12]
    assert log['previous_command']['executed_position_delta'] == [[0., 0., 0.], [0., 0., 0.]]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_real_controller_stops_bad_geometry_and_retains_truthful_ablation_state(real_predictor):
    from src.control.service_v2.controller import ServiceV2Controller
    predictor, digest, _ = real_predictor
    controller = ServiceV2Controller('service-v2-no-online-service-state', fixture_config(), predictor, METHOD, digest, 'f' * 64)
    logs = []
    for frame in service_frames([60] * 7 + [0] * 6):
        frame['positions'] = geometry_frame()['positions']
        _, log = controller.act(frame)
        logs.append(log)
    assert logs[-1]['service_state']['mode'] == 'RESTORE'
    assert logs[-1]['prediction_outside_training_support'] is True
    assert not any(log['start'] and log['context'] != 'forecast' for log in logs)
    bad = service_frames([60] * 14)[-1]
    bad['positions'][1][0] = float('nan')
    action, log = controller.act(bad)
    assert not np.any(action) and log['observation_error']
    bad_route = service_frames([60] * 15)[-1]
    bad_route['service_observation']['route'] = None
    action, log = controller.act(bad_route)
    assert not np.any(action) and log['observation_error']
    assert log['service_state']['step'] == 15
    assert log['service_state']['mode'] == 'UNKNOWN'
    assert log['service_state']['ratios'] == dict(A1=None, A5=None, X5=None, Q5=None)
    assert log['service_state']['healthy_evidence'] is False
    events = log['service_state']['event_onsets']
    resumed = service_frames([60] * 16)[-1]
    action, log = controller.act(resumed)
    assert not np.any(action)
    assert log['service_state']['mode'] == 'UNKNOWN'
    assert log['service_state']['event_onsets'] == events

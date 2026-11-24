import numpy as np

from src.control.service_v9.motion import Motion, position_action, swept_safe, widest
from src.control.service_v9.study import V9, scenario_config
from src.control.service_v9 import protocol
from src.control.service_v3.protocol import build_phases


def test_widest_uses_relay_path_and_does_not_confuse_missing_edge():
    q = np.array([[0, .8, .2], [.8, 0, .7], [.2, .7, 0]])
    assert widest(q, 0, 2) == .7
    q[1, 2] = q[2, 1] = 0
    assert widest(q, 0, 2) == .2


def test_swept_check_catches_crossing_even_when_endpoints_separated():
    assert not swept_safe(np.array([[0., 0, 0], [10., 0, 0]]), np.array([[10., 0, 0], [-10., 0, 0]]), 5.)


def test_motion_is_budget_limited():
    c = scenario_config(dict(regime="n5", context="gradual"))
    f = dict(positions=[[0, 0, 100], [10, 0, 100], [20, 0, 100], [30, 0, 100], [200, 0, 100]],
             velocities=[[0, 0, 0]] * 5, travel=[0, 399.5, 400, 399., 0])
    action, safe, _ = position_action(f, c)
    assert safe
    lengths = np.linalg.norm(action.reshape(-1, 3) * c.max_speed, axis=1)
    np.testing.assert_allclose(lengths, [.5, 0, 1.])


def test_predictive_and_reactive_share_motion_law_but_trigger_differently():
    c = scenario_config(dict(regime="n5", context="gradual"))
    f = dict(positions=[[0, 0, 100], [10, 0, 100], [20, 0, 100], [30, 0, 100], [200, 0, 100]],
             velocities=[[0, 0, 0]] * 5, travel=[0] * 5)
    good = np.ones((5, 5)) - np.eye(5)
    weak = good * .6
    reactive, forecast = Motion(c), Motion(c, True)
    for step in (3, 4):
        f["step"] = step
        observed_action, _ = reactive.act(f, good)
        predictive_action, _ = forecast.act(f, good, weak)
    assert not np.any(observed_action) and np.any(predictive_action)
    reactive2, forecast2 = Motion(c), Motion(c, True)
    for step in (3, 4):
        f["step"] = step
        a, _ = reactive2.act(f, weak)
        b, _ = forecast2.act(f, weak, weak)
        np.testing.assert_array_equal(a, b)


def test_registered_scenario_is_bound_into_cells():
    plan = dict(phases=build_phases(V9))
    cells = protocol._cells(plan, "validation", V9)
    assert len(cells) == 80 and all(c["config"]["mission_speed"] == 2. for c in cells)
    assert all(c["config"]["mission_start_step"] == (30 if c["context"] == "gradual" else 1000) for c in cells)

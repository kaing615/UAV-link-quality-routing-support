"""Policy differences use one hand-checked chain planner and observed graphs."""
import numpy as np

from src.control.prevention.config import PreventionConfig
from src.control.prevention.controller import PreventionController, target_plan


def frame(triangle=False, outage=False):
    return dict(step=1, positions=[[0, 100, 100], [100, 200, 100], [200, 100, 100]],
                velocities=[[0, 0, 0]]*3, travel=[999, 0, 999],
                rx_ratio=[[0, 1, int(triangle)], [1, 0, 1], [int(triangle), 1, 0]],
                rssi=[[0]*3 for _ in range(3)], rssi_mask=[[False]*3 for _ in range(3)],
                components=2 if outage else 1)


class Risk:
    def __init__(self, risk_at=20): self.risk_at = risk_at
    def predict(self, current, previous, comm_range):
        return {h: {(0, 1): float(h >= self.risk_at)} for h in (3, 5, 10, 20)}


def config(**kw): return PreventionConfig(num_uavs=3, source=0, destination=2, **kw)


def test_hold_reactive_and_geometry_share_plan():
    f, c = frame(), config()
    assert not np.any(PreventionController('hold', c).act(f)[0])
    reactive = PreventionController('reactive', c)
    assert not np.any(reactive.act(f)[0])
    f = frame(outage=True); f['step'] = 2
    assert np.any(reactive.act(f)[0])
    action, diagnostic = PreventionController('geometry', c).act(frame())
    np.testing.assert_allclose(action, [0, -1, 0])
    assert diagnostic['required_time_s'] == 15.5  # 100/8 + 3 routing
    assert target_plan(frame(), c)[1] == 12.5


def test_deadline_uses_within_horizon_bound_and_ablation_differs():
    c = config()
    # Risk first crosses at H20; 15.5 seconds needed: wait until H<=15.5.
    controller = PreventionController('predictive-learned', c, predictor=Risk(20))
    action, d = controller.act(frame())
    assert not np.any(action) and d['reason'] == 'risk_outside_action_window'
    action, d = PreventionController('learned-no-deadline', c, predictor=Risk(20)).act(frame())
    assert np.any(action) and d['trigger_horizon_s'] == 20
    action, d = PreventionController('predictive-learned', c, predictor=Risk(10)).act(frame())
    assert np.any(action) and d['trigger_horizon_s'] == 10
    assert d['action_window_reached'] is True and d['first_action_step'] == 1
    assert d['planning_slack_s'] == -5.5  # urgency bound; no exact failure-time claim


def test_bridge_only_vs_all_edges_and_outage_recovery():
    c = config()
    action, _ = PreventionController('predictive-learned', c, predictor=Risk(3)).act(frame(True))
    assert not np.any(action)
    action, _ = PreventionController('learned-no-criticality', c, predictor=Risk(3)).act(frame(True))
    assert np.any(action)
    action, d = PreventionController('predictive-simple', c).act(frame(outage=True))
    assert np.any(action) and d['reason'] == 'observed_outage'


def test_planner_speed_only_simulator_owns_safety_and_budget():
    f = frame(); f['travel'][1] = 400
    action, _ = PreventionController('geometry', config()).act(f)
    assert np.linalg.norm(action) <= 1
    assert np.any(action)  # simulator enforces shared budget/swept safety


def test_simple_never_calls_supplied_learned_and_learned_policies_keep_it():
    from src.control.prevention.prediction import SimplePredictor
    class LearnedSpy(Risk):
        calls = 0
        def predict(self, *args):
            self.calls += 1
            return super().predict(*args)
    learned = LearnedSpy(3)
    simple = PreventionController('predictive-simple', config(), predictor=learned)
    assert isinstance(simple.predictor, SimplePredictor)
    simple_action, _ = simple.act(frame())
    assert learned.calls == 0 and not np.any(simple_action)
    for policy in ('predictive-learned', 'learned-no-criticality', 'learned-no-deadline'):
        controller = PreventionController(policy, config(), predictor=learned)
        assert controller.predictor is learned
        assert np.any(controller.act(frame())[0])
    assert learned.calls == 3

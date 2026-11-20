"""Frame-only rule controllers with a shared explicit geometric target planner.

Earliest H whose within-H event risk crosses the frozen threshold is an urgency
bound, not a failure timestamp. Deadline mode starts motion when H <= maximum
relay travel time + routing_delay_s. H-required_time is logged as planning slack;
negative slack signals urgency, NOT a guaranteed miss (the event time is unknown).
The no-deadline ablation starts on any threshold crossing, including distant H.
No model scores arbitrary candidate actions. Simulator owns all budgets/safety.
"""
import numpy as np

from .features import bridge_edges, measured_edges
from .prediction import HORIZONS, SimplePredictor

POLICIES = ('hold', 'reactive', 'geometry', 'predictive-simple', 'predictive-learned',
            'learned-no-criticality', 'learned-no-deadline')


def target_plan(frame, config):
    """Return normalized relay velocities and max straight-line travel time.

    Rank relays by projection on the current source-destination line; place them
    at equal fractions between endpoints. Use at most one second of max_speed.
    Positions/velocities of mission anchors are observed, never extrapolated from
    scenario.json. Bounds, swept separation and remaining budget are enforced by
    the shared live simulator, identically for all policies.
    """
    positions = np.asarray(frame['positions'], float)
    start, end = positions[config.source], positions[config.destination]
    axis = end-start
    ordered = sorted(config.relay_ids, key=lambda i: (float((positions[i]-start) @ axis), i))
    targets = {node: start+(rank+1)/(len(ordered)+1)*axis for rank, node in enumerate(ordered)}
    actions, distances = [], []
    for node in config.relay_ids:
        delta = targets[node]-positions[node]
        distance = float(np.linalg.norm(delta)); distances.append(distance)
        actions.extend((delta/max(config.max_speed, distance)).tolist())
    return np.asarray(actions, dtype=float), max(distances, default=0.)/config.max_speed


class PreventionController:
    def __init__(self, policy, config, predictor=None, threshold=.5, routing_delay_s=3.):
        if policy not in POLICIES: raise ValueError('Unknown prevention policy: '+policy)
        if not np.isfinite(threshold) or not 0 < threshold < 1: raise ValueError('Threshold must be in (0,1)')
        if not np.isfinite(routing_delay_s) or routing_delay_s < 0: raise ValueError('Invalid routing delay')
        if policy in ('predictive-learned', 'learned-no-criticality', 'learned-no-deadline') and predictor is None:
            raise ValueError('A learned predictor is required')
        self.policy, self.config = policy, config
        self.predictor = predictor if policy in ('predictive-learned', 'learned-no-criticality', 'learned-no-deadline') else SimplePredictor()
        self.threshold, self.routing_delay_s = threshold, routing_delay_s
        self.previous = None
        self.outage_seen = False
        self.was_moving = False
        self.first_action_step = None

    def act(self, frame):
        if self.previous is not None and frame['step'] != self.previous['step']+1:
            raise ValueError('Controller needs consecutive frames; create a new controller per episode')
        plan, travel_time = target_plan(frame, self.config)
        required = travel_time+self.routing_delay_s
        self.outage_seen |= frame['components'] > 1
        diagnostic = dict(step=frame['step'], policy=self.policy, reason='hold',
                          threshold=self.threshold, routing_delay_s=self.routing_delay_s,
                          travel_time_s=travel_time, required_time_s=required,
                          critical_edges=sorted(bridge_edges(frame)), edge_risks=[],
                          trigger_edge=None, trigger_probability=None, trigger_horizon_s=None,
                          planning_slack_s=None, action_window_reached=False,
                          forecast_semantics='event_within_H_under_current_evolution',
                          intervention_started=False, first_action_step=self.first_action_step)
        active = False
        if self.policy == 'geometry': active, diagnostic['reason'] = True, 'geometry_tracking'
        elif self.policy == 'reactive':
            active = self.outage_seen
            diagnostic['reason'] = 'observed_outage' if active else 'waiting_for_outage'
        elif self.policy != 'hold':
            # Continue recovery after an observed outage; never end an episode.
            if self.outage_seen:
                active, diagnostic['reason'] = True, 'observed_outage'
            else:
                edges = set(measured_edges(frame)) if self.policy == 'learned-no-criticality' else bridge_edges(frame)
                risks = self.predictor.predict(frame, self.previous, self.config.comm_range)
                candidates = []
                for horizon in HORIZONS:
                    for edge in sorted(edges):
                        probability = risks[horizon].get(edge, 0.)
                        if not np.isfinite(probability) or not 0 <= probability <= 1:
                            raise ValueError('Invalid forecast probability')
                        diagnostic['edge_risks'].append(dict(edge=edge, horizon_s=horizon, probability=probability))
                        if probability >= self.threshold: candidates.append((horizon, -probability, edge))
                if candidates:
                    horizon, negative_risk, edge = min(candidates)
                    active = self.policy == 'learned-no-deadline' or horizon <= required
                    diagnostic.update(trigger_edge=edge, trigger_probability=-negative_risk, trigger_horizon_s=horizon,
                                      planning_slack_s=horizon-required, action_window_reached=horizon <= required,
                                      reason='forecast_urgent' if active else 'risk_outside_action_window')
                else: diagnostic['reason'] = 'no_critical_risk'
        action = plan if active else np.zeros_like(plan)
        moving = bool(np.linalg.norm(action) > 1e-8)
        if moving and self.first_action_step is None: self.first_action_step = frame['step']
        diagnostic.update(intervention_started=moving and not self.was_moving,
                          first_action_step=self.first_action_step, commanded_motion=moving)
        self.was_moving, self.previous = moving, frame
        return action, diagnostic

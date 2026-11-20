"""Causal service decisions with bounded, fixed-target motion bursts."""
from copy import deepcopy
from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from time import perf_counter

import numpy as np

from .config import ADOPTED_SPEC_SHA256, ServiceV2Config
from .features import _array, service_features, service_feature_diagnostics
from .observation import ServiceHistory, ServiceState, full_window
from .planner import V2_POLICIES, PlanAssessment, _measure, _step, enumerate_plans
from .prediction import ServicePredictor, _predict_details, _valid_hash, _validate_config


@dataclass(frozen=True)
class MotionMemory:
    mode: str = 'HOLD'
    crossing_streak: int = 0
    low_risk_streak: int = 0
    remaining_burst_commands: int = 0
    remaining_cooldown_intervals: int = 0
    active_plan: dict | None = None
    context: str | None = None
    crossing_steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class MotionDecision:
    action: np.ndarray
    diagnostic: dict
    next_memory: MotionMemory


def advance_motion(memory, state, risks, assessment, frame, config, method, mode,
                   action_gate=None):
    """Pure transition. Service state, forecasts and measured geometry are inputs."""
    if mode not in V2_POLICIES:
        raise ValueError('Unsupported v2 policy')
    memory = deepcopy(memory)
    ungated = mode == 'service-v2-no-motion-gate'
    online = mode != 'service-v2-no-online-service-state'
    zero = np.zeros(3 * len(config.relay_ids))
    valid_risks = (set(risks) == set(method.horizons)
        and all(type(risks[h]) in (int, float) and np.isfinite(risks[h]) and 0 <= risks[h] <= 1 for h in risks))
    crossing = min((h for h in method.horizons if valid_risks and risks[h] >= method.risk_on), default=None)
    diagnostic = dict(motion_mode='HOLD', context=memory.context, reason='hold',
        start=False, termination=False, risk_crossing_horizon=crossing,
        crossing_steps=list(memory.crossing_steps), prediction_valid=valid_risks,
        proposed_displacement=zero.reshape(-1, 3).tolist(), sent_action=zero.tolist(),
        proposed_action=zero.tolist(),
        predicted_displacement=zero.reshape(-1, 3).tolist(), plan=memory.active_plan,
        training_eligibility_support=state.forecast_eligible)

    def result(action, next_memory, reason, motion_mode='HOLD'):
        diagnostic.update(reason=reason, motion_mode=motion_mode,
            context=next_memory.context if next_memory.context else diagnostic['context'],
            sent_action=action.tolist())
        return MotionDecision(action, diagnostic, next_memory)

    def stop(reason):
        diagnostic['termination'] = memory.mode == 'BURST'
        if memory.mode == 'BURST' and not ungated:
            next_memory = MotionMemory(mode='COOLDOWN', remaining_cooldown_intervals=method.cooldown_intervals - 1)
            return result(zero, next_memory, reason, 'COOLDOWN')
        return result(zero, MotionMemory(), reason)

    if not ungated and memory.mode == 'COOLDOWN':
        remaining = max(0, memory.remaining_cooldown_intervals - 1)
        next_memory = MotionMemory(mode='COOLDOWN', remaining_cooldown_intervals=remaining) if remaining else MotionMemory()
        return result(zero, next_memory, 'cooldown', 'COOLDOWN')
    try:
        valid_observation = full_window(frame, config, method)
        p, v, budget = _measure(frame, config)
    except (ValueError, KeyError, TypeError, AttributeError):
        return stop('invalid_observation')
    if not valid_observation or state.mode == 'UNKNOWN':
        return stop('invalid_observation')
    if type(frame.get('safety_rejected')) is not bool:
        return stop('invalid_observation')
    if frame['safety_rejected']:
        return stop('simulator_safety_rejected')
    if frame['step'] >= config.steps:
        return stop('episode_end')
    restore = online and state.mode == 'RESTORE'
    eligible = (state.forecast_eligible if online else state.ever_armed and state.valid_history)
    eligible = eligible and mode != 'service-v2-no-forecast'
    if ungated:
        memory = MotionMemory()
    if memory.mode == 'BURST':
        if restore:
            memory = replace(memory, context=state.activation_context)
        elif online and memory.context != 'forecast' and state.recovered:
            return stop('recovered')
        if memory.context == 'forecast':
            if not valid_risks:
                return stop('invalid_prediction')
            low = memory.low_risk_streak + 1 if all(risks[h] < method.risk_off for h in method.horizons) else 0
            memory = replace(memory, low_risk_streak=low)
            if low >= method.forecast_confirmation_frames:
                return stop('low_risk')
        plan = memory.active_plan
    else:
        streak = memory.crossing_streak + 1 if eligible and crossing is not None else 0
        crossing_steps = (*memory.crossing_steps, frame['step'])[-method.forecast_confirmation_frames:] if streak else ()
        memory = replace(memory, crossing_streak=streak, crossing_steps=crossing_steps)
        diagnostic['crossing_steps'] = list(crossing_steps)
        plan = assessment.chosen
        if not restore and not (eligible and crossing is not None and
                streak >= (1 if ungated else method.forecast_confirmation_frames)):
            return result(zero, memory, 'forecast_confirmation' if streak else 'service_ineligible' if not eligible else 'no_crossing')
        if plan is None:
            return result(zero, memory, 'no_plan')
        diagnostic.update(tau=plan['tau'], urgency_limit=plan['tau'] + method.routing_allowance + method.confirmation_allowance)
        if not restore and crossing > diagnostic['urgency_limit'] + method.numerical_tolerance:
            return result(zero, memory, 'warning_not_urgent')
        if not restore and action_gate is not None:
            try:
                allowed, gate_diagnostic = action_gate(
                    risks=risks, plan=plan, frame=frame, state=state)
            except (ValueError, KeyError, TypeError, AttributeError):
                return result(zero, memory, 'invalid_action_gate')
            diagnostic['action_value_gate'] = gate_diagnostic
            if not allowed:
                return result(zero, memory, 'action_value_gate')
        memory = replace(memory, mode='BURST', remaining_burst_commands=method.burst_commands,
            active_plan=deepcopy(plan), context=state.activation_context if restore else 'forecast')
        diagnostic['start'] = True
    diagnostic.update(plan=deepcopy(plan), context=memory.context)
    targets = plan['targets']
    stop_distance = method.numerical_tolerance if ungated else method.stop_deadband
    distances = {int(key): float(np.linalg.norm(np.asarray(target) - p[int(key)])) for key, target in targets.items()}
    if all(d <= stop_distance for d in distances.values()):
        return stop('targets_reached')
    if not any(budget[i] > 0 and distance > stop_distance for i, distance in distances.items()):
        return stop('no_movable_budget')
    q, _, action, safe = _step(p, v, budget, targets, config, method, stop_distance)
    diagnostic['proposed_displacement'] = [(q[i] - p[i]).tolist() for i in config.relay_ids]
    diagnostic['proposed_action'] = action.tolist()
    if not safe:
        return stop('swept_separation')
    if not np.any(action):
        return stop('no_motion')
    diagnostic['predicted_displacement'] = diagnostic['proposed_displacement']
    remaining = memory.remaining_burst_commands - 1
    if ungated:
        next_memory = MotionMemory()
    elif remaining == 0:
        next_memory = MotionMemory(mode='COOLDOWN', remaining_cooldown_intervals=method.cooldown_intervals)
        diagnostic.update(termination=True, stop_after_command='burst_cap')
    else:
        next_memory = replace(memory, remaining_burst_commands=remaining)
    return result(action, next_memory, 'restoration_motion' if memory.context != 'forecast' else 'forecast_motion', 'BURST')


class ServiceV2Controller:
    """Own causal histories; external hashes identify the caller's frozen files.

    Decisions use only copied observed frames. Execution evidence is attributed
    on the next consecutive observation, never inferred from an action norm.
    """

    def __init__(self, policy, config, predictor, method, model_hash, method_hash,
                 action_gate=None):
        if policy not in V2_POLICIES:
            raise ValueError('Unsupported v2 controller role')
        _validate_config(config)
        if not isinstance(method, ServiceV2Config) or method != ServiceV2Config():
            raise ValueError('Unsupported v2 method')
        if not isinstance(predictor, ServicePredictor):
            raise ValueError('A validated service predictor is required')
        if not _valid_hash(model_hash) or not _valid_hash(method_hash):
            raise ValueError('Frozen model and method SHA256 identities required')
        self.policy, self.config, self.predictor, self.method = policy, config, predictor, method
        self.model_hash, self.method_hash = model_hash, method_hash
        if action_gate is not None and not callable(action_gate):
            raise ValueError('action_gate must be callable')
        self.action_gate = action_gate
        artifact = predictor.artifact
        self.artifact_hash = hashlib.sha256(json.dumps(artifact, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.artifact = artifact
        self.history = ServiceHistory(config, method)
        self.frames = deque(maxlen=6)
        self.memory = MotionMemory()
        self.previous_frame, self.previous_decision = None, None
        self.state = ServiceState(-1, 'UNARMED', False, None, False, False, False,
            False, False, False, False, 'initialization', False,
            dict(A1=None, A5=None, X5=None, Q5=None), {}, {})

    def _execution(self, frame):
        if self.previous_frame is None:
            return None
        prior = self.previous_frame
        evidence = dict(command_step=prior.get('step'), observation_step=frame.get('step'),
            sent_action=self.previous_decision.action.tolist(),
            predicted_displacement=self.previous_decision.diagnostic['predicted_displacement'],
            executed_position_delta=None, executed_travel_delta=None,
            observation_consecutive=False, observed_safety_rejected=frame.get('safety_rejected'), reason=None)
        if type(frame.get('step')) is not int or type(prior.get('step')) is not int or frame['step'] != prior['step'] + 1:
            evidence['reason'] = 'nonconsecutive_observation'
            return evidence
        evidence['observation_consecutive'] = True
        try:
            n = self.config.num_uavs
            p, old = _array(frame, 'positions', (n, 3)), _array(prior, 'positions', (n, 3))
            travel, old_travel = _array(frame, 'travel', (n,)), _array(prior, 'travel', (n,))
            evidence['executed_position_delta'] = [(p[i] - old[i]).tolist() for i in self.config.relay_ids]
            evidence['executed_travel_delta'] = [float(travel[i] - old_travel[i]) for i in self.config.relay_ids]
        except ValueError as exc:
            evidence['reason'] = str(exc)
        return evidence

    def act(self, frame):
        started = perf_counter()
        frame = deepcopy(frame)
        execution = self._execution(frame)
        observation_error = None
        try:
            self.state = self.history.update(frame)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            observation_error = str(exc)
            self.state = replace(self.state,
                step=frame['step'] if type(frame.get('step')) is int else self.state.step,
                mode='UNKNOWN', forecast_eligible=False, valid_history=False,
                healthy_evidence=False, recovered=False, degraded_confirmed=False,
                severe_confirmed=False, outage_confirmed=False, route_failure_confirmed=False,
                activation_context='unknown_service', ratios=dict(A1=None, A5=None, X5=None, Q5=None))
            # An unidentifiable frame breaks consecutive history while keeping
            # the irreversible arming latch and any confirmed restoration.
            previous = self.history
            self.history = ServiceHistory(self.config, self.method)
            self.history.ever_armed, self.history.first_armed_step = previous.ever_armed, previous.first_armed_step
            self.history.restoration, self.history.restoration_context = previous.restoration, previous.restoration_context
            self.history.onsets = deepcopy(previous.onsets)
            self.history.confirmations = deepcopy(previous.confirmations)
            self.history.rebuilding = True
            self.frames.clear()
        self.frames.append(frame)
        feature_started = perf_counter()
        features, feature_diagnostic, prediction_detail, risks = None, None, None, {}
        feature_error, prediction_error = None, None
        try:
            n = self.config.num_uavs
            _measure(frame, self.config)
            probes = _array(frame, 'rx_ratio', (n, n))
            _array(frame, 'rssi', (n, n))
            _array(frame, 'rssi_mask', (n, n), boolean=True)
            if np.any((probes < 0) | (probes > 1)):
                raise ValueError('Invalid measured probes')
            if self.state.valid_history:
                features = service_features(list(self.frames), self.config).tolist()
                feature_diagnostic = service_feature_diagnostics(list(self.frames), self.config)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            feature_error = str(exc)
            observation_error = observation_error or feature_error
        feature_elapsed = perf_counter() - feature_started
        predictor_started = perf_counter()
        if features is not None:
            try:
                risks = self.predictor.predict(list(self.frames), self.config)
                prediction_detail = _predict_details(np.asarray(features), self.artifact)
            except (ValueError, KeyError, TypeError) as exc:
                prediction_error = str(exc)
                observation_error = observation_error or prediction_error
        predictor_elapsed = perf_counter() - predictor_started
        planner_started = perf_counter()
        try:
            assessment = enumerate_plans(frame, self.config, self.method, self.policy)
        except (ValueError, KeyError, TypeError) as exc:
            assessment = PlanAssessment([], None, {})
            observation_error = observation_error or str(exc)
        planner_elapsed = perf_counter() - planner_started
        decision_state = replace(self.state, mode='UNKNOWN', forecast_eligible=False) if observation_error else self.state
        motion_started = perf_counter()
        decision = advance_motion(self.memory, decision_state, risks, assessment, frame,
            self.config, self.method, self.policy, action_gate=self.action_gate)
        motion_elapsed = perf_counter() - motion_started
        self.memory = decision.next_memory
        log = dict(decision.diagnostic)
        log.update(policy=self.policy, step=frame.get('step'), service_state=asdict(self.state),
            risks={str(k): v for k, v in risks.items()}, prediction_detail=prediction_detail,
            features=features, feature_diagnostic=feature_diagnostic,
            prediction_outside_training_support=bool(risks) and not self.state.forecast_eligible,
            model_hash=self.model_hash, method_hash=self.method_hash,
            predictor_artifact_canonical_hash=self.artifact_hash, adopted_spec_sha256=ADOPTED_SPEC_SHA256,
            observation_error=observation_error, feature_error=feature_error, prediction_error=prediction_error,
            candidates=assessment.candidates, rejection_counts=assessment.rejection_counts,
            previous_command=execution, next_motion_memory=asdict(self.memory),
            timing_seconds=dict(features=feature_elapsed, predictor=predictor_elapsed,
                planner=planner_elapsed, motion=motion_elapsed, decision=perf_counter() - started))
        self.previous_frame, self.previous_decision = frame, decision
        return decision.action.copy(), log

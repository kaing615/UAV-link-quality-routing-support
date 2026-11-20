"""Opt-in operating-point and action-value gates for the next study revision.

The formal v2 experiment used a fixed 0.50 risk threshold and did not estimate
the value of an intervention.  This module keeps those results untouched while
providing pure, validation-only utilities for a new v3 experiment.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np


def _binary_arrays(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError('labels and probabilities must be non-empty 1-D arrays of equal length')
    if not np.all(np.isin(y, (0, 1))) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('labels must be binary and probabilities must be finite in [0, 1]')
    return y.astype(np.int8), p


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float | None
    recall: float | None
    false_positive_rate: float | None
    utility: float
    selection_source: str = 'validation-only'


def _point(y, p, threshold, positive_reward, false_positive_cost, false_negative_cost):
    predicted = p >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    tn = int(np.sum(~predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    utility = tp * positive_reward - fp * false_positive_cost - fn * false_negative_cost
    return OperatingPoint(float(threshold), tp, fp, tn, fn, precision, recall, fpr,
                          float(utility))


def select_operating_point(labels, probabilities, *, min_recall=0.0,
                           positive_reward=1.0, false_positive_cost=1.0,
                           false_negative_cost=1.0):
    """Select a threshold using validation rows only.

    Threshold selection is deterministic.  It never accepts formal/test rows;
    callers should pass a validation split and retain the returned counts and
    costs in the next phase manifest.
    """
    y, p = _binary_arrays(labels, probabilities)
    values = sorted(set(float(x) for x in p), reverse=True)
    candidates = [1.0, *values, 0.0]
    for value in (min_recall, positive_reward, false_positive_cost, false_negative_cost):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError('selection costs and min_recall must be finite and non-negative')
    if min_recall > 1:
        raise ValueError('min_recall must be in [0, 1]')
    feasible = [
        _point(y, p, threshold, positive_reward, false_positive_cost, false_negative_cost)
        for threshold in candidates
    ]
    feasible = [point for point in feasible if point.recall is not None and point.recall >= min_recall]
    if not feasible:
        raise ValueError('no threshold satisfies min_recall')
    # Prefer utility, then a higher threshold (fewer interventions), then a
    # deterministic confusion-count ordering.
    chosen = max(feasible, key=lambda point: (
        point.utility, point.threshold, point.tp, -point.fp))
    return chosen


@dataclass(frozen=True)
class ActionValueGate:
    """Gate an intervention using expected service gain minus motion cost.

    ``predicted_gain`` is a validation-trained estimate of the PDR gain from
    the proposed action, not the geometric planner gain.  It must be supplied
    by a separate action-value model in a future phase.
    """

    risk_threshold: float = .5
    risk_horizon: int = 10
    min_predicted_gain: float = .0
    motion_cost_weight: float = .001
    min_net_benefit: float = .0
    max_motion_m: float = 400.0

    def __post_init__(self):
        values = (self.risk_threshold, self.risk_horizon, self.min_predicted_gain,
                  self.motion_cost_weight, self.min_net_benefit, self.max_motion_m)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError('action-value gate parameters must be finite')
        if not 0 <= self.risk_threshold <= 1:
            raise ValueError('risk_threshold must be in [0, 1]')
        if type(self.risk_horizon) is not int or self.risk_horizon <= 0:
            raise ValueError('risk_horizon must be a positive integer')
        if self.min_predicted_gain < 0 or self.motion_cost_weight < 0 or self.max_motion_m <= 0:
            raise ValueError('action-value gate bounds are invalid')

    def evaluate(self, risk, predicted_gain, motion_m):
        values = (risk, predicted_gain, motion_m)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError('risk, predicted_gain and motion_m must be finite')
        if not 0 <= risk <= 1 or motion_m < 0:
            raise ValueError('risk must be in [0, 1] and motion_m must be non-negative')
        net_benefit = float(predicted_gain - self.motion_cost_weight * motion_m)
        allowed = bool(
            risk >= self.risk_threshold and
            predicted_gain >= self.min_predicted_gain and
            motion_m <= self.max_motion_m and
            net_benefit >= self.min_net_benefit
        )
        return dict(allowed=allowed, risk=float(risk), predicted_gain=float(predicted_gain),
                    motion_m=float(motion_m), net_benefit=net_benefit,
                    risk_threshold=self.risk_threshold,
                    min_predicted_gain=self.min_predicted_gain,
                    motion_cost_weight=self.motion_cost_weight,
                    min_net_benefit=self.min_net_benefit,
                    max_motion_m=self.max_motion_m)

    def __call__(self, *, risks, plan, frame, state):
        """Evaluate a proposed plan using its externally supplied action value.

        The plan must contain ``predicted_service_gain``.  Refusing absent
        values prevents the v3 controller from silently treating geometric
        gain as PDR gain.
        """
        if not isinstance(plan, dict) or 'predicted_service_gain' not in plan:
            return False, dict(reason='missing_predicted_service_gain')
        horizon = self.risk_horizon
        if horizon not in risks:
            return False, dict(reason='missing_risk_horizon', horizon=horizon)
        risk = risks[horizon]
        result = self.evaluate(risk, plan['predicted_service_gain'], plan.get('total_motion', 0.0))
        result['horizon'] = horizon
        return result['allowed'], result


def operating_point_dict(point):
    """Serialize an operating point into strict JSON-friendly values."""
    if not isinstance(point, OperatingPoint):
        raise TypeError('point must be an OperatingPoint')
    return asdict(point)

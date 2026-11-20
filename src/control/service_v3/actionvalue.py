"""Tuning-only selection for v3: horizon, risk threshold, burst PDR-gain model, motion cost.

Every function is pure over already validated traces. Procedure and constants
are fixed by docs/service_v3_protocol.md section 4; nothing here reads
validation or test data.
"""
from dataclasses import asdict
import math

import numpy as np

from ..service_v2.optimization import ActionValueGate, select_operating_point

LABEL_WINDOW_S = 30
MIN_RECALL = .30
RIDGE_ALPHA = 1.0
LAMBDA_GRID = (0., 1e-4, 2e-4, 5e-4, 1e-3)
MAX_MOTION_M = 400.
FEATURES = ('risk_10', 'risk_20', 'A1', 'A5', 'X5', 'Q5', 'total_motion', 'gain', 'tau')


def feature_vector(risks, ratios, plan):
    """Same vector at gate time (live objects) and from logged start decisions."""
    risks = {int(k): v for k, v in (risks or {}).items()}
    values = [risks.get(10), risks.get(20), *((ratios or {}).get(k) for k in ('A1', 'A5', 'X5', 'Q5')),
              *((plan or {}).get(k) for k in ('total_motion', 'gain', 'tau'))]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        return None
    return [float(v) for v in values]


def logged_features(decision):
    return feature_vector(decision.get('risks'), (decision.get('service_state') or {}).get('ratios'), decision.get('plan'))


def window_pdr(frames, start, window=LABEL_WINDOW_S):
    """Application PDR over (start, start+window]; None when the window is unavailable."""
    end = start + window
    if start < 0 or end >= len(frames):
        return None
    a, b = (frames[t]['service_observation']['cumulative'] for t in (start, end))
    offered = b['offered_packets'] - a['offered_packets']
    return (b['sink_rx_packets'] - a['sink_rx_packets']) / offered if offered > 0 else None


def forecast_starts(decisions):
    return [d for d in decisions if d.get('start') is True and d.get('context') == 'forecast']


def burst_labels(policy_frames, decisions, hold_frames, group):
    """y = PDR_policy(t, t+30] - PDR_hold(t, t+30] on the same seed (common random numbers)."""
    rows, excluded = [], {}
    for d in forecast_starts(decisions):
        t = d['step']; x = logged_features(d)
        a, b = window_pdr(policy_frames, t), window_pdr(hold_frames, t)
        reason = 'missing_feature' if x is None else 'window_unavailable' if a is None or b is None else None
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        rows.append(dict(group=list(group), step=t, features=x, y=a - b, motion=float(d['plan']['total_motion'])))
    return rows, excluded


class Ridge:
    def __init__(self, mean, scale, coef, intercept, alpha=RIDGE_ALPHA):
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.coef, self.intercept, self.alpha = np.asarray(coef, float), float(intercept), float(alpha)

    @classmethod
    def fit(cls, X, y, alpha=RIDGE_ALPHA):
        X, y = np.asarray(X, float), np.asarray(y, float)
        if X.ndim != 2 or X.shape[1] != len(FEATURES) or len(X) != len(y) or len(y) < 2:
            raise ValueError('Ridge needs at least two labelled bursts')
        mean = X.mean(axis=0); scale = X.std(axis=0); scale[scale < 1e-12] = 1.
        Z = (X - mean) / scale; ybar = y.mean()
        coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ybar))  # intercept unpenalised
        return cls(mean, scale, coef, ybar, alpha)

    def predict(self, X):
        return ((np.atleast_2d(np.asarray(X, float)) - self.mean) / self.scale) @ self.coef + self.intercept

    def to_dict(self):
        return dict(features=list(FEATURES), alpha=self.alpha, mean=self.mean.tolist(), scale=self.scale.tolist(),
                    coef=self.coef.tolist(), intercept=self.intercept)

    @classmethod
    def from_dict(cls, value):
        if value.get('features') != list(FEATURES): raise ValueError('Ridge feature order mismatch')
        return cls(value['mean'], value['scale'], value['coef'], value['intercept'], value['alpha'])


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return float(spearmanr(a, b).statistic)


def loso_predictions(rows):
    """Leave-one-seed-pair-out: static/gradual tuning twins leave together."""
    groups = sorted({tuple(r['group']) for r in rows})
    if len(groups) < 2: raise ValueError('Leave-one-out needs two seed pairs with bursts')
    predictions = [None] * len(rows)
    for g in groups:
        train = [r for r in rows if tuple(r['group']) != g]
        model = Ridge.fit([r['features'] for r in train], [r['y'] for r in train])
        for i, r in enumerate(rows):
            if tuple(r['group']) == g:
                predictions[i] = float(model.predict(r['features'])[0])
    return predictions


def lambda_value(rows, predictions, lam):
    allowed = [p >= 0 and r['motion'] <= MAX_MOTION_M and p - lam * r['motion'] >= 0 for r, p in zip(rows, predictions)]
    return dict(lambda_per_m=lam, allowed=sum(allowed), realized_gain_sum=float(sum(r['y'] for r, a in zip(rows, allowed) if a)))


def select_lambda(rows, predictions):
    table = [lambda_value(rows, predictions, lam) for lam in LAMBDA_GRID]
    best = max(table, key=lambda t: (t['realized_gain_sum'], t['lambda_per_m']))  # tie -> larger lambda
    return best['lambda_per_m'], table


def select_horizon_threshold(scores):
    """scores: v2 forecast_scores() on tuning HOLD. PR-AUC picks the horizon (tie -> 20)."""
    auc = {h: scores['horizons'][str(h)]['pr_auc'] for h in (10, 20)}
    horizon = max((20, 10), key=lambda h: -math.inf if auc[h] is None else auc[h])
    origins = scores['horizons'][str(horizon)]['origins']
    point = select_operating_point([o['label'] for o in origins], [o['projected_risk'] for o in origins],
                                   min_recall=MIN_RECALL)
    if not 0 < point.threshold <= 1: raise ValueError('Degenerate risk threshold; v3 stops (protocol 4.2)')
    return horizon, auc, point


def gate_params(selection):
    return dict(risk_threshold=selection['risk_on'], risk_horizon=selection['horizon'],
                min_predicted_gain=0., motion_cost_weight=selection['lambda_per_m'],
                min_net_benefit=0., max_motion_m=MAX_MOTION_M)


class GainGate:
    """ActionValueGate fed by the tuning ridge model instead of planner geometry."""

    def __init__(self, selection):
        self.model = Ridge.from_dict(selection['ridge'])
        self.gate = ActionValueGate(**gate_params(selection))

    def predicted_gain(self, risks, ratios, plan):
        x = feature_vector(risks, ratios, plan)
        return None if x is None else float(self.model.predict(x)[0])

    def __call__(self, *, risks, plan, frame, state):
        gain = self.predicted_gain(risks, state.ratios, plan)
        if gain is None:
            return False, dict(reason='missing_feature')
        return self.gate(risks=risks, plan=dict(plan, predicted_service_gain=gain), frame=frame, state=state)


def build_selection(scores, label_rows, excluded, *, tuning_freeze_sha256, predictor_sha256):
    horizon, auc, point = select_horizon_threshold(scores)
    if len(label_rows) < 2: raise ValueError('Too few labelled forecast bursts for the action-value model')
    loso = loso_predictions(label_rows)
    lam, table = select_lambda(label_rows, loso)
    y = [r['y'] for r in label_rows]
    sst = float(sum((v - np.mean(y)) ** 2 for v in y)); sse = float(sum((v - p) ** 2 for v, p in zip(y, loso)))
    ridge = Ridge.fit([r['features'] for r in label_rows], y)
    selection = dict(schema='service-v3-tuning-selection-v1', tuning_freeze_sha256=tuning_freeze_sha256,
        predictor_sha256=predictor_sha256, horizon=horizon, pr_auc={str(h): v for h, v in auc.items()},
        operating_point=asdict(point), min_recall=MIN_RECALL, risk_on=point.threshold, risk_off=point.threshold / 2,
        label_window_seconds=LABEL_WINDOW_S, labelled_bursts=len(label_rows), excluded_bursts=excluded,
        label_rows=label_rows, ridge=ridge.to_dict(),
        loso=dict(predictions=loso, r2=1 - sse / sst if sst > 0 else None, spearman=spearman(loso, y)),
        lambda_grid=table, lambda_per_m=lam)
    selection['gate'] = gate_params(selection)
    return selection

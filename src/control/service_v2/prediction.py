"""Future-only service labels, deterministic weighted logistic/Platt artifact.

Directory/collector loading belongs to the study layer. Inputs here are
explicit episode objects; no historical edge target or fitting data is reused.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import numpy as np

from ..prevention.config import PreventionConfig
from .config import ADOPTED_SPEC_SHA256, POLICIES, ServiceV2Config
from .features import FEATURE_NAMES, FEATURE_TRANSFORMS, FEATURE_VERSION, service_features
from .observation import ServiceHistory, full_window

METHOD = ServiceV2Config()
HORIZONS = METHOD.horizons
ARTIFACT_SCHEMA = 'service-decline-logistic-v2'
TARGET = dict(name='future-only-sustained-arrival-offer-decline',
    risk_set='causally_armed_READY_and_current_healthy_evidence', ratio='sum(R)/sum(O)',
    rolling_windows=5, threshold=.90, consecutive_frames=3, earliest_ratio_offset=5,
    earliest_confirmation_offset=7, future_invalid='uniform_censor_entire_horizon', route_only_target=False)
SOLVER = dict(name='weighted_damped_newton', l2=.01, calibration_l2=.001,
    intercept_penalized=False, max_iterations=100, update_infinity_tolerance=1e-7, minimum_line_search=1e-8)
SUPPORTED_CONFIG = dict(num_uavs=[4, 5], source=0, destination='num_uavs-1',
    spacing_fractions={'4': .15, '5': .10}, comm_range=240., max_speed=8., budget=400.,
    min_separation=5., x_max=1800., y_max=1200., z_min=50., z_max=150., warmup=10., steps=240)
NORMALIZATION = dict(fit='training_rows_only', weighting='equal_total_per_nonempty_trajectory',
                     std='weighted_population', replace_scale_below=1e-8, replacement_scale=1.)


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Artifact must be finite strict JSON') from exc


def _valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _validate_config(config):
    if not isinstance(config, PreventionConfig):
        raise ValueError('Unsupported scenario config type')
    fixed = {k: v for k, v in SUPPORTED_CONFIG.items()
             if k not in ('num_uavs', 'destination', 'spacing_fractions')}
    if (config.num_uavs not in (4, 5) or config.destination != config.num_uavs - 1
            or any(getattr(config, k) != v for k, v in fixed.items())
            or config.initial_spacing_fraction != SUPPORTED_CONFIG['spacing_fractions'][str(config.num_uavs)]):
        raise ValueError('Unsupported scenario config/range')


@dataclass
class ServiceEpisode:
    identity: str
    phase: str
    policy: str
    regime: str
    context: str
    layout_seed: int
    radio_seed: int
    config: PreventionConfig
    frames: list[dict[str, object]]
    source_hashes: dict[str, str]
    trace_sha256: str


@dataclass(frozen=True)
class DeclineLabel:
    value: int | None
    reason: str
    target_confirmation_step: int | None
    ordinary_onset_step: int | None


def future_decline_label(episode, origin_step, horizon):
    frames, states = _episode_states(episode)
    return _decline_label(episode, frames, states, origin_step, horizon)


def _episode_states(episode):
    frames = {}
    states = {}
    history = ServiceHistory(episode.config, ServiceV2Config())
    previous_step = -1
    for frame in episode.frames:
        step = frame.get('step')
        if type(step) is not int or step <= previous_step:
            raise ValueError('Episode frames must have unique increasing steps')
        frames[step] = frame
        states[step] = history.update(frame)
        previous_step = step
    return frames, states


def _decline_label(episode, frames, states, origin_step, horizon):
    method = ServiceV2Config()
    if type(horizon) is not int or horizon not in method.horizons:
        raise ValueError('Unsupported horizon')
    if type(origin_step) is not int or origin_step not in states:
        raise ValueError('Unknown origin step')
    if not states[origin_step].forecast_eligible:
        return DeclineLabel(None, 'ineligible_origin', None, None)
    # Censor the entire horizon before searching for a positive triple.
    for step in range(origin_step + 1, origin_step + horizon + 1):
        if step not in frames:
            return DeclineLabel(None, 'right_censored', None, None)
        try:
            valid = full_window(frames[step], episode.config, method, frames.get(step - 1))
        except (ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            return DeclineLabel(None, 'invalid_future_window', None, None)
    for k in range(origin_step + 5, origin_step + horizon - 1):
        if all(states[k + j].ratios['A5'] is not None and states[k + j].ratios['A5'] < .90 for j in (0, 1, 2)):
            onset = k
            while onset - 1 in states and states[onset - 1].ratios['A5'] is not None and states[onset - 1].ratios['A5'] < .90:
                onset -= 1
            return DeclineLabel(1, 'future_decline', k + 2, onset)
    return DeclineLabel(0, 'no_future_decline', None, None)


@dataclass(frozen=True)
class SupervisedRows:
    x: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    episode_id: tuple[str, ...]
    origin_step: np.ndarray
    counts: dict


def build_service_samples(episodes, horizon):
    if type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError('Unsupported horizon')
    episodes = list(episodes)
    if len({e.identity for e in episodes}) != len(episodes):
        raise ValueError('Duplicate episode identity')
    x, y, weights, identities, origins = [], [], [], [], []
    counts = dict(horizon=horizon, episodes=len(episodes), rows=0, positive_rows=0, negative_rows=0,
        positive_episodes=0, negative_episodes=0, positive_groups=[], negative_groups=[],
        ineligible_origin=0, right_censored=0, invalid_future_window=0,
        zero_eligible_episodes=[], per_episode={})
    groups = {0: set(), 1: set()}
    for ep in episodes:
        _validate_config(ep.config)
        frames, states = _episode_states(ep)
        start = len(y)
        ec = dict(positive_rows=0, negative_rows=0, eligible_rows=0,
                  ineligible_origin=0, right_censored=0, invalid_future_window=0)
        for step in frames:
            label = _decline_label(ep, frames, states, step, horizon)
            if label.value is None:
                ec[label.reason] += 1
                counts[label.reason] += 1
                continue
            causal_history = [frames[t] for t in range(max(0, step - 5), step + 1) if t in frames]
            x.append(service_features(causal_history, ep.config))
            y.append(label.value)
            identities.append(ep.identity)
            origins.append(step)
            ec['eligible_rows'] += 1
            ec['positive_rows' if label.value else 'negative_rows'] += 1
            groups[label.value].add((ep.layout_seed, ep.radio_seed))
        number = len(y) - start
        if number:
            weights.extend([1. / number] * number)
        else:
            counts['zero_eligible_episodes'].append(ep.identity)
        counts['per_episode'][ep.identity] = ec
    counts['rows'] = len(y)
    counts['positive_rows'] = sum(y)
    counts['negative_rows'] = len(y) - sum(y)
    for value, name in ((0, 'negative'), (1, 'positive')):
        counts[name + '_episodes'] = sum(ec[name + '_rows'] > 0 for ec in counts['per_episode'].values())
        counts[name + '_groups'] = [list(pair) for pair in sorted(groups[value])]
    return SupervisedRows(np.asarray(x, dtype=float).reshape((-1, 25)), np.asarray(y, dtype=float),
        np.asarray(weights, dtype=float), tuple(identities), np.asarray(origins, dtype=int), counts)


def validate_partitions(partitions):
    """Both seed identifiers are disjoint between phases, not just pairs."""
    seen = {}
    for phase, episodes in partitions.items():
        if phase not in ('train', 'calibration', 'development', 'pilot', 'formal'):
            raise ValueError('Invalid partition phase')
        entries = list(episodes)
        identities, cells, layout_pairs, radio_pairs = set(), set(), {}, {}
        for e in entries:
            if e.phase != phase:
                raise ValueError('Recorded episode phase conflicts with partition')
            if not isinstance(e.identity, str) or not e.identity:
                raise ValueError('Missing episode identity')
            if e.identity in identities:
                raise ValueError('Duplicate episode identity')
            identities.add(e.identity)
            if any(type(getattr(e, k)) is not int or not 0 <= getattr(e, k) < 2**31 for k in ('layout_seed', 'radio_seed')):
                raise ValueError('Invalid seed identity')
            if e.policy not in POLICIES or e.context not in ('static', 'gradual') or not isinstance(e.regime, str) or not e.regime:
                raise ValueError('Invalid policy/context/regime identity')
            if phase in ('train', 'calibration', 'development') and e.policy not in ('hold', 'geometry'):
                raise ValueError('Unsupported collection policy')
            cell = (e.layout_seed, e.radio_seed, e.context, e.policy)
            if cell in cells:
                raise ValueError('Duplicate within-split paired cell')
            cells.add(cell)
            # Collection phases pair both contexts within a split. Pilot and
            # formal allocate independent static/gradual seed inventories.
            seed_context = e.context if phase in ('pilot', 'formal') else None
            for name, pairs, identifier, other in (
                ('layout_seed', layout_pairs, e.layout_seed, (e.radio_seed, e.regime, seed_context)),
                ('radio_seed', radio_pairs, e.radio_seed, (e.layout_seed, e.regime, seed_context))):
                if identifier in pairs and pairs[identifier] != other:
                    raise ValueError(f'Unpaired within-split seed/context reuse: {name}')
                pairs[identifier] = other
        current = dict(identity=identities, layout_seed=set(layout_pairs), radio_seed=set(radio_pairs))
        for previous in seen.values():
            for name in current:
                if current[name] & previous[name]:
                    raise ValueError(f'Partition overlap: {name}')
        seen[phase] = current


def _sigmoid(value):
    value = np.asarray(value, dtype=float)
    # logaddexp avoids overflow and does not impose an artificial logit cap.
    return np.exp(-np.logaddexp(0., -value))


def _fit_logistic(x, y, weight, l2, *, max_iterations=100):
    """Weighted L2 logistic Newton fit. No fallback on a numerical failure."""
    x, y, weight = np.asarray(x, float), np.asarray(y, float), np.asarray(weight, float)
    if (x.ndim != 2 or y.shape != (len(x),) or weight.shape != y.shape or not len(y)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(weight).all()
            or np.any(weight <= 0) or set(y) != {0., 1.} or not np.isfinite(l2) or l2 <= 0):
        raise ValueError('Invalid finite weighted two-class logistic inputs')
    weights = weight / weight.sum()
    design = np.column_stack((np.ones(len(x)), x))
    coef = np.zeros(design.shape[1])
    penalty = np.full(len(coef), l2)
    penalty[0] = 0.

    def objective(value):
        with np.errstate(over='ignore', invalid='ignore'):
            logits = design @ value
            return float(np.dot(weights, np.logaddexp(0., logits) - y * logits) + .5 * np.dot(penalty, value * value))

    for iteration in range(1, max_iterations + 1):
        with np.errstate(over='ignore', invalid='ignore'):
            probabilities = _sigmoid(design @ coef)
            gradient = design.T @ (weights * (probabilities - y)) + penalty * coef
            hessian = (design.T * (weights * probabilities * (1. - probabilities))) @ design + np.diag(penalty)
        if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
            raise ValueError('Nonfinite logistic fit')
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Singular logistic solve') from exc
        if not np.isfinite(step).all():
            raise ValueError('Nonfinite Newton step')
        rate, old = 1., objective(coef)
        if not np.isfinite(old):
            raise ValueError('Nonfinite logistic objective')
        while rate >= METHOD.solver_min_line_search:
            candidate = coef - rate * step
            candidate_loss = objective(candidate)
            if np.isfinite(candidate_loss) and candidate_loss <= old:
                break
            rate *= .5
        else:
            raise ValueError('Logistic line search failed to converge')
        coef = candidate
        update = float(np.max(np.abs(rate * step)))
        if update < METHOD.solver_update_tolerance:
            if not np.isfinite(coef).all():
                raise ValueError('Nonfinite fitted coefficients')
            return coef, dict(converged=True, iterations=iteration, objective=candidate_loss,
                              final_update_infinity_norm=update, final_line_search_rate=rate)
    raise ValueError(f'Logistic failed to converge after {max_iterations} iterations')


def _class_gate(counts, phase):
    row_min = METHOD.train_rows_per_class if phase == 'train' else METHOD.calibration_rows_per_class
    group_min = METHOD.train_groups_per_class if phase == 'train' else METHOD.calibration_groups_per_class
    if any(counts[f'{name}_rows'] < row_min or len(counts[f'{name}_groups']) < group_min for name in ('positive', 'negative')):
        raise ValueError(f'{phase} class-deficiency counts: {json.dumps(counts, sort_keys=True)}')


def _metadata(e):
    _validate_config(e.config)
    if (not isinstance(e.source_hashes, dict) or not e.source_hashes
            or any(not isinstance(k, str) or not k or not _valid_hash(v) for k, v in e.source_hashes.items())
            or not _valid_hash(e.trace_sha256)):
        raise ValueError('Invalid source/binary/trace hashes')
    return dict(identity=e.identity, phase=e.phase, policy=e.policy, regime=e.regime, context=e.context,
        layout_seed=e.layout_seed, radio_seed=e.radio_seed, config=asdict(e.config),
        source_hashes=deepcopy(e.source_hashes), trace_sha256=e.trace_sha256)


def _predict_details(x, artifact):
    result, previous = {}, 0.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for horizon in HORIZONS:
            m = artifact['models'][str(horizon)]
            logit = float(((x - np.asarray(m['mean'])) / np.asarray(m['scale'])) @ np.asarray(m['weights']) + m['intercept'])
            mapped_logit = m['calibration'][0] + m['calibration'][1] * logit
            if not np.isfinite(logit) or not np.isfinite(mapped_logit):
                raise ValueError('Nonfinite predictor output')
            mapped = float(_sigmoid(mapped_logit))
            projected = max(previous, mapped)
            result[str(horizon)] = dict(raw_logit=logit, mapped_risk=mapped, projected_risk=projected)
            previous = projected
    return result


def fit_service_predictor(train, calibration):
    train, calibration = list(train), list(calibration)
    if not train or not calibration:
        raise ValueError('Nonempty training and calibration partitions required')
    validate_partitions({'train': train, 'calibration': calibration})
    a = dict(schema=ARTIFACT_SCHEMA, feature_version=FEATURE_VERSION, feature_names=list(FEATURE_NAMES),
        feature_transforms=list(FEATURE_TRANSFORMS), comm_range=240., horizons=list(HORIZONS),
        adopted_spec_sha256=ADOPTED_SPEC_SHA256, target=deepcopy(TARGET), solver=deepcopy(SOLVER),
        supported_config=deepcopy(SUPPORTED_CONFIG), normalization=deepcopy(NORMALIZATION),
        method=_json_copy(asdict(METHOD)), monotone_projection='cumulative_max',
        splits=dict(train=[_metadata(e) for e in train], calibration=[_metadata(e) for e in calibration]),
        models={}, calibration_diagnostics={})
    hold = [e for e in calibration if e.policy == 'hold']
    all_calibration_rows = {}
    for horizon in HORIZONS:
        rows, crows = build_service_samples(train, horizon), build_service_samples(hold, horizon)
        _class_gate(rows.counts, 'train')
        _class_gate(crows.counts, 'calibration')
        mean = np.average(rows.x, axis=0, weights=rows.weight)
        scale = np.sqrt(np.average((rows.x - mean) ** 2, axis=0, weights=rows.weight))
        scale[scale < METHOD.minimum_scale] = 1.
        coef, fit = _fit_logistic((rows.x - mean) / scale, rows.y, rows.weight, METHOD.logistic_l2)
        logits = ((crows.x - mean) / scale) @ coef[1:] + coef[0]
        cal, calfit = _fit_logistic(logits[:, None], crows.y, crows.weight, METHOD.calibration_l2)
        a['models'][str(horizon)] = dict(mean=mean.tolist(), scale=scale.tolist(), weights=coef[1:].tolist(),
            intercept=float(coef[0]), calibration=cal.tolist(), train_counts=rows.counts,
            calibration_counts=crows.counts, train_fit=fit, calibration_fit=calfit)
        all_calibration_rows[horizon] = build_service_samples(calibration, horizon)
    records, metrics = {}, {}
    for horizon, rows in all_calibration_rows.items():
        for x, identity, origin, label in zip(rows.x, rows.episode_id, rows.origin_step, rows.y):
            key = (identity, int(origin))
            if key not in records:
                episode = next(e for e in calibration if e.identity == identity)
                records[key] = dict(identity=identity, origin_step=int(origin), policy=episode.policy,
                    context=episode.context, predictions=_predict_details(x, a), labels={})
            records[key]['labels'][str(horizon)] = int(label)
        for policy in ('hold', 'geometry'):
            indexes = [i for i, identity in enumerate(rows.episode_id)
                       if next(e for e in calibration if e.identity == identity).policy == policy]
            if not indexes:
                metrics[f'{policy}-{horizon}'] = dict(rows=0, brier=None, log_loss=None, prevalence=None)
                continue
            probabilities = np.asarray([records[(rows.episode_id[i], int(rows.origin_step[i]))]['predictions'][str(horizon)]['projected_risk'] for i in indexes])
            labels, weights = rows.y[indexes], rows.weight[indexes]
            p = np.clip(probabilities, 1e-15, 1 - 1e-15)
            metrics[f'{policy}-{horizon}'] = dict(rows=len(indexes),
                brier=float(np.average((probabilities - labels) ** 2, weights=weights)),
                log_loss=float(np.average(-labels * np.log(p) - (1 - labels) * np.log1p(-p), weights=weights)),
                prevalence=float(np.average(labels, weights=weights)))
    a['calibration_diagnostics'] = dict(fitted_policy='hold', diagnostic_policy='geometry',
        metrics_after_projection=metrics, rows=[records[k] for k in sorted(records)],
        all_policy_counts={str(h): r.counts for h, r in all_calibration_rows.items()})
    return ServicePredictor(a)


class ServicePredictor:
    def __init__(self, artifact):
        self._artifact = _json_copy(artifact)
        try:
            self._validate()
        except (KeyError, TypeError, OverflowError, AttributeError) as exc:
            raise ValueError('Invalid service predictor schema') from exc

    @property
    def artifact(self):
        return deepcopy(self._artifact)

    def _validate(self):
        a = self._artifact
        fixed = dict(schema=ARTIFACT_SCHEMA, feature_version=FEATURE_VERSION, feature_names=list(FEATURE_NAMES),
            feature_transforms=list(FEATURE_TRANSFORMS), comm_range=240., horizons=list(HORIZONS),
            adopted_spec_sha256=ADOPTED_SPEC_SHA256, target=TARGET, solver=SOLVER, supported_config=SUPPORTED_CONFIG,
            normalization=NORMALIZATION, method=_json_copy(asdict(METHOD)), monotone_projection='cumulative_max')
        if not isinstance(a, dict) or set(a) != set(fixed) | {'splits', 'models', 'calibration_diagnostics'}:
            raise ValueError('Invalid service predictor schema')
        if any(a[k] != value for k, value in fixed.items()):
            raise ValueError('Incompatible service predictor schema/feature/config contract')
        if set(a['splits']) != {'train', 'calibration'} or any(not e for e in a['splits'].values()):
            raise ValueError('Invalid split provenance')
        episodes = {}
        metadata_keys = {'identity', 'phase', 'policy', 'regime', 'context', 'layout_seed', 'radio_seed', 'config', 'source_hashes', 'trace_sha256'}
        for phase, entries in a['splits'].items():
            episodes[phase] = []
            for entry in entries:
                if set(entry) != metadata_keys:
                    raise ValueError('Invalid episode provenance schema')
                e = ServiceEpisode(**{**entry, 'config': PreventionConfig(**entry['config']), 'frames': []})
                _metadata(e)
                episodes[phase].append(e)
        validate_partitions(episodes)
        if set(a['models']) != {str(h) for h in HORIZONS}:
            raise ValueError('Invalid artifact horizons')
        for horizon in HORIZONS:
            model = a['models'][str(horizon)]
            if set(model) != {'mean', 'scale', 'weights', 'intercept', 'calibration', 'train_counts', 'calibration_counts', 'train_fit', 'calibration_fit'}:
                raise ValueError('Invalid model schema')
            for key in ('mean', 'scale', 'weights'):
                if not isinstance(model[key], list) or any(type(v) not in (int, float) for v in model[key]):
                    raise ValueError('Invalid numeric model vector')
                vector = np.asarray(model[key], dtype=float)
                if vector.shape != (25,) or not np.isfinite(vector).all() or (key == 'scale' and np.any(vector <= 0)):
                    raise ValueError('Invalid finite model vector/scale')
            if type(model['intercept']) not in (int, float) or not np.isfinite(model['intercept']):
                raise ValueError('Invalid model intercept')
            if not isinstance(model['calibration'], list) or any(type(v) not in (int, float) for v in model['calibration']):
                raise ValueError('Invalid numeric calibration coefficients')
            cal = np.asarray(model['calibration'], float)
            if cal.shape != (2,) or not np.isfinite(cal).all():
                raise ValueError('Invalid Platt calibration coefficients')
            for phase in ('train', 'calibration'):
                fit = model[phase + '_fit']
                if (set(fit) != {'converged', 'iterations', 'objective', 'final_update_infinity_norm', 'final_line_search_rate'}
                        or fit['converged'] is not True or type(fit['iterations']) is not int or not 1 <= fit['iterations'] <= 100
                        or type(fit['objective']) not in (int, float) or fit['objective'] < 0
                        or not 0 <= fit['final_update_infinity_norm'] < 1e-7 or not 1e-8 <= fit['final_line_search_rate'] <= 1):
                    raise ValueError('Invalid converged solver evidence')
                selected = episodes[phase] if phase == 'train' else [e for e in episodes[phase] if e.policy == 'hold']
                _validate_counts(model[phase + '_counts'], horizon, selected)
                _class_gate(model[phase + '_counts'], phase)
        _validate_diagnostics(a, episodes['calibration'])

    @classmethod
    def load(cls, path):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f'Duplicate JSON field: {key}')
                result[key] = value
            return result
        def nonfinite(value):
            raise ValueError(f'Nonfinite JSON constant: {value}')
        return cls(json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=nonfinite, object_pairs_hook=pairs))

    def save_exclusive(self, path):
        path = Path(path)
        if path.exists():
            raise ValueError('Predictor output already exists')
        self._validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open('x', encoding='utf-8') as stream:
                json.dump(self._artifact, stream, indent=2, allow_nan=False)
        except FileExistsError as exc:
            raise ValueError('Predictor output already exists') from exc

    def predict(self, history, config):
        _validate_config(config)
        details = _predict_details(service_features(history, config), self._artifact)
        return {h: details[str(h)]['projected_risk'] for h in HORIZONS}

    def validate_scenarios(self, scenarios):
        """Scenario objects require layout_seed, radio_seed, and config mapping."""
        for scenario in scenarios:
            if not isinstance(scenario, dict) or not {'layout_seed', 'radio_seed', 'config'} <= set(scenario):
                raise ValueError('Invalid scenario schema')
            try:
                _validate_config(PreventionConfig(**scenario['config']))
            except (TypeError, ValueError) as exc:
                raise ValueError('Unsupported scenario config') from exc
            for name in ('layout_seed', 'radio_seed'):
                if type(scenario[name]) is not int or not 0 <= scenario[name] < 2**31:
                    raise ValueError(f'Invalid scenario {name}')
                if any(scenario[name] == e[name] for split in self._artifact['splits'].values() for e in split):
                    raise ValueError(f'Scenario overlap with fitted splits: {name}')


def _validate_counts(counts, horizon, episodes):
    expected_keys = {'horizon', 'episodes', 'rows', 'positive_rows', 'negative_rows', 'positive_episodes',
        'negative_episodes', 'positive_groups', 'negative_groups', 'ineligible_origin', 'right_censored',
        'invalid_future_window', 'zero_eligible_episodes', 'per_episode'}
    if not isinstance(counts, dict) or set(counts) != expected_keys or counts['horizon'] != horizon or counts['episodes'] != len(episodes):
        raise ValueError('Invalid sample counts schema')
    episode_map = {e.identity: e for e in episodes}
    if set(counts['per_episode']) != set(episode_map):
        raise ValueError('Sample counts/provenance identity mismatch')
    ekeys = {'positive_rows', 'negative_rows', 'eligible_rows', 'ineligible_origin', 'right_censored', 'invalid_future_window'}
    for ec in counts['per_episode'].values():
        if set(ec) != ekeys or any(type(v) is not int or v < 0 for v in ec.values()) or ec['eligible_rows'] != ec['positive_rows'] + ec['negative_rows']:
            raise ValueError('Invalid per-episode sample counts')
    for name in ('positive_rows', 'negative_rows', 'ineligible_origin', 'right_censored', 'invalid_future_window'):
        if type(counts[name]) is not int or counts[name] != sum(ec[name] for ec in counts['per_episode'].values()):
            raise ValueError('Inconsistent sample counts')
    if counts['rows'] != counts['positive_rows'] + counts['negative_rows']:
        raise ValueError('Inconsistent row counts')
    zero = [e.identity for e in episodes if counts['per_episode'][e.identity]['eligible_rows'] == 0]
    if counts['zero_eligible_episodes'] != zero:
        raise ValueError('Inconsistent zero-eligible episode counts')
    for name in ('positive', 'negative'):
        contributing = [episode_map[k] for k, ec in counts['per_episode'].items() if ec[name + '_rows'] > 0]
        groups = [list(pair) for pair in sorted({(e.layout_seed, e.radio_seed) for e in contributing})]
        if counts[name + '_episodes'] != len(contributing) or counts[name + '_groups'] != groups:
            raise ValueError('Inconsistent episode/seed-pair class counts')


def _validate_diagnostics(a, episodes):
    d = a['calibration_diagnostics']
    if (set(d) != {'fitted_policy', 'diagnostic_policy', 'metrics_after_projection', 'rows', 'all_policy_counts'}
            or d['fitted_policy'] != 'hold' or d['diagnostic_policy'] != 'geometry'
            or set(d['all_policy_counts']) != {'10', '20'}):
        raise ValueError('Invalid calibration diagnostic schema')
    by_id = {e.identity: e for e in episodes}
    for h in HORIZONS:
        _validate_counts(d['all_policy_counts'][str(h)], h, episodes)
    keys, class_counts = set(), {str(h): {e.identity: [0, 0] for e in episodes} for h in HORIZONS}
    for row in d['rows']:
        if set(row) != {'identity', 'origin_step', 'policy', 'context', 'predictions', 'labels'}:
            raise ValueError('Invalid diagnostic row schema')
        identity, origin = row['identity'], row['origin_step']
        if identity not in by_id or type(origin) is not int or not 0 <= origin <= by_id[identity].config.steps or (identity, origin) in keys:
            raise ValueError('Invalid diagnostic row identity/origin')
        keys.add((identity, origin))
        if row['policy'] != by_id[identity].policy or row['context'] != by_id[identity].context:
            raise ValueError('Diagnostic policy/context mismatch')
        if set(row['predictions']) != {'10', '20'} or not row['labels'] or not set(row['labels']) <= {'10', '20'}:
            raise ValueError('Invalid diagnostic horizons')
        previous = 0.
        for h in HORIZONS:
            values = row['predictions'][str(h)]
            if (set(values) != {'raw_logit', 'mapped_risk', 'projected_risk'}
                    or any(type(v) not in (int, float) for v in values.values())):
                raise ValueError('Invalid diagnostic predictions')
            m = a['models'][str(h)]
            mapped = float(_sigmoid(m['calibration'][0] + m['calibration'][1] * values['raw_logit']))
            projected = max(previous, mapped)
            if not 0 <= values['mapped_risk'] <= 1 or not 0 <= values['projected_risk'] <= 1 or abs(values['mapped_risk'] - mapped) > 1e-12 or abs(values['projected_risk'] - projected) > 1e-12:
                raise ValueError('Inconsistent calibration/projection diagnostics')
            previous = projected
            if str(h) in row['labels']:
                value = row['labels'][str(h)]
                if type(value) is not int or value not in (0, 1):
                    raise ValueError('Invalid diagnostic label')
                class_counts[str(h)][identity][value] += 1
    for h in HORIZONS:
        counts = d['all_policy_counts'][str(h)]['per_episode']
        hold_counts = a['models'][str(h)]['calibration_counts']['per_episode']
        for identity in by_id:
            if class_counts[str(h)][identity] != [counts[identity]['negative_rows'], counts[identity]['positive_rows']]:
                raise ValueError('Diagnostic row counts mismatch')
            if by_id[identity].policy == 'hold' and counts[identity] != hold_counts[identity]:
                raise ValueError('HOLD calibration counts disagree with stored diagnostic labels/counts')
    metrics = d['metrics_after_projection']
    if set(metrics) != {f'{policy}-{h}' for policy in ('hold', 'geometry') for h in HORIZONS}:
        raise ValueError('Invalid diagnostic metrics')
    for name, value in metrics.items():
        if set(value) != {'rows', 'brier', 'log_loss', 'prevalence'} or type(value['rows']) is not int or value['rows'] < 0:
            raise ValueError('Invalid diagnostic metrics schema')
        policy, horizon = name.rsplit('-', 1)
        rows = [row for row in d['rows'] if row['policy'] == policy and horizon in row['labels']]
        if value['rows'] != len(rows):
            raise ValueError('Diagnostic metric row count disagrees with stored labels')
        if value['rows'] == 0:
            if any(value[k] is not None for k in ('brier', 'log_loss', 'prevalence')):
                raise ValueError('Undefined empty diagnostic metrics must be null')
            continue
        if (any(type(value[k]) not in (int, float) for k in ('brier', 'log_loss', 'prevalence'))
                or not 0 <= value['brier'] <= 1 or not 0 <= value['prevalence'] <= 1 or value['log_loss'] < 0):
            raise ValueError('Invalid diagnostic metrics range/type')
        labels = np.asarray([row['labels'][horizon] for row in rows], dtype=float)
        risks = np.asarray([row['predictions'][horizon]['projected_risk'] for row in rows], dtype=float)
        # The stored labeled rows supply both the outcome and each trajectory's
        # denominator. A long trajectory must not receive extra total weight.
        counts = d['all_policy_counts'][horizon]['per_episode']
        weights = np.asarray([1. / counts[row['identity']]['eligible_rows'] for row in rows])
        p = np.clip(risks, 1e-15, 1 - 1e-15)
        expected = dict(brier=float(np.average((risks - labels) ** 2, weights=weights)),
            log_loss=float(np.average(-labels * np.log(p) - (1 - labels) * np.log1p(-p), weights=weights)),
            prevalence=float(np.average(labels, weights=weights)))
        if any(abs(value[k] - expected[k]) > 1e-12 for k in expected):
            raise ValueError('Diagnostic metric values disagree with projected trajectory-weighted rows')

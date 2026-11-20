"""Causal within-H event risk, NumPy logistic fitting and strict JSON artifacts.

Public APIs: load_traces(root, expected_split=None), build_samples(episodes, horizon),
fit_predictor(train, calibration, l2=.01), LearnedPredictor.load(path),
predict(current, previous, comm_range) -> {horizon: {(i,j): probability}}.
All labels require the full future horizon. Episodes sharing layout OR radio
seeds across splits are refused. Multiple collection policies per split are OK.
Calibration is Platt scaling on independent episodes, never on test data.
Risks across horizons receive a cumulative-max projection for event nesting.
These probabilities are neither exact event times nor action counterfactuals.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

import numpy as np

from .features import FEATURE_VERSION, FEATURE_NAMES, edge_features, measured_edges

HORIZONS = (3, 5, 10, 20)
ARTIFACT_SCHEMA = 'prevention-logistic-json-v1'


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant: '+value)))


@dataclass
class EpisodeTrace:
    episode_id: str
    layout_seed: int
    radio_seed: int
    comm_range: float
    frames: list
    source_hashes: dict
    collection_policy: str = 'unknown'
    collection_split: str = 'unknown'


def validate_partitions(train, calibration, test=()):
    """Refuse repeated episode identity or either scenario seed across splits."""
    splits = (list(train), list(calibration), list(test))
    for i, first in enumerate(splits):
        for second in splits[i+1:]:
            for attr in ('episode_id', 'layout_seed', 'radio_seed'):
                if {getattr(t, attr) for t in first} & {getattr(t, attr) for t in second}:
                    raise ValueError(f'Episode split overlap: {attr}')


def _recorded_split(value, source):
    if value not in ('train', 'calibration', 'test'):
        raise ValueError(f'Invalid or missing recorded split: {source}')
    return value


def _require_partition(episodes, expected_split, *, allow_unknown=False):
    _recorded_split(expected_split, 'intended partition')
    for episode in episodes:
        if episode.collection_split != expected_split and not (allow_unknown and episode.collection_split == 'unknown'):
            raise ValueError(f'Recorded split {episode.collection_split} conflicts with intended partition {expected_split}')


def _manifest_episode_path(row, run, root):
    """Relocate only via an explicitly recorded original output root, never names."""
    value = row.get('episode_dir')
    if not isinstance(value, str) or not value: raise ValueError('Missing manifest episode directory')
    flavor = PureWindowsPath if re.match(r'^[A-Za-z]:[\\/]', value) else PurePosixPath
    recorded = flavor(value)
    if recorded.is_absolute():
        native = Path(value)
        if native.is_absolute() and native.resolve().is_relative_to(root):
            candidate = native.resolve()
        else:
            original = run.get('output_root') or run.get('provenance', {}).get('arguments', {}).get('output')
            if not original: raise ValueError('Absolute episode path needs recorded output root for relocation')
            try: relative = recorded.relative_to(flavor(original))
            except ValueError as exc: raise ValueError('Manifest episode outside recorded output root') from exc
            if '..' in relative.parts: raise ValueError('Unsafe manifest episode path')
            candidate = (root/Path(*relative.parts)).resolve()
    else:
        if '..' in recorded.parts: raise ValueError('Unsafe relative manifest episode path')
        candidate = (root/Path(*recorded.parts)).resolve()
    if not candidate.is_relative_to(root): raise ValueError('Manifest episode escapes collector directory')
    return candidate


def _collector_index(root):
    """Validate whole enclosing/nested collectors before selecting a subtree."""
    manifests = set(root.rglob('episodes.json'))
    run_paths = set(root.rglob('run_config.json'))
    for parent in (root, *root.parents):
        if (parent/'episodes.json').is_file(): manifests.add(parent/'episodes.json')
        if (parent/'run_config.json').is_file(): run_paths.add(parent/'run_config.json')
    for run_path in run_paths:
        if strict_json(run_path).get('schema') == 'prevention-grid-v1':
            manifest = run_path.with_name('episodes.json')
            if not manifest.is_file(): raise ValueError(f'Missing collector episode manifest: {manifest}')
            manifests.add(manifest)
    index = {}
    for manifest in sorted(manifests):
        run_path = manifest.with_name('run_config.json')
        if not run_path.is_file(): raise ValueError(f'Missing collector run manifest: {run_path}')
        rows, run = strict_json(manifest), strict_json(run_path)
        if run.get('schema') != 'prevention-grid-v1': raise ValueError('Incompatible collector manifest schema')
        split = _recorded_split(run.get('split'), run_path)
        if not rows or any(row.get('status') != 'ok' for row in rows):
            raise ValueError(f'Technical error or empty trace collection: {manifest}')
        from .evaluation import validate_paired_grid
        validate_paired_grid(rows, run['policies'], [tuple(s) for s in run['scenarios']])
        declared = set()
        for row in rows:
            episode_dir = _manifest_episode_path(row, run, manifest.parent.resolve())
            path = episode_dir/'transitions.jsonl'
            if path in declared or path in index: raise ValueError('Duplicate manifest episode directory')
            declared.add(path)
            if not path.is_file(): raise ValueError(f'Missing manifest trace: {path}')
            if not isinstance(row.get('trace_sha256'), str) or row['trace_sha256'] != sha256(path):
                raise ValueError(f'Manifest trace hash mismatch: {path}')
            config = strict_json(episode_dir/'config.json')
            metadata_path = episode_dir/'policy_metadata.json'
            if not metadata_path.is_file(): raise ValueError('Missing collector policy/split metadata')
            metadata = strict_json(metadata_path)
            if metadata.get('collection_policy') != row['policy']: raise ValueError('Manifest episode policy mismatch')
            if _recorded_split(metadata.get('split'), metadata_path) != split: raise ValueError('Run/episode split conflict')
            for key in ('layout_seed', 'radio_seed'):
                if config.get(key) != row[key] or metadata.get(key) != row[key]:
                    raise ValueError(f'Manifest episode seed identity mismatch: {key}')
            if 'config' in run and config['config'] != run['config']: raise ValueError('Run/episode configuration mismatch')
            index[path] = (row, run, manifest, run_path)
        actual = {p.resolve() for p in manifest.parent.rglob('transitions.jsonl')}
        if actual != declared: raise ValueError(f'Unlisted or missing collector traces: {manifest}')
    return index, bool(manifests)


def load_traces(root, expected_split=None):
    """Read successful complete collector traces recursively; refuse any run errors.

    expected_split requires a known matching recorded train/calibration/test role.
    Validate full enclosing collector manifests even when selecting one policy;
    declared traces, hashes, identities and discovered files must agree exactly.
    Absolute paths relocate only through recorded output_root (or CLI provenance
    output); relative paths stay within the collector. No basename heuristics.
    """
    root = Path(root).resolve()
    index, has_manifests = _collector_index(root)
    paths = sorted(root.rglob('transitions.jsonl'))
    if not paths: raise ValueError(f'No prevention episode traces under {root}')
    traces = []
    for path in paths:
        if has_manifests and path not in index: raise ValueError(f'Unlisted trace outside declared collector grid: {path}')
        config_path = path.with_name('config.json')
        config = strict_json(config_path)
        if path.with_name('setup_error.json').exists(): raise ValueError(f'Setup error: {path.parent}')
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        frames = [record['frame'] for record in records]
        steps = config['config']['steps']
        if [f['step'] for f in frames] != list(range(steps+1)):
            raise ValueError(f'Incomplete horizon in {path}')
        # Validate offline protocol, not just dimensions used by a particular feature.
        from ..bridge import validate_frame
        from .config import PreventionConfig
        c = PreventionConfig(**config['config'])
        for frame_index, f in enumerate(frames): validate_frame(f, c, frame_index)
        if not records[-1].get('truncated') or frames[0]['components'] != 1:
            raise ValueError(f'Invalid complete prevention episode: {path}')
        hashes = {str(path.resolve()): sha256(path), str(config_path.resolve()): sha256(config_path)}
        if path in index:
            _, _, manifest, run_path = index[path]
            hashes[str(manifest)] = sha256(manifest)
            hashes[str(run_path)] = sha256(run_path)
        policy_path = path.with_name('policy_metadata.json')
        policy = 'unknown'
        split = 'unknown'
        if policy_path.is_file():
            metadata = strict_json(policy_path)
            policy = metadata['collection_policy']
            split = _recorded_split(metadata.get('split'), policy_path)
            for key in ('layout_seed', 'radio_seed'):
                if key in metadata and metadata[key] != config[key]: raise ValueError('Episode metadata seed identity mismatch')
            hashes[str(policy_path.resolve())] = sha256(policy_path)
        traces.append(EpisodeTrace(path.parent.name, config['layout_seed'], config['radio_seed'],
                                  c.comm_range, frames, hashes, policy, split))
    if len({t.episode_id for t in traces}) != len(traces): raise ValueError('Duplicate episode IDs')
    if expected_split is not None: _require_partition(traces, expected_split)
    return traces


def build_samples(episodes, horizon):
    if horizon not in HORIZONS: raise ValueError('Unsupported horizon')
    x, y, ids = [], [], []
    for episode in episodes:
        frames = episode.frames
        if [f['step'] for f in frames] != list(range(len(frames))): raise ValueError('Nonconsecutive episode frames')
        edge_sets = [set(measured_edges(f)) for f in frames]
        for t in range(len(frames)-horizon):
            for edge in sorted(edge_sets[t]):
                x.append(edge_features(frames[t], frames[t-1] if t else None, edge, episode.comm_range, horizon))
                y.append(int(any(edge not in edges for edges in edge_sets[t+1:t+horizon+1])))
                ids.append(episode.episode_id)
    return np.asarray(x, dtype=float).reshape((-1, len(FEATURE_NAMES))), np.asarray(y, dtype=float), ids


def _sigmoid(value):
    value = np.clip(value, -40., 40.)
    return 1./(1.+np.exp(-value))


def _fit_logistic(x, y, l2):
    """Damped Newton logistic regression; L2 on weights, unpenalized intercept."""
    if len(np.unique(y)) != 2: raise ValueError('both classes required for train and calibration at every horizon')
    design = np.column_stack((np.ones(len(x)), x))
    coef = np.zeros(design.shape[1]); penalty = np.full(len(coef), l2); penalty[0] = 0

    def objective(w):
        z = design @ w
        return float(np.mean(np.logaddexp(0., z)-y*z)+.5*np.sum(penalty*w*w))

    for _ in range(100):
        p = _sigmoid(design @ coef)
        gradient = design.T @ (p-y)/len(y)+penalty*coef
        hessian = (design.T * (p*(1-p))) @ design/len(y)+np.diag(penalty+1e-8)
        step = np.linalg.solve(hessian, gradient)
        rate, old = 1., objective(coef)
        while rate > 1e-8 and objective(coef-rate*step) > old: rate *= .5
        coef -= rate*step
        if np.max(np.abs(rate*step)) < 1e-7: break
    if not np.isfinite(coef).all(): raise ValueError('Nonfinite fitted coefficients')
    return coef


def _split_metadata(episodes):
    return [dict(episode_id=e.episode_id, layout_seed=e.layout_seed, radio_seed=e.radio_seed,
                 source_hashes=e.source_hashes, collection_policy=e.collection_policy,
                 collection_split=e.collection_split) for e in episodes]


def fit_predictor(train, calibration, l2=.01):
    train, calibration = list(train), list(calibration)
    _require_partition(train, 'train', allow_unknown=True)
    _require_partition(calibration, 'calibration', allow_unknown=True)
    validate_partitions(train, calibration)
    if not np.isfinite(l2) or l2 <= 0: raise ValueError('Positive finite L2 required')
    if not train or not calibration: raise ValueError('Both train and calibration episodes required')
    if len({e.comm_range for e in train+calibration}) != 1: raise ValueError('Incompatible communication ranges')
    artifact = dict(schema=ARTIFACT_SCHEMA, feature_version=FEATURE_VERSION, feature_names=list(FEATURE_NAMES),
                    horizons=list(HORIZONS), model_kind='logistic', l2=l2, calibration_l2=.001,
                    monotone_projection='cumulative_max', comm_range=train[0].comm_range,
                    splits=dict(train=_split_metadata(train), calibration=_split_metadata(calibration)), models={})
    for horizon in HORIZONS:
        x, y, _ = build_samples(train, horizon)
        cx, cy, _ = build_samples(calibration, horizon)
        if len(np.unique(y)) != 2 or len(np.unique(cy)) != 2:
            raise ValueError(f'both classes required in train and calibration for horizon {horizon}; collect new train/calibration seeds')
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-8] = 1.
        coef = _fit_logistic((x-mean)/scale, y, l2)
        logits = ((cx-mean)/scale) @ coef[1:]+coef[0]
        calibration_coef = _fit_logistic(logits[:, None], cy, .001)
        artifact['models'][str(horizon)] = dict(mean=mean.tolist(), scale=scale.tolist(), weights=coef[1:].tolist(),
                    intercept=float(coef[0]), calibration=calibration_coef.tolist(),
                    train_samples=len(y), train_positive=int(y.sum()), calibration_samples=len(cy), calibration_positive=int(cy.sum()))
    return LearnedPredictor(artifact)


class LearnedPredictor:
    def __init__(self, artifact):
        self.artifact = deepcopy(artifact)
        a = self.artifact
        if (a.get('schema') != ARTIFACT_SCHEMA or a.get('feature_version') != FEATURE_VERSION
                or a.get('feature_names') != list(FEATURE_NAMES) or a.get('horizons') != list(HORIZONS)
                or a.get('model_kind') != 'logistic' or a.get('monotone_projection') != 'cumulative_max'):
            raise ValueError('Incompatible predictor schema or feature version')
        try:
            try: json.dumps(a, allow_nan=False)
            except ValueError as exc: raise ValueError('Nonfinite predictor artifact') from exc
            if not np.isfinite(a['comm_range']) or a['comm_range'] <= 0: raise ValueError('Invalid range')
            if set(a['models']) != {str(h) for h in HORIZONS}: raise ValueError('Invalid horizons')
            for h in HORIZONS:
                m = a['models'][str(h)]
                for key in ('weights', 'mean', 'scale'):
                    v = np.asarray(m[key], float)
                    if v.shape != (len(FEATURE_NAMES),) or not np.isfinite(v).all(): raise ValueError('Invalid finite model shape')
                if (np.asarray(m['scale']) <= 0).any(): raise ValueError('Invalid normalization scale')
                if np.asarray(m['calibration'], float).shape != (2,) or not np.isfinite(m['calibration']).all():
                    raise ValueError('Invalid calibration')
                if not np.isfinite(m['intercept']): raise ValueError('Nonfinite intercept')
                for split in ('train', 'calibration'):
                    total, positive = m[split+'_samples'], m[split+'_positive']
                    if type(total) is not int or type(positive) is not int or not 0 < positive < total:
                        raise ValueError('Both classes required in artifact')
            splits = a['splits']
            if set(splits) != {'train', 'calibration'}: raise ValueError('Invalid splits')
            for split_name, split in splits.items():
                if not split: raise ValueError('Empty split provenance')
                for e in split:
                    if e.get('collection_split', 'unknown') not in (split_name, 'unknown'):
                        raise ValueError('Recorded artifact split conflicts with intended partition')
                    if (not isinstance(e['episode_id'], str) or not e['episode_id'] or not e['source_hashes']
                            or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in e['source_hashes'].values())
                            or any(type(e[k]) is not int or not 0 <= e[k] < 2**31 for k in ('layout_seed', 'radio_seed'))):
                        raise ValueError('Invalid episode provenance')
            validate_partitions(*[[type('Identity', (), e) for e in splits[k]] for k in ('train', 'calibration')])
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError('Invalid predictor schema') from exc

    @classmethod
    def load(cls, path): return cls(strict_json(path))

    def save(self, path):
        path = Path(path)
        if path.exists(): raise ValueError('Predictor output already exists; use a new file')
        LearnedPredictor(self.artifact)  # refuse mutation/nonfinite JSON before opening output
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream: json.dump(self.artifact, stream, indent=2, allow_nan=False)

    def validate_test_scenarios(self, scenarios):
        test_layout, test_radio = zip(*scenarios) if scenarios else ((), ())
        for episodes in self.artifact['splits'].values():
            if ({e['layout_seed'] for e in episodes} & set(test_layout)
                    or {e['radio_seed'] for e in episodes} & set(test_radio)):
                raise ValueError('Test scenario overlap with train/calibration episodes')

    def predict(self, current, previous, comm_range):
        if comm_range != self.artifact['comm_range']: raise ValueError('Incompatible predictor communication range')
        result, previous_risk = {}, {}
        for horizon in HORIZONS:
            m = self.artifact['models'][str(horizon)]
            result[horizon] = {}
            for edge in measured_edges(current):
                x = edge_features(current, previous, edge, comm_range, horizon)
                logit = ((x-m['mean'])/m['scale']) @ m['weights'] + m['intercept']
                probability = float(_sigmoid(m['calibration'][0]+m['calibration'][1]*logit))
                result[horizon][edge] = max(probability, previous_risk.get(edge, 0.))
            previous_risk = result[horizon]
        return result


class SimplePredictor:
    """Fixed extrapolation heuristic; no fitting, counterfactual or future mission input."""
    def predict(self, current, previous, comm_range):
        result, previous_risk = {}, {}
        for horizon in HORIZONS:
            result[horizon] = {}
            for edge in measured_edges(current):
                x = edge_features(current, previous, edge, comm_range, horizon)
                distance = x[0]+max(0., x[1])*horizon/comm_range
                projected_rx = x[2]+min(0., x[3])*horizon
                probability = float(max(_sigmoid(12*(distance-.9)), _sigmoid(8*(.5-projected_rx))))
                result[horizon][edge] = max(probability, previous_risk.get(edge, 0.))
            previous_risk = result[horizon]
        return result

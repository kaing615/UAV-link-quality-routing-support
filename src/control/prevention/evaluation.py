"""Reproducible prevention collector, paired evaluator and held-out reliability.

Stable public APIs:
  run_grid(*, config, launcher, output, policies, scenarios, predictor=None,
           threshold=.5, routing_delay_s=3, split='test', provenance=None,
           env_factory=Ns3PreventionEnv) -> summary dictionary
  validate_paired_grid(records, policies, scenarios) -> None or ValueError
  episode_metrics(frames, config, episode_dir, decisions) -> metrics dictionary
  evaluate_predictor(model, episodes, threshold=.5, bins=10) -> reliability

Scenarios are explicit (layout_seed, radio_seed) pairs. Separate fresh output
directories may shard a grid. Merge records in memory and call validation with
the complete intended policy/scenario grid before any complete comparison.
Disconnected fractions count frames1..steps, one measured preceding interval
each; frame0 is setup and only supplies the initial state for event counts.
"""
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .controller import POLICIES, PreventionController
from .environment import Ns3PreventionEnv
from .features import measured_edges
from .prediction import HORIZONS, sha256, _require_partition


def require_empty_output(path):
    path = Path(path)
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise ValueError('Prevention output must be an empty directory or a new path')


def validate_grid(policies, scenarios, supported_policies=POLICIES):
    policies, scenarios = list(policies), [tuple(s) for s in scenarios]
    if not policies or len(set(policies)) != len(policies) or any(p not in supported_policies for p in policies):
        raise ValueError('Explicit nonempty unique supported policies required')
    if not scenarios or len(set(scenarios)) != len(scenarios): raise ValueError('Empty or duplicate scenario grid')
    if any(len(s) != 2 or any(type(v) is not int or not 0 <= v < 2**31 for v in s) for s in scenarios):
        raise ValueError('Scenario seeds must be integer pairs in [0,2**31)')
    return policies, scenarios


def validate_paired_grid(records, policies, scenarios):
    policies, scenarios = validate_grid(policies, scenarios)
    if any(row.get('status') != 'ok' for row in records): raise ValueError('Technical/setup errors invalidate complete paired comparison')
    actual = Counter((row['policy'], row['layout_seed'], row['radio_seed']) for row in records)
    expected = Counter((p, layout, radio) for p in policies for layout, radio in scenarios)
    if actual != expected: raise ValueError('Missing, unexpected or duplicate paired grid cells')


def episode_metrics(frames, config, episode_dir, decisions):
    if [f['step'] for f in frames] != list(range(config.steps+1)):
        raise ValueError('Full configured horizon is required for prevention metrics')
    samples = frames[1:]
    disconnected = [f['components'] > 1 for f in frames]
    failures = [i for i in range(1, len(frames)) if disconnected[i] and not disconnected[i-1]]
    with (Path(episode_dir)/'closed_loop_metrics.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1: raise ValueError('Expected one complete simulator metrics row')
    raw = rows[0]
    pdr = float(raw['pdr']); received = int(raw['rx_packets'])
    if not np.isfinite(pdr) or not 0 <= pdr <= 1 or received < 0: raise ValueError('Invalid simulator PDR/received count')
    delay = float(raw['mean_delay_ms']) if received else None
    if delay is not None and (not np.isfinite(delay) or delay < 0): raise ValueError('Invalid measured delay')
    if 'offered_packets' in raw:
        offered = int(raw['offered_packets'])
        if offered < 0 or (offered and abs(received/offered-pdr) > 1e-5): raise ValueError('Simulator PDR disagrees with packet counts')
    travel = [frames[-1]['travel'][i] for i in config.relay_ids]
    return dict(disconnected_fraction=float(np.mean(disconnected[1:])), partition_count=len(failures),
                first_partition_step=next((f['step'] for f in samples if f['components'] > 1), None),
                source_dest_disconnected_fraction=float(np.mean([not f['source_dest_connected'] for f in samples])),
                olsr_unusable_fraction=float(np.mean([not f['olsr_route_usable'] for f in samples])),
                pdr=pdr, mean_delay_ms=delay, total_relay_travel=float(sum(travel)), max_relay_travel=float(max(travel)),
                rejected_actions=sum(bool(f['safety_rejected']) for f in samples),
                intervention_starts=sum(bool(d['intervention_started']) for d in decisions),
                first_action_step=next((d['step'] for d in decisions if d.get('commanded_motion')), None),
                elapsed_steps=config.steps,
                **{k: int(raw[k]) for k in ('offered_packets', 'tx_packets', 'rx_packets') if k in raw})


def source_provenance():
    """Hash implementation and simulator sources using repository-relative names."""
    root = Path(__file__).resolve().parents[3]
    paths = list((root/'src/control/prevention').glob('*.py'))
    paths += [root/p for p in ('src/control/bridge.py', 'simulation/ns3/uav-olsr-dataset.cc',
                              'simulation/ns3/recovery-control.h', 'simulation/ns3/recovery-protocol.h',
                              'scripts/train/train_prevention_predictor.py', 'scripts/analysis/evaluate_prevention.py')]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def _write(path, data, *, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def paired_summary(records, policies, scenarios, bootstrap_seed=20261111, bootstrap_samples=2000):
    """Scenario-paired bootstrap against first requested policy; no time-window IID."""
    validate_paired_grid(records, policies, scenarios)
    metrics = ('disconnected_fraction', 'source_dest_disconnected_fraction', 'olsr_unusable_fraction',
               'partition_count', 'pdr', 'mean_delay_ms', 'total_relay_travel', 'max_relay_travel',
               'rejected_actions', 'intervention_starts')
    lookup = {(r['policy'], r['layout_seed'], r['radio_seed']): r for r in records}
    means = {}
    for policy in policies:
        rows = [lookup[(policy, *s)] for s in scenarios]
        means[policy] = {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
                         if any(r[key] is not None for r in rows) else None for key in metrics}
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(scenarios), size=(bootstrap_samples, len(scenarios)))
    differences = {}
    for policy in policies[1:]:
        delta = np.array([lookup[(policy, *s)]['disconnected_fraction']-
                          lookup[(policies[0], *s)]['disconnected_fraction'] for s in scenarios])
        low, high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        differences[policy] = dict(reference=policies[0], mean_difference=float(delta.mean()),
                                   scenario_bootstrap_95pct_ci=[float(low), float(high)])
    return dict(scenario_count=len(scenarios), means=means, disconnected_fraction_differences=differences,
                bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples,
                statistical_unit='scenario', interval_limit='pilot percentile bootstrap; one scenario gives no variability estimate')


def run_grid(*, config, launcher, output, policies, scenarios, predictor=None, threshold=.5,
             routing_delay_s=3., split='test', provenance=None, env_factory=Ns3PreventionEnv,
             controller_factory=None, supported_policies=POLICIES):
    output = Path(output).resolve()
    require_empty_output(output)  # before construction/reset/any simulator launch
    policies, scenarios = validate_grid(policies, scenarios, supported_policies)
    controller_factory = controller_factory or PreventionController
    if split not in ('train', 'calibration', 'test'): raise ValueError('Invalid split')
    learned = any(p in ('predictive-learned', 'learned-no-criticality', 'learned-no-deadline') for p in policies)
    if learned and predictor is None: raise ValueError('Learned policies need a model')
    if predictor is not None:
        predictor.validate_test_scenarios(scenarios)
        if config.comm_range != predictor.artifact['comm_range']: raise ValueError('Incompatible predictor communication range')
    # Validate controller settings before launch, even if only hold was requested.
    controller_factory(policies[0], config, predictor=predictor, threshold=threshold, routing_delay_s=routing_delay_s)
    output.mkdir(parents=True, exist_ok=True)
    run = dict(schema='prevention-grid-v1', output_root=str(output), config=asdict(config), launcher=list(launcher),
               policies=policies, scenarios=scenarios, split=split, threshold=threshold, routing_delay_s=routing_delay_s,
               provenance=provenance or {}, forecast_semantics='within-H event risk, no exact failure time or action counterfactual',
               metrics_intervals='frames 1..steps; frame0 setup; travel excludes both mission anchors')
    _write(output/'run_config.json', run, exclusive=True)
    records = []
    for policy in policies:
        for layout, radio in scenarios:
            episode_root = output/'episodes'/policy/f'layout_{layout}_radio_{radio}'
            row = dict(policy=policy, layout_seed=layout, radio_seed=radio, status='technical_error',
                       error=None, episode_dir=None)
            env, frames, decisions, phase = None, [], [], 'setup'
            try:
                episode_root.mkdir(parents=True, exist_ok=False)
                env = env_factory(config=config, launcher=launcher, output_root=episode_root)
                env.reset(options=dict(layout_seed=layout, radio_seed=radio))
                row['episode_dir'] = str(env.episode_output_dir.resolve())
                _write(env.episode_output_dir/'policy_metadata.json',
                       dict(collection_policy=policy, split=split, layout_seed=layout, radio_seed=radio), exclusive=True)
                frames.append(env.last_frame)
                controller = controller_factory(policy, config, predictor=predictor,
                                                  threshold=threshold, routing_delay_s=routing_delay_s)
                phase = 'episode'
                with (episode_root/'decisions.jsonl').open('x', encoding='utf-8') as stream:
                    for step in range(config.steps):
                        action, decision = controller.act(env.last_frame)
                        stream.write(json.dumps(decision, allow_nan=False)+'\n'); stream.flush()
                        decisions.append(decision)
                        _, _, terminated, truncated, _ = env.step(action)
                        frames.append(env.last_frame)
                        if terminated or truncated != (step == config.steps-1):
                            raise ValueError('Unexpected termination; full horizon required')
                env.close()
                row.update(episode_metrics(frames, config, env.episode_output_dir, decisions))
                row.update(status='ok', trace_sha256=sha256(env.episode_output_dir/'transitions.jsonl'))
            except Exception as exc:
                row.update(status='setup_error' if phase == 'setup' else 'technical_error', error=f'{type(exc).__name__}: {exc}')
                if env is not None and env.episode_output_dir is not None: row['episode_dir'] = str(env.episode_output_dir.resolve())
            finally:
                if env is not None:
                    try: env.close()
                    except Exception as exc: row.update(status='technical_error', error=f'Close failure: {exc}')
            records.append(row)
            _write(output/'episodes.json', records)
            print(f'{policy} layout={layout} radio={radio}: {row["status"]}', flush=True)
    try:
        if supported_policies != POLICIES and any(r['status'] != 'ok' for r in records):
            raise ValueError('Technical/setup errors invalidate complete paired comparison')
        comparison = paired_summary(records, policies, scenarios) if supported_policies == POLICIES else None
        summary = dict(paired_valid=True, paired_error=None, comparison=comparison)
    except ValueError as exc:
        summary = dict(paired_valid=False, paired_error=str(exc), comparison=None)
    summary.update(episode_count=len(records), errors=[r for r in records if r['status'] != 'ok'])
    _write(output/'summary.json', summary)
    return summary


def reliability_metrics(labels, probabilities, threshold=.5, bins=10):
    labels, p = np.asarray(labels, float), np.asarray(probabilities, float)
    if (labels.ndim != 1 or labels.shape != p.shape or len(labels) == 0
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
        raise ValueError('Finite nonempty matched binary labels and probabilities required')
    if type(bins) is not int or bins < 1 or not 0 < threshold < 1: raise ValueError('Invalid bins/threshold')
    clipped = np.clip(p, 1e-12, 1-1e-12)
    rows = []
    for index in range(bins):
        selected = (p >= index/bins) & ((p < (index+1)/bins) if index < bins-1 else (p <= 1))
        rows.append(dict(lower=index/bins, upper=(index+1)/bins, count=int(selected.sum()),
                         mean_probability=float(p[selected].mean()) if selected.any() else None,
                         observed_fraction=float(labels[selected].mean()) if selected.any() else None))
    positive, negative = labels == 1, labels == 0
    return dict(samples=len(p), positives=int(positive.sum()), brier=float(np.mean((p-labels)**2)),
                log_loss=float(-np.mean(labels*np.log(clipped)+(1-labels)*np.log(1-clipped))),
                threshold=threshold, recall=float(np.mean(p[positive] >= threshold)) if positive.any() else None,
                false_positive_rate=float(np.mean(p[negative] >= threshold)) if negative.any() else None, bins=rows)


def evaluate_predictor(model, episodes, threshold=.5, bins=10):
    episodes = list(episodes)
    if not episodes: raise ValueError('No test episodes')
    _require_partition(episodes, 'test', allow_unknown=True)
    model.validate_test_scenarios([(e.layout_seed, e.radio_seed) for e in episodes])
    all_labels = {h: [] for h in HORIZONS}; all_p = {h: [] for h in HORIZONS}; per_episode = []
    by_policy = {}
    for episode in episodes:
        labels = {h: [] for h in HORIZONS}; probabilities = {h: [] for h in HORIZONS}
        frames = episode.frames
        edges = [set(measured_edges(f)) for f in frames]
        for t in range(len(frames)-min(HORIZONS)):
            risks = model.predict(frames[t], frames[t-1] if t else None, episode.comm_range)
            for h in HORIZONS:
                if t+h >= len(frames): continue
                for edge in sorted(edges[t]):
                    labels[h].append(int(any(edge not in future for future in edges[t+1:t+h+1])))
                    probabilities[h].append(risks[h][edge])
        results = {}
        for h in HORIZONS:
            results[str(h)] = reliability_metrics(labels[h], probabilities[h], threshold, bins) if labels[h] else None
            all_labels[h].extend(labels[h]); all_p[h].extend(probabilities[h])
        group = by_policy.setdefault(episode.collection_policy,
                                     {h: dict(labels=[], probabilities=[]) for h in HORIZONS})
        for h in HORIZONS:
            group[h]['labels'].extend(labels[h]); group[h]['probabilities'].extend(probabilities[h])
        per_episode.append(dict(episode_id=episode.episode_id, layout_seed=episode.layout_seed, radio_seed=episode.radio_seed,
                                collection_policy=episode.collection_policy, source_hashes=episode.source_hashes, horizons=results))
    return dict(schema='prevention-reliability-v1',
                semantics='observed within-H failures on supplied policy traces; interventions can prevent failures; not counterfactual false alarms',
                statistical_unit='episode/scenario; pooled windows are descriptive, not independent trials',
                horizons={str(h): reliability_metrics(all_labels[h], all_p[h], threshold, bins) for h in HORIZONS},
                by_collection_policy={policy: {str(h): reliability_metrics(group[h]['labels'], group[h]['probabilities'], threshold, bins)
                                             if group[h]['labels'] else None for h in HORIZONS}
                                      for policy, group in by_policy.items()},
                episodes=per_episode)

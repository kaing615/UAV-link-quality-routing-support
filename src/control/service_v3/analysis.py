"""Pre-registered v3 analyses: validation go/no-go (section 5) and locked test (section 6)."""
from pathlib import Path

import numpy as np

from ..service_v2.protocol import reserve, sha256, strict_json, write_json

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20261120
GO_PDR_MARGIN = -.005
METRICS = ('actual_udp_pdr', 'total_relay_travel_m')


def _ok(row):
    return row is not None and row.get('status') == 'ok' and all(row.get(m) is not None for m in METRICS)


def blocks(rows, policies, context='gradual'):
    """Per regime: complete scenario blocks (every arm observed) and shared bootstrap draws."""
    rng = np.random.default_rng(BOOTSTRAP_SEED); lookup = {}; out = {}
    for r in rows:
        lookup[(r['regime'], r['context'], r['layout_seed'], r['radio_seed']), r['policy']] = r
    for regime in ('n4', 'n5'):
        keys = sorted({k for (k, _) in lookup if k[0] == regime and k[1] == context})
        complete = [k for k in keys if all(_ok(lookup.get((k, p))) for p in policies)]
        draws = rng.integers(0, len(complete), size=(BOOTSTRAP_SAMPLES, len(complete))) if complete else None
        out[regime] = dict(keys=keys, complete=complete, draws=draws)
    return lookup, out


def contrast(block_data, a, b, metric):
    """Equal-weight n4/n5 mean of paired (a - b), CI from scenario-block bootstrap."""
    lookup, strata = block_data; regimes = {}; samples = []
    for regime, group in strata.items():
        if not group['complete']: return dict(arm=a, comparator=b, metric=metric, point=None, ci95=None, regimes=regimes)
        delta = np.array([lookup[k, a][metric] - lookup[k, b][metric] for k in group['complete']])
        samples.append(delta[group['draws']].mean(axis=1))
        bounds = None
        if metric == 'actual_udp_pdr':  # missing-PDR sensitivity over every planned block
            get = lambda k, p, fill: lookup[k, p][metric] if _ok(lookup.get((k, p))) else fill
            bounds = [float(np.mean([get(k, a, 0) - get(k, b, 1) for k in group['keys']])),
                      float(np.mean([get(k, a, 1) - get(k, b, 0) for k in group['keys']]))]
        regimes[regime] = dict(planned=len(group['keys']), complete=len(group['complete']), point=float(delta.mean()),
                               missing_sensitivity=bounds)
    pooled = (samples[0] + samples[1]) / 2
    return dict(arm=a, comparator=b, metric=metric, point=float(np.mean([r['point'] for r in regimes.values()])),
                ci95=[float(v) for v in np.quantile(pooled, [.025, .975])], regimes=regimes)


def burst_predictions(rows, selection, policies):
    """Ridge prediction from the logged start decision vs realized matched-window gain."""
    from .actionvalue import Ridge
    from .study import labelled_bursts
    labels, excluded = labelled_bursts(rows, policies)
    model = Ridge.from_dict(selection['ridge'])
    for item in labels:
        item['predicted'] = float(model.predict(item['features'])[0])
    return labels, excluded


def go_decision(rows, selection, policies):
    from .actionvalue import spearman
    labels, excluded = burst_predictions(rows, selection, ('service-v3-threshold', 'service-v3-gate'))
    rho = spearman([b['predicted'] for b in labels], [b['y'] for b in labels])
    data = blocks(rows, policies)
    pdr = contrast(data, 'service-v3-gate', 'service-v2-no-forecast', 'actual_udp_pdr')
    motion = contrast(data, 'service-v3-gate', 'service-v2', 'total_relay_travel_m')
    criteria = dict(
        spearman_positive=dict(value=rho, passed=rho is not None and rho > 0),
        pdr_not_worse_than_margin=dict(value=pdr['point'], margin=GO_PDR_MARGIN,
                                       passed=pdr['point'] is not None and pdr['point'] >= GO_PDR_MARGIN),
        motion_not_above_v2=dict(value=motion['point'], passed=motion['point'] is not None and motion['point'] <= 0))
    return dict(go=all(c['passed'] for c in criteria.values()), criteria=criteria, bursts=len(labels),
                excluded_bursts=excluded, contrasts=dict(pdr=pdr, motion=motion), burst_rows=labels)


def fixed_sequence(data):
    """Section 6: interpret a step only if every earlier step succeeded."""
    steps = [('pdr_gate_vs_no_forecast', contrast(data, 'service-v3-gate', 'service-v2-no-forecast', 'actual_udp_pdr'), 'lower_gt_0'),
             ('motion_gate_vs_v2', contrast(data, 'service-v3-gate', 'service-v2', 'total_relay_travel_m'), 'upper_lt_0'),
             ('pdr_gate_vs_hold', contrast(data, 'service-v3-gate', 'hold', 'actual_udp_pdr'), 'lower_gt_0')]
    out = []; open_gate = True
    for name, c, rule in steps:
        ci = c['ci95']
        success = ci is not None and (ci[0] > 0 if rule == 'lower_gt_0' else ci[1] < 0)
        out.append(dict(name=name, rule=rule, contrast=c, success=success, interpretable=open_gate))
        open_gate = open_gate and success
    return out


def _safety(rows):
    from ..service_v2.analysis import episode_endpoints
    from .study import episode
    totals = {}
    for r in rows:
        e = episode_endpoints(episode(r, 'v3'), r['decisions'])
        t = totals.setdefault(r['policy'], dict(forecast_starts_before_arming=0, simulator_safety_rejections=0,
                                                budget_rejections=0, gate_blocks=0, forecast_starts=0))
        for k in ('forecast_starts_before_arming', 'simulator_safety_rejections', 'budget_rejections', 'forecast_starts'):
            t[k] += e[k]
        t['gate_blocks'] += sum(d.get('reason') == 'action_value_gate' for d in r['decisions'])
    return totals


def analyze(protocol_path, run_root, output):
    from .study import load_traces
    phase, report, rows = load_traces(protocol_path, run_root)
    if phase['phase'] not in ('validation', 'test'): raise ValueError('Analysis is for validation/test')
    package = Path(phase['_path']).parent; selection = strict_json(package/'selection.json')
    out = reserve(output); policies = phase['policies']
    slim = [{k: v for k, v in r.items() if k not in ('frames', 'decisions')} for r in rows]
    result = dict(schema=f"service-v3-{phase['phase']}-analysis-v1", freeze_sha256=phase['_freeze_sha256'],
                  selection_sha256=phase['selection_sha256'], planned_cells=report['planned_cells'],
                  ok_cells=report['ok_cells'], safety=_safety(rows), rows=slim,
                  bootstrap=dict(samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, unit='gradual scenario block, all arms kept together'))
    if phase['phase'] == 'validation':
        decision = go_decision(rows, selection, policies)
        result['go'] = decision
        write_json(out/'go.json', dict(schema='service-v3-go-v1', go=decision['go'], criteria=decision['criteria'],
                                       validation_freeze_sha256=phase['_freeze_sha256']))
    else:
        data = blocks(rows, policies)
        result['fixed_sequence'] = fixed_sequence(data)
        result['descriptive'] = [contrast(data, a, b, m) for a, b in
                                 (('service-v3-threshold', 'service-v2'), ('service-v3-threshold', 'service-v2-no-forecast'),
                                  ('service-v2', 'service-v2-no-forecast')) for m in METRICS]
        result['static_descriptive'] = [contrast(blocks(rows, policies, 'static'), 'service-v3-gate', b, 'actual_udp_pdr')
                                        for b in ('service-v2-no-forecast', 'hold')]
        result['forecast_at_threshold'] = forecast_at_threshold(rows, selection, package/'new-model.json')
    write_json(out/'analysis.json', result)
    write_json(out/'manifest.json', dict(files={p.name: sha256(p) for p in out.iterdir() if p.is_file()}))
    return result


def forecast_at_threshold(rows, selection, model_path):
    """Descriptive: predictor confusion at the frozen tau on test HOLD (chosen horizon)."""
    from dataclasses import asdict
    from ..service_v2.analysis import forecast_scores
    from ..service_v2.optimization import _point
    from ..service_v2.prediction import ServicePredictor
    from .study import episode
    scores = forecast_scores([episode(r, 'test') for r in rows if r['policy'] == 'hold'], ServicePredictor.load(model_path))
    head = scores['horizons'][str(selection['horizon'])]
    y = np.array([o['label'] for o in head['origins']], dtype=np.int8); p = np.array([o['projected_risk'] for o in head['origins']])
    point = asdict(_point(y, p, selection['risk_on'], 1., 1., 1.)) if len(y) else None
    return dict(horizon=selection['horizon'], threshold=selection['risk_on'], operating_point=point, pr_auc=head['pr_auc'],
                false_warning_fraction_at_v2_threshold_050=scores['false_warning_fraction'])

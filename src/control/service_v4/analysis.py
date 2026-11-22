"""Pre-registered v4 analyses (docs/service_v4_protocol.md sections 5-6)."""
from pathlib import Path

from ..service_v2.protocol import reserve, sha256, write_json
from ..service_v3.analysis import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, METRICS, blocks, contrast
from .study import ROUTE_ARMS, V4, load, route_integrity

MIN_ENGAGED_EPISODE_SHARE = .5


def mechanism(rows):
    """Per arm: how often the override was used and PDR on pinned vs OLSR seconds (gradual)."""
    out = {}
    for policy in sorted({r['policy'] for r in rows}):
        chosen = [r for r in rows if r['policy'] == policy and r['context'] == 'gradual']
        pinned = dict(offered=0, rx=0, seconds=0); live = dict(offered=0, rx=0, seconds=0)
        engaged = changes = 0
        for r in chosen:
            routes = [(d.get('route_control') or {}).get('route', []) for d in r['decisions']]
            engaged += any(routes)
            changes += sum(a != b for a, b in zip([[]] + routes, routes))
            for t, route in enumerate(routes):  # decision t governs window (t, t+1] = frame t+1
                w = r['frames'][t + 1]['service_observation']['window']
                acc = pinned if route else live
                acc['offered'] += w['offered_packets']; acc['rx'] += w['sink_rx_packets']; acc['seconds'] += 1
        pdr = lambda a: a['rx'] / a['offered'] if a['offered'] else None
        out[policy] = dict(gradual_episodes=len(chosen), engaged_episodes=engaged,
                           engaged_share=engaged / len(chosen) if chosen else None,
                           route_changes_per_episode=changes / len(chosen) if chosen else None,
                           pinned_seconds=pinned['seconds'], pinned_pdr=pdr(pinned), olsr_seconds=live['seconds'], olsr_pdr=pdr(live))
    return out


def go_decision(rows, policies, integrity_ok, mech):
    data = blocks(rows, policies)
    pdr = contrast(data, 'lq-route', 'hold', 'actual_udp_pdr')
    criteria = dict(
        pdr_route_above_hold=dict(value=pdr['point'], passed=pdr['point'] is not None and pdr['point'] > 0),
        override_engaged=dict(value=mech['lq-route']['engaged_share'], minimum=MIN_ENGAGED_EPISODE_SHARE,
                              passed=(mech['lq-route']['engaged_share'] or 0) >= MIN_ENGAGED_EPISODE_SHARE),
        routes_applied_as_commanded=dict(passed=integrity_ok))
    return dict(go=all(c['passed'] for c in criteria.values()), criteria=criteria, pdr=pdr)


def fixed_sequence(data):
    steps = [('pdr_route_vs_hold', contrast(data, 'lq-route', 'hold', 'actual_udp_pdr')),
             ('pdr_route_restore_vs_restore', contrast(data, 'lq-route-restore', 'service-v2-no-forecast', 'actual_udp_pdr')),
             ('pdr_route_restore_vs_hold', contrast(data, 'lq-route-restore', 'hold', 'actual_udp_pdr'))]
    out, open_gate = [], True
    for name, c in steps:
        success = c['ci95'] is not None and c['ci95'][0] > 0
        out.append(dict(name=name, rule='lower_gt_0', contrast=c, success=success, interpretable=open_gate))
        open_gate = open_gate and success
    return out


def analyze(protocol_path, run_root, output):
    from ..service_v3.analysis import _safety
    from ..service_v3.study import load_traces
    phase, report, rows = load_traces(protocol_path, run_root, loader=load)
    if phase['phase'] not in ('validation', 'test'): raise ValueError('Analysis is for validation/test')
    out = reserve(output); policies = phase['policies']
    integrity = {r['cell_id']: route_integrity(r) for r in rows if r['policy'] in ROUTE_ARMS}
    integrity_ok = all(v['ok'] for v in integrity.values())
    mech = mechanism(rows)
    data = blocks(rows, policies)
    slim = [{k: v for k, v in r.items() if k not in ('frames', 'decisions')} for r in rows]
    result = dict(schema=f"service-v4-{phase['phase']}-analysis-v1", freeze_sha256=phase['_freeze_sha256'],
                  selection_sha256=phase['selection_sha256'], planned_cells=report['planned_cells'], ok_cells=report['ok_cells'],
                  route_integrity_ok=integrity_ok, route_integrity=integrity, mechanism=mech, safety=_safety(rows),
                  descriptive=[contrast(data, a, b, m) for a, b in
                               (('lq-route', 'hold'), ('lq-route-restore', 'service-v2-no-forecast'),
                                ('service-v2-no-forecast', 'hold'), ('lq-route-restore', 'lq-route')) for m in METRICS],
                  static_descriptive=[contrast(blocks(rows, policies, 'static'), 'lq-route', 'hold', 'actual_udp_pdr')],
                  bootstrap=dict(samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, unit='gradual scenario block, all arms kept together'),
                  rows=slim)
    if phase['phase'] == 'validation':
        decision = go_decision(rows, policies, integrity_ok, mech)
        result['go'] = decision
        write_json(out/'go.json', dict(schema=V4.schema('go'), go=decision['go'], criteria=decision['criteria'],
                                       validation_freeze_sha256=phase['_freeze_sha256']))
    else:
        result['fixed_sequence'] = fixed_sequence(data)
    write_json(out/'analysis.json', result)
    write_json(out/'manifest.json', dict(files={p.name: sha256(p) for p in out.iterdir() if p.is_file()}))
    return result

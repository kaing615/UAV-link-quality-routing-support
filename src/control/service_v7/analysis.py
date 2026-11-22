"""Pre-registered v7 analyses (docs/service_v7_protocol.md sections 4-6)."""
import numpy as np

from ..service_v2.protocol import reserve, sha256, write_json
from ..service_v3.analysis import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, blocks, contrast
from ..service_v4.study import route_integrity
from ..service_v5.analysis import route_mix
from .study import (HELLO_ARMS, REGIMES, ROUTE_ARMS, V7, background, central_signalling_bytes, flags_ok, load,
                    selected_hello, startup_blackout, transmissions, write_selection)

STEPS = ('pdr_lq_route_vs_olsr_etx_fast', 'lq-route', 'olsr-etx-fast'), \
        ('pdr_lq_route_delay_vs_olsr_etx_fast', 'lq-route-delay', 'olsr-etx-fast'), \
        ('pdr_lq_route_vs_olsr_etx', 'lq-route', 'olsr-etx')


def _blocks(rows, policies, context='gradual'):
    return blocks(rows, policies, context, regimes=REGIMES)


def technical(rows, hello):
    """Checks shared by every phase; any failure is a technical failure, not a result."""
    integrity = {r['cell_id']: route_integrity(r) for r in rows if r['policy'] in ROUTE_ARMS}
    tx = {r['cell_id']: transmissions(r) for r in rows}
    bg = {r['cell_id']: background(r) for r in rows}
    return dict(route_integrity_ok=all(v['ok'] for v in integrity.values()), route_integrity=integrity,
                flags_ok=all(flags_ok(r, hello) for r in rows), startup_blackouts=sorted(r['cell_id'] for r in rows if startup_blackout(r)),
                transmissions_ok=all(v is not None for v in tx.values()), background_ok=all(v is not None for v in bg.values())), tx, bg


def overhead(rows, tx):
    """Mean IP-layer bytes per second per arm (gradual): OLSR and probes measured, central reports estimated."""
    out = {}
    for policy in sorted({r['policy'] for r in rows}):
        sel = [r for r in rows if r['policy'] == policy and r['context'] == 'gradual']
        if not sel: continue
        seconds = len(sel[0]['frames']) - 1
        mean = lambda kind: float(np.mean([tx[r['cell_id']][kind][1] for r in sel])) / seconds
        entry = dict(olsr_Bps=mean('olsr'), probe_Bps=mean('probe'), data_Bps=mean('data'), background_Bps=mean('background'))
        if policy in ROUTE_ARMS:
            entry['central_estimate_Bps'] = float(np.mean([central_signalling_bytes(r) for r in sel])) / seconds
            entry['method_total_Bps'] = entry['olsr_Bps'] + entry['probe_Bps'] + entry['central_estimate_Bps']
        else:
            entry['method_total_Bps'] = entry['olsr_Bps']
        out[policy] = entry
    return out


def with_background_pdr(rows, bg):
    return [dict(r, background_pdr=bg[r['cell_id']][0] / bg[r['cell_id']][1]) for r in rows]


def select(protocol_path, run_root, output):
    """Tuning: HELLO interval with the highest equal-weight n5/n7 gradual PDR; tie -> longer interval."""
    from ..service_v3.study import load_traces
    phase, report, rows = load_traces(protocol_path, run_root, loader=load)
    if phase['phase'] != 'tuning': raise ValueError('select is for the tuning phase')
    checks, tx, bg = technical(rows, None)
    if not (checks['flags_ok'] and checks['transmissions_ok'] and checks['background_ok']) or checks['startup_blackouts']:
        raise ValueError('Tuning technical check failed: ' + str({k: v for k, v in checks.items() if k != 'route_integrity'}))
    data = _blocks(rows, list(HELLO_ARMS))
    lookup, strata = data
    table = []
    for arm, hello in HELLO_ARMS.items():
        per = {g: float(np.mean([lookup[k, arm]['actual_udp_pdr'] for k in strata[g]['complete']])) for g in REGIMES}
        table.append(dict(arm=arm, hello_interval_s=hello, pdr=float(np.mean(list(per.values()))), regimes=per,
                          olsr_Bps=overhead(rows, tx)[arm]['olsr_Bps']))
    best = max(table, key=lambda t: (t['pdr'], t['hello_interval_s']))
    write_selection(output, best['hello_interval_s'], table, phase['_freeze_sha256'])
    return best, table


def fixed_sequence(data):
    out, gate = [], True
    for name, a, b in STEPS:
        c = contrast(data, a, b, 'actual_udp_pdr')
        success = c['ci95'] is not None and c['ci95'][0] > 0
        out.append(dict(name=name, rule='lower_gt_0', contrast=c, success=success, interpretable=gate))
        gate = gate and success
    return out


def analyze(protocol_path, run_root, output):
    from ..service_v3.study import load_traces
    phase, report, rows = load_traces(protocol_path, run_root, loader=load)
    if phase['phase'] == 'tuning': raise ValueError('Use select for tuning')
    out = reserve(output); policies = phase['policies']
    checks, tx, bg = technical(rows, selected_hello(phase))
    over = overhead(rows, tx)
    mix = route_mix(rows)
    rows_bg = with_background_pdr(rows, bg) if checks['background_ok'] else rows
    data = _blocks(rows_bg, policies)
    pairs = [('lq-route', 'olsr-etx-fast'), ('lq-route-delay', 'olsr-etx-fast'), ('lq-route', 'olsr-etx'),
             ('olsr-etx-fast', 'olsr-etx'), ('olsr-etx', 'hold'), ('lq-route', 'hold'), ('lq-route', 'lq-route-delay')]
    metrics = ['actual_udp_pdr'] + (['background_pdr'] if checks['background_ok'] else [])
    slim = [{k: v for k, v in r.items() if k not in ('frames', 'decisions')} for r in rows_bg]
    result = dict(schema=f"service-v7-{phase['phase']}-analysis-v1", freeze_sha256=phase['_freeze_sha256'],
                  selection_sha256=phase['selection_sha256'], planned_cells=report['planned_cells'], ok_cells=report['ok_cells'],
                  **checks, overhead=over, route_mix=mix,
                  descriptive=[contrast(data, a, b, m) for a, b in pairs for m in metrics],
                  static_descriptive=[contrast(_blocks(rows_bg, policies, 'static'), a, b, 'actual_udp_pdr') for a, b in pairs[:3]],
                  bootstrap=dict(samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
                                 unit='gradual scenario block, all arms kept together, equal weight n5/n7'),
                  rows=slim)
    if phase['phase'] == 'validation':
        fast = over.get('olsr-etx-fast', {}).get('olsr_Bps', 0.); slow = over.get('olsr-etx', {}).get('olsr_Bps', 0.)
        criteria = dict(routes_applied_as_commanded=dict(passed=checks['route_integrity_ok']),
                        arm_flags_exact=dict(passed=checks['flags_ok']),
                        no_startup_blackout=dict(cells=checks['startup_blackouts'], passed=not checks['startup_blackouts']),
                        transmissions_counted=dict(passed=checks['transmissions_ok']),
                        background_measured=dict(passed=checks['background_ok']),
                        fast_hello_active=dict(olsr_etx_Bps=slow, olsr_etx_fast_Bps=fast, passed=fast > 1.5 * slow))
        go = all(c['passed'] for c in criteria.values())
        result['go'] = dict(go=go, criteria=criteria)
        write_json(out/'go.json', dict(schema=V7.schema('go'), go=go, criteria=criteria,
                                       validation_freeze_sha256=phase['_freeze_sha256']))
    else:
        result['fixed_sequence'] = fixed_sequence(data)
    write_json(out/'analysis.json', result)
    write_json(out/'manifest.json', dict(files={p.name: sha256(p) for p in out.iterdir() if p.is_file()}))
    return result

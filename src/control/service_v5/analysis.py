"""Pre-registered v5 analyses (docs/service_v5_protocol.md sections 4-5)."""
from collections import Counter

from ..service_v2.protocol import reserve, sha256, write_json
from ..service_v3.analysis import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, METRICS, blocks, contrast
from ..service_v4.study import route_integrity
from .study import V5, etx_active, load


def route_mix(rows):
    """Per arm, gradual: share of seconds on a direct / relay / no OLSR route and losses by kind."""
    out = {}
    for policy in sorted({r['policy'] for r in rows}):
        seconds, lost = Counter(), Counter()
        for r in rows:
            if r['policy'] != policy or r['context'] != 'gradual': continue
            for f in r['frames'][1:]:
                s = f['service_observation']; route = s['route']
                kind = 'none' if not route['success'] else 'direct' if len(route['nodes']) == 2 else 'relay'
                seconds[kind] += 1; lost[kind] += s['window']['offered_packets'] - s['window']['sink_rx_packets']
        total = sum(seconds.values()) or 1
        out[policy] = dict(seconds_share={k: v / total for k, v in seconds.items()}, lost_packets=dict(lost))
    return out


def go_decision(rows, integrity_ok, etx_ok, mix):
    direct = lambda p: mix[p]['seconds_share'].get('direct', 0.)
    criteria = dict(
        routes_applied_as_commanded=dict(passed=integrity_ok),
        olsr_etx_active_only_in_its_arm=dict(passed=etx_ok),
        olsr_etx_reduces_direct_route_time=dict(hold=direct('hold'), olsr_etx=direct('olsr-etx'),
                                               passed=direct('olsr-etx') < direct('hold')))
    return dict(go=all(c['passed'] for c in criteria.values()), criteria=criteria)


def fixed_sequence(data):
    steps = [('pdr_lq_route_vs_olsr_etx', contrast(data, 'lq-route', 'olsr-etx', 'actual_udp_pdr')),
             ('pdr_olsr_etx_vs_hold', contrast(data, 'olsr-etx', 'hold', 'actual_udp_pdr'))]
    out, open_gate = [], True
    for name, c in steps:
        success = c['ci95'] is not None and c['ci95'][0] > 0
        out.append(dict(name=name, rule='lower_gt_0', contrast=c, success=success, interpretable=open_gate))
        open_gate = open_gate and success
    return out


def analyze(protocol_path, run_root, output):
    from ..service_v3.study import load_traces
    phase, report, rows = load_traces(protocol_path, run_root, loader=load)
    out = reserve(output); policies = phase['policies']
    integrity = {r['cell_id']: route_integrity(r) for r in rows if r['policy'] == 'lq-route'}
    integrity_ok = all(v['ok'] for v in integrity.values())
    etx_ok = all(etx_active(r) for r in rows)
    mix = route_mix(rows)
    data = blocks(rows, policies)
    slim = [{k: v for k, v in r.items() if k not in ('frames', 'decisions')} for r in rows]
    result = dict(schema=f"service-v5-{phase['phase']}-analysis-v1", freeze_sha256=phase['_freeze_sha256'],
                  selection_sha256=phase['selection_sha256'], planned_cells=report['planned_cells'], ok_cells=report['ok_cells'],
                  route_integrity_ok=integrity_ok, route_integrity=integrity, olsr_etx_flags_ok=etx_ok, route_mix=mix,
                  descriptive=[contrast(data, a, b, m) for a, b in
                               (('lq-route', 'olsr-etx'), ('olsr-etx', 'hold'), ('lq-route', 'hold')) for m in METRICS],
                  static_descriptive=[contrast(blocks(rows, policies, 'static'), a, b, 'actual_udp_pdr')
                                      for a, b in (('lq-route', 'olsr-etx'), ('olsr-etx', 'hold'))],
                  bootstrap=dict(samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, unit='gradual scenario block, all arms kept together'),
                  rows=slim)
    if phase['phase'] == 'validation':
        decision = go_decision(rows, integrity_ok, etx_ok, mix)
        result['go'] = decision
        write_json(out/'go.json', dict(schema=V5.schema('go'), go=decision['go'], criteria=decision['criteria'],
                                       validation_freeze_sha256=phase['_freeze_sha256']))
    else:
        result['fixed_sequence'] = fixed_sequence(data)
    write_json(out/'analysis.json', result)
    write_json(out/'manifest.json', dict(files={p.name: sha256(p) for p in out.iterdir() if p.is_file()}))
    return result

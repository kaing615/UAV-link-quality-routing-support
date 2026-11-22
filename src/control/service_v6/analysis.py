"""Pre-registered v6 analyses (docs/service_v6_protocol.md sections 4-5); v5 analyses plus ARP checks."""
from ..service_v2.protocol import reserve, sha256, write_json
from ..service_v3.analysis import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, METRICS, blocks, contrast
from ..service_v4.study import route_integrity
from ..service_v5.analysis import fixed_sequence, go_decision, route_mix
from .study import V6, arp_active, etx_active, load, startup_blackout


def analyze(protocol_path, run_root, output):
    from ..service_v3.study import load_traces
    phase, report, rows = load_traces(protocol_path, run_root, loader=load)
    out = reserve(output); policies = phase['policies']
    integrity = {r['cell_id']: route_integrity(r) for r in rows if r['policy'] == 'lq-route'}
    integrity_ok = all(v['ok'] for v in integrity.values())
    etx_ok = all(etx_active(r) for r in rows)
    arp_ok = all(arp_active(r) for r in rows)
    blackouts = sorted(r['cell_id'] for r in rows if startup_blackout(r))
    mix = route_mix(rows)
    data = blocks(rows, policies)
    slim = [{k: v for k, v in r.items() if k not in ('frames', 'decisions')} for r in rows]
    result = dict(schema=f"service-v6-{phase['phase']}-analysis-v1", freeze_sha256=phase['_freeze_sha256'],
                  selection_sha256=phase['selection_sha256'], planned_cells=report['planned_cells'], ok_cells=report['ok_cells'],
                  route_integrity_ok=integrity_ok, route_integrity=integrity, olsr_etx_flags_ok=etx_ok,
                  arp_fix_ok=arp_ok, startup_blackouts=blackouts, route_mix=mix,
                  descriptive=[contrast(data, a, b, m) for a, b in
                               (('lq-route', 'olsr-etx'), ('olsr-etx', 'hold'), ('lq-route', 'hold')) for m in METRICS],
                  static_descriptive=[contrast(blocks(rows, policies, 'static'), a, b, 'actual_udp_pdr')
                                      for a, b in (('lq-route', 'olsr-etx'), ('olsr-etx', 'hold'))],
                  bootstrap=dict(samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, unit='gradual scenario block, all arms kept together'),
                  rows=slim)
    if phase['phase'] == 'validation':
        decision = go_decision(rows, integrity_ok, etx_ok, mix)
        decision['criteria'].update(arp_fix_in_every_episode=dict(passed=arp_ok),
                                    no_startup_blackout=dict(cells=blackouts, passed=not blackouts))
        decision['go'] = all(c['passed'] for c in decision['criteria'].values())
        result['go'] = decision
        write_json(out/'go.json', dict(schema=V6.schema('go'), go=decision['go'], criteria=decision['criteria'],
                                       validation_freeze_sha256=phase['_freeze_sha256']))
    else:
        result['fixed_sequence'] = fixed_sequence(data)
    write_json(out/'analysis.json', result)
    write_json(out/'manifest.json', dict(files={p.name: sha256(p) for p in out.iterdir() if p.is_file()}))
    return result

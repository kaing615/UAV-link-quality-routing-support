"""Where are packets lost, and what does a burst change? Read-only diagnosis of
complete service collections (v2 formal, v3 tuning/validation); fits nothing.

Per second: window offered/received, OLSR route (direct source->destination,
via relays, or none) and the worst probe rx_ratio on the route.  Per burst:
event-aligned PDR and route mix versus the HOLD trace of the same seed.
"""
import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path

import numpy as np

OFFSETS = range(-10, 61)


def episode_files(run_root, row):
    attempt = Path(run_root) / row['cell_id'] / f"attempt-{row['attempt']}"
    episode = attempt / row['episode_path']
    return episode / 'transitions.jsonl', next(episode.parent.glob('decisions.jsonl'))


def seconds(transitions):
    out = []
    for line in Path(transitions).read_text(encoding='utf-8').splitlines():
        f = json.loads(line)['frame']
        if f['step'] == 0:
            continue
        s = f['service_observation']; route = s['route']
        nodes = route['nodes'] if route['success'] else []
        hops = {(h['from'], h['to']): min(h['rx_ratio_forward'], h['rx_ratio_reverse']) for h in s['route_probe_hops']}
        worst = min(hops.values()) if hops else None
        kind = 'none' if not nodes else 'direct' if len(nodes) == 2 else 'relay'
        out.append(dict(step=f['step'], offered=s['window']['offered_packets'], rx=s['window']['sink_rx_packets'],
                        kind=kind, hops=len(nodes) - 1 if nodes else None, worst=worst, nodes=tuple(nodes)))
    last = -99
    for i, x in enumerate(out):  # seconds since the OLSR route last changed
        if i and x['nodes'] != out[i - 1]['nodes']: last = x['step']
        age = x['step'] - last
        x['age'] = 'changed_now' if age == 0 else 'changed_1-5s' if age <= 5 else 'stable_gt5s'
    return out


def pdr_table(series_list, key):
    acc = defaultdict(lambda: [0, 0, 0])
    for series in series_list:
        for x in series:
            a = acc[key(x)]; a[0] += 1; a[1] += x['offered']; a[2] += x['rx']
    lost = sum(a[1] - a[2] for a in acc.values()) or 1
    return {' '.join(map(str, k)): dict(seconds=a[0], pdr=a[2] / a[1] if a[1] else None, share_of_loss=(a[1] - a[2]) / lost)
            for k, a in sorted(acc.items(), key=lambda kv: str(kv[0]))}


def loss_by_route(series):
    acc = defaultdict(lambda: dict(seconds=0, offered=0, lost=0, worst=[]))
    for s in series:
        a = acc[s['kind']]; a['seconds'] += 1; a['offered'] += s['offered']; a['lost'] += s['offered'] - s['rx']
        if s['worst'] is not None: a['worst'].append(s['worst'])
    return acc


def summarize_route(acc):
    total_lost = sum(a['lost'] for a in acc.values()) or 1
    return {k: dict(seconds=a['seconds'], offered=a['offered'], lost=a['lost'], share_of_loss=a['lost'] / total_lost,
                    pdr=1 - a['lost'] / a['offered'] if a['offered'] else None,
                    median_worst_link=float(np.median(a['worst'])) if a['worst'] else None) for k, a in acc.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True, help='validation.json (complete collection)')
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in json.loads(args.validation.read_text(encoding='utf-8'))['rows'] if r['status'] == 'ok']
    series, decisions = {}, {}
    for r in rows:
        t, d = episode_files(args.run_root, r)
        series[r['cell_id']] = seconds(t)
        decisions[r['cell_id']] = [json.loads(line) for line in d.read_text(encoding='utf-8').splitlines()]
    key = lambda r: (r['regime'], r['context'], r['layout_seed'], r['radio_seed'])
    hold = {key(r): r['cell_id'] for r in rows if r['policy'] == 'hold'}

    routes = {}
    for policy in sorted({r['policy'] for r in rows}):
        for context in ('gradual', 'static'):
            acc = defaultdict(lambda: dict(seconds=0, offered=0, lost=0, worst=[]))
            for r in rows:
                if r['policy'] == policy and r['context'] == context:
                    for k, a in loss_by_route(series[r['cell_id']]).items():
                        for f in ('seconds', 'offered', 'lost'): acc[k][f] += a[f]
                        acc[k]['worst'] += a['worst']
            routes[f'{context}/{policy}'] = summarize_route(acc)

    route_age, link_bins, changes = {}, {}, {}
    for policy in sorted({r['policy'] for r in rows}):
        chosen = [series[r['cell_id']] for r in rows if r['policy'] == policy and r['context'] == 'gradual']
        route_age[policy] = pdr_table(chosen, lambda x: (x['age'], x['kind']))
        link_bins[policy] = pdr_table(chosen, lambda x: (x['kind'], None if x['worst'] is None else min(int(x['worst'] * 10), 9) / 10))
        changes[policy] = sum(x['age'] == 'changed_now' for c in chosen for x in c) / max(1, len(chosen))

    aligned = defaultdict(lambda: defaultdict(list))   # (policy, context) -> offset -> deltas
    route_after = defaultdict(lambda: defaultdict(int))
    bursts = []
    for r in rows:
        if r['policy'] == 'hold' or r['context'] != 'gradual': continue
        mine, base = series[r['cell_id']], series[hold[key(r)]]
        by_step = {s['step']: s for s in mine}; base_step = {s['step']: s for s in base}
        for d in decisions[r['cell_id']]:
            if d.get('start') is not True: continue
            context = 'forecast' if d.get('context') == 'forecast' else 'restoration'
            t = d['step']
            for k in OFFSETS:
                a, b = by_step.get(t + k), base_step.get(t + k)
                if a and b and a['offered'] and b['offered']:
                    aligned[r['policy'], context][k].append(a['rx'] / a['offered'] - b['rx'] / b['offered'])
            before = by_step.get(t); window = [by_step[t + k] for k in range(1, 31) if t + k in by_step]
            hold_window = [base_step[t + k] for k in range(1, 31) if t + k in base_step]
            for s in window: route_after[r['policy'], context][s['kind']] += 1
            chain = (d.get('plan') or {}).get('chain')
            bursts.append(dict(cell=r['cell_id'], policy=r['policy'], context=context, step=t,
                               route_before=before['kind'] if before else None,
                               plan_chain=chain,
                               relay_seconds_after=sum(s['kind'] == 'relay' for s in window),
                               relay_seconds_hold=sum(s['kind'] == 'relay' for s in hold_window),
                               direct_seconds_after=sum(s['kind'] == 'direct' for s in window),
                               direct_seconds_hold=sum(s['kind'] == 'direct' for s in hold_window),
                               pdr_after=sum(s['rx'] for s in window) / max(1, sum(s['offered'] for s in window)),
                               pdr_hold=sum(s['rx'] for s in hold_window) / max(1, sum(s['offered'] for s in hold_window))))
    curves = {f'{p}/{c}': {str(k): dict(n=len(v), mean=float(np.mean(v))) for k, v in sorted(offsets.items()) if v}
              for (p, c), offsets in aligned.items()}
    report = dict(schema='motion-harm-diagnosis-v1', source=str(args.validation), cells=len(rows),
                  loss_by_route=routes, route_age=route_age, link_quality_bins=link_bins,
                  route_changes_per_gradual_episode=changes, burst_aligned_pdr_minus_hold=curves,
                  route_mix_30s_after_burst={f'{p}/{c}': dict(v) for (p, c), v in route_after.items()}, bursts=bursts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    for policy, n in changes.items(): print('route changes/episode', policy, round(n, 1))
    for name, table in routes.items():
        if name.startswith('gradual/'):
            print(name, {k: (v['seconds'], round(v['share_of_loss'], 3), v['pdr'] and round(v['pdr'], 3), v['median_worst_link']) for k, v in table.items()})


if __name__ == '__main__':
    main()

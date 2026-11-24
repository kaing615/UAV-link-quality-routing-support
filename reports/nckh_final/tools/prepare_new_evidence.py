"""Audit and plot new locked studies without changing their analysis outputs."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(version):
    base = ROOT / f'reports/uav_service_{version}/results'
    manifest = read(base / 'export-manifest.json')['files']
    for name, digest in manifest.items():
        assert sha(base / name) == digest, name
    phases = [phase for phase in ('validation', 'test') if (base / f'analysis/{phase}/analysis.json').exists()]
    assert phases and phases[0] == 'validation'
    validations = {}
    analyses = {}
    for phase in phases:
        validation = read(base / f'runs/{phase}-validation.json')
        assert validation['complete'] and validation['planned_cells'] == validation['ok_cells']
        assert all(row['status'] == 'ok' for row in validation['rows'])
        validations[phase] = validation
        analysis = read(base / f'analysis/{phase}/analysis.json')
        assert sha(base / f'analysis/{phase}/analysis.json') == read(base / f'analysis/{phase}/manifest.json')['files']['analysis.json']
        analyses[phase] = analysis
    go = analyses['validation']['go']['go']
    assert ('test' in phases) == go
    phase = 'test' if go else 'validation'
    analysis = analyses[phase]
    groups = defaultdict(dict)
    for row in analysis['rows']:
        key = (row['context'], row['regime'], row['layout_seed'], row['radio_seed'])
        assert row['policy'] not in groups[key]
        groups[key][row['policy']] = row
    arms = sorted(next(iter(groups.values())))
    assert all(sorted(group) == arms for group in groups.values())
    paired = []
    for comparison in analysis.get('fixed_sequence', []):
        c = comparison['contrast']
        values = []
        for key, group in sorted(groups.items()):
            if key[0] != 'gradual':
                continue
            diff = group[c['arm']][c['metric']] - group[c['comparator']][c['metric']]
            values.append(diff)
            paired.append(dict(comparison=comparison['name'], regime=key[1], layout_seed=key[2], radio_seed=key[3], difference=diff))
        assert len(values) == 40
        assert abs(float(np.mean(values)) - c['point']) < 1e-10
    means = {}
    spread = {}
    metrics = ['actual_udp_pdr', 'background_pdr', 'disconnected_seconds', 'no_olsr_route_seconds', 'total_relay_travel_m']
    if version == 'v9':
        metrics += ['longest_outage_seconds', 'motion_starts', 'safety_blocks']
    for context in ('static', 'gradual'):
        means[context] = {arm: {metric: float(np.mean([group[arm][metric] for key, group in groups.items() if key[0] == context]))
                              for metric in metrics} for arm in arms}
        spread[context] = {arm: {metric: float(np.std([group[arm][metric] for key, group in groups.items() if key[0] == context], ddof=1))
                               for metric in metrics} for arm in arms}
    supplement = read(base / 'descriptive-supplement.json')
    result = dict(version=version, phase=phase, validation_go=analyses['validation']['go'],
        counts={p: v['ok_cells'] for p, v in validations.items()},
        analysis_sha256=sha(base / f'analysis/{phase}/analysis.json'),
        fixed_sequence=analysis.get('fixed_sequence'), means=means, episode_sd=spread,
        technical=analysis.get('technical', {k: analysis.get(k) for k in (
            'route_integrity_ok', 'flags_ok', 'transmissions_ok', 'background_ok', 'forecast_traces_ok', 'relays_stationary', 'startup_blackouts')}),
        route_integrity_count=len(analysis['route_integrity']),
        route_changes={arm:dict(episodes=len([r for r in analysis['rows'] if r['policy']==arm and r['cell_id'] in analysis['route_integrity']]),
            commanded_changes=sum(analysis['route_integrity'][r['cell_id']]['commanded_changes'] for r in analysis['rows']
                                  if r['policy']==arm and r['cell_id'] in analysis['route_integrity']),
            override_seconds=sum(analysis['route_integrity'][r['cell_id']]['override_seconds'] for r in analysis['rows']
                                  if r['policy']==arm and r['cell_id'] in analysis['route_integrity'])) for arm in arms},
        forecast_evaluation=analysis.get('forecast_evaluation'),
        timings=supplement['phases'][phase]['decision_timing_ms'],
        timing_note=supplement['phases'][phase]['timing_note'],
        hashes={p: sha(base / p) for p in manifest if p.endswith(('analysis.json', 'go.json', 'freeze.json'))},
        queue_completions={p:read(base / f'evidence/queue-{p}/queue-finished.json') for p in phases})
    if version == 'v9':
        motions=supplement['phases'][phase]['motion_episodes']
        assert len(motions)==len(analysis['rows'])
        result['motion_supplement']={}
        for arm in arms:
            subset=[row for row in motions if row['policy']==arm and row['context']=='gradual']
            both=[row for row in subset if row['first_motion_step'] is not None and row['first_outage_step'] is not None]
            result['motion_supplement'][arm]=dict(
                episodes=len(subset), moving_episodes=sum(row['first_motion_step'] is not None for row in subset),
                motion_before_outage=sum(row['first_motion_step']<row['first_outage_step'] for row in both),
                both_motion_and_outage=len(both), no_outage_episodes=sum(row['first_outage_step'] is None for row in subset),
                safety_blocked_decisions=sum(row['safety_blocked_decisions'] for row in subset),
                simulator_rejections=sum(row['simulator_safety_rejections'] for row in subset),
                max_relay_travel_m=max(max(row['relay_travel_m'].values()) for row in subset),
                episodes_with_exhausted_relay=sum(any(row['budget_exhausted_frames'].values()) for row in subset))
    evidence = REPORT / 'evidence'
    evidence.mkdir(exist_ok=True)
    (evidence / f'{version}-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if paired:
        with (evidence / f'{version}-paired.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True})
    if paired:
        primary = analysis['fixed_sequence'][0]
        values = [row for row in paired if row['comparison'] == primary['name']]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), sharey=True)
        for ax, regime in zip(axes, ('n5', 'n7')):
            subset = [row for row in values if row['regime'] == regime]
            scale = 100 if version == 'v8' else 1
            ax.scatter(range(1, 21), [row['difference'] * scale for row in subset], color='#237b79', s=25)
            ax.axhline(0, color='#222222', linewidth=.8)
            ax.set_title(f'{regime}: 20 khối gradual')
            ax.set_xlabel('Kịch bản theo layout seed')
            ax.set_xticks([1,5,10,15,20])
        axes[0].set_ylabel('PDR GNN − hiện tại (điểm %)' if version == 'v8' else 'Mất kết nối: reactive − GNN (giây)')
        fig.tight_layout()
        for suffix in ('png','pdf'):
            fig.savefig(REPORT / f'images/fig_{version}_paired.{suffix}', dpi=220, bbox_inches='tight')
        plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k in ('version','phase','counts','means','technical','timings')}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    import sys
    for version in sys.argv[1:] or ['v8','v9']:
        audit(version)

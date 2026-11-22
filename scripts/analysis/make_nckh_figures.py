"""Figures for the NCKH final report, drawn only from committed result files.

python scripts/analysis/make_nckh_figures.py  ->  reports/nckh_final/images/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/nckh_final/images'
V4 = ROOT / 'reports/uav_service_v4/results'
V5, V6 = (ROOT / f'reports/uav_service_v{n}/results' for n in (5, 6))
# One fixed colour per arm in every figure (validated categorical slots 1-4).
ARM = {'hold': ('HOLD (OLSR)', '#2a78d6'), 'service-v2-no-forecast': ('Di chuyển phục hồi', '#eb6834'),
       'lq-route': ('lq-route (ghim route)', '#1baf7a'), 'lq-route-restore': ('lq-route + phục hồi', '#eda100'),
       'olsr-etx': ('OLSR-ETX (phân tán)', '#8a5cd0')}
INK, MUTED, GRID = '#1f1f1e', '#5f5e5a', '#e6e5e0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True, 'savefig.dpi': 200})


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, bbox_inches='tight'); plt.close(fig)


def loss_by_route():
    table = json.loads((ROOT / 'reports/uav_service_v3/diagnosis/v2-formal.json').read_text())['loss_by_route']['gradual/hold']
    names = {'direct': 'Đi thẳng nguồn → đích', 'relay': 'Qua relay', 'none': 'Không có route'}
    kinds = ['direct', 'relay', 'none']
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    shares = [table[k]['share_of_loss'] * 100 for k in kinds]
    ax.barh([names[k] for k in kinds][::-1], shares[::-1], color='#2a78d6', height=.55)
    for y, k in enumerate(kinds[::-1]):
        t = table[k]
        ax.text(t['share_of_loss'] * 100 + 1, y, f"{t['share_of_loss']*100:.0f}% gói mất · {t['seconds']/sum(table[x]['seconds'] for x in kinds)*100:.0f}% thời gian · PDR {t['pdr']:.3f}",
                va='center', color=INK, fontsize=9)
    ax.set_xlim(0, 100); ax.set_xlabel('Phần trăm số gói bị mất (HOLD, gradual, formal v2)'); ax.grid(axis='y', visible=False)
    save(fig, 'fig_loss_by_route.png')


def tuning():
    s = json.loads((V4 / 'tuning-selection.json').read_text())
    integrity = s['route_integrity']
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    margins = [t['margin'] for t in s['margins']]
    for t in s['margins']:
        lo, hi = (v * 100 for v in t['ci95'])
        a.errorbar(t['margin'], t['point'] * 100, yerr=[[t['point'] * 100 - lo], [hi - t['point'] * 100]], fmt='o',
                   color='#1baf7a', ms=8, capsize=4, lw=2)
        a.annotate(f"+{t['point']*100:.2f}", (t['margin'], t['point'] * 100), xytext=(8, 0), textcoords='offset points',
                   va='center', color=INK, fontsize=9)
    a.axhline(0, color=MUTED, lw=1); a.set_xticks(margins); a.set_xlabel('Margin m (ETX)')
    a.set_ylabel('Chênh PDR so với HOLD (điểm %)'); a.set_title('PDR (gradual, CI 95%)', fontsize=10, color=INK)
    changes = [sum(v['commanded_changes'] for k, v in integrity.items() if k.endswith('__' + t['arm'])) for t in s['margins']]
    b.bar([str(m) for m in margins], changes, color='#1baf7a', width=.5)
    for x, c in enumerate(changes): b.text(x, c + 8, str(c), ha='center', color=INK, fontsize=9)
    b.set_xlabel('Margin m (ETX)'); b.set_ylabel('Số lần đổi route (32 episode)'); b.set_title('Đổi route', fontsize=10, color=INK)
    b.grid(axis='x', visible=False)
    save(fig, 'fig_v4_tuning.png')


def arms_and_pairs():
    rows = [r for r in json.loads((V4 / 'analysis/test/analysis.json').read_text())['rows'] if r['context'] == 'gradual']
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    width = .2
    for i, arm in enumerate(('hold', 'service-v2-no-forecast', 'lq-route', 'lq-route-restore')):
        label, color = ARM[arm]
        means = [sum(r['actual_udp_pdr'] for r in rows if r['regime'] == g and r['policy'] == arm) / 20 for g in ('n4', 'n5')]
        xs = [g + (i - 1.5) * (width + .02) for g in range(2)]
        ax.bar(xs, means, width, color=color, label=label)
        for x, m in zip(xs, means): ax.text(x, m + .004, f'{m:.3f}', ha='center', fontsize=7.5, color=INK, rotation=90)
    ax.set_xticks(range(2), ['n4 (2 relay)', 'n5 (3 relay)']); ax.set_ylim(.8, 1.02); ax.set_ylabel('PDR trung bình (20 kịch bản)')
    ax.legend(ncol=4, fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.14)); ax.grid(axis='x', visible=False)
    save(fig, 'fig_v4_arms.png')

    lookup = {(r['regime'], r['layout_seed'], r['policy']): r['actual_udp_pdr'] for r in rows}
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for y, (arm, base) in enumerate((('lq-route', 'hold'), ('lq-route-restore', 'service-v2-no-forecast'))):
        for g, offset in (('n4', -.12), ('n5', .12)):
            d = [(lookup[(g, l, arm)] - lookup[(g, l, base)]) * 100 for (gg, l, p) in lookup if gg == g and p == base]
            ax.scatter(d, [y + offset] * len(d), s=36, color=ARM[arm][1], edgecolor='white', linewidth=1, zorder=3,
                       marker='o' if g == 'n4' else 's')
            ax.text(51, y + offset, f'{g}: tăng {sum(v > 0 for v in d)}/{len(d)}', va='center', fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks([0, 1], ['lq-route − HOLD', 'lq-route+phục hồi −\nDi chuyển phục hồi']); ax.set_ylim(-.5, 1.5); ax.set_xlim(-45, 50)
    ax.set_xlabel('Chênh PDR theo cặp kịch bản (điểm %); tròn = n4, vuông = n5'); ax.grid(axis='y', visible=False)
    save(fig, 'fig_v4_paired.png')


def outlier():
    data = json.loads((V4 / 'outlier-test-n5-gradual-18.json').read_text())['arms']
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for arm in ('hold', 'lq-route', 'lq-route-restore'):
        sec = data[arm]['seconds']
        xs, ys = [], []
        for b in range(0, 240, 10):
            chunk = sec[b:b + 10]
            xs.append(b + 5); ys.append(sum(s['rx'] for s in chunk) / max(1, sum(s['offered'] for s in chunk)))
        ax.plot(xs, ys, color=ARM[arm][1], lw=2, marker='o', ms=4, label=ARM[arm][0])
    commands = data['lq-route']['route_commands']; start = None
    for c in commands + [dict(step=240, route=[])]:
        if c['route'] and start is None: start = c['step']
        elif not c['route'] and start is not None:
            ax.axvspan(start, c['step'], color='#1baf7a', alpha=.08, lw=0); start = None
    ax.set_xlabel('Thời gian (giây)'); ax.set_ylabel('PDR theo cửa sổ 10 giây'); ax.set_ylim(-.03, 1.05); ax.set_xlim(0, 240)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.text(160, .45, 'vùng nền xanh: lq-route\nđang ghim route', fontsize=8, color=MUTED)
    save(fig, 'fig_v4_outlier.png')


def v5_v6_paired():
    """lq-route - olsr-etx per gradual scenario, v5 (ARP default) vs v6 (ARP DeadTimeout 1 s)."""
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for y, (name, root) in enumerate((('v6 (đã sửa ARP)', V6), ('v5 (ARP mặc định)', V5))):
        rows = [r for r in json.loads((root / 'analysis/test/analysis.json').read_text())['rows'] if r['context'] == 'gradual']
        pdr = {(r['regime'], r['layout_seed'], r['policy']): r['actual_udp_pdr'] for r in rows}
        for g, offset in (('n4', -.12), ('n5', .12)):
            d = [(pdr[(g, l, 'lq-route')] - pdr[(g, l, 'olsr-etx')]) * 100 for (gg, l, p) in pdr if gg == g and p == 'hold']
            ax.scatter(d, [y + offset] * len(d), s=34, color=ARM['lq-route'][1], edgecolor='white', linewidth=1, zorder=3,
                       marker='o' if g == 'n4' else 's')
            ax.text(51, y + offset, f'{g}: lq-route hơn {sum(v > 0 for v in d)}/{len(d)}', va='center', fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=1); ax.set_yticks([0, 1], ['v6 (đã sửa ARP)', 'v5 (ARP mặc định)'])
    ax.set_ylim(-.5, 1.5); ax.set_xlim(-60, 50); ax.grid(axis='y', visible=False)
    ax.set_xlabel('PDR lq-route − OLSR-ETX theo kịch bản (điểm %); tròn = n4, vuông = n5')
    save(fig, 'fig_v56_paired.png')


def v6_loss():
    mix = json.loads((V6 / 'analysis/test/analysis.json').read_text())['route_mix']
    kinds = (('direct', 'Đi thẳng', '#d95f45'), ('relay', 'Qua relay', '#eda100'), ('none', 'Không có route', '#8c8a83'))
    arms = ['hold', 'olsr-etx', 'lq-route']
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    left = [0] * len(arms)
    for kind, label, color in kinds:
        vals = [mix[a]['lost_packets'].get(kind, 0) / 1000 for a in arms]
        ax.barh([ARM[a][0] for a in arms][::-1], vals[::-1], left=left[::-1], color=color, height=.55, label=label)
        left = [l + v for l, v in zip(left, vals)]
    for y, total in enumerate(left[::-1]): ax.text(total + .5, y, f'{total:.1f}'.replace('.', ',') + ' nghìn', va='center', fontsize=8.5, color=INK)
    ax.set_xlabel('Số gói bị mất (nghìn), test v6, 40 kịch bản gradual')
    ax.set_xlim(0, max(left) * 1.18); ax.grid(axis='y', visible=False)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.3))
    save(fig, 'fig_v6_loss.png')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    loss_by_route(); tuning(); arms_and_pairs(); outlier(); v5_v6_paired(); v6_loss()
    print(sorted(p.name for p in OUT.glob('*.png')))

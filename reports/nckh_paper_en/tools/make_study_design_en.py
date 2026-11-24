"""English version of the v8/v9 design figure for the international manuscript.

Same geometry and wording as reports/nckh_final/tools/make_study_design.py, only
the labels are translated. Depicts no unmeasured effect.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'images').mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, ax = plt.subplots(figsize=(10.4, 5.0))
ax.set_xlim(0, 10.4)
ax.set_ylim(0, 5)
ax.axis('off')


def box(x, y, w, h, text, color='#eef2f4', edge='#52606a', size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.06',
                                linewidth=.8, facecolor=color, edgecolor=edge))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, linespacing=1.5)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12,
                                 linewidth=1, color='#52606a'))


ax.text(.05, 4.75, 'V8  •  Value of forecasting for route selection',
        fontsize=12, fontweight='bold', color='#195d60')
box(.05, 3.6, 2.05, .8, 'Current and past\nobservations')
box(2.65, 3.36, 2.55, 1.3, 'Current EWMA\nGNN 5 s forecast\nMLP 5 s forecast', color='#e8f3ef')
box(5.75, 3.6, 2.0, .8, 'Same ETX\nroute-pinning rule')
box(8.3, 3.6, 2.05, .8, 'Closed-loop PDR\nRelays held still')
arrow(2.14, 4, 2.6, 4)
arrow(5.24, 4, 5.7, 4)
arrow(7.79, 4, 8.25, 4)
ax.text(5.2, 3.0, 'System reference: OLSR-ETX-fast; not used to replace the primary endpoint.',
        ha='center', fontsize=9, color='#404b52')

ax.text(.05, 2.45, 'V9  •  Value of forecasting for motion triggering',
        fontsize=12, fontweight='bold', color='#195d60')
box(.05, 1.13, 2.05, .8, 'Same routing on\ncurrent measurements')
box(2.65, .89, 2.55, 1.3, 'Reactive trigger\nGNN trigger\nMLP trigger', color='#e8f3ef')
box(5.75, 1.13, 2.0, .8, 'Same relay\nplacement rule')
box(8.3, 1.13, 2.05, .8, 'Disconnected seconds\nPDR and travelled distance', size=9)
arrow(2.14, 1.53, 2.6, 1.53)
arrow(5.24, 1.53, 5.7, 1.53)
arrow(7.79, 1.53, 8.25, 1.53)
ax.text(5.2, .5, 'Additional reference: relays held still. Forecaster weights carried over from v8.',
        ha='center', fontsize=9, color='#404b52')
ax.text(5.2, .08, 'GNN/MLP trained by supervised learning; action rules fixed. All control runs inside ns-3.',
        ha='center', fontsize=9, color='#404b52')
fig.tight_layout(pad=.5)
for extension in ('pdf', 'png'):
    fig.savefig(ROOT / f'images/fig_v8_v9_design.{extension}', dpi=220, bbox_inches='tight')
print('Wrote English v8/v9 design figures')

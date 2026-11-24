"""English version of the v9 motion example figure for the international manuscript.

Reads the same audited selection and traces as
reports/nckh_final/tools/plot_motion_example.py; only the labels are translated.
The example episode is the preselected one, not chosen for a favourable result.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
FINAL = ROOT / 'reports/nckh_final'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    (OUT / 'images').mkdir(exist_ok=True)
    base = ROOT / 'reports/uav_service_v9/results'
    phase = read(FINAL / 'evidence/v9-audit.json')['phase']
    supplement = read(base / 'descriptive-supplement.json')['phases'][phase]
    arms = ['route-only', 'reactive-motion', 'gnn-motion', 'mlp-motion']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.0), sharex=True, sharey=True)
    metadata = {}
    for ax, arm in zip(axes.ravel(), arms):
        trace = base / supplement['example_traces'][arm]
        frames = [json.loads(line)['frame'] for line in trace.read_text().splitlines()]
        decisions = [json.loads(line) for line
                     in (trace.parent.parent / 'decisions.jsonl').read_text().splitlines()]
        positions = np.asarray([frame['positions'] for frame in frames])
        steps = [frame['step'] for frame in frames]
        for node, color in zip(range(5), ['#343434', '#216e98', '#2d947b', '#d08a29', '#343434']):
            ax.plot(steps, positions[:, node, 1],
                    label='Source/destination' if node == 0 else f'Relay {node}' if node != 4 else None,
                    color=color, linewidth=1.4, linestyle='--' if node in (0, 4) else '-')
        for frame in frames[1:]:
            if not frame['source_dest_connected']:
                ax.axvspan(frame['step'] - 1, frame['step'], color='#a92734', alpha=.13, linewidth=0)
        first = next((d['step'] for d in decisions if d.get('commanded_motion')), None)
        if first is not None:
            ax.axvline(first, color='#6f52a1', linewidth=1, linestyle=':')
        metadata[arm] = dict(first_motion_step=first,
                             disconnected_seconds=sum(not f['source_dest_connected'] for f in frames[1:]),
                             trace=str(trace.relative_to(ROOT)))
        ax.set_title(arm)
        ax.grid(alpha=.15)
    for ax in axes[-1]:
        ax.set_xlabel('Control time (s)')
    for ax in axes[:, 0]:
        ax.set_ylabel('y coordinate (m; endpoint separation axis)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.03), frameon=False)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / f'images/fig_v9_motion_example.{extension}', dpi=220, bbox_inches='tight')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()

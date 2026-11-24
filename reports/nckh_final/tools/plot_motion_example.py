"""Plot the preselected first n5 gradual example, including all four arms."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    base = ROOT / 'reports/uav_service_v9/results'
    phase = read(REPORT / 'evidence/v9-audit.json')['phase']
    supplement = read(base / 'descriptive-supplement.json')['phases'][phase]
    arms = ['route-only','reactive-motion','gnn-motion','mlp-motion']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(10.4,6.0),sharex=True,sharey=True)
    metadata = {}
    for ax, arm in zip(axes.ravel(),arms):
        trace = base / supplement['example_traces'][arm]
        frames = [json.loads(line)['frame'] for line in trace.read_text().splitlines()]
        decisions = [json.loads(line) for line in (trace.parent.parent / 'decisions.jsonl').read_text().splitlines()]
        config = read(trace.parent / 'config.json')
        positions = np.asarray([frame['positions'] for frame in frames])
        steps = [frame['step'] for frame in frames]
        for node, color in zip(range(5),['#343434','#216e98','#2d947b','#d08a29','#343434']):
            ax.plot(steps,positions[:,node,1],label='Nguồn/đích' if node==0 else f'Relay {node}' if node!=4 else None,
                    color=color,linewidth=1.4,linestyle='--' if node in (0,4) else '-')
        for frame in frames[1:]:
            if not frame['source_dest_connected']:
                ax.axvspan(frame['step']-1,frame['step'],color='#a92734',alpha=.13,linewidth=0)
        first = next((d['step'] for d in decisions if d.get('commanded_motion')),None)
        if first is not None:
            ax.axvline(first,color='#6f52a1',linewidth=1,linestyle=':')
        metadata[arm] = dict(first_motion_step=first, disconnected_seconds=sum(not f['source_dest_connected'] for f in frames[1:]),
                             trace=str(trace.relative_to(ROOT)))
        ax.set_title(arm)
        ax.grid(alpha=.15)
    for ax in axes[-1]:
        ax.set_xlabel('Thời gian điều khiển (s)')
    for ax in axes[:,0]:
        ax.set_ylabel('Tọa độ y (m; trục đầu mút tách xa)')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.03),frameon=False)
    fig.tight_layout()
    for extension in ('png','pdf'):
        fig.savefig(REPORT / f'images/fig_v9_motion_example.{extension}',dpi=220,bbox_inches='tight')
    (REPORT / 'evidence/v9-motion-example.json').write_text(json.dumps(dict(phase=phase,
        selection=supplement['example_selection'],arms=metadata),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

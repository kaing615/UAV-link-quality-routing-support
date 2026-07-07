from __future__ import annotations
import argparse
import fnmatch
import json
from pathlib import Path
import pandas as pd
import torch
from src.models.gnn.edge_gnn import EdgeAwareSAGEEdgeClassifier, GATEdgeClassifier, GraphSAGEEdgeClassifier
from src.training.gnn.common import evaluate_split, make_loader
_MODELS = {'graphsage': GraphSAGEEdgeClassifier, 'gat': GATEdgeClassifier, 'edge-sage': EdgeAwareSAGEEdgeClassifier}
NODE_IN = 8
EDGE_IN = 7

def refresh_run(model_dir: Path, run_dir: Path, device: torch.device) -> None:
    metadata = json.loads((run_dir / 'metadata.json').read_text(encoding='utf-8'))
    model_id = metadata['model_id']
    base_id = model_id.replace('-noedge', '')
    model = _MODELS[base_id](node_in_channels=NODE_IN, edge_in_channels=EDGE_IN, hidden_channels=metadata['hidden_channels'], num_layers=metadata['num_layers'], dropout=metadata['dropout'], use_edge_features=metadata['use_edge_features']).to(device)
    model.load_state_dict(torch.load(run_dir / 'best_model.pt', weights_only=True))
    graph_root = Path('data/graph_dataset') / run_dir.name / 'graph_dataset'
    threshold = float(metadata.get('threshold', 0.5))
    rows = []
    for split in ['val', 'test']:
        loader = make_loader(graph_root / f'{split}.pt', batch_size=16, shuffle=False)
        metrics, _ = evaluate_split(model, loader, device, model_id, metadata['model_name'], split, threshold=threshold)
        rows.append(metrics)
    pd.DataFrame(rows).to_csv(run_dir / 'metrics.csv', index=False)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Refresh GNN metrics.csv with AUC + inference time.')
    p.add_argument('--gnn-root', type=Path, default=Path('outputs/gnn'))
    p.add_argument('--model-pattern', type=str, default='*')
    p.add_argument('--run-pattern', type=str, default='*')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    refreshed = failures = 0
    for model_dir in sorted((p for p in args.gnn_root.iterdir() if p.is_dir())):
        if not fnmatch.fnmatch(model_dir.name, args.model_pattern):
            continue
        for run_dir in sorted((p for p in model_dir.iterdir() if p.is_dir())):
            if not fnmatch.fnmatch(run_dir.name, args.run_pattern):
                continue
            if not (run_dir / 'best_model.pt').exists():
                continue
            try:
                refresh_run(model_dir, run_dir, device)
                refreshed += 1
            except Exception as exc:
                print(f'[FAIL] {model_dir.name}/{run_dir.name}: {exc}')
                failures += 1
    print(f'[OK] refreshed={refreshed} failures={failures}')
    if failures:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
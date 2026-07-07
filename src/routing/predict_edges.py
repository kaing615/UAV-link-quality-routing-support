from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from src.models.gnn.edge_gnn import EdgeAwareSAGEEdgeClassifier, GATEdgeClassifier, GraphSAGEEdgeClassifier
_MODELS = {'graphsage': GraphSAGEEdgeClassifier, 'gat': GATEdgeClassifier, 'edge-sage': EdgeAwareSAGEEdgeClassifier}
NODE_IN = 8
EDGE_IN = 7

def predict_edges(run_name: str, model_id: str='edge-sage', output_csv: Path | None=None) -> Path:
    model_dir = Path('outputs/gnn') / model_id / run_name
    metadata = json.loads((model_dir / 'metadata.json').read_text(encoding='utf-8'))
    base_id = model_id.replace('-noedge', '')
    model_cls = _MODELS[base_id]
    model = model_cls(node_in_channels=NODE_IN, edge_in_channels=EDGE_IN, hidden_channels=metadata['hidden_channels'], num_layers=metadata['num_layers'], dropout=metadata['dropout'], use_edge_features=metadata['use_edge_features'])
    model.load_state_dict(torch.load(model_dir / 'best_model.pt', weights_only=True))
    model.eval()
    test_pt = Path('data/graph_dataset') / run_name / 'graph_dataset' / 'test.pt'
    graphs = torch.load(test_pt, weights_only=False)
    rows = []
    with torch.no_grad():
        for g in graphs:
            logits = model(g['x'], g['edge_index'], g['edge_attr'], g['edge_label_index'], g['edge_attr'][::2])
            scores = torch.sigmoid(logits).numpy()
            node_ids = g['node_ids']
            eli = g['edge_label_index'].numpy()
            for k in range(eli.shape[1]):
                rows.append({'time': int(g['time']), 'src': int(node_ids[eli[0, k]]), 'dst': int(node_ids[eli[1, k]]), 'y_true': int(g['edge_label'][k]), 'pred_score': float(scores[k])})
    if output_csv is None:
        output_csv = Path('outputs/routing') / run_name / f'edge_predictions_{model_id}.csv'
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Export per-edge GNN stability scores for routing.')
    p.add_argument('--run-name', type=str, required=True)
    p.add_argument('--model', type=str, default='edge-sage')
    p.add_argument('--output', type=Path, default=None)
    return p.parse_args()
if __name__ == '__main__':
    args = parse_args()
    out = predict_edges(args.run_name, model_id=args.model, output_csv=args.output)
    print(f'[OK] wrote {out}')
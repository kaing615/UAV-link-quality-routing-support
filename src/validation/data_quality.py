from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

NODE_FEATURE_RANGES = {0: ('x', -5000, 5000), 1: ('y', -5000, 5000), 2: ('z', 0, 1000), 3: ('vx', -100, 100), 4: ('vy', -100, 100), 5: ('vz', -50, 50), 6: ('speed', 0, 200), 7: ('degree', 0, 50)}
EDGE_FEATURE_RANGES = {0: ('distance', 0, 5000), 1: ('rssi', -120, 0), 2: ('snr', -10, 60), 3: ('delay', 0, 1000), 4: ('packet_loss', 0, 1), 5: ('relative_speed', 0, 300), 6: ('throughput', 0, 100)}

def check_tensor_quality(tensor: torch.Tensor, name: str, ranges: dict) -> list[dict]:
    issues = []
    if torch.isnan(tensor).any():
        nan_count = int(torch.isnan(tensor).sum())
        issues.append({'severity': 'error', 'check': 'nan_detection', 'feature_group': name, 'message': f'{nan_count} NaN values found in {name}'})
    if torch.isinf(tensor).any():
        inf_count = int(torch.isinf(tensor).sum())
        issues.append({'severity': 'error', 'check': 'inf_detection', 'feature_group': name, 'message': f'{inf_count} Inf values found in {name}'})
    for col_idx, (feat_name, vmin, vmax) in ranges.items():
        if col_idx >= tensor.shape[1]:
            continue
        col = tensor[:, col_idx]
        below = int((col < vmin).sum())
        above = int((col > vmax).sum())
        if below > 0 or above > 0:
            issues.append({'severity': 'warning', 'check': 'range_violation', 'feature': feat_name, 'message': f'{feat_name}: {below} below {vmin}, {above} above {vmax}', 'actual_range': [float(col.min()), float(col.max())]})
    return issues

def check_class_balance(labels: torch.Tensor, run_name: str) -> list[dict]:
    issues = []
    n_total = len(labels)
    n_pos = int(labels.sum())
    ratio = n_pos / max(n_total, 1)
    if ratio > 0.95 or ratio < 0.05:
        issues.append({'severity': 'warning', 'check': 'class_imbalance', 'run_name': run_name, 'message': f'Extreme class imbalance: positive_ratio={ratio:.4f} ({n_pos}/{n_total})', 'positive_ratio': ratio})
    if n_total < 10:
        issues.append({'severity': 'warning', 'check': 'insufficient_samples', 'run_name': run_name, 'message': f'Very few labeled edges: {n_total}'})
    return issues

def check_graph_integrity(graphs: list[dict], run_name: str, split: str) -> list[dict]:
    issues = []
    for i, g in enumerate(graphs):
        n_nodes = g['x'].shape[0]
        edge_index = g['edge_index']
        if n_nodes < 2:
            issues.append({'severity': 'error', 'check': 'too_few_nodes', 'run_name': run_name, 'message': f'{split} graph {i}: only {n_nodes} node(s)'})
        if edge_index.max() >= n_nodes:
            issues.append({'severity': 'error', 'check': 'invalid_edge_index', 'run_name': run_name, 'message': f'{split} graph {i}: edge_index max ({edge_index.max()}) >= n_nodes ({n_nodes})'})
        if edge_index.min() < 0:
            issues.append({'severity': 'error', 'check': 'negative_edge_index', 'run_name': run_name, 'message': f'{split} graph {i}: negative edge indices found'})
    return issues

def validate_run(run_dir: Path) -> dict:
    graph_dir = run_dir / 'graph_dataset'
    run_name = run_dir.name
    result = {'run_name': run_name, 'issues': [], 'stats': {}}
    for split in ('train', 'val', 'test'):
        pt_path = graph_dir / f'{split}.pt'
        if not pt_path.exists():
            result['issues'].append({'severity': 'error', 'check': 'missing_split', 'message': f'Missing {split}.pt in {run_name}'})
            continue
        graphs = torch.load(pt_path, weights_only=False)
        result['stats'][f'{split}_graphs'] = len(graphs)
        all_x = torch.cat([g['x'] for g in graphs])
        all_edge_attr = torch.cat([g['edge_attr'] for g in graphs])
        all_labels = torch.cat([g['edge_label'] for g in graphs])
        result['issues'].extend(check_tensor_quality(all_x, f'{split}/node_features', NODE_FEATURE_RANGES))
        result['issues'].extend(check_tensor_quality(all_edge_attr, f'{split}/edge_features', EDGE_FEATURE_RANGES))
        result['issues'].extend(check_class_balance(all_labels, run_name))
        result['issues'].extend(check_graph_integrity(graphs, run_name, split))
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data/graph_dataset'))
    parser.add_argument('--output', type=Path, default=Path('reports/data_quality.json'))
    parser.add_argument('--fail-on-error', action='store_true', default=False)
    args = parser.parse_args()
    run_dirs = sorted(d for d in args.data_dir.iterdir() if d.is_dir() and (d / 'graph_dataset').exists()) if args.data_dir.exists() else []
    report = {'total_runs': len(run_dirs), 'runs': [], 'summary': {}}
    n_errors = 0
    n_warnings = 0
    for run_dir in run_dirs:
        result = validate_run(run_dir)
        report['runs'].append(result)
        for issue in result['issues']:
            if issue['severity'] == 'error':
                n_errors += 1
            else:
                n_warnings += 1
    report['summary'] = {'total_runs_checked': len(run_dirs), 'total_errors': n_errors, 'total_warnings': n_warnings, 'status': 'FAIL' if n_errors > 0 else 'WARN' if n_warnings > 0 else 'PASS'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"[DATA QUALITY] {report['summary']['status']}: {n_errors} errors, {n_warnings} warnings across {len(run_dirs)} runs")
    print(f'               Report: {args.output}')
    if args.fail_on_error and n_errors > 0:
        sys.exit(1)
if __name__ == '__main__':
    main()

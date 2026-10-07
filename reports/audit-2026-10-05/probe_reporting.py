"""Reproduce reporting defects without training or modifying experiment data."""
import ast
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


def source_function(file, name):
    tree = ast.parse(Path(file).read_text(encoding='utf-8'))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {'pd': pd, 'np': np, 'Path': Path, 'json': json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), file, 'exec'), namespace)
    return namespace[name]


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    metadata = source_function('scripts/analysis/analyze_multihorizon_stage6.py', 'load_run_metadata')
    names = []
    for scenario, radio_range in [('baseline', 240), ('fast', 240), ('sparse', 160), ('dense', 300)]:
        name = f'stress_{scenario}_rwp_s60001'
        names.append(name)
        (root / name).mkdir()
        (root / name / 'scenario.json').write_text(json.dumps({
            'num_uavs': 20, 'comm_range': radio_range, 'x_limit': [0, 800],
            'y_limit': [0, 800], 'mobility_model': 'random-waypoint',
        }))
    result = metadata(root, names)
    assert set(result.density_group) == {'sparse'}
    print('DENSITY: all four controlled scenarios, including dense, are classified as sparse')

reference = source_function('scripts/train/run_stage6_benchmark.py', 'with_full_ablation_reference')
common = {'target': 'qos', 'horizon': 1, 'split': 'test'}
ablation = pd.DataFrame([{**common, 'run_name': 'r1', 'model_id': 'edge-sage-noedge'}])
full = pd.DataFrame([{**common, 'run_name': name, 'model_id': 'edge-sage'} for name in ['r1', 'r2']])
result = reference(ablation, full)
assert result.groupby('model_id').run_name.nunique().to_dict() == {'edge-sage': 2, 'edge-sage-noedge': 1}
print('ABLATION: full model includes 2 runs while ablation includes 1 run')

for file in ['src/training/gnn/train_gnn.py', 'src/training/gnn/train_gnn_loro.py']:
    tree = ast.parse(Path(file).read_text(encoding='utf-8'))
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'metadata' for target in node.targets))
    keys = {key.value for key in assignment.value.keys}
    assert 'use_edge_features' not in keys
    assert {'use_message_edge_features', 'use_decoder_edge_features'} <= keys
    print(f'METADATA: {file} omits use_edge_features required by downstream readers')

path = Path('reports/stage6/loro_summary.csv')
if path.exists():
    data = pd.read_csv(path)
    selected = data[(data.run_name == 'ns3big_001_rwp_s17296_n18_c211_t108') &
                    (data.target == 'qos') & (data.split == 'test')]
    print('EXISTING LORO COHORTS:')
    print(selected[['model_id', 'horizon', 'n_samples']].to_string(index=False))

print('All audit reproductions matched the reported behavior.')

"""Reproduce serving contract failures with synthetic inputs, no deployed model."""
import ast
from contextlib import chdir
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from pydantic import ValidationError
from src.serving import app as serving
from src.serving.schemas import PredictionRequest
from src.routing.predict_edges import predict_edges


class Scores:
    def __call__(self, x, edge_index, edge_attr, edge_label_index, labeled_edge_attr):
        return torch.zeros(edge_label_index.shape[1])


payload = {
    'nodes': [{'node_id': node, 'x': 0, 'y': 0, 'z': 50, 'vx': 0, 'vy': 0,
               'vz': 0, 'degree': 1, 'load': 0.1} for node in (1, 2)],
    'edges': [{'src': 1, 'dst': 2, 'distance': 100, 'rssi': -60, 'snr': 25,
               'delay': 1, 'packet_loss': 0.01, 'relative_speed': 0, 'throughput': 1}],
    'query_edges': [(999, 2), (1, 2)],
}
with patch.object(serving, '_model', Scores()), patch.object(serving, '_model_id', 'audit'):
    try:
        serving.predict(PredictionRequest.model_validate(payload))
    except IndexError as error:
        print(f'QUERY: valid schema causes {type(error).__name__}: {error}')
    else:
        raise AssertionError('Expected query index mismatch')

tree = ast.parse(Path('src/dashboard/app.py').read_text(encoding='utf-8'))
sample = next(node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'sample_data' for target in node.targets))
try:
    PredictionRequest.model_validate(ast.literal_eval(sample))
except ValidationError as error:
    assert len(error.errors()) == 3
    assert all(item['loc'][-1] == 'load' for item in error.errors())
    print('DASHBOARD: all 3 sample nodes fail validation because load is missing')
else:
    raise AssertionError('Expected missing load validation failure')

with tempfile.TemporaryDirectory() as directory, chdir(directory):
    output = Path('outputs/gnn/edge-sage/audit')
    output.mkdir(parents=True)
    (output / 'metadata.json').write_text(json.dumps({
        'hidden_channels': 128, 'num_layers': 2, 'dropout': 0.3,
        'edge_mode': 'full', 'use_message_edge_features': True,
        'use_decoder_edge_features': True,
    }))
    try:
        predict_edges('audit')
    except KeyError as error:
        assert error.args == ('use_edge_features',)
        print(f'ARTIFACT: current metadata causes KeyError: {error}')
    else:
        raise AssertionError('Expected metadata contract failure')

print('All serving reproductions matched the reported behavior.')

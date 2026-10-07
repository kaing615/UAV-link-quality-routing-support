"""Run the two existing shell lifecycle test functions on Linux without pytest."""
from pathlib import Path
import runpy
import tempfile

cases = runpy.run_path('tests/test_serving_model_lifecycle.py')
for name, function in cases.items():
    if name.startswith('test_'):
        with tempfile.TemporaryDirectory(prefix='uav-audit-') as directory:
            function(Path(directory))
        print(f'PASS {name}')

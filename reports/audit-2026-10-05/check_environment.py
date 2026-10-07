import importlib.util
import json
import sys

print(sys.version)
print(json.dumps({name: bool(importlib.util.find_spec(name)) for name in (
    'pytest', 'ruff', 'numpy', 'pandas', 'torch', 'torch_geometric', 'sklearn',
    'xgboost', 'fastapi', 'dvc', 'matplotlib', 'psutil', 'tabulate', 'httpx',
)}))

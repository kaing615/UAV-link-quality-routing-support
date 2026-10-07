"""Probe the GDrive authentication import without credentials or network calls."""
from importlib.metadata import version
import json

result = {'versions': {name: version(name) for name in ('dvc', 'dvc-gdrive', 'PyDrive2', 'pyOpenSSL', 'cryptography')}}
try:
    import pydrive2.auth
except Exception as error:
    result['auth_import'] = f'{type(error).__name__}: {error}'
else:
    result['auth_import'] = 'PASS'
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['auth_import'] == 'PASS' else 1)

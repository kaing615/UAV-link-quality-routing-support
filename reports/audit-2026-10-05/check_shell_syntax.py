"""Read-only Bash syntax audit; normalize checkout CRLF in memory."""
import json
from pathlib import Path
import subprocess

results = []
for path in sorted(Path('scripts').rglob('*.sh')):
    result = subprocess.run(['bash', '-n'], input=path.read_text(encoding='utf-8'), text=True,
                            encoding='utf-8', errors='replace', capture_output=True)
    results.append({'file': str(path), 'exit_code': result.returncode,
                    'diagnostic': result.stderr.strip()})
print(json.dumps(results, indent=2))
print(f"Checked {len(results)} scripts; failures={sum(row['exit_code'] != 0 for row in results)}")

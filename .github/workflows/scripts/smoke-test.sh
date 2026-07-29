#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-https://api-staging.example.com}"

curl --fail --silent --show-error --max-time 30 "$BASE_URL/health" | tee /tmp/health.out

python - <<'PY' "$BASE_URL"
import json
import sys
import urllib.request

base_url = sys.argv[1]
payload = {
    "nodes": [
        {"node_id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "speed": 0.0, "degree": 1},
        {"node_id": 2, "x": 1.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "speed": 0.0, "degree": 1},
    ],
    "edges": [{"src": 1, "dst": 2, "distance": 10.0, "rssi": -60.0, "snr": 20.0, "delay": 1.0, "packet_loss": 0.0, "relative_speed": 0.0, "throughput": 100.0}],
    "query_edges": [[1, 2]],
}
req = urllib.request.Request(
    f"{base_url}/predict",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as resp:
    body = json.loads(resp.read().decode())
    if not isinstance(body.get("predictions", []), list):
        raise SystemExit("predict response did not contain predictions")
PY

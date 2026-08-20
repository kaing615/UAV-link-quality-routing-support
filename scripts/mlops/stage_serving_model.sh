#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

POINTER="deploy/serving_model.json"
ARTIFACT="deploy/serving_model_artifact"
DEST="models"

[[ -f "$POINTER" ]] || {
  echo "[ERROR] No promoted serving model. Run scripts/mlops/promote_model.sh first." >&2
  exit 1
}

dvc pull "${ARTIFACT}.dvc"

for file in best_model.pt metadata.json; do
  [[ -f "${ARTIFACT}/${file}" ]] || {
    echo "[ERROR] ${ARTIFACT}/${file} not found after dvc pull." >&2
    exit 1
  }
done

python3 - "$POINTER" "${ARTIFACT}/metadata.json" <<'PY'
import json
import sys

pointer = json.load(open(sys.argv[1], encoding="utf-8"))
metadata = json.load(open(sys.argv[2], encoding="utf-8"))
for key in ("model_id", "run_name"):
    if pointer.get(key) != metadata.get(key):
        raise SystemExit(f"Promoted model mismatch for {key}: {pointer.get(key)!r} != {metadata.get(key)!r}")
print(f"[STAGE] model_id={pointer['model_id']} run={pointer['run_name']}")
PY

mkdir -p "${DEST}"
cp "${ARTIFACT}/best_model.pt" "${DEST}/best_model.pt"
cp "${ARTIFACT}/metadata.json" "${DEST}/metadata.json"

echo "[OK] staged promoted model -> ${DEST}/"
ls -la "${DEST}/"

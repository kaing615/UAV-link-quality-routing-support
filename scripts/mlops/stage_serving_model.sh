#!/usr/bin/env bash
# Stage the promoted serving model into ./models/ for the inference image build.
#
# Reads the model pointer from deploy/serving_model.json (or explicit args),
# pulls the artifact from the DVC remote, and copies best_model.pt + metadata.json
# into ./models/ — the directory baked into the serve image (MODEL_DIR=/app/models).
#
# Usage:
#   ./scripts/mlops/stage_serving_model.sh                          # use deploy/serving_model.json
#   ./scripts/mlops/stage_serving_model.sh edge-sage ns3big_001_... # explicit override
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

POINTER="deploy/serving_model.json"

if [[ $# -ge 2 ]]; then
  MODEL_ID="$1"
  RUN_NAME="$2"
else
  MODEL_ID="$(python3 -c "import json;print(json.load(open('${POINTER}'))['model_id'])")"
  RUN_NAME="$(python3 -c "import json;print(json.load(open('${POINTER}'))['run_name'])")"
fi

SRC="outputs/gnn/${MODEL_ID}/${RUN_NAME}"
DEST="models"

echo "[STAGE] model_id=${MODEL_ID} run=${RUN_NAME}"

# Pull the GNN outputs from the DVC remote (no-op if already cached locally).
dvc pull outputs/gnn 2>/dev/null || dvc pull

if [[ ! -f "${SRC}/best_model.pt" ]]; then
  echo "[ERROR] ${SRC}/best_model.pt not found after dvc pull." >&2
  exit 1
fi

mkdir -p "${DEST}"
cp "${SRC}/best_model.pt" "${DEST}/best_model.pt"
cp "${SRC}/metadata.json" "${DEST}/metadata.json"

echo "[OK] staged ${MODEL_ID}/${RUN_NAME} -> ${DEST}/"
ls -la "${DEST}/"

#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:?Usage: promote_model.sh <model_id> <run_name> [macro_f1]}"
RUN_NAME="${2:?Usage: promote_model.sh <model_id> <run_name> [macro_f1]}"
MACRO_F1="${3:-unknown}"

VERSION="$(date +%Y%m%d_%H%M%S)"
TAG="model-v${VERSION}"
MSG="Promote ${MODEL_ID}/${RUN_NAME} (macro_f1=${MACRO_F1})"

echo "[PROMOTE] ${MSG}"
echo "          tag: ${TAG}"

dvc push

python3 - "${MODEL_ID}" "${RUN_NAME}" <<'PY'
import json, sys

with open("deploy/serving_model.json", "w") as f:
    json.dump({"model_id": sys.argv[1], "run_name": sys.argv[2]}, f, indent=2)
    f.write("\n")
PY
git add deploy/serving_model.json
git commit -m "${MSG}" || echo "[INFO] pointer unchanged, skipping commit"

git tag -a "${TAG}" -m "${MSG}"
git push origin HEAD
git push origin "${TAG}"

echo "[OK] Model promoted as ${TAG}"
echo "     CI will dvc pull + bake the model into the inference image; ArgoCD syncs the new tag."

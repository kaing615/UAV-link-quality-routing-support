#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:?Usage: promote_model.sh <model_dir> [macro_f1]}"
MACRO_F1="${2:-unknown}"
ARTIFACT="deploy/serving_model_artifact"
POINTER="deploy/serving_model.json"

for file in best_model.pt metadata.json; do
  [[ -f "${MODEL_DIR}/${file}" ]] || {
    echo "[ERROR] Missing ${MODEL_DIR}/${file}" >&2
    exit 1
  }
done

MODEL_ID="$(python3 -c "import json;print(json.load(open('${MODEL_DIR}/metadata.json'))['model_id'])")"
RUN_NAME="$(python3 -c "import json;print(json.load(open('${MODEL_DIR}/metadata.json'))['run_name'])")"

VERSION="$(date +%Y%m%d_%H%M%S)"
TAG="model-v${VERSION}"
MSG="Promote ${MODEL_ID}/${RUN_NAME} (macro_f1=${MACRO_F1})"

echo "[PROMOTE] ${MSG}"
echo "          tag: ${TAG}"

mkdir -p "$ARTIFACT"
cp "${MODEL_DIR}/best_model.pt" "${ARTIFACT}/best_model.pt"
cp "${MODEL_DIR}/metadata.json" "${ARTIFACT}/metadata.json"
dvc add "$ARTIFACT"
dvc push "${ARTIFACT}.dvc"

python3 - "$POINTER" "${MODEL_ID}" "${RUN_NAME}" <<'PY'
import json, sys

with open(sys.argv[1], "w") as f:
    json.dump({"model_id": sys.argv[2], "run_name": sys.argv[3]}, f, indent=2)
    f.write("\n")
PY
git add "$POINTER" "${ARTIFACT}.dvc" deploy/.gitignore
git commit -m "${MSG}" || echo "[INFO] pointer unchanged, skipping commit"

git tag -a "${TAG}" -m "${MSG}"
git push origin HEAD
git push origin "${TAG}"

echo "[OK] Model promoted as ${TAG}"
echo "     CI will pull only the promoted artifact and bake it into the inference image."

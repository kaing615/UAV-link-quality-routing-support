#!/usr/bin/env bash
# Usage: ./scripts/mlops/promote_model.sh <model_id> <run_name> [macro_f1]
# Example: ./scripts/mlops/promote_model.sh edge-sage ns3big_042 0.9234
set -euo pipefail

MODEL_ID="${1:?Usage: promote_model.sh <model_id> <run_name> [macro_f1]}"
RUN_NAME="${2:?Usage: promote_model.sh <model_id> <run_name> [macro_f1]}"
MACRO_F1="${3:-unknown}"

VERSION="$(date +%Y%m%d_%H%M%S)"
TAG="model-v${VERSION}"
MSG="Promote ${MODEL_ID}/${RUN_NAME} (macro_f1=${MACRO_F1})"

echo "[PROMOTE] ${MSG}"
echo "          tag: ${TAG}"

# Ensure DVC outputs are pushed
dvc push

# Tag in git
git tag -a "${TAG}" -m "${MSG}"
git push origin "${TAG}"

echo "[OK] Model promoted as ${TAG}"
echo "     ArgoCD will pick up the new tag on next sync."

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATTERN="${1:-*}"
RUN_PATTERN="${2:-batch_*}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="python3"
elif [[ -x "${PROJECT_ROOT}/simulation/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/simulation/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"

echo "[INFO] project_root=${PROJECT_ROOT}"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] model_pattern=${MODEL_PATTERN}"
echo "[INFO] run_pattern=${RUN_PATTERN}"

"${PYTHON_BIN}" -m src.evaluation.aggregate_baseline_metrics \
  --model-pattern "${MODEL_PATTERN}" \
  --run-pattern "${RUN_PATTERN}"

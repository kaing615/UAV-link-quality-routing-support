#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="python3"
AGGREGATE_MODULE="src.evaluation.aggregate_all_metrics"

MODEL_PATTERN="${1:-*}"
RUN_PATTERN="${2:-*}"

echo "[INFO] project_root=${PROJECT_ROOT}"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] model_pattern=${MODEL_PATTERN}"
echo "[INFO] run_pattern=${RUN_PATTERN}"

exec "${PYTHON_BIN}" -m "${AGGREGATE_MODULE}" \
  --model-pattern "${MODEL_PATTERN}" \
  --run-pattern "${RUN_PATTERN}"

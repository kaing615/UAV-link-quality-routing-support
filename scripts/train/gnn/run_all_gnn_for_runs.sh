#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNS_ROOT="${PROJECT_ROOT}/data/graph_dataset"
GNN_MODULE="src.training.gnn.train_gnn"
RUN_PATTERN="${1:-*}"
MODEL_TYPE="${2:-graphsage}"
NOEDGE="${3:-}"

EXTRA_FLAG=""
if [[ "${NOEDGE}" == "noedge" ]]; then
  EXTRA_FLAG="--no-edge-features"
fi

if [[ ! -f "${PROJECT_ROOT}/src/training/gnn/train_gnn.py" ]]; then
  echo "GNN training script not found: ${PROJECT_ROOT}/src/training/gnn/train_gnn.py"
  exit 1
fi

if [[ ! -d "${RUNS_ROOT}" ]]; then
  echo "Runs root not found: ${RUNS_ROOT}"
  exit 1
fi

PYTHON_BIN="python3"

matching_runs=()
while IFS= read -r run_dir; do
  matching_runs+=("${run_dir}")
done < <(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${RUN_PATTERN}" | sort)

if [[ ${
  echo "No run directories matched pattern '${RUN_PATTERN}' under ${RUNS_ROOT}"
  exit 1
fi

echo "[INFO] project_root=${PROJECT_ROOT}"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] runs_root=${RUNS_ROOT}"
echo "[INFO] run_pattern=${RUN_PATTERN}"
echo "[INFO] model_type=${MODEL_TYPE}"
echo "[INFO] matched_runs=${

failures=0

for run_dir in "${matching_runs[@]}"; do
  run_name="$(basename "${run_dir}")"

  echo
  echo "============================================================"
  echo "[GNN - ${MODEL_TYPE}] ${run_name}"
  echo "============================================================"

  if "${PYTHON_BIN}" -m "${GNN_MODULE}" --run-name "${run_name}" --model "${MODEL_TYPE}" ${EXTRA_FLAG}; then
    echo "[OK] GNN (${MODEL_TYPE}) completed for ${run_name}"
  else
    echo "[FAIL] GNN (${MODEL_TYPE}) failed for ${run_name}"
    failures=$((failures + 1))
  fi
done

echo
echo "============================================================"
echo "[SUMMARY]"
echo "============================================================"
echo "- matched_runs : ${
echo "- failures     : ${failures}"

if [[ ${failures} -gt 0 ]]; then
  exit 1
fi

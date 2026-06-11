#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNS_ROOT="${PROJECT_ROOT}/data/graph_dataset"
MLP_MODULE="src.training.baselines.mlp_baseline"
RUN_PATTERN="${1:-*}"

if [[ ! -f "${PROJECT_ROOT}/src/training/baselines/mlp_baseline.py" ]]; then
  echo "MLP baseline script not found: ${PROJECT_ROOT}/src/training/baselines/mlp_baseline.py"
  exit 1
fi

if [[ ! -d "${RUNS_ROOT}" ]]; then
  echo "Runs root not found: ${RUNS_ROOT}"
  exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="python3"
elif [[ -x "${PROJECT_ROOT}/simulation/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/simulation/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

matching_runs=()
while IFS= read -r run_dir; do
  matching_runs+=("${run_dir}")
done < <(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${RUN_PATTERN}" | sort)

if [[ ${#matching_runs[@]} -eq 0 ]]; then
  echo "No run directories matched pattern '${RUN_PATTERN}' under ${RUNS_ROOT}"
  exit 1
fi

echo "[INFO] project_root=${PROJECT_ROOT}"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] runs_root=${RUNS_ROOT}"
echo "[INFO] run_pattern=${RUN_PATTERN}"
echo "[INFO] matched_runs=${#matching_runs[@]}"

failures=0

for run_dir in "${matching_runs[@]}"; do
  run_name="$(basename "${run_dir}")"

  echo
  echo "============================================================"
  echo "[MLP] ${run_name}"
  echo "============================================================"

  if "${PYTHON_BIN}" -m "${MLP_MODULE}" --run-name "${run_name}"; then
    echo "[OK] MLP completed for ${run_name}"
  else
    echo "[FAIL] MLP failed for ${run_name}"
    failures=$((failures + 1))
  fi
done

echo
echo "============================================================"
echo "[SUMMARY]"
echo "============================================================"
echo "- matched_runs : ${#matching_runs[@]}"
echo "- failures     : ${failures}"

if [[ ${failures} -gt 0 ]]; then
  exit 1
fi

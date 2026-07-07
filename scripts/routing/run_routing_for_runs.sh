#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNS_ROOT="${PROJECT_ROOT}/data/graph_dataset"
RUN_PATTERN="${1:-*}"
MODEL_ID="${2:-edge-sage}"
HORIZON="${HORIZON:-5}"
P_TH="${P_TH:-0.0}"

cd "${PROJECT_ROOT}"

matching_runs=()
while IFS= read -r run_dir; do
  matching_runs+=("$(basename "${run_dir}")")
done < <(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${RUN_PATTERN}" | sort)

if [[ ${#matching_runs[@]} -eq 0 ]]; then
  echo "No run directories matched pattern '${RUN_PATTERN}' under ${RUNS_ROOT}"
  exit 1
fi

echo "[INFO] runs=${#matching_runs[@]} model=${MODEL_ID} horizon=${HORIZON} p_th=${P_TH}"

failures=0
for run_name in "${matching_runs[@]}"; do
  echo
  echo "============================================================"
  echo "[ROUTING] ${run_name}"
  echo "============================================================"
  if python3 -m src.routing.predict_edges --run-name "${run_name}" --model "${MODEL_ID}" \
     && python3 -m src.routing.replay_eval --run-name "${run_name}" \
          --gnn-predictions "outputs/routing/${run_name}/edge_predictions_${MODEL_ID}.csv" \
          --horizon "${HORIZON}" --p-th "${P_TH}"; then
    echo "[OK] ${run_name}"
  else
    echo "[FAIL] ${run_name}"
    failures=$((failures + 1))
  fi
done

echo
python3 -m src.routing.aggregate_routing --pattern "${RUN_PATTERN}"

echo
echo "[SUMMARY] runs=${#matching_runs[@]} failures=${failures}"
[[ ${failures} -eq 0 ]] || exit 1

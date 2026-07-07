#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNS_ROOT="${PROJECT_ROOT}/data/graph_dataset"

RUN_PATTERN="${1:-ns3big_*}"
MODEL_ID="${MODEL_ID:-edge-sage}"
P_TH_VALUES="${P_TH_VALUES:-0.1,0.2,0.3,0.4,0.5,0.6,0.7}"
HORIZON="${HORIZON:-5}"
JOBS="${JOBS:-4}"

cd "${PROJECT_ROOT}"

runs=()
while IFS= read -r run_dir; do
  run_name="$(basename "${run_dir}")"
  pred="${PROJECT_ROOT}/outputs/routing/${run_name}/edge_predictions_${MODEL_ID}.csv"
  if [[ -f "${pred}" ]]; then
    runs+=("${run_name}")
  else
    echo "[SKIP] ${run_name} — predictions not found, run run_routing_for_runs.sh first"
  fi
done < <(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${RUN_PATTERN}" | sort)

if [[ ${
  echo "[ERROR] No runs with predictions matched '${RUN_PATTERN}'"
  exit 1
fi

echo "[INFO] runs=${

_eval_run() {
  local run_name="$1"
  local pred="${PROJECT_ROOT}/outputs/routing/${run_name}/edge_predictions_${MODEL_ID}.csv"
  if python3 -m src.routing.replay_eval \
      --run-name "${run_name}" \
      --gnn-predictions "${pred}" \
      --horizon "${HORIZON}" \
      --p-th "${P_TH_VALUES}" 2>&1 | sed "s/^/  [${run_name}] /"; then
    echo "[OK]   ${run_name}"
  else
    echo "[FAIL] ${run_name}"
    return 1
  fi
}
export -f _eval_run
export PROJECT_ROOT MODEL_ID P_TH_VALUES HORIZON

failures=0
printf '%s\n' "${runs[@]}" \
  | xargs -P "${JOBS}" -I{} bash -c '_eval_run "$@"' _ {} \
  || failures=$?

echo
echo "============================================================"
echo "[SWEEP] replay done — generating pth_sweep.csv + chart"
echo "============================================================"

python3 -m src.routing.plot_pth_sweep \
  --routing-root outputs/routing \
  --output-dir outputs/aggregates/routing \
  --title "Trade-off giữa an toàn tuyến và duy trì liên thông theo p_th (Edge-SAGE)"

echo
echo "============================================================"
echo "[SUMMARY] runs=${
echo "  pth_sweep.csv   → outputs/aggregates/routing/pth_sweep.csv"
echo "  pth_tradeoff    → outputs/aggregates/routing/pth_tradeoff.png"
echo "============================================================"

[[ ${failures} -eq 0 ]] || exit 1

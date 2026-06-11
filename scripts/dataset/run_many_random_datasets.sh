#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_one_dataset.sh"

COUNT="${1:-10}"
PREFIX="${2:-batch}"

if ! [[ "${COUNT}" =~ ^[0-9]+$ ]] || [[ "${COUNT}" -le 0 ]]; then
  echo "COUNT must be a positive integer."
  echo "Usage: $0 [COUNT] [PREFIX]"
  echo "Example: $0 10 exp01"
  exit 1
fi

if [[ ! -x "${PIPELINE_SCRIPT}" ]]; then
  echo "Pipeline script not found or not executable: ${PIPELINE_SCRIPT}"
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
declare -a mobility_options=("random-waypoint" "gauss-markov")

echo "[INFO] project_root=${PROJECT_ROOT}"
echo "[INFO] count=${COUNT}"
echo "[INFO] prefix=${PREFIX}"
echo "[INFO] timestamp=${timestamp}"

for ((i = 1; i <= COUNT; i++)); do
  seed=$((10000 + RANDOM + i))
  mobility="${mobility_options[$((RANDOM % ${#mobility_options[@]}))]}"
  num_uavs=$((6 + RANDOM % 5))
  comm_range=$((180 + RANDOM % 101))
  time_steps=$((80 + RANDOM % 71))
  rwp_speed_min=$((2 + RANDOM % 4))
  rwp_speed_max=$((rwp_speed_min + 3 + RANDOM % 5))

  if [[ "${mobility}" == "random-waypoint" ]]; then
    mobility_tag="rwp"
  else
    mobility_tag="gm"
  fi

  run_name="${PREFIX}_${timestamp}_$(printf '%02d' "${i}")_${mobility_tag}_s${seed}_n${num_uavs}_c${comm_range}_t${time_steps}"

  echo
  echo "============================================================"
  echo "[DATASET ${i}/${COUNT}]"
  echo "run_name=${run_name}"
  echo "seed=${seed}"
  echo "mobility=${mobility}"
  echo "num_uavs=${num_uavs}"
  echo "comm_range=${comm_range}"
  echo "time_steps=${time_steps}"
  echo "rwp_speed_range=(${rwp_speed_min}, ${rwp_speed_max})"
  echo "============================================================"

  SIM_NUM_UAVS="${num_uavs}" \
  SIM_COMM_RANGE="${comm_range}" \
  SIM_TIME_STEPS="${time_steps}" \
  SIM_RWP_SPEED_MIN="${rwp_speed_min}" \
  SIM_RWP_SPEED_MAX="${rwp_speed_max}" \
  bash "${PIPELINE_SCRIPT}" "${run_name}" "${seed}" "${mobility}"
done

echo
echo "[OK] Generated ${COUNT} datasets."
echo "- Raw root          : ${PROJECT_ROOT}/data/raw_snapshots"
echo "- Preprocessed root : ${PROJECT_ROOT}/data/graph_dataset"

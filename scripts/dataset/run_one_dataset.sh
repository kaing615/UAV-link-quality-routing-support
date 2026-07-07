#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_NAME [SEED] [MOBILITY_MODEL]"
  echo "Example: $0 seed_42_rwp 42 random-waypoint"
  exit 1
fi

RUN_NAME="$1"
SEED="${2:-42}"
MOBILITY_MODEL="${3:-random-waypoint}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="python3"
elif [[ -x "${PROJECT_ROOT}/simulation/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/simulation/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "[INFO] project_root=${PROJECT_ROOT}"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] run_name=${RUN_NAME}"
echo "[INFO] seed=${SEED}"
echo "[INFO] mobility_model=${MOBILITY_MODEL}"
if [[ -n "${SIM_NUM_UAVS:-}" ]]; then
  echo "[INFO] sim_num_uavs=${SIM_NUM_UAVS}"
fi
if [[ -n "${SIM_COMM_RANGE:-}" ]]; then
  echo "[INFO] sim_comm_range=${SIM_COMM_RANGE}"
fi
if [[ -n "${SIM_TIME_STEPS:-}" ]]; then
  echo "[INFO] sim_time_steps=${SIM_TIME_STEPS}"
fi
if [[ -n "${SIM_RWP_SPEED_MIN:-}" ]] || [[ -n "${SIM_RWP_SPEED_MAX:-}" ]]; then
  echo "[INFO] sim_rwp_speed_min=${SIM_RWP_SPEED_MIN:-unset}"
  echo "[INFO] sim_rwp_speed_max=${SIM_RWP_SPEED_MAX:-unset}"
fi

cd "${PROJECT_ROOT}"

echo
echo "[1/4] Running simulation"
SIM_SEED="${SEED}" \
SIM_RUN_NAME="${RUN_NAME}" \
SIM_MOBILITY_MODEL="${MOBILITY_MODEL}" \
"${PYTHON_BIN}" simulation/main.py

echo
echo "[2/4] Running graph preprocessing"
"${PYTHON_BIN}" -m src.preprocessing.run_preprocessing \
  --nodes "data/raw_snapshots/${RUN_NAME}/nodes.csv" \
  --edges "data/raw_snapshots/${RUN_NAME}/edges.csv" \
  --output-root "data/graph_dataset/${RUN_NAME}"

echo
echo "[3/4] Standardizing non-GNN baseline data"
"${PYTHON_BIN}" src/preprocessing/non-gnn/standardize_baseline_data.py --run-name "${RUN_NAME}"

echo
echo "[4/4] Handling class imbalance"
"${PYTHON_BIN}" src/preprocessing/non-gnn/handle_imbalance.py --run-name "${RUN_NAME}"

echo
echo "[OK] Pipeline completed for ${RUN_NAME}"
echo "- Raw data            : data/raw_snapshots/${RUN_NAME}"
echo "- Preprocessed data   : data/graph_dataset/${RUN_NAME}"
echo "- Standardized data   : data/graph_dataset/${RUN_NAME}/baseline_standardized"
echo "- Imbalance outputs   : data/graph_dataset/${RUN_NAME}/baseline_standardized/imbalance"

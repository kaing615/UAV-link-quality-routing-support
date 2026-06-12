#!/usr/bin/env bash
# Generate one dataset with the ns-3 simulator (real 802.11 + OLSR stack)
# and run the same preprocessing chain as run_one_dataset.sh.
#
# Usage: ./run_one_dataset_ns3.sh RUN_NAME [SEED] [MOBILITY_MODEL]
#   MOBILITY_MODEL: random-waypoint (default) | gauss-markov
#
# Optional env overrides (same names as the Python simulator pipeline):
#   SIM_NUM_UAVS, SIM_COMM_RANGE, SIM_TIME_STEPS,
#   SIM_RWP_SPEED_MIN, SIM_RWP_SPEED_MAX, SIM_X_MAX, SIM_Y_MAX

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NS3_SRC_DIR="${PROJECT_ROOT}/simulation/ns3"
NS3_BIN="${NS3_SRC_DIR}/build/uav-olsr-dataset"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_NAME [SEED] [MOBILITY_MODEL]"
  echo "Example: $0 ns3_seed_42_rwp 42 random-waypoint"
  exit 1
fi

RUN_NAME="$1"
SEED="${2:-42}"
MOBILITY_MODEL="${3:-random-waypoint}"

NUM_UAVS="${SIM_NUM_UAVS:-6}"
COMM_RANGE="${SIM_COMM_RANGE:-243}"
TIME_STEPS="${SIM_TIME_STEPS:-145}"
RWP_SPEED_MIN="${SIM_RWP_SPEED_MIN:-3}"
RWP_SPEED_MAX="${SIM_RWP_SPEED_MAX:-8}"
X_MAX="${SIM_X_MAX:-500}"
Y_MAX="${SIM_Y_MAX:-500}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="python3"
elif [[ -x "${PROJECT_ROOT}/simulation/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/simulation/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "[INFO] project_root=${PROJECT_ROOT}"
echo "[INFO] run_name=${RUN_NAME}"
echo "[INFO] seed=${SEED}"
echo "[INFO] mobility_model=${MOBILITY_MODEL}"
echo "[INFO] num_uavs=${NUM_UAVS} comm_range=${COMM_RANGE} time_steps=${TIME_STEPS} area=${X_MAX}x${Y_MAX}"

# Build the ns-3 scenario binary on first use
if [[ ! -x "${NS3_BIN}" ]]; then
  echo
  echo "[0/4] Building ns-3 scenario (first run)"
  cmake -B "${NS3_SRC_DIR}/build" -S "${NS3_SRC_DIR}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${NS3_SRC_DIR}/build"
fi

cd "${PROJECT_ROOT}"

OUTPUT_DIR="data/raw_snapshots/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo
echo "[1/4] Running ns-3 simulation"
"${NS3_BIN}" \
  --runName="${RUN_NAME}" \
  --numUavs="${NUM_UAVS}" \
  --timeSteps="${TIME_STEPS}" \
  --seed="${SEED}" \
  --mobility="${MOBILITY_MODEL}" \
  --commRange="${COMM_RANGE}" \
  --rwpSpeedMin="${RWP_SPEED_MIN}" \
  --rwpSpeedMax="${RWP_SPEED_MAX}" \
  --xMax="${X_MAX}" \
  --yMax="${Y_MAX}" \
  --outputDir="${OUTPUT_DIR}" \
  --enableAnim=false

echo
echo "[2/4] Running graph preprocessing"
"${PYTHON_BIN}" -m src.preprocessing.run_preprocessing \
  --nodes "${OUTPUT_DIR}/nodes.csv" \
  --edges "${OUTPUT_DIR}/edges.csv" \
  --output-root "data/graph_dataset/${RUN_NAME}"

echo
echo "[3/4] Standardizing non-GNN baseline data"
"${PYTHON_BIN}" src/preprocessing/non-gnn/standardize_baseline_data.py --run-name "${RUN_NAME}"

echo
echo "[4/4] Handling class imbalance"
"${PYTHON_BIN}" src/preprocessing/non-gnn/handle_imbalance.py --run-name "${RUN_NAME}"

echo
echo "[OK] ns-3 pipeline completed for ${RUN_NAME}"
echo "- Raw data            : data/raw_snapshots/${RUN_NAME}"
echo "- Preprocessed data   : data/graph_dataset/${RUN_NAME}"

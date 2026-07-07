#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNS_ROOT="${PROJECT_ROOT}/data/graph_dataset"
PYTHON_BIN="python3"
BALANCED_IDS="${BALANCED_IDS:-007 012 035 046 077 084}"

cd "${PROJECT_ROOT}"

runs=()
for id in ${BALANCED_IDS}; do
  match="$(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "*_${id}_*" -exec basename {} \; | sort | head -1)"
  if [[ -z "${match}" ]]; then
    echo "[FAIL] no run directory matched id '${id}'"
    exit 1
  fi
  runs+=("${match}")
done

echo "[INFO] LORO folds over ${#runs[@]} runs:"
printf '       %s\n' "${runs[@]}"

failures=0

for test_run in "${runs[@]}"; do
  train_runs=""
  for r in "${runs[@]}"; do
    if [[ "${r}" != "${test_run}" ]]; then
      train_runs="${train_runs:+${train_runs},}${r}"
    fi
  done

  echo
  echo "============================================================"
  echo "[FOLD] test_run=${test_run}"
  echo "============================================================"

  for model in graphsage gat; do
    echo "--- GNN ${model} ---"
    "${PYTHON_BIN}" -m src.training.gnn.train_gnn_loro \
      --test-run "${test_run}" --train-runs "${train_runs}" \
      --model "${model}" --hidden 64 --num-layers 2 --dropout 0.3 \
      --epochs 200 --patience 25 || failures=$((failures + 1))
  done

  echo "--- GNN edge-sage ---"
  "${PYTHON_BIN}" -m src.training.gnn.train_gnn_loro \
    --test-run "${test_run}" --train-runs "${train_runs}" \
    --model edge-sage --hidden 128 --num-layers 2 --dropout 0.3 \
    --lr 5e-4 --epochs 300 --patience 30 --lr-scheduler || failures=$((failures + 1))

  for model in xgb mlp logreg rf threshold; do
    echo "--- baseline ${model} ---"
    "${PYTHON_BIN}" -m src.training.baselines.loro_baselines \
      --test-run "${test_run}" --train-runs "${train_runs}" \
      --model "${model}" || failures=$((failures + 1))
  done
done

echo
echo "============================================================"
echo "[SUMMARY] folds=${#runs[@]} failures=${failures}"
echo "============================================================"

if [[ ${failures} -gt 0 ]]; then
  exit 1
fi

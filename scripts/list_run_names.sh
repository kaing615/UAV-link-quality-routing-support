#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNS_ROOT="${PROJECT_ROOT}/data/graph_dataset"
PATTERN="${1:-*}"

if [[ ! -d "${RUNS_ROOT}" ]]; then
  echo "Runs root not found: ${RUNS_ROOT}"
  exit 1
fi

found=0
while IFS= read -r run_dir; do
  basename "${run_dir}"
  found=1
done < <(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${PATTERN}" | sort)

if [[ "${found}" -eq 0 ]]; then
  echo "No RUN_NAME matched pattern '${PATTERN}' in ${RUNS_ROOT}"
  exit 1
fi

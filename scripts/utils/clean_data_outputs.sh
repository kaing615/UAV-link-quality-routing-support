#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

FORCE=0
PATTERN='*'
for arg in "$@"; do
  case "${arg}" in
    --force) FORCE=1 ;;
    *) PATTERN="${arg}" ;;
  esac
done

targets=()
for root in data/raw_snapshots data/graph_dataset outputs/plots; do
  [[ -d "${root}" ]] || continue
  while IFS= read -r d; do targets+=("${d}"); done \
    < <(find "${root}" -mindepth 1 -maxdepth 1 -type d -name "${PATTERN}" | sort)
done
for root in outputs/baselines outputs/gnn outputs/loro; do
  [[ -d "${root}" ]] || continue
  while IFS= read -r d; do targets+=("${d}"); done \
    < <(find "${root}" -mindepth 2 -maxdepth 2 -type d -name "${PATTERN}" | sort)
done
[[ -d outputs/aggregates ]] && targets+=("outputs/aggregates")

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "Nothing matched pattern '${PATTERN}'. Nothing to delete."
  exit 0
fi

total_size="$(du -shc "${targets[@]}" 2>/dev/null | tail -1 | cut -f1)"

echo "Pattern : ${PATTERN}"
echo "Targets : ${#targets[@]} directories (${total_size})"
echo
echo "Per-root counts:"
for root in data/raw_snapshots data/graph_dataset outputs/baselines outputs/gnn outputs/loro outputs/plots; do
  n=0
  for t in "${targets[@]}"; do [[ "${t}" == "${root}"/* ]] && n=$((n + 1)); done
  [[ ${n} -gt 0 ]] && echo "  ${root}: ${n}"
done
for t in "${targets[@]}"; do
  [[ "${t}" == "outputs/aggregates" ]] && echo "  outputs/aggregates: all"
done

if [[ ${FORCE} -ne 1 ]]; then
  echo
  read -r -p "Delete all of the above? Type 'yes' to confirm: " answer
  if [[ "${answer}" != "yes" ]]; then
    echo "Aborted. Nothing deleted."
    exit 1
  fi
fi

for t in "${targets[@]}"; do
  rm -rf "${t}"
done

echo
echo "[OK] Deleted ${#targets[@]} directories (${total_size})."

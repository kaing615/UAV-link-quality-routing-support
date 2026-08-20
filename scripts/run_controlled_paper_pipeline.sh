#!/usr/bin/env bash
set -euo pipefail

MULTIHORIZON_DATA="data/multihorizon_controlled"
MULTIHORIZON_OUTPUT="outputs/multihorizon_controlled"
REPORT_ROOT="reports/controlled_paper"
STAGE6_OUTPUT="outputs/controlled_stage6"
CLOSED_LOOP_OUTPUT="outputs/closed_loop_controlled"
CLOSED_LOOP_RUNS="${CLOSED_LOOP_RUNS:-10}"
WITHIN_RUNS_PER_SCENARIO="${WITHIN_RUNS_PER_SCENARIO:-10}"
LORO_FOLDS_PER_SCENARIO="${LORO_FOLDS_PER_SCENARIO:-5}"
ABLATION_RUNS_PER_SCENARIO="${ABLATION_RUNS_PER_SCENARIO:-10}"
GNN_EPOCHS="${GNN_EPOCHS:-200}"
GNN_PATIENCE="${GNN_PATIENCE:-20}"
NS3_BINARY="simulation/ns3/build/uav-olsr-dataset"

for name in CLOSED_LOOP_RUNS WITHIN_RUNS_PER_SCENARIO LORO_FOLDS_PER_SCENARIO ABLATION_RUNS_PER_SCENARIO; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || {
    echo "$name must be a positive integer" >&2
    exit 2
  }
done

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' \
    '1. Resume/generate 100 paired seeds x 4 controlled scenarios' \
    '2. Validate controlled raw and graph data' \
    '3. Build QoS/survival multi-horizon datasets (k=1,2,3,5)' \
    "4. Train within-run models on ${WITHIN_RUNS_PER_SCENARIO} runs/scenario; retain all 400 datasets" \
    "5. Run LORO on ${LORO_FOLDS_PER_SCENARIO} folds/scenario, ablation on ${ABLATION_RUNS_PER_SCENARIO} runs/scenario, and full cross-mobility" \
    '6. Run multi-horizon routing replay and paired statistics' \
    "7. Run inference resources and closed-loop ns-3 on ${CLOSED_LOOP_RUNS} baseline runs" \
    '8. Generate publication figures and tables'
  exit 0
fi

mkdir -p "$(dirname "$NS3_BINARY")" "$REPORT_ROOT"
if [[ -x /app/simulation/ns3/build/uav-olsr-dataset ]]; then
  cp /app/simulation/ns3/build/uav-olsr-dataset "$NS3_BINARY"
  chmod +x "$NS3_BINARY"
fi
[[ -x "$NS3_BINARY" ]] || {
  echo "Missing ns-3 binary: $NS3_BINARY" >&2
  exit 1
}

python -m scripts.dataset.run_controlled_scenarios --start-index 1 --end-index 100
python -m scripts.analysis.summarize_controlled_scenarios --fail-on-check
python -m src.validation.data_quality \
  --output reports/controlled_scenarios/data_quality.json --fail-on-error

python -m scripts.dataset.build_multihorizon_pilot \
  --pattern 'stress_*' --limit 400 \
  --output-root "$MULTIHORIZON_DATA" \
  --summary "$REPORT_ROOT/multihorizon_dataset_summary.csv"
python -m scripts.dataset.analyze_threshold_sensitivity \
  --pilot-root "$MULTIHORIZON_DATA" \
  --output-dir "$REPORT_ROOT/threshold_sensitivity"
python -m src.training.baselines.persistence_baseline \
  --data-root "$MULTIHORIZON_DATA" \
  --output-root "$MULTIHORIZON_OUTPUT/persistence" \
  --summary "$REPORT_ROOT/persistence_summary.csv"
python -m scripts.train.run_multihorizon_benchmark \
  --data-root "$MULTIHORIZON_DATA" \
  --output-root "$MULTIHORIZON_OUTPUT" \
  --summary "$REPORT_ROOT/multihorizon_benchmark_summary.csv" \
  --runs-per-scenario "$WITHIN_RUNS_PER_SCENARIO" \
  --gnn-epochs "$GNN_EPOCHS" --gnn-patience "$GNN_PATIENCE"

python -m scripts.analysis.analyze_multihorizon_stage6 \
  --summary "$REPORT_ROOT/multihorizon_benchmark_summary.csv" \
  --output-dir "$REPORT_ROOT/stage6"
python -m scripts.train.run_stage6_benchmark \
  --data-root "$MULTIHORIZON_DATA" \
  --output-root "$STAGE6_OUTPUT" \
  --reports-root "$REPORT_ROOT/stage6" \
  --benchmark-summary "$REPORT_ROOT/multihorizon_benchmark_summary.csv" \
  --loro-runs-per-scenario "$LORO_FOLDS_PER_SCENARIO" \
  --ablation-runs-per-scenario "$ABLATION_RUNS_PER_SCENARIO" \
  --gnn-epochs "$GNN_EPOCHS" --gnn-patience "$GNN_PATIENCE"

python -m src.routing.multihorizon_eval \
  --data-root "$MULTIHORIZON_DATA" \
  --model-root "$MULTIHORIZON_OUTPUT" \
  --routing-root outputs/routing_multihorizon_controlled \
  --reports-root "$REPORT_ROOT/routing_multihorizon" \
  --runs-per-scenario "$WITHIN_RUNS_PER_SCENARIO"

mapfile -t sample_runs < <(
  find "$MULTIHORIZON_DATA/survival/k3" -mindepth 1 -maxdepth 1 \
    -type d -name 'stress_baseline_*' -printf '%f\n' | sort -t_ -k4.2n | sed -n "1,${CLOSED_LOOP_RUNS}p"
)
[[ "${#sample_runs[@]}" -eq "$CLOSED_LOOP_RUNS" ]] || {
  echo "Expected $CLOSED_LOOP_RUNS complete baseline runs, found ${#sample_runs[@]}" >&2
  exit 1
}

python -m scripts.analysis.benchmark_inference_resources \
  --data-root "$MULTIHORIZON_DATA" \
  --output-root "$MULTIHORIZON_OUTPUT" \
  --report-dir "$REPORT_ROOT/inference_benchmark" \
  --runs "${sample_runs[@]}"

for run_name in "${sample_runs[@]}"; do
  python -m src.routing.closed_loop \
    --binary "$NS3_BINARY" \
    --baseline-run "$run_name" \
    --data-root "$MULTIHORIZON_DATA" \
    --model-root "$MULTIHORIZON_OUTPUT" \
    --output-root "$CLOSED_LOOP_OUTPUT" \
    --reports-root "$REPORT_ROOT/closed_loop" \
    --strategies olsr hop delay persistence logreg xgb edge-sage \
    --target survival --horizon 3 --cost-mode neglog
done

python -m scripts.analysis.generate_stage7_artifacts \
  --within "$REPORT_ROOT/multihorizon_benchmark_summary_aggregate.csv" \
  --within-detail "$REPORT_ROOT/multihorizon_benchmark_summary.csv" \
  --stage6-root "$REPORT_ROOT/stage6" \
  --output-dir "$REPORT_ROOT/artifacts"
python -m scripts.analysis.generate_closed_loop_artifacts \
  --summary "$REPORT_ROOT/closed_loop/summary.csv" \
  --paired "$REPORT_ROOT/closed_loop/paired_comparisons.csv" \
  --output-dir "$REPORT_ROOT/closed_loop/artifacts"

echo "[OK] controlled paper pipeline completed: $REPORT_ROOT"

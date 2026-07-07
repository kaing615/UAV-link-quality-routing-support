#!/usr/bin/env bash
#
# Sweep routing replay với các horizon khác nhau để xem prediction-assisted
# routing có cải thiện ở horizon xa hơn không.
#
# Usage:
#   ./sweep_horizon.sh <pattern> <model> <horizons>
#   ./sweep_horizon.sh 'ns3big_*' edge-sage '1,3,5,10'
#
# Research question:
#   - GNN có vượt tabular ở t+k>1 không?
#   - Prediction có giá trị hơn ở horizon xa hơn không?
#   - Edge-aware features giúp ở horizon nào?

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNS_ROOT="${PROJECT_ROOT}/data/graph_dataset"
RUN_PATTERN="${1:-*}"
MODEL_ID="${2:-edge-sage}"
HORIZONS="${3:-1,3,5,10}"
IFS=',' read -ra HORIZON_ARRAY <<< "${HORIZONS}"

cd "${PROJECT_ROOT}"

# Validate runs exist
matching_runs=()
while IFS= read -r run_dir; do
  matching_runs+=("$(basename "${run_dir}")")
done < <(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${RUN_PATTERN}" | sort)

if [[ ${#matching_runs[@]} -eq 0 ]]; then
  echo "No run directories matched pattern '${RUN_PATTERN}' under ${RUNS_ROOT}"
  exit 1
fi

echo "[INFO] runs=${#matching_runs[@]}, horizons=${HORIZONS}"

# Create output dir for horizon sweep
SWEEP_DIR="${PROJECT_ROOT}/outputs/aggregates/routing/horizon_sweep"
mkdir -p "${SWEEP_DIR}"

# For each horizon, run replay and aggregate
declare -A HORIZON_SUMMARIES

for H in "${HORIZON_ARRAY[@]}"; do
  echo
  echo "============================================================"
  echo "[HORIZON H=${H}]"
  echo "============================================================"

  HORIZON_DIR="${PROJECT_ROOT}/outputs/aggregates/routing/horizon_h${H}"
  mkdir -p "${HORIZON_DIR}"

  # Run replay for each run with this horizon
  for run_name in "${matching_runs[@]}"; do
    GNN_PREDS="${PROJECT_ROOT}/outputs/routing/${run_name}/edge_predictions_${MODEL_ID}.csv"

    if [[ ! -f "${GNN_PREDS}" ]]; then
      echo "[SKIP] ${run_name}: predictions not found"
      continue
    fi

    echo "[H=${H}] ${run_name}"

    python3 -m src.routing.replay_eval \
      --run-name "${run_name}" \
      --gnn-predictions "${GNN_PREDS}" \
      --horizon "${H}" \
      --p-th "0.0" \
      > /dev/null 2>&1 || true
  done

  # Aggregate this horizon
  echo "[H=${H}] Aggregating..."
  python3 -m src.routing.aggregate_routing \
    --routing-root "${PROJECT_ROOT}/outputs/routing" \
    --output-dir "${HORIZON_DIR}" \
    --pattern "${RUN_PATTERN}" \
    --title "Routing Strategy Comparison (H=${H})" \
    --filename "summary.csv" \
    > /dev/null 2>&1 || true

  if [[ -f "${HORIZON_DIR}/summary_by_strategy.csv" ]]; then
    cp "${HORIZON_DIR}/summary_by_strategy.csv" "${SWEEP_DIR}/h${H}_summary.csv"
    echo "[H=${H}] summary_by_strategy.csv saved"
  fi
done

# Combine all horizons into single CSV
echo
echo "============================================================"
echo "[COMBINING HORIZON RESULTS]"
echo "============================================================"

python3 << 'PYEOF'
import pandas as pd
from pathlib import Path

SWEEP_DIR = Path("outputs/aggregates/routing/horizon_sweep")
HORIZONS = [1, 3, 5, 10]

combined_rows = []
for H in HORIZONS:
    csv_path = SWEEP_DIR / f"h{H}_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["horizon"] = H
        combined_rows.append(df)

if combined_rows:
    combined = pd.concat(combined_rows, ignore_index=True)
    combined.to_csv(SWEEP_DIR / "horizon_sweep.csv", index=False)
    print(f"[OK] horizon_sweep.csv: {len(combined)} rows")

    # Also print summary table
    for strat in ["hop", "xgb", "gnn"]:
        strat_df = combined[combined["strategy"] == strat]
        if not strat_df.empty:
            print(f"\n{strat.upper()} by horizon:")
            for _, row in strat_df.iterrows():
                h = int(row["horizon"])
                lt = row["mean_route_lifetime_mean"]
                pdr = row["mean_realized_pdr_t1_mean"]
                ch = row["mean_route_changes_mean"]
                print(f"  H={h}: lifetime={lt:.3f}, pdr={pdr:.3f}, changes={ch:.2f}")
else:
    print("[WARN] No horizon summaries found")
PYEOF

# Plot horizon tradeoff
echo
echo "[PLOTTING] horizon_tradeoff.png"

python3 << 'PYEOF'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

SWEEP_DIR = Path("outputs/aggregates/routing/horizon_sweep")
CSV_PATH = SWEEP_DIR / "horizon_sweep.csv"

if not CSV_PATH.exists():
    print("[SKIP] horizon_sweep.csv not found")
    exit(0)

df = pd.read_csv(CSV_PATH)

# Only plot hop, xgb, gnn
strategies = ["hop", "xgb", "gnn"]
labels = {"hop": "Shortest-Hop", "xgb": "XGBoost", "gnn": "GNN (Edge-SAGE)"}
colors = {"hop": "#9e9e9e", "xgb": "#ff7f0e", "gnn": "#2ca02c"}

metrics = [
    ("mean_route_lifetime_mean", "Route Lifetime (steps)"),
    ("mean_realized_pdr_t1_mean", "Realized PDR @ t+1"),
    ("mean_route_changes_mean", "Route Changes / session"),
]

horizons = sorted(df["horizon"].unique())

fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
for ax, (col, title) in zip(axes, metrics):
    for strat in strategies:
        strat_df = df[df["strategy"] == strat].sort_values("horizon")
        if strat_df.empty:
            continue
        means = strat_df[col].values
        stds = [strat_df[f"{col[:-5]}std"].values[0] if f"{col[:-5]}std" in strat_df.columns else 0
                for _ in means]
        ax.errorbar(strat_df["horizon"].values, means, yerr=stds,
                   label=labels[strat], color=colors[strat], marker='o', capsize=4)
    ax.set_xlabel("Horizon H")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.set_xticks(horizons)
    ax.grid(alpha=0.3)
    ax.legend()

fig.suptitle("Routing Performance vs Horizon", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(SWEEP_DIR / "horizon_tradeoff.png", dpi=150)
plt.close(fig)
print(f"[OK] horizon_tradeoff.png saved")
PYEOF

echo
echo "[DONE] Horizon sweep complete"
echo "Results: outputs/aggregates/routing/horizon_sweep/"

#!/usr/bin/env bash
# v10 runner (docs/service_v10_protocol.md). Usage: service_v10_runner.sh <action>
set -euo pipefail
W=/home/sac/uav-service/2026-11-24-service-v10
V7=/home/sac/uav-service/2026-11-23-service-v7
V5=/home/sac/uav-service/2026-11-22-service-v5
PY=/home/sac/uav-service/2026-11-11-path-control/.venv/bin/python
MODEL=/home/sac/uav-service/2026-11-12-service-prevention-v2/reports/uav_service_prevention_v2_2026-11-12/model/predictor.json
LOCK=/home/sac/uav-service/2026-11-12-service-prevention-v2/global-simulations.lock
BIN=$V7/repo/build/recovery/uav-olsr-dataset
E="env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin LD_LIBRARY_PATH=$V5/runtime/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
NS3ID="ns-3.45 + OLSR-ETX patch 9cd7c5ec083e404f443832cc88f92f102ef57ef18230df66a4d34e4725550454; libns3.45-olsr-default.so c3777857f66cf8094a6c74c48fc802c6456bcad9999abdaac33408792ea8bc84; v7 binary 90f068b63d3b610389ec2d68223b23867b7988963efe9d77bb09a6fbee2a3e33 (UAVTX counter)"
CCID="GNU C++13.3.0; CMake Release build in WSL Ubuntu 24.04 against patched ns-3.45"
REF=reports/uav_prevention_pilot_2026-11-11/predictor.json
cd "$W/repo"
no_simulations() {
  matches=$(ps -eo pid=,comm= | awk '$2 ~ /^uav-olsr/')
  if [ -n "$matches" ]; then printf '%s\n' "$matches"; exit 1; fi
}
launch() {  # $1 package dir name, $2 run name
  no_simulations
  test ! -e "$W/runs/$2"; test ! -e "$W/evidence/queue-$2"
  mkdir -p "$W/evidence"
  (set -o noclobber; date -u +%FT%TZ > "$W/evidence/queue-$2-launched.txt")
  nohup $E "$PY" orchestration/service_v10_queue.py --protocol "$W/packages/$1/protocol.json" --run-root "$W/runs/$2" --evidence "$W/evidence/queue-$2" --lock "$LOCK" > "$W/evidence/queue-$2.log" 2>&1 < /dev/null &
  printf 'queue_pid=%s\n' "$!"
}
case "$1" in
  preflight)
    no_simulations; df -h "$HOME"
    sha256sum -c "$V7/repo/build/recovery/source.sha256"
    sha256sum "$BIN" "$V5/runtime/lib/libns3.45-olsr-default.so" "$MODEL" reports/uav_service_v10/*.json docs/service_v10_protocol.md ;;
  freeze-tuning)
    $E "$PY" -m src.control.service_v10.tune_cli freeze --phase validation --selection reports/uav_service_v10/tuning-model-selection.json --seed-plan reports/uav_service_v10/tuning-seed-plan.json --model "$MODEL" --reference-model $REF --launcher "$BIN" --ns3-identity "$NS3ID" --compiler-identity "$CCID" --output "$W/packages/tuning" ;;
  launch-tuning) launch tuning tuning ;;
  validate-tuning) $E "$PY" -m src.control.service_v10.tune_cli validate --protocol "$W/packages/tuning/protocol.json" --run-root "$W/runs/tuning" --output "$W/runs/tuning-validation.json" ;;
  analyze-tuning) $E "$PY" -m src.control.service_v10.tune_cli analyze --protocol "$W/packages/tuning/protocol.json" --run-root "$W/runs/tuning" --output "$W/analysis/tuning" ;;
  freeze-validation)
    $E "$PY" -m src.control.service_v10.cli freeze --phase validation --selection "$W/analysis/tuning/tuning-selection.json" --seed-plan reports/uav_service_v10/seed-plan.json --model "$MODEL" --reference-model $REF --launcher "$BIN" --ns3-identity "$NS3ID" --compiler-identity "$CCID" --output "$W/packages/validation" ;;
  freeze-test)
    $E "$PY" -m src.control.service_v10.cli freeze --phase test --selection "$W/analysis/tuning/tuning-selection.json" --inherits "$W/packages/validation/protocol.json" --go "$W/analysis/validation/go.json" --seed-plan reports/uav_service_v10/seed-plan.json --model "$MODEL" --reference-model $REF --launcher "$BIN" --ns3-identity "$NS3ID" --compiler-identity "$CCID" --output "$W/packages/test" ;;
  launch-validation) launch validation validation ;;
  launch-test) launch test test ;;
  validate-validation|validate-test)
    phase=${1#validate-}
    $E "$PY" -m src.control.service_v10.cli validate --protocol "$W/packages/$phase/protocol.json" --run-root "$W/runs/$phase" --output "$W/runs/$phase-validation.json" ;;
  analyze-validation|analyze-test)
    phase=${1#analyze-}
    $E "$PY" -m src.control.service_v10.cli analyze --protocol "$W/packages/$phase/protocol.json" --run-root "$W/runs/$phase" --output "$W/analysis/$phase" ;;
  *) printf 'Unknown runner action: %s\n' "$1" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
set -euo pipefail
W=/home/sac/uav-service/2026-11-23-service-v8
V7=/home/sac/uav-service/2026-11-23-service-v7
V5=/home/sac/uav-service/2026-11-22-service-v5
PY=/home/sac/uav-service/2026-11-11-path-control/.venv/bin/python
MODEL=/home/sac/uav-service/2026-11-12-service-prevention-v2/reports/uav_service_prevention_v2_2026-11-12/model/predictor.json
LOCK=/home/sac/uav-service/2026-11-12-service-prevention-v2/global-simulations.lock
BIN=$V7/repo/build/recovery/uav-olsr-dataset
E="env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin LD_LIBRARY_PATH=$V5/runtime/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
NS3ID="ns-3.45 + OLSR-ETX patch 9cd7c5ec083e404f443832cc88f92f102ef57ef18230df66a4d34e4725550454; libns3.45-olsr-default.so c3777857f66cf8094a6c74c48fc802c6456bcad9999abdaac33408792ea8bc84; v7 binary 90f068b63d3b610389ec2d68223b23867b7988963efe9d77bb09a6fbee2a3e33 (UAVTX counter)"
CCID="GNU C++13.3.0; CMake Release build in WSL Ubuntu 24.04 against patched ns-3.45"
cd "$W/repo"
no_simulations() {
  matches=$(ps -eo pid=,comm= | awk '$2 ~ /^uav-olsr/')
  if [ -n "$matches" ]; then printf '%s\n' "$matches"; exit 1; fi
}
case "$1" in
  preflight)
    no_simulations
    df -h "$HOME"
    sha256sum -c "$V7/repo/build/recovery/source.sha256"
    sha256sum "$BIN" "$V5/runtime/lib/libns3.45-olsr-default.so" "$MODEL" "$W/selection.json"
    $E "$PY" -c 'import json,hashlib,pathlib; p=pathlib.Path("/home/sac/uav-service/2026-11-22-service-v5/runtime"); m=json.loads((p/"runtime_manifest.json").read_text())["files"]; bad=[k for k,v in m.items() if hashlib.sha256((p/"lib"/k).read_bytes()).hexdigest()!=v]; print("MANIFEST_OK" if not bad else bad); assert not bad'
    ;;
  smoke)
    no_simulations
    $E "$PY" scripts/analysis/smoke_service_v8.py --selection "$W/selection.json" --reference-model reports/uav_prevention_pilot_2026-11-11/predictor.json --launcher "$BIN" --output "$W/smoke" --lock "$LOCK"
    ;;
  freeze-validation)
    $E "$PY" -m src.control.service_v8.cli freeze --phase validation --selection "$W/selection.json" --seed-plan reports/uav_service_v8/seed-plan.json --model "$MODEL" --reference-model reports/uav_prevention_pilot_2026-11-11/predictor.json --launcher "$BIN" --ns3-identity "$NS3ID" --compiler-identity "$CCID" --output "$W/packages/validation"
    ;;
  freeze-test)
    $E "$PY" -m src.control.service_v8.cli freeze --phase test --selection "$W/selection.json" --inherits "$W/packages/validation/protocol.json" --go "$W/analysis/validation/go.json" --seed-plan reports/uav_service_v8/seed-plan.json --model "$MODEL" --reference-model reports/uav_prevention_pilot_2026-11-11/predictor.json --launcher "$BIN" --ns3-identity "$NS3ID" --compiler-identity "$CCID" --output "$W/packages/test"
    ;;
  dry-validation|dry-test)
    phase=${1#dry-}
    $E "$PY" orchestration/service_v8_queue.py --protocol "$W/packages/$phase/protocol.json" --run-root "$W/runs/$phase" --evidence "$W/evidence/queue-$phase-dry" --lock "$LOCK" --dry-run
    ;;
  launch-validation|launch-test)
    phase=${1#launch-}
    no_simulations
    test ! -e "$W/runs/$phase"
    test ! -e "$W/evidence/queue-$phase"
    mkdir -p "$W/evidence"
    (set -o noclobber; date -u +%FT%TZ > "$W/evidence/queue-$phase-launched.txt")
    nohup $E "$PY" orchestration/service_v8_queue.py --protocol "$W/packages/$phase/protocol.json" --run-root "$W/runs/$phase" --evidence "$W/evidence/queue-$phase" --lock "$LOCK" > "$W/evidence/queue-$phase.log" 2>&1 < /dev/null &
    printf 'queue_pid=%s\n' "$!"
    ;;
  validate-validation|validate-test)
    phase=${1#validate-}
    $E "$PY" -m src.control.service_v8.cli validate --protocol "$W/packages/$phase/protocol.json" --run-root "$W/runs/$phase" --output "$W/runs/$phase-validation.json"
    ;;
  analyze-validation|analyze-test)
    phase=${1#analyze-}
    $E "$PY" -m src.control.service_v8.cli analyze --protocol "$W/packages/$phase/protocol.json" --run-root "$W/runs/$phase" --output "$W/analysis/$phase"
    ;;
  *) printf 'Unknown runner action: %s\n' "$1" >&2; exit 2 ;;
esac

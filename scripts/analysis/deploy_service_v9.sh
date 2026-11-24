#!/usr/bin/env bash
set -euo pipefail
W=/home/sac/uav-service/2026-11-23-service-v9
V7=/home/sac/uav-service/2026-11-23-service-v7
V8=/home/sac/uav-service/2026-11-23-service-v8
test ! -e "$W/repo"
git clone --no-hardlinks "$V7/repo" "$W/repo"
tar -xzf "$W/development/v8-source-bundle.tar.gz" -C "$W/repo"
tar -xzf "$W/development/v9-source-bundle.tar.gz" -C "$W/repo"
cp "$W/repo/reports/uav_service_v9/selection.json" "$W/selection.json"
mkdir -p "$W/repo/build/recovery"
cp "$V7/repo/build/recovery/uav-olsr-dataset" "$W/repo/build/recovery/uav-olsr-dataset"
cp "$V7/repo/build/recovery/source.sha256" "$W/repo/build/recovery/source.sha256"
sed 's/service-v8/service-v9/g; s/service_v8/service_v9/g' "$V8/development/runner.sh" > "$W/development/runner.sh"
cp "$V8/development/monitor.py" "$W/development/monitor.py"
cd "$W/repo"
git rev-parse HEAD
sha256sum -c build/recovery/source.sha256
sha256sum "$W/selection.json" reports/uav_service_v9/seed-plan.json docs/service_v9_protocol.md
df -h "$HOME"

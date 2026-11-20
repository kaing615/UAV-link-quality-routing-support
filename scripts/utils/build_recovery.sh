#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
options=(-S simulation/ns3 -B build/recovery -DCMAKE_BUILD_TYPE=Release)
if [[ -n "${NS3_PREFIX:-}" ]]; then options+=("-DNS3_PREFIX=$NS3_PREFIX"); fi
cmake "${options[@]}"
cmake --build build/recovery -j "${BUILD_JOBS:-2}"
sha256sum simulation/ns3/uav-olsr-dataset.cc simulation/ns3/CMakeLists.txt \
    simulation/ns3/recovery-control.h simulation/ns3/recovery-protocol.h \
    simulation/ns3/routing-diagnostics.h \
    simulation/ns3/service-observation.h \
    build/recovery/uav-olsr-dataset > build/recovery/source.sha256

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}/simulation/ns3"

# Image binaries carry checksums of both their source and executable.
if [[ ! -f build/CMakeCache.txt && -x build/uav-olsr-dataset && -f build/source.sha256 ]]; then
  sha256sum --check build/source.sha256 >/dev/null || {
    echo "ns-3 binary/source mismatch; rebuild the research image before generating data" >&2
    exit 1
  }
else
  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
  cmake --build build
  sha256sum uav-olsr-dataset.cc CMakeLists.txt build/uav-olsr-dataset > build/source.sha256
fi

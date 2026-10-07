"""Verify image checksum enforcement and local incremental CMake execution."""

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    script = Path("scripts/utils/ensure_ns3_binary.sh").read_text()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        helper = root / "scripts" / "utils" / "ensure_ns3_binary.sh"
        helper.parent.mkdir(parents=True)
        helper.write_text(script)
        source = root / "simulation" / "ns3"
        (source / "build").mkdir(parents=True)
        for name in ("uav-olsr-dataset.cc", "CMakeLists.txt", "build/uav-olsr-dataset"):
            (source / name).write_text("original")
        (source / "build" / "uav-olsr-dataset").chmod(0o755)
        checksums = "".join(
            f"{hashlib.sha256((source / name).read_bytes()).hexdigest()}  {name}\n"
            for name in ("uav-olsr-dataset.cc", "CMakeLists.txt", "build/uav-olsr-dataset")
        )
        (source / "build" / "source.sha256").write_text(checksums)
        subprocess.run(["bash", str(helper)], check=True)
        (source / "uav-olsr-dataset.cc").write_text("changed")
        failed = subprocess.run(["bash", str(helper)], capture_output=True, text=True)
        assert failed.returncode != 0 and "binary/source mismatch" in failed.stderr
        tools = root / "tools"
        tools.mkdir()
        cmake = tools / "cmake"
        cmake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CMAKE_CHECK_LOG"\n')
        cmake.chmod(0o755)
        (source / "build" / "CMakeCache.txt").touch()
        log = root / "cmake.log"
        env = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "CMAKE_CHECK_LOG": str(log)}
        subprocess.run(["bash", str(helper)], env=env, check=True)
        subprocess.run(["bash", str(helper)], env=env, check=True)
        assert log.read_text().splitlines() == ["-B build -S . -DCMAKE_BUILD_TYPE=Release", "--build build"] * 2
        print("PASS: image source/binary checksums and local incremental build")


if __name__ == "__main__":
    main()

from pathlib import Path

import pytest

from src.utils.experiment import validate_ns3_binary


def test_ns3_provenance_detects_motion_header_change(tmp_path: Path):
    import hashlib

    source = tmp_path / "ns3"
    binary = source / "build/uav-olsr-dataset"
    binary.parent.mkdir(parents=True)
    files = {
        "uav-olsr-dataset.cc": "source",
        "CMakeLists.txt": "cmake",
        "recovery-control.h": "first motion implementation",
        "recovery-protocol.h": "protocol",
        "build/uav-olsr-dataset": "binary",
    }
    for name, content in files.items():
        (source / name).write_text(content)
    manifest = "".join(f"{hashlib.sha256((source / name).read_bytes()).hexdigest()}  {name}\n" for name in files)
    binary.with_name("source.sha256").write_text(manifest)
    validate_ns3_binary(binary, source)
    (source / "recovery-control.h").write_text("changed motion implementation")
    with pytest.raises(ValueError, match="mismatch"):
        validate_ns3_binary(binary, source)

"""Identify independent runs and reuse only matching experiment outputs."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path


def seed_group(run_name: str) -> str:
    match = re.fullmatch(r"stress_.+_(?:rwp|gm|ctrl)_s(\d+)", run_name)
    return f"controlled-seed-{int(match[1])}" if match else run_name


@lru_cache(maxsize=4096)
def _file_digest(path: str, size: int, modified_ns: int) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(config: dict, paths: list[Path]) -> dict:
    files = {}
    for path in paths:
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            if "__pycache__" in candidate.parts or candidate.is_dir():
                continue
            resolved = candidate.resolve()
            stat = resolved.stat()
            files[str(resolved)] = _file_digest(str(resolved), stat.st_size, stat.st_mtime_ns)
    payload = {"config": config, "files": files}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"fingerprint": digest, **payload}


def cache_matches(manifest: Path, expected: dict, outputs: list[Path]) -> bool:
    if not all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")) == expected
    except (OSError, ValueError):
        return False


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_ns3_binary(binary: Path, source_root: Path = Path("simulation/ns3")) -> None:
    manifest_path = binary.with_name("source.sha256")
    if not manifest_path.is_file():
        raise ValueError(f"Missing ns-3 build provenance: {manifest_path}; run scripts/utils/ensure_ns3_binary.sh")
    try:
        expected = {
            name: digest for digest, name in (line.split(maxsplit=1) for line in manifest_path.read_text().splitlines())
        }
    except ValueError as error:
        raise ValueError(f"Invalid ns-3 build provenance: {manifest_path}") from error
    inputs = {
        "uav-olsr-dataset.cc": source_root / "uav-olsr-dataset.cc",
        "CMakeLists.txt": source_root / "CMakeLists.txt",
        "build/uav-olsr-dataset": binary,
    }
    current = fingerprint({}, list(inputs.values()))["files"]
    if any(expected.get(name) != current[str(path.resolve())] for name, path in inputs.items()):
        raise ValueError("ns-3 binary/source mismatch; rebuild before running experiments")


def validate_test_predictions(run_root: Path, predictions_csv: Path) -> None:
    """Require the exact held-out edge identities and labels for every model."""
    import pandas as pd

    expected = pd.read_csv(run_root / "features" / "edges_labeled.csv")
    splits = pd.read_csv(run_root / "splits" / "time_splits.csv")
    expected = expected[expected["time"].isin(splits.loc[splits["split"] == "test", "time"])]
    actual = pd.read_csv(predictions_csv)
    identity = ["time", "src", "dst"]
    if not set(identity) <= set(actual):
        from src.routing.predict_edges import attach_prediction_identity

        identified = predictions_csv.with_name(f"{predictions_csv.stem}_identified.csv")
        attach_prediction_identity(predictions_csv, run_root / "graph_dataset" / "test.pt", identified)
        actual = pd.read_csv(identified)
    if "y_true" in actual:
        actual = actual.rename(columns={"y_true": "label"})
    columns = [*identity, "label"]
    for frame in (expected, actual):
        if frame.duplicated(identity).any():
            raise ValueError(f"Duplicate test edge identities: {predictions_csv}")
    expected = expected[columns].sort_values(identity).reset_index(drop=True)
    actual = actual[columns].sort_values(identity).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False)
    except AssertionError as error:
        raise ValueError(f"Test prediction cohort differs from labeled test edges: {predictions_csv}") from error

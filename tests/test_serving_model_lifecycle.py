from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _copy_script(root: Path, name: str) -> None:
    destination = root / "scripts" / "mlops" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        (Path("scripts/mlops") / name).read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )


def _fake_command(root: Path, name: str) -> None:
    command = root / "bin" / name
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$PWD/{name}_calls.txt"\n',
        encoding="utf-8",
        newline="\n",
    )
    command.chmod(0o755)


def test_staging_pulls_only_the_promoted_serving_artifact(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    artifact = root / "deploy" / "serving_model_artifact"
    artifact.mkdir(parents=True)
    _copy_script(root, "stage_serving_model.sh")
    _fake_command(root, "dvc")
    (root / "deploy" / "serving_model.json").write_text(
        json.dumps({"model_id": "edge-sage", "run_name": "run-001"}),
        encoding="utf-8",
    )
    (artifact / "best_model.pt").write_bytes(b"weights")
    (artifact / "metadata.json").write_text(
        json.dumps({"model_id": "edge-sage", "run_name": "run-001"}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            "-c",
            'PATH="$PWD/bin:$PATH" bash scripts/mlops/stage_serving_model.sh',
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (root / "models" / "best_model.pt").read_bytes() == b"weights"
    assert (root / "models" / "metadata.json").exists()
    assert (root / "dvc_calls.txt").read_text(encoding="utf-8").splitlines() == [
        "pull deploy/serving_model_artifact.dvc"
    ]


def test_promotion_publishes_a_dedicated_serving_artifact(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = root / "trained-model"
    source.mkdir(parents=True)
    _copy_script(root, "promote_model.sh")
    _fake_command(root, "dvc")
    _fake_command(root, "git")
    (source / "best_model.pt").write_bytes(b"new-weights")
    (source / "metadata.json").write_text(
        json.dumps({"model_id": "edge-sage", "run_name": "run-002"}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            "-c",
            'PATH="$PWD/bin:$PATH" bash scripts/mlops/promote_model.sh trained-model 0.91',
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = root / "deploy" / "serving_model_artifact"
    assert (artifact / "best_model.pt").read_bytes() == b"new-weights"
    assert json.loads((root / "deploy" / "serving_model.json").read_text(encoding="utf-8")) == {
        "model_id": "edge-sage",
        "run_name": "run-002",
    }
    assert (root / "dvc_calls.txt").read_text(encoding="utf-8").splitlines() == [
        "add deploy/serving_model_artifact",
        "push deploy/serving_model_artifact.dvc",
    ]

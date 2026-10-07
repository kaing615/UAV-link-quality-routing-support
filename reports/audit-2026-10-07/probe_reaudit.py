"""Reproduce audit findings with temporary data; never train or change real artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from scripts.analysis import benchmark_inference_resources as resources  # noqa: E402
from scripts.analysis.analyze_multihorizon_stage6 import paired_model_comparisons  # noqa: E402
from scripts.dataset.run_controlled_scenarios import build_jobs  # noqa: E402
from scripts.train import run_multihorizon_benchmark as within  # noqa: E402
from scripts.train import run_stage6_benchmark as stage6  # noqa: E402

from src.models.gnn.edge_gnn import GATEdgeClassifier  # noqa: E402
from src.routing import replay_eval  # noqa: E402
from src.serving import app as serving  # noqa: E402
from src.serving.schemas import PredictionRequest  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def dataset(path: Path) -> None:
    graph_dir = path / "graph_dataset"
    graph_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (graph_dir / f"{split}.pt").touch()


def metric(model: str, run: str, value: float = 0.7) -> dict:
    return {"model_id": model, "run_name": run, "split": "test", "macro_f1": value, "pr_auc": value}


def check_controlled_folds(tmp: Path) -> None:
    jobs = build_jobs(1, 2)
    for job in jobs:
        dataset(tmp / "data" / "qos" / "k1" / job.run_name)
    calls = []
    with patch.object(stage6, "_run", side_effect=lambda command: calls.append(command)):
        stage6.run_stage6(
            tmp / "data", tmp / "raw", tmp / "outputs", tmp / "reports", {"loro"}, ["logreg"],
            {"qos"}, {1}, False, 1, 1, tmp / "unused.csv", 1, 1,
        )
    command = calls[0]
    test_run = command[command.index("--test-runs") + 1]
    train_runs = command[command.index("--train-runs") + 1].split(",")
    siblings = [run for run in train_runs if run.rsplit("_s", 1)[1] == test_run.rsplit("_s", 1)[1]]
    assert len(siblings) == 3
    print("CONFIRMED controlled LORO train contains same-seed siblings:", test_run, siblings)
    print("CONFIRMED 8 scenario runs in this fixture have only 2 seed groups")


def check_persistence_filter(tmp: Path) -> None:
    dataset(tmp / "data" / "qos" / "k1" / "run-a")
    for model in (*within.MODELS, "persistence"):
        write_csv(tmp / "outputs" / model / "qos" / "k1" / "run-a" / "metrics.csv", [metric(model, "run-a")])
    result = within.run_benchmark(tmp / "data", tmp / "outputs", tmp / "summary.csv", list(within.MODELS))
    assert "persistence" not in set(result.model_id)
    comparisons = paired_model_comparisons(result)
    missing = comparisons[comparisons.reference_strategy.eq("persistence")]
    assert not missing.empty and missing.n_pairs.eq(0).all()
    print("CONFIRMED current within-run summary drops Persistence; comparisons have n_pairs=0")


def check_stage6_scope(tmp: Path) -> None:
    dataset(tmp / "data" / "qos" / "k1" / "run-a")
    write_csv(tmp / "outputs" / "loro" / "logreg" / "qos" / "k1" / "run-a" / "metrics.csv", [metric("logreg", "run-a")])
    write_csv(tmp / "outputs" / "loro" / "xgb" / "survival" / "k5" / "old-run" / "metrics.csv", [metric("xgb", "old-run")])
    stage6.run_stage6(
        tmp / "data", tmp / "raw", tmp / "outputs", tmp / "reports", {"loro"}, ["logreg"],
        {"qos"}, {1}, False, 1, 1, tmp / "unused.csv", 1, 1,
    )
    result = pd.read_csv(tmp / "reports" / "loro_summary.csv")
    assert "old-run" in set(result.run_name)
    print("CONFIRMED Stage 6 qos/k1/logreg request includes old survival/k5/xgb results")


def check_resource_cache(tmp: Path) -> None:
    detail = tmp / "reports" / "details" / "logreg__run-a.json"
    detail.parent.mkdir(parents=True)
    detail.write_text(json.dumps({"model_id": "logreg", "run_name": "run-a", "target": "qos", "horizon": 1, "threads": 1}))
    args = argparse.Namespace(
        runs=["run-a"], models=["logreg"], target="survival", horizon=5, threads=4,
        data_root=tmp / "data", output_root=tmp / "outputs", report_dir=tmp / "reports",
        warmups=1, repeats=2, force=False,
    )
    captured = []
    with patch.object(resources, "_write_outputs", side_effect=lambda frame, _: captured.append(frame)), patch.object(
        resources.subprocess, "run", side_effect=AssertionError("Cache unexpectedly recomputed")
    ):
        resources.run_benchmark(args)
    row = captured[0].iloc[0]
    assert (row.target, row.horizon, row.threads) == ("qos", 1, 1)
    print("CONFIRMED inference survival/k5/4 threads request reuses qos/k1/1 thread cache")


def replay_fixture(tmp: Path, *, snr: float = 25.0) -> tuple[Path, Path, Path]:
    raw_root = tmp / "raw"
    write_csv(raw_root / "run-a" / "edges.csv", [
        {"time": time, "src": 0, "dst": 1, "connected": 1, "snr": snr, "packet_loss": 0.01, "delay": 1.0}
        for time in (0, 1)
    ])
    split = tmp / "splits.csv"
    write_csv(split, [{"time": 0, "split": "test"}])
    predictions = tmp / "predictions.csv"
    write_csv(predictions, [{"time": 0, "src": 0, "dst": 1, "pred_score": 0.9}])
    return raw_root, split, predictions


def check_replay_horizon(tmp: Path) -> None:
    raw, split, predictions = replay_fixture(tmp)
    _, details = replay_eval.evaluate_run(
        "run-a", None, horizon=5, prediction_csvs={"edge-sage": predictions}, split_csv=split,
        raw_root=raw, output_dir=tmp / "results", include_olsr=False, target="qos", prediction_horizon=5,
    )
    frame = pd.read_csv(details)
    assert frame.horizon.eq(1).all() and frame.survival_at_horizon.eq(1).all()
    print("CONFIRMED requested routing H=5 reports survival=1 using only H=1")


def check_serving_image_default(tmp: Path) -> None:
    model_dir = tmp / "model"
    model_dir.mkdir()
    torch.save(GATEdgeClassifier(edge_in_channels=7, hidden_channels=8).state_dict(), model_dir / "best_model.pt")
    (model_dir / "metadata.json").write_text(json.dumps({"model_id": "gat", "hidden_channels": 8, "num_layers": 2}))
    with patch.dict("os.environ", {"MODEL_DIR": str(model_dir), "MODEL_ID": "edge-sage"}):
        try:
            serving.load_model()
        except ValueError as error:
            assert "disagrees with artifact gat" in str(error)
            print("CONFIRMED Dockerfile MODEL_ID default blocks GAT artifact:", error)
        else:
            raise AssertionError("Expected image default to conflict with artifact")


def check_nonfinite_input(tmp: Path) -> None:
    request = PredictionRequest(
        nodes=[{"node_id": node, "x": 0, "y": 0, "z": 0, "vx": 0, "vy": 0, "vz": 0, "degree": 1, "load": 0} for node in (0, 1)],
        edges=[{"src": 0, "dst": 1, "distance": 1e100, "rssi": -60, "snr": 30, "delay": 1, "packet_loss": 0.01, "relative_speed": 0, "throughput": 1}],
    )
    assert torch.isinf(torch.tensor(request.edges[0].distance, dtype=torch.float32))
    torch.manual_seed(42)
    model = GATEdgeClassifier(edge_in_channels=7, hidden_channels=8).eval()
    with patch.object(serving, "_model", model), patch.object(serving, "_model_id", "gat"):
        try:
            serving.predict(request)
        except ValidationError as error:
            assert "stability_score" in str(error)
        else:
            raise AssertionError("Expected non-finite model output to fail response validation")
    print("CONFIRMED accepted finite JSON number 1e100 overflows float32 and breaks prediction response")


def check_existing_loro_artifacts() -> None:
    path = ROOT / "reports" / "stage6" / "loro_summary.csv"
    if not path.exists():
        print("UNVERIFIED old LORO artifacts absent")
        return
    frame = pd.read_csv(path)
    test = frame[frame["split"].eq("test")]
    counts = test.groupby(["target", "horizon", "run_name"])["n_samples"].nunique()
    mismatches = counts[counts.gt(1)]
    assert not mismatches.empty
    print("CONFIRMED existing LORO artifacts still have unequal sample counts:", len(mismatches), "coordinates")
    first = test[test.run_name.eq("ns3big_001_rwp_s17296_n18_c211_t108")]
    print(first[["target", "horizon", "model_id", "n_samples"]].to_string(index=False))


def main() -> None:
    for check in (
        check_controlled_folds, check_persistence_filter, check_stage6_scope, check_resource_cache,
        check_replay_horizon, check_serving_image_default, check_nonfinite_input,
    ):
        with tempfile.TemporaryDirectory(prefix="uav-reaudit-", dir=ROOT / ".cache") as directory:
            check(Path(directory))
    check_existing_loro_artifacts()


if __name__ == "__main__":
    main()

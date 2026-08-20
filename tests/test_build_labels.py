from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.gnn.build_labels import build_labeled_edges, parse_args


def _write_edge_history(path: Path) -> Path:
    pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4, 5],
            "src": [1] * 6,
            "dst": [2] * 6,
            "connected": [1, 0, 1, 1, 1, 1],
            "snr": [20.0, 0.0, 20.0, 20.0, 10.0, 20.0],
            "packet_loss": [0.01, 1.0, 0.01, 0.01, 0.01, 0.01],
            "delay": [1.0, 99.0, 1.0, 1.0, 1.0, 1.0],
        }
    ).to_csv(path, index=False)
    return path


def test_qos_horizon_uses_endpoint_and_removes_incomplete_tail(tmp_path: Path):
    output = tmp_path / "labeled.csv"

    build_labeled_edges(
        _write_edge_history(tmp_path / "edges.csv"),
        output,
        target="qos",
        horizon=2,
    )

    labeled = pd.read_csv(output)
    assert labeled["time"].tolist() == [0, 2, 3]
    assert labeled["label"].tolist() == [1, 0, 1]


def test_default_remains_qos_at_t_plus_one(tmp_path: Path):
    output = tmp_path / "labeled.csv"

    build_labeled_edges(_write_edge_history(tmp_path / "edges.csv"), output)

    labeled = pd.read_csv(output)
    assert labeled["time"].tolist() == [0, 2, 3, 4]
    assert labeled["label"].tolist() == [0, 1, 0, 1]


def test_survival_horizon_requires_continuous_connection(tmp_path: Path):
    output = tmp_path / "labeled.csv"

    build_labeled_edges(
        _write_edge_history(tmp_path / "edges.csv"),
        output,
        target="survival",
        horizon=2,
    )

    labeled = pd.read_csv(output)
    assert labeled["time"].tolist() == [0, 2, 3]
    assert labeled["label"].tolist() == [0, 1, 1]


def test_common_max_horizon_only_restricts_input_support(tmp_path: Path):
    output = tmp_path / "labeled.csv"

    build_labeled_edges(
        _write_edge_history(tmp_path / "edges.csv"),
        output,
        target="qos",
        horizon=1,
        common_max_horizon=5,
    )

    labeled = pd.read_csv(output)
    assert labeled["time"].tolist() == [0]
    assert labeled["label"].tolist() == [0]
    assert set(labeled["target"]) == {"qos"}
    assert set(labeled["horizon"]) == {1}
    assert set(labeled["support_horizon"]) == {5}


@pytest.mark.parametrize(
    "options",
    [
        {"target": "other"},
        {"horizon": 0},
        {"horizon": 4},
        {"horizon": 3, "common_max_horizon": 2},
    ],
)
def test_build_labeled_edges_rejects_unsupported_options(tmp_path: Path, options: dict):
    with pytest.raises(ValueError):
        build_labeled_edges(
            _write_edge_history(tmp_path / "edges.csv"),
            tmp_path / "labeled.csv",
            **options,
        )


def test_cli_accepts_label_selection(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_labels.py",
            "--edges-features",
            "edges.csv",
            "--output",
            "labeled.csv",
            "--target",
            "survival",
            "--horizon",
            "3",
            "--common-max-horizon",
            "5",
        ],
    )

    args = parse_args()

    assert (args.target, args.horizon, args.common_max_horizon) == ("survival", 3, 5)

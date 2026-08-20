"""Tests for leakage-safe temporal splits."""

import pandas as pd
import pytest

from src.preprocessing.common.split_dataset import build_time_split


def _write_times(path, count: int) -> None:
    pd.DataFrame({"time": range(count)}).to_csv(path, index=False)


def test_time_split_purges_one_step_at_each_boundary_by_default(tmp_path):
    labeled_csv = tmp_path / "edges_labeled.csv"
    _write_times(labeled_csv, 20)

    split_csv = build_time_split(labeled_csv, tmp_path / "splits", train_ratio=0.5, val_ratio=0.25)

    splits = pd.read_csv(split_csv).set_index("time")["split"].to_dict()
    assert splits == {
        **dict.fromkeys(range(9), "train"),
        9: "purged",
        **dict.fromkeys(range(10, 14), "val"),
        14: "purged",
        **dict.fromkeys(range(15, 20), "test"),
    }


def test_time_split_purges_requested_horizon_at_both_boundaries(tmp_path):
    labeled_csv = tmp_path / "edges_labeled.csv"
    _write_times(labeled_csv, 20)

    split_csv = build_time_split(
        labeled_csv,
        tmp_path / "splits",
        train_ratio=0.5,
        val_ratio=0.25,
        horizon=2,
    )

    splits = pd.read_csv(split_csv).set_index("time")["split"].to_dict()
    assert splits == {
        **dict.fromkeys(range(8), "train"),
        8: "purged",
        9: "purged",
        **dict.fromkeys(range(10, 13), "val"),
        13: "purged",
        14: "purged",
        **dict.fromkeys(range(15, 20), "test"),
    }


@pytest.mark.parametrize("horizon", [0, -1])
def test_time_split_rejects_non_positive_horizon(tmp_path, horizon):
    labeled_csv = tmp_path / "edges_labeled.csv"
    _write_times(labeled_csv, 20)

    with pytest.raises(ValueError, match="horizon must be positive"):
        build_time_split(labeled_csv, tmp_path / "splits", horizon=horizon)


def test_time_split_rejects_horizon_that_empties_a_split(tmp_path):
    labeled_csv = tmp_path / "edges_labeled.csv"
    _write_times(labeled_csv, 12)

    with pytest.raises(ValueError, match="Not enough time steps"):
        build_time_split(labeled_csv, tmp_path / "splits", train_ratio=0.5, val_ratio=0.25, horizon=3)

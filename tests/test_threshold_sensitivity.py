"""Behavioral check for QoS label-threshold sensitivity."""

from __future__ import annotations

import pandas as pd
from scripts.dataset.analyze_threshold_sensitivity import evaluate_threshold_grid


def test_threshold_grid_reports_split_level_prevalence_and_reference_agreement():
    labeled = pd.DataFrame(
        {
            "time": [1, 2, 3, 4],
            "connected_next": [1, 1, 1, 0],
            "snr_next": [20.0, 16.0, 20.0, 30.0],
            "packet_loss_next": [0.05, 0.05, 0.15, 0.0],
            "delay_next": [8.0, 8.0, 8.0, 1.0],
        }
    )
    splits = pd.DataFrame({"time": [1, 2, 3, 4], "split": ["train", "train", "val", "val"]})

    result = evaluate_threshold_grid(
        labeled.merge(splits, on="time"),
        run_name="run-1",
        horizon=1,
        snr_values=[15.0, 18.0],
        loss_values=[0.10],
        delay_values=[10.0],
    )

    reference_val = result[
        (result["split"] == "val") & (result["tau_snr"] == 18.0)
    ].iloc[0]
    loose_train = result[
        (result["split"] == "train") & (result["tau_snr"] == 15.0)
    ].iloc[0]

    assert reference_val["n_samples"] == 2
    assert reference_val["positive_ratio"] == 0.0
    assert reference_val["agreement_with_reference"] == 1.0
    assert loose_train["positive_ratio"] == 1.0
    assert loose_train["agreement_with_reference"] == 0.5

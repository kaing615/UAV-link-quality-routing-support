"""Tests for run-level paired statistical inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.paired_statistics import bootstrap_mean_ci, holm_adjust, paired_comparisons


def test_bootstrap_mean_ci_is_exact_for_constant_values():
    low, high = bootstrap_mean_ci(np.array([2.5, 2.5, 2.5]), n_resamples=200, seed=7)
    assert (low, high) == (2.5, 2.5)


def test_holm_adjust_matches_step_down_definition():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_paired_comparisons_uses_only_shared_runs_and_reports_effect():
    detailed = pd.DataFrame(
        [
            {"run_name": "r1", "strategy": "hop", "mean_route_lifetime": 1.0},
            {"run_name": "r1", "strategy": "gnn", "mean_route_lifetime": 2.0},
            {"run_name": "r2", "strategy": "hop", "mean_route_lifetime": 2.0},
            {"run_name": "r2", "strategy": "gnn", "mean_route_lifetime": 3.0},
            {"run_name": "r3", "strategy": "hop", "mean_route_lifetime": 3.0},
            {"run_name": "r3", "strategy": "gnn", "mean_route_lifetime": 4.0},
            {"run_name": "gnn_only", "strategy": "gnn", "mean_route_lifetime": 9.0},
        ]
    )

    result = paired_comparisons(
        detailed,
        comparisons=[("gnn", "hop")],
        metrics={"mean_route_lifetime": True},
        n_resamples=500,
        seed=11,
    )

    row = result.iloc[0]
    assert row["n_pairs"] == 3
    assert row["reference_mean"] == 2.0
    assert row["comparator_mean"] == 3.0
    assert row["mean_delta"] == 1.0
    assert row["ci95_low"] == 1.0
    assert row["ci95_high"] == 1.0
    assert row["higher_is_better"]
    assert row["primary_endpoint"]
    assert row["p_holm"] >= row["p_raw"]

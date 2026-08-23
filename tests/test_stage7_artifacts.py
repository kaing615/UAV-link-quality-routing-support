"""Publication-artifact generation contracts."""

from __future__ import annotations

import pandas as pd
import pytest
import scripts.analysis.generate_temporal_artifacts as temporal_artifacts
from scripts.analysis.generate_stage7_artifacts import (
    combine_paired_effects,
    combine_protocol_aggregates,
    paired_ablation_effects,
    write_tables,
)
from scripts.analysis.generate_temporal_artifacts import generate_temporal_artifacts, temporal_effects


def test_combine_protocol_aggregates_keeps_only_test_rows():
    within = pd.DataFrame(
        [
            {
                "model_id": "edge-sage",
                "target": "qos",
                "horizon": 1,
                "split": "val",
                "n_runs": 10,
                "macro_f1_mean": 0.9,
            },
            {
                "model_id": "edge-sage",
                "target": "qos",
                "horizon": 1,
                "split": "test",
                "n_runs": 10,
                "macro_f1_mean": 0.8,
            },
        ]
    )
    loro = pd.DataFrame(
        [{"model_id": "edge-sage", "target": "qos", "horizon": 1, "split": "test", "n_runs": 10, "macro_f1_mean": 0.7}]
    )
    cross = pd.DataFrame(
        [{"model_id": "edge-sage", "target": "qos", "horizon": 1, "split": "test", "n_runs": 10, "macro_f1_mean": 0.6}]
    )

    result = combine_protocol_aggregates(within, loro, cross)

    assert result["protocol"].tolist() == ["Within-run", "LORO", "Cross-mobility"]
    assert result["macro_f1_mean"].tolist() == [0.8, 0.7, 0.6]
    assert set(result["split"]) == {"test"}


def test_paired_ablation_effects_uses_shared_runs_and_full_as_reference():
    detailed = pd.DataFrame(
        [
            {
                "model_id": "edge-sage",
                "target": "qos",
                "horizon": 1,
                "split": "test",
                "run_name": "r1",
                "macro_f1": 0.8,
            },
            {
                "model_id": "edge-sage-noedge",
                "target": "qos",
                "horizon": 1,
                "split": "test",
                "run_name": "r1",
                "macro_f1": 0.5,
            },
            {
                "model_id": "edge-sage",
                "target": "qos",
                "horizon": 1,
                "split": "test",
                "run_name": "r2",
                "macro_f1": 0.9,
            },
            {
                "model_id": "edge-sage-noedge",
                "target": "qos",
                "horizon": 1,
                "split": "test",
                "run_name": "r2",
                "macro_f1": 0.4,
            },
            {
                "model_id": "edge-sage-noedge",
                "target": "qos",
                "horizon": 1,
                "split": "test",
                "run_name": "unpaired",
                "macro_f1": 0.1,
            },
        ]
    )

    row = paired_ablation_effects(detailed, n_resamples=100).iloc[0]

    assert row["reference_strategy"] == "edge-sage"
    assert row["comparator_strategy"] == "edge-sage-noedge"
    assert row["n_pairs"] == 2
    assert row["mean_delta"] == pytest.approx(-0.4)


def test_write_tables_creates_a_missing_output_directory(tmp_path):
    metric = {
        "protocol": "Within-run",
        "model_id": "edge-sage",
        "target": "qos",
        "horizon": 1,
        "n_runs": 10,
        "macro_f1_mean": 0.8,
        "macro_f1_ci95_low": 0.7,
        "macro_f1_ci95_high": 0.9,
        "pr_auc_mean": 0.85,
    }
    paired = pd.DataFrame(
        [
            {
                "protocol": "Within-run",
                "target": "qos",
                "horizon": 1,
                "reference_strategy": "logreg",
                "comparator_strategy": "edge-sage",
                "n_pairs": 10,
                "mean_delta": -0.02,
                "ci95_low": -0.03,
                "ci95_high": -0.01,
                "p_holm": 0.04,
            }
        ]
    )
    worst = pd.DataFrame(
        [
            {
                "model_id": "edge-sage",
                "target": "qos",
                "horizon": 1,
                "group_type": "mobility",
                "group_value": "gauss-markov",
                "n_runs": 5,
                "metric_mean": 0.7,
            }
        ]
    )
    output_dir = tmp_path / "missing" / "tables"

    write_tables(pd.DataFrame([metric]), pd.DataFrame([metric]), paired, worst, output_dir)

    assert (output_dir / "protocol_summary.csv").exists()
    assert (output_dir / "worst_group_summary.tex").exists()


def test_combine_paired_effects_recomputes_pairs_from_complete_detail():
    rows = []
    for run, reference in (("r1", 0.7), ("r2", 0.8)):
        rows.extend(
            [
                {
                    "model_id": "logreg",
                    "target": "qos",
                    "horizon": 1,
                    "split": "test",
                    "run_name": run,
                    "macro_f1": reference,
                    "pr_auc": 0.8,
                },
                {
                    "model_id": "xgb",
                    "target": "qos",
                    "horizon": 1,
                    "split": "test",
                    "run_name": run,
                    "macro_f1": reference - 0.1,
                    "pr_auc": 0.7,
                },
                {
                    "model_id": "edge-sage",
                    "target": "qos",
                    "horizon": 1,
                    "split": "test",
                    "run_name": run,
                    "macro_f1": reference + 0.05,
                    "pr_auc": 0.85,
                },
            ]
        )
    detail = pd.DataFrame(rows)

    result = combine_paired_effects(detail, detail, detail, pd.DataFrame())
    edge_vs_logreg = result[(result["reference_strategy"] == "logreg") & (result["comparator_strategy"] == "edge-sage")]

    assert set(edge_vs_logreg["protocol"]) == {"Within-run", "LORO", "Cross-mobility"}
    assert set(edge_vs_logreg["n_pairs"]) == {2}


def test_temporal_effects_requires_exact_static_coordinates_and_uses_paired_deltas():
    static = pd.DataFrame(
        [
            {"target": "survival", "horizon": 3, "run_name": "r1", "split": "test", "macro_f1": 0.6, "pr_auc": 0.7},
            {"target": "survival", "horizon": 3, "run_name": "r2", "split": "test", "macro_f1": 0.7, "pr_auc": 0.8},
        ]
    )
    temporal = static.copy()
    temporal["macro_f1"] += 0.1
    temporal["pr_auc"] += 0.05

    result = temporal_effects(static, temporal, n_resamples=100)

    assert set(result["metric"]) == {"macro_f1", "pr_auc"}
    assert set(result["n_pairs"]) == {2}
    assert result.set_index("metric").loc["macro_f1", "mean_delta"] == pytest.approx(0.1)

    with pytest.raises(ValueError, match="missing temporal coordinates"):
        temporal_effects(static, temporal.iloc[:1], n_resamples=100)


def test_temporal_effects_preserves_each_paired_comparison_holm_family(monkeypatch):
    detail = pd.DataFrame(
        [
            {"target": "survival", "horizon": horizon, "run_name": "r1", "split": split, "macro_f1": 0.6, "pr_auc": 0.7}
            for horizon, split in ((3, "val"), (5, "test"))
        ]
    )

    def paired(*_args, **_kwargs):
        return pd.DataFrame(
            [
                {"metric": "macro_f1", "p_raw": 0.01, "p_holm": 0.02},
                {"metric": "pr_auc", "p_raw": 0.04, "p_holm": 0.04},
            ]
        )

    monkeypatch.setattr(temporal_artifacts, "paired_comparisons", paired)
    result = temporal_effects(detail, detail)

    assert result.groupby("metric")["p_holm"].apply(set).to_dict() == {
        "macro_f1": {0.02},
        "pr_auc": {0.04},
    }


def test_temporal_artifacts_select_the_static_edge_sage_counterpart(tmp_path):
    static = pd.DataFrame(
        [
            {
                "model_id": "logreg",
                "target": "survival",
                "horizon": 3,
                "run_name": "r1",
                "split": "test",
                "macro_f1": 0.4,
                "pr_auc": 0.5,
            },
            {
                "model_id": "edge-sage",
                "target": "survival",
                "horizon": 3,
                "run_name": "r1",
                "split": "test",
                "macro_f1": 0.6,
                "pr_auc": 0.7,
            },
        ]
    )
    temporal = pd.DataFrame(
        [
            {
                "model_id": "edge-sage-gru",
                "target": "survival",
                "horizon": 3,
                "run_name": "r1",
                "split": "test",
                "macro_f1": 0.7,
                "pr_auc": 0.8,
            }
        ]
    )

    detailed, _ = generate_temporal_artifacts(static, temporal, tmp_path, n_resamples=100)

    assert detailed["macro_f1_static"].tolist() == [0.6]
    assert (tmp_path / "temporal_static_detailed.csv").exists()
    assert (tmp_path / "temporal_static_aggregate.csv").exists()
    assert (tmp_path / "tables" / "temporal_static_effects.tex").exists()
    assert (tmp_path / "figures" / "temporal_static_comparison.png").exists()
    assert (tmp_path / "figures" / "temporal_static_comparison.pdf").exists()

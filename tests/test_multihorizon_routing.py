"""Multi-horizon routing replay contracts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.routing.multihorizon_eval import aggregate_multihorizon_results, discover_coordinates
from src.routing.predict_edges import attach_prediction_identity
from src.routing.replay_eval import evaluate_run


def test_routing_coordinates_can_use_the_same_balanced_controlled_subset(tmp_path: Path) -> None:
    for run in (
        "stress_baseline_rwp_s60001",
        "stress_baseline_gm_s60002",
        "stress_fast_rwp_s60001",
        "stress_fast_gm_s60002",
    ):
        (tmp_path / "qos" / "k1" / run / "graph_dataset").mkdir(parents=True)

    selected = discover_coordinates(tmp_path, runs_per_scenario=1)

    assert [run_root.name for _, _, run_root in selected] == [
        "stress_baseline_rwp_s60001",
        "stress_fast_rwp_s60001",
    ]


def test_replay_uses_horizon_scores_on_the_same_sessions(tmp_path: Path) -> None:
    run_name = "run_a"
    raw_root = tmp_path / "raw"
    run_root = raw_root / run_name
    run_root.mkdir(parents=True)

    rows = []
    edges_by_time = {
        0: [(0, 1), (1, 3), (0, 2), (2, 4), (4, 3)],
        1: [(0, 1), (0, 2), (2, 4), (4, 3)],
        2: [(0, 1), (0, 2), (2, 4), (4, 3)],
    }
    for time, edges in edges_by_time.items():
        for src, dst in edges:
            rows.append(
                {
                    "time": time,
                    "src": src,
                    "dst": dst,
                    "connected": 1,
                    "snr": 25.0,
                    "packet_loss": 0.01,
                    "delay": 1.0,
                }
            )
    pd.DataFrame(rows).to_csv(run_root / "edges.csv", index=False)

    split_csv = tmp_path / "time_splits.csv"
    pd.DataFrame({"time": [0], "split": ["test"]}).to_csv(split_csv, index=False)

    prediction_csv = tmp_path / "edge_sage_predictions.csv"
    pd.DataFrame(
        [
            {"time": 0, "src": 0, "dst": 1, "pred_score": 0.05},
            {"time": 0, "src": 1, "dst": 3, "pred_score": 0.05},
            {"time": 0, "src": 0, "dst": 2, "pred_score": 0.95},
            {"time": 0, "src": 2, "dst": 4, "pred_score": 0.95},
            {"time": 0, "src": 4, "dst": 3, "pred_score": 0.95},
        ]
    ).to_csv(prediction_csv, index=False)

    summary_csv, details_csv = evaluate_run(
        run_name,
        None,
        horizon=2,
        output_dir=tmp_path / "routing",
        prediction_csvs={"edge-sage": prediction_csv},
        split_csv=split_csv,
        raw_root=raw_root,
        weight_mode="neglog",
        include_olsr=False,
        target="survival",
        prediction_horizon=2,
    )

    details = pd.read_csv(details_csv)
    summary = pd.read_csv(summary_csv)
    session_columns = ["time", "src", "dst"]
    session_sets = [
        set(group[session_columns].itertuples(index=False, name=None)) for _, group in details.groupby("strategy")
    ]

    assert session_sets and all(sessions == session_sets[0] for sessions in session_sets[1:])
    pair = details[(details["src"] == 0) & (details["dst"] == 3)].set_index("strategy")
    assert pair.loc["hop", "route_lifetime"] == 0
    assert pair.loc["edge-sage", "route_lifetime"] == 2
    assert pair.loc["edge-sage", "survival_at_horizon"] == 1
    assert pair.loc["edge-sage", "realized_pdr_at_horizon"] > 0.97
    assert set(summary["strategy"]) == {"hop", "delay", "edge-sage"}
    assert set(summary["prediction_horizon"]) == {2}
    assert set(summary["target"]) == {"survival"}
    assert set(summary["weight_mode"]) == {"neglog"}


def test_attach_prediction_identity_uses_graph_edge_order(tmp_path: Path) -> None:
    test_pt = tmp_path / "test.pt"
    torch.save(
        [
            {
                "time": 7,
                "node_ids": [10, 20, 30],
                "edge_label_index": torch.tensor([[0, 1], [1, 2]]),
                "edge_label": torch.tensor([1, 0]),
            }
        ],
        test_pt,
    )
    predictions_csv = tmp_path / "test_predictions.csv"
    pd.DataFrame(
        {
            "y_true": [1, 0],
            "y_pred": [1, 1],
            "pred_score": [0.9, 0.6],
        }
    ).to_csv(predictions_csv, index=False)

    output_csv = attach_prediction_identity(predictions_csv, test_pt, tmp_path / "identified.csv")

    identified = pd.read_csv(output_csv)
    assert identified[["time", "src", "dst"]].to_dict("records") == [
        {"time": 7, "src": 10, "dst": 20},
        {"time": 7, "src": 20, "dst": 30},
    ]
    assert identified["pred_score"].tolist() == [0.9, 0.6]


def test_aggregate_multihorizon_results_pairs_runs_within_each_coordinate(tmp_path: Path) -> None:
    routing_root = tmp_path / "routing"
    metrics = {
        "route_found_rate": 1.0,
        "mean_hops": 2.0,
        "mean_e2e_delay_ms": 2.0,
        "mean_est_pdr": 0.9,
        "survival_at_1": 0.8,
        "mean_realized_pdr_t1": 0.8,
        "mean_route_changes": 1.0,
        "disconnected_rate": 0.0,
        "survival_at_horizon": 0.5,
        "mean_realized_pdr_at_horizon": 0.5,
    }
    for run_name, hop_lifetime, edge_lifetime in [("run_a", 1.0, 2.0), ("run_b", 2.0, 4.0)]:
        run_dir = routing_root / "survival" / "k2" / "neglog" / run_name
        run_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "run_name": run_name,
                    "strategy": "hop",
                    "target": "survival",
                    "prediction_horizon": 2,
                    "weight_mode": "neglog",
                    "n_sessions": 10,
                    "mean_route_lifetime": hop_lifetime,
                    **metrics,
                },
                {
                    "run_name": run_name,
                    "strategy": "edge-sage",
                    "target": "survival",
                    "prediction_horizon": 2,
                    "weight_mode": "neglog",
                    "n_sessions": 10,
                    "mean_route_lifetime": edge_lifetime,
                    **metrics,
                },
            ]
        ).to_csv(run_dir / "summary.csv", index=False)

    detailed, aggregate, paired = aggregate_multihorizon_results(routing_root, tmp_path / "reports")

    edge = aggregate[aggregate["strategy"] == "edge-sage"].iloc[0]
    lifetime = paired[
        (paired["comparator_strategy"] == "edge-sage")
        & (paired["reference_strategy"] == "hop")
        & (paired["metric"] == "mean_route_lifetime")
    ].iloc[0]
    assert len(detailed) == 4
    assert edge["n_runs"] == 2
    assert lifetime["n_pairs"] == 2
    assert lifetime["mean_delta"] == 1.5
    assert lifetime["p_holm"] == lifetime["p_raw"]

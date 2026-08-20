"""Trace-driven ns-3 closed-loop routing contracts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.routing.closed_loop import (
    aggregate_closed_loop_results,
    build_ns3_command,
    build_route_plan,
)


def test_route_plan_uses_selected_predictor_and_shared_pair(tmp_path: Path) -> None:
    run_name = "run_a"
    raw_root = tmp_path / "raw"
    run_root = raw_root / run_name
    run_root.mkdir(parents=True)
    rows = []
    for time in (0, 1, 2):
        edges = [(0, 1), (0, 2), (2, 4), (4, 3)]
        if time == 0:
            edges.insert(1, (1, 3))
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
    predictions_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"time": 0, "src": 0, "dst": 1, "pred_score": 0.05},
            {"time": 0, "src": 1, "dst": 3, "pred_score": 0.05},
            {"time": 0, "src": 0, "dst": 2, "pred_score": 0.95},
            {"time": 0, "src": 2, "dst": 4, "pred_score": 0.95},
            {"time": 0, "src": 4, "dst": 3, "pred_score": 0.95},
        ]
    ).to_csv(predictions_csv, index=False)

    learned_plan = build_route_plan(
        run_name,
        "edge-sage",
        source=0,
        destination=3,
        horizon=2,
        weight_mode="neglog",
        raw_root=raw_root,
        split_csv=split_csv,
        predictions_csv=predictions_csv,
        output_csv=tmp_path / "learned.csv",
    )
    hop_plan = build_route_plan(
        run_name,
        "hop",
        source=0,
        destination=3,
        horizon=2,
        weight_mode="neglog",
        raw_root=raw_root,
        split_csv=split_csv,
        output_csv=tmp_path / "hop.csv",
    )

    learned = pd.read_csv(learned_plan)
    hop = pd.read_csv(hop_plan)
    assert learned.loc[0, "route_path"] == "0->2->4->3"
    assert hop.loc[0, "route_path"] == "0->1->3"
    assert learned.loc[0, ["time", "source", "destination", "route_found"]].tolist() == [0, 0, 3, 1]
    assert learned.loc[0, "prediction_horizon"] == 2


def test_ns3_command_replays_original_scenario_with_measured_data_flow(tmp_path: Path) -> None:
    scenario = {
        "run_name": "run_a",
        "num_uavs": 5,
        "time_steps": 20,
        "seed": 17,
        "mobility_model": "gauss-markov",
        "x_limit": [0.0, 500.0],
        "y_limit": [0.0, 600.0],
        "z_limit": [50.0, 150.0],
        "comm_range": 240.0,
        "gauss_markov_alpha": 0.85,
        "rwp_speed_range": [3.0, 8.0],
        "tx_power_dbm": 20.0,
        "reference_path_loss_db": 40.0,
        "path_loss_exponent": 2.2,
        "noise_floor_dbm": -90.0,
        "warmup_s": 10.0,
    }

    command = build_ns3_command(
        scenario,
        binary=Path("simulation/ns3/build/uav-olsr-dataset"),
        route_plan=tmp_path / "plan.csv",
        output_dir=tmp_path / "output",
        strategy="edge-sage",
        target="survival",
        horizon=3,
        cost_mode="neglog",
        source=0,
        destination=3,
        app_rate_kbps=256.0,
        packet_size=512,
    )

    assert "--seed=17" in command
    assert "--mobility=gauss-markov" in command
    assert "--routingStrategy=edge-sage" in command
    assert "--predictionTarget=survival" in command
    assert "--predictionHorizon=3" in command
    assert "--enableDataFlow=true" in command
    assert "--appRateKbps=256.0" in command


def test_closed_loop_aggregation_pairs_strategies_by_baseline_run(tmp_path: Path) -> None:
    output_root = tmp_path / "closed_loop"
    for run_name, olsr_pdr, edge_pdr in [("run_a", 0.4, 0.6), ("run_b", 0.5, 0.8)]:
        for strategy, pdr in [("olsr", olsr_pdr), ("edge-sage", edge_pdr)]:
            output_dir = output_root / run_name / "survival" / "k3" / "neglog" / strategy
            output_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "strategy": strategy,
                        "target": "survival",
                        "horizon": 3,
                        "cost_mode": "neglog",
                        "seed": 1,
                        "source": 0,
                        "destination": 3,
                        "app_rate_kbps": 256.0,
                        "packet_size": 512,
                        "tx_packets": 100,
                        "rx_packets": int(100 * pdr),
                        "lost_packets": int(100 * (1 - pdr)),
                        "pdr": pdr,
                        "mean_delay_ms": 2.0,
                        "throughput_mbps": pdr,
                        "route_changes": 1,
                        "plan_steps": 10,
                        "route_found_rate": 1.0,
                    }
                ]
            ).to_csv(output_dir / "closed_loop_metrics.csv", index=False)

    detailed, summary, paired = aggregate_closed_loop_results(output_root, tmp_path / "reports")

    effect = paired[
        (paired["metric"] == "pdr")
        & (paired["reference_strategy"] == "olsr")
        & (paired["comparator_strategy"] == "edge-sage")
    ].iloc[0]
    assert len(detailed) == 4
    assert summary[summary["strategy"] == "edge-sage"].iloc[0]["n_runs"] == 2
    assert effect["n_pairs"] == 2
    assert effect["mean_delta"] == 0.25

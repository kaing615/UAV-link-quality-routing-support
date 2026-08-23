"""Validate extension scenario interventions from raw ns-3 artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCENARIOS = ("baseline", "ctrl", "load")
EXTENSION_SCENARIOS = ("ctrl", "load")
RUN_PATTERN = re.compile(r"^stress_(baseline|ctrl|load)_(rwp|gm)_s(\d+)$")


def _required_files(scenario: str) -> tuple[str, ...]:
    required = ("scenario.json", "nodes.csv", "edges.csv", "traffic_log.csv")
    if scenario == "load":
        return (*required, "background_flow_metrics.csv")
    return required


def _mean_connected(edges: pd.DataFrame, column: str) -> float:
    connected = edges[edges["connected"].astype(int) == 1]
    if connected.empty:
        return float("nan")
    return float(connected[column].mean())


def _pick_time(times: list[int], target: int) -> int:
    if not times:
        raise ValueError("No edge snapshots available")
    return min(times, key=lambda time: (abs(time - target), time))


def _snapshot_metrics(
    edges: pd.DataFrame,
    *,
    time: int,
    num_uavs: int,
    group_a: set[int],
    group_b: set[int],
) -> tuple[bool, int, int]:
    snapshot = edges[(edges["time"].astype(int) == time) & (edges["connected"].astype(int) == 1)]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_uavs))
    graph.add_edges_from((int(row.src), int(row.dst)) for row in snapshot.itertuples())
    components = nx.number_connected_components(graph)
    cross_group = 0
    for row in snapshot.itertuples():
        src = int(row.src)
        dst = int(row.dst)
        if (src in group_a and dst in group_b) or (src in group_b and dst in group_a):
            cross_group += 1
    return components == 1, components, cross_group


def _partition_metrics(raw_dir: Path, metadata: dict[str, object]) -> dict[str, float | bool]:
    if metadata.get("scenario_mode") != "partition-reconnect":
        return {}
    partition_start = metadata.get("partition_start_step")
    partition_end = metadata.get("partition_end_step")
    reconnect_step = metadata.get("reconnect_step")
    groups = metadata.get("partition_groups") or {}
    if partition_start is None or partition_end is None or reconnect_step is None:
        return {}
    edges = pd.read_csv(raw_dir / "edges.csv", usecols=["time", "src", "dst", "connected"])
    traffic = pd.read_csv(raw_dir / "traffic_log.csv", usecols=["time", "reachable"])
    times = sorted(edges["time"].astype(int).unique().tolist())
    group_a = {int(node) for node in groups.get("group_a", [])}
    group_b = {int(node) for node in groups.get("group_b", [])}
    pre_time = _pick_time(times, int(partition_start) - 1)
    middle_time = _pick_time(times, (int(partition_start) + int(partition_end)) // 2)
    post_time = _pick_time(times, int(reconnect_step))
    pre_connected, pre_components, pre_cross_group = _snapshot_metrics(
        edges,
        time=pre_time,
        num_uavs=int(metadata["num_uavs"]),
        group_a=group_a,
        group_b=group_b,
    )
    middle_connected, middle_components, middle_cross_group = _snapshot_metrics(
        edges,
        time=middle_time,
        num_uavs=int(metadata["num_uavs"]),
        group_a=group_a,
        group_b=group_b,
    )
    post_connected, post_components, post_cross_group = _snapshot_metrics(
        edges,
        time=post_time,
        num_uavs=int(metadata["num_uavs"]),
        group_a=group_a,
        group_b=group_b,
    )
    partition_window = traffic[
        traffic["time"].astype(int).between(int(partition_start), int(reconnect_step) - 1, inclusive="both")
    ]
    recovery_window = traffic[traffic["time"].astype(int) >= int(reconnect_step)]
    recovering = recovery_window[recovery_window["reachable"].astype(int) == 1]
    recovery_steps = float("nan")
    if not recovering.empty:
        recovery_steps = float(int(recovering["time"].astype(int).iloc[0]) - int(reconnect_step))
    return {
        "pre_time": float(pre_time),
        "middle_time": float(middle_time),
        "post_time": float(post_time),
        "pre_connected": bool(pre_connected),
        "pre_components": float(pre_components),
        "pre_cross_group_links": float(pre_cross_group),
        "middle_connected": bool(middle_connected),
        "middle_components": float(middle_components),
        "middle_cross_group_links": float(middle_cross_group),
        "post_connected": bool(post_connected),
        "post_components": float(post_components),
        "post_cross_group_links": float(post_cross_group),
        "traffic_partition_outage": bool(
            not partition_window.empty and (partition_window["reachable"].astype(int) == 0).any()
        ),
        "recovery_steps": recovery_steps,
    }


def _load_metrics(raw_dir: Path, scenario: str) -> dict[str, float]:
    if scenario != "load":
        return {}
    metrics = pd.read_csv(raw_dir / "background_flow_metrics.csv")
    aggregate = metrics["aggregate_throughput_mbps"].replace([np.inf, -np.inf], np.nan)
    fairness = metrics["jain_fairness"].replace([np.inf, -np.inf], np.nan)
    return {
        "measured_flow_count": float(len(metrics)),
        "observed_offered_rate_kbps": float(metrics["observed_offered_rate_kbps"].sum()),
        "aggregate_throughput_mbps": float(aggregate.dropna().iloc[0]) if not aggregate.dropna().empty else float("nan"),
        "jain_fairness": float(fairness.dropna().iloc[0]) if not fairness.dropna().empty else float("nan"),
    }


def collect_inventory(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_dir in sorted(path for path in raw_root.glob("stress_*") if path.is_dir()):
        match = RUN_PATTERN.match(raw_dir.name)
        if not match:
            continue
        scenario, mobility_tag, seed_text = match.groups()
        row: dict[str, object] = {
            "run_name": raw_dir.name,
            "scenario": scenario,
            "seed": int(seed_text),
            "mobility": "random-waypoint" if mobility_tag == "rwp" else "gauss-markov",
            "raw_complete": False,
            "complete": False,
        }
        required = [raw_dir / name for name in _required_files(scenario)]
        complete = all(path.exists() and path.stat().st_size > 0 for path in required)
        row["raw_complete"] = complete
        row["complete"] = complete
        if complete:
            metadata = json.loads((raw_dir / "scenario.json").read_text(encoding="utf-8"))
            edges = pd.read_csv(
                raw_dir / "edges.csv",
                usecols=["time", "src", "dst", "connected", "delay", "packet_loss"],
            )
            row.update(
                {
                    "scenario_mode": metadata.get("scenario_mode"),
                    "num_uavs": int(metadata["num_uavs"]),
                    "time_steps": int(metadata["time_steps"]),
                    "source_id": int(metadata.get("source_id", 0)),
                    "dest_id": int(metadata.get("dest_id", 0)),
                    "connected_edge_rate": float(edges["connected"].mean()),
                    "mean_connected_delay_ms": _mean_connected(edges, "delay"),
                    "mean_connected_packet_loss": _mean_connected(edges, "packet_loss"),
                }
            )
            row.update(_partition_metrics(raw_dir, metadata))
            row.update(_load_metrics(raw_dir, scenario))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        group = inventory[inventory["scenario"] == scenario]
        row: dict[str, object] = {
            "scenario": scenario,
            "runs": int(len(group)),
            "complete_runs": int(group["complete"].sum()) if not group.empty else 0,
            "unique_seeds": int(group["seed"].nunique()) if not group.empty else 0,
            "rwp_runs": int((group["mobility"] == "random-waypoint").sum()) if not group.empty else 0,
            "gm_runs": int((group["mobility"] == "gauss-markov").sum()) if not group.empty else 0,
        }
        for metric in (
            "connected_edge_rate",
            "mean_connected_delay_ms",
            "mean_connected_packet_loss",
            "observed_offered_rate_kbps",
            "aggregate_throughput_mbps",
            "jain_fairness",
            "recovery_steps",
        ):
            row[f"{metric}_mean"] = float(group[metric].mean()) if metric in group and not group.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_baseline_load(inventory: pd.DataFrame) -> pd.DataFrame:
    baseline = inventory[(inventory["scenario"] == "baseline") & inventory["complete"]].copy()
    load = inventory[(inventory["scenario"] == "load") & inventory["complete"]].copy()
    baseline = baseline.rename(
        columns={
            "run_name": "baseline_run_name",
            "connected_edge_rate": "baseline_connected_edge_rate",
            "mean_connected_delay_ms": "baseline_mean_connected_delay_ms",
            "mean_connected_packet_loss": "baseline_mean_connected_packet_loss",
        }
    )
    load = load.rename(
        columns={
            "run_name": "load_run_name",
            "connected_edge_rate": "load_connected_edge_rate",
            "mean_connected_delay_ms": "load_mean_connected_delay_ms",
            "mean_connected_packet_loss": "load_mean_connected_packet_loss",
            "measured_flow_count": "load_measured_flow_count",
            "observed_offered_rate_kbps": "load_observed_offered_rate_kbps",
            "aggregate_throughput_mbps": "load_aggregate_throughput_mbps",
            "jain_fairness": "load_jain_fairness",
        }
    )
    paired = baseline.merge(load, on=["seed", "mobility"], how="inner", validate="one_to_one")
    if paired.empty:
        return paired
    paired["delay_delta_ms"] = (
        paired["load_mean_connected_delay_ms"] - paired["baseline_mean_connected_delay_ms"]
    )
    paired["packet_loss_delta"] = (
        paired["load_mean_connected_packet_loss"] - paired["baseline_mean_connected_packet_loss"]
    )
    paired["connected_edge_rate_delta"] = (
        paired["load_connected_edge_rate"] - paired["baseline_connected_edge_rate"]
    )
    return paired


def build_checks(inventory: pd.DataFrame, expected_per_scenario: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in EXTENSION_SCENARIOS:
        group = inventory[inventory["scenario"] == scenario]
        complete_runs = int(group["complete"].sum()) if not group.empty else 0
        rows.append(
            {
                "check": f"{scenario}_complete_runs",
                "observed": complete_runs,
                "expected": expected_per_scenario,
                "status": "PASS" if complete_runs == expected_per_scenario else "FAIL",
                "details": "",
            }
        )

    paired = _paired_baseline_load(inventory)
    rows.append(
        {
            "check": "paired_baseline_load_seed_sets",
            "observed": int(len(paired)),
            "expected": expected_per_scenario,
            "status": "PASS" if len(paired) == expected_per_scenario else "FAIL",
            "details": "",
        }
    )

    ctrl = inventory[(inventory["scenario"] == "ctrl") & inventory["complete"]].copy()
    partition_pass = pd.Series(False, index=ctrl.index)
    if not ctrl.empty:
        partition_pass = (
            ctrl["pre_connected"].astype(bool)
            & ctrl["middle_components"].ge(2)
            & ctrl["middle_cross_group_links"].eq(0)
            & ctrl["post_connected"].astype(bool)
        )
    rows.append(
        {
            "check": "partition_sequence",
            "observed": int(partition_pass.sum()),
            "expected": expected_per_scenario,
            "status": "PASS" if int(partition_pass.sum()) == expected_per_scenario else "FAIL",
            "details": (
                f"middle_components_mean={ctrl['middle_components'].mean():.3f}; "
                f"middle_cross_group_links_mean={ctrl['middle_cross_group_links'].mean():.3f}"
                if not ctrl.empty
                else ""
            ),
        }
    )

    recovery_pass = pd.Series(False, index=ctrl.index)
    recovery_observed = float("nan")
    if not ctrl.empty:
        recovery_pass = ctrl["traffic_partition_outage"].astype(bool) & np.isfinite(ctrl["recovery_steps"])
        recovery_observed = float(ctrl.loc[recovery_pass, "recovery_steps"].mean()) if recovery_pass.any() else float("nan")
    rows.append(
        {
            "check": "partition_recovery",
            "observed": recovery_observed,
            "expected": "finite recovery after reconnect",
            "status": "PASS" if int(recovery_pass.sum()) == expected_per_scenario else "FAIL",
            "details": (
                f"recovered_runs={int(recovery_pass.sum())}/{expected_per_scenario}; "
                f"max_recovery_steps={ctrl['recovery_steps'].max():.3f}"
                if not ctrl.empty and ctrl["recovery_steps"].notna().any()
                else f"recovered_runs={int(recovery_pass.sum())}/{expected_per_scenario}"
            ),
        }
    )

    load = inventory[(inventory["scenario"] == "load") & inventory["complete"]].copy()
    multiple_flow_pass = pd.Series(False, index=load.index)
    if not load.empty:
        multiple_flow_pass = (
            load["measured_flow_count"].gt(1)
            & load["observed_offered_rate_kbps"].gt(0)
            & np.isfinite(load["aggregate_throughput_mbps"])
            & np.isfinite(load["jain_fairness"])
        )
    rows.append(
        {
            "check": "load_multiple_flows",
            "observed": int(multiple_flow_pass.sum()),
            "expected": expected_per_scenario,
            "status": "PASS" if int(multiple_flow_pass.sum()) == expected_per_scenario else "FAIL",
            "details": (
                f"measured_flows_mean={load['measured_flow_count'].mean():.3f}; "
                f"observed_offered_kbps_mean={load['observed_offered_rate_kbps'].mean():.3f}"
                if not load.empty
                else ""
            ),
        }
    )

    channel_effect_pass = pd.Series(False, index=paired.index)
    if not paired.empty:
        channel_effect_pass = (
            (paired["delay_delta_ms"].gt(0) | paired["packet_loss_delta"].gt(0))
            & paired["load_measured_flow_count"].gt(1)
            & paired["load_observed_offered_rate_kbps"].gt(0)
            & np.isfinite(paired["load_aggregate_throughput_mbps"])
            & np.isfinite(paired["load_jain_fairness"])
        )
    rows.append(
        {
            "check": "load_channel_effect",
            "observed": int(channel_effect_pass.sum()),
            "expected": expected_per_scenario,
            "status": "PASS" if int(channel_effect_pass.sum()) == expected_per_scenario else "FAIL",
            "details": (
                f"delay_delta_ms_mean={paired['delay_delta_ms'].mean():.3f}; "
                f"packet_loss_delta_mean={paired['packet_loss_delta'].mean():.6f}; "
                f"connected_edge_rate_delta_mean={paired['connected_edge_rate_delta'].mean():.6f}; "
                f"observed_offered_kbps_mean={paired['load_observed_offered_rate_kbps'].mean():.3f}; "
                f"aggregate_throughput_mbps_mean={paired['load_aggregate_throughput_mbps'].mean():.3f}; "
                f"jain_fairness_mean={paired['load_jain_fairness'].mean():.6f}"
                if not paired.empty
                else ""
            ),
        }
    )
    return pd.DataFrame(rows)


def _write_figure(inventory: pd.DataFrame, output_dir: Path) -> None:
    summary = summarize_inventory(inventory)
    paired = _paired_baseline_load(inventory)
    ctrl = inventory[(inventory["scenario"] == "ctrl") & inventory["complete"]]
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    scenario_summary = summary[summary["scenario"].isin(SCENARIOS)]
    axes[0, 0].bar(scenario_summary["scenario"].str.title(), scenario_summary["complete_runs"], color=["#7f7f7f", "#d62728", "#4c78a8"])
    axes[0, 0].set_title("Complete runs")
    axes[0, 0].grid(axis="y", alpha=0.25)

    if ctrl.empty:
        axes[0, 1].text(0.5, 0.5, "No ctrl runs", ha="center", va="center")
        axes[0, 1].set_axis_off()
    else:
        axes[0, 1].bar(
            ["Pre connected", "Mid components", "Post connected", "Recovery steps"],
            [
                float(ctrl["pre_connected"].astype(int).mean()),
                float(ctrl["middle_components"].mean()),
                float(ctrl["post_connected"].astype(int).mean()),
                float(ctrl["recovery_steps"].mean()),
            ],
            color=["#2ca02c", "#d62728", "#2ca02c", "#ff7f0e"],
        )
        axes[0, 1].set_title("Partition / recovery")
        axes[0, 1].grid(axis="y", alpha=0.25)

    if paired.empty:
        axes[1, 0].text(0.5, 0.5, "No baseline/load pairs", ha="center", va="center")
        axes[1, 0].set_axis_off()
    else:
        axes[1, 0].bar(
            ["Delay delta", "Loss delta", "Edge-rate delta"],
            [
                float(paired["delay_delta_ms"].mean()),
                float(paired["packet_loss_delta"].mean()),
                float(paired["connected_edge_rate_delta"].mean()),
            ],
            color=["#ff7f0e", "#d62728", "#4c78a8"],
        )
        axes[1, 0].set_title("Load vs baseline")
        axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].set_axis_off()
    if paired.empty:
        message = "No measured load pairs"
    else:
        message = (
            f"Observed offered load: {paired['load_observed_offered_rate_kbps'].mean():.1f} kbps\n"
            f"Aggregate throughput: {paired['load_aggregate_throughput_mbps'].mean():.3f} Mbps\n"
            f"Jain fairness: {paired['load_jain_fairness'].mean():.3f}"
        )
    axes[1, 1].text(0.05, 0.95, message, ha="left", va="top")
    fig.suptitle("Extension manipulation checks")
    fig.tight_layout()
    fig.savefig(figures / "extension_manipulation_checks.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "extension_manipulation_checks.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_extension_report(
    raw_root: Path,
    output_dir: Path,
    expected_per_scenario: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory = collect_inventory(raw_root)
    if inventory.empty:
        raise FileNotFoundError(f"No extension-compatible runs found under {raw_root}")
    summary = summarize_inventory(inventory)
    checks = build_checks(inventory, expected_per_scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "run_inventory.csv", index=False)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    checks.to_csv(output_dir / "manipulation_checks.csv", index=False)
    _write_figure(inventory, output_dir)

    paired = _paired_baseline_load(inventory)
    status = "PASS" if (checks["status"] == "PASS").all() else "FAIL"
    measured_load_line = "No baseline/load pairs available."
    if not paired.empty:
        measured_load_line = (
            f"Measured offered load averaged **{paired['load_observed_offered_rate_kbps'].mean():.1f} kbps**; "
            f"paired load runs changed connected-edge delay by **{paired['delay_delta_ms'].mean():.3f} ms**, "
            f"packet loss by **{paired['packet_loss_delta'].mean():.6f}**, and connected-edge rate by "
            f"**{paired['connected_edge_rate_delta'].mean():.6f}** while delivering "
            f"**{paired['load_aggregate_throughput_mbps'].mean():.3f} Mbps** aggregate throughput "
            f"with Jain fairness **{paired['load_jain_fairness'].mean():.6f}**."
        )
    report = (
        "# Extension Scenario Execution Report\n\n"
        f"Overall status: **{status}**\n\n"
        f"Expected runs per extension scenario: **{expected_per_scenario}**  \n"
        f"Inventory rows: **{len(inventory)}**\n\n"
        "## Conclusions\n\n"
        f"{measured_load_line}\n\n"
        "## Scenario summary\n\n"
        f"{summary.to_markdown(index=False)}\n\n"
        "## Acceptance checks\n\n"
        f"{checks.to_markdown(index=False)}\n"
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return inventory, summary, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_snapshots"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/controlled_scenarios/extensions"),
    )
    parser.add_argument("--expected-per-scenario", type=int, default=10)
    parser.add_argument("--fail-on-check", action="store_true")
    args = parser.parse_args()
    _, _, checks = generate_extension_report(
        args.raw_root,
        args.output_dir,
        expected_per_scenario=args.expected_per_scenario,
    )
    failed = int((checks["status"] == "FAIL").sum())
    print(f"[EXTENSIONS] {'PASS' if failed == 0 else 'FAIL'}: {failed} failed checks")
    print(f"Report: {args.output_dir / 'REPORT.md'}")
    if args.fail_on_check and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

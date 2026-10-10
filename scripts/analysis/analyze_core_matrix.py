"""Measure core interventions and held-seed prediction/routing outcomes without pooling edges as seeds."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from src.evaluation.paired_statistics import bootstrap_mean_ci
from src.routing.replay_eval import build_strategy_graph, load_raw_edges, path_valid, shortest_path
from src.utils.experiment import validate_test_predictions


def topology_series(edges: pd.DataFrame, num_uavs: int) -> tuple[pd.DataFrame, dict[int, set[tuple[int, int]]]]:
    rows, connected = [], {}
    previous = None
    for time, snapshot in edges.groupby("time", sort=True):
        links = {tuple(sorted((int(row.src), int(row.dst))))
                 for row in snapshot[snapshot["connected"] == 1].itertuples()}
        graph = nx.Graph()
        graph.add_nodes_from(range(num_uavs))
        graph.add_edges_from(links)
        union = links | previous if previous is not None else set()
        rows.append({"time": int(time), "mean_degree": 2 * len(links) / num_uavs,
                     "components": nx.number_connected_components(graph),
                     "churn": len(links ^ previous) / len(union) if union else np.nan})
        connected[int(time)] = links
        previous = links
    return pd.DataFrame(rows), connected


def manipulation_report(jobs, raw_root: Path, report: Path, *, require_complete: bool = True) -> pd.DataFrame:
    rows = []
    for job in jobs:
        directory = raw_root / job.run_name
        metadata = json.loads((directory / "scenario.json").read_text())
        if metadata.get("core_matrix") != asdict(job):
            raise ValueError(f"Scenario configuration does not match the plan: {job.run_name}")
        nodes = pd.read_csv(directory / "nodes.csv")
        edges = pd.read_csv(directory / "edges.csv")
        traffic = pd.read_csv(directory / "traffic_log.csv")
        series, _ = topology_series(edges, int(metadata["num_uavs"]))
        required_times = set(range(int(metadata["time_steps"])))
        if set(series["time"]) != required_times or set(nodes["time"]) != required_times:
            raise ValueError(f"Incomplete snapshot support: {job.run_name}")
        if not np.isfinite(nodes[["x", "y", "z", "speed"]].to_numpy()).all():
            raise ValueError(f"Nonfinite telemetry: {job.run_name}")
        row = {"run_name": job.run_name, "seed": job.seed, "family": job.family,
               "condition": job.condition, "speed_mean": nodes["speed"].mean(),
               "altitude_min": nodes["z"].min(), "altitude_max": nodes["z"].max(),
               "mean_degree": series["mean_degree"].mean(), "components": series["components"].mean(),
               "churn": series["churn"].mean(), "physical_found_rate": traffic["reachable"].mean(),
               "measured_loss": edges.loc[edges["connected"] == 1, "packet_loss"].mean(),
               "scenario_check": True}
        for key, expected in (("comm_range", job.comm_range), ("gauss_markov_alpha", job.alpha)):
            row["scenario_check"] &= bool(np.isclose(metadata[key], expected))
        row["scenario_check"] &= metadata["rwp_speed_range"] == [job.speed_min, job.speed_max]
        row["scenario_check"] &= metadata["z_limit"] == [job.z_min, job.z_max]
        row["scenario_check"] &= row["altitude_min"] >= job.z_min - .01 and row["altitude_max"] <= job.z_max + .01
        if job.flows:
            flows = pd.read_csv(directory / "background_flow_metrics.csv")
            observed = flows["observed_offered_rate_kbps"].sum()
            configured = 6000 * job.offered_fraction
            row.update(offered_fraction=job.offered_fraction, offered_rate_kbps=observed,
                       network_loss=flows["lost_packets"].sum() / max(flows["tx_packets"].sum(), 1),
                       throughput_mbps=flows["throughput_mbps"].sum(),
                       jain_fairness=flows["jain_fairness"].iloc[0],
                       queue_samples=flows["queue_samples"].sum(),
                       queue_delay_ms=np.average(flows.loc[flows["queue_samples"] > 0, "queue_delay_ms"],
                                                 weights=flows.loc[flows["queue_samples"] > 0, "queue_samples"])
                       if flows["queue_samples"].sum() else np.nan)
            row["scenario_check"] &= len(flows) == job.flows and flows["flow_index"].nunique() == job.flows
            row["scenario_check"] &= abs(observed / configured - 1) <= .02
            row["scenario_check"] &= set(flows["traffic_pattern"]) == {job.traffic_pattern}
            # A flow dropped before MAC service can have no queue observations.
            row["scenario_check"] &= bool((flows.loc[flows["rx_packets"] > 0, "queue_samples"] > 0).all())
        if job.family == "partition" and job.level != "control":
            start = int(metadata["group_separation_step"])
            reunion = int(metadata["reconnect_step"])
            middle = (start + reunion) // 2
            # Explicitly verify the intended connectivity sequence, not just event metadata.
            pre = series.loc[series["time"] == max(0, start - 1), "components"].iloc[0]
            mid = series.loc[series["time"] == middle, "components"].iloc[0]
            post = series.loc[series["time"] == min(reunion + 2, metadata["time_steps"] - 1), "components"].iloc[0]
            if job.level == "partition":
                early = series.loc[series["time"] < int(metadata["partition_start_step"]), "components"].median()
                separated = series.loc[series["time"] == start + 1, "components"].iloc[0]
                row["scenario_check"] &= early == 1 and separated >= 2 and post == 1
            else:
                row["scenario_check"] &= pre >= 2 and mid == 1 and post >= 2
            row.update(pre_components=pre, middle_components=mid, post_components=post)
        rows.append(row)
    frame = pd.DataFrame(rows)
    report.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report / "manipulation_runs.csv", index=False)
    checks = []
    for condition, group in frame.groupby("condition"):
        queue_observed_runs = int((group.get("queue_samples", pd.Series(dtype=float)) > 0).sum())
        checks.append({"condition": condition, "n_seeds": group["seed"].nunique(),
                       "queue_observed_runs": queue_observed_runs,
                       "pass": bool(group["seed"].nunique() >= 10 and group["scenario_check"].all()
                                    and (group["family"].iloc[0] != "load" or queue_observed_runs > 0))})
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(report / "manipulation_checks.csv", index=False)
    if require_complete and not check_frame["pass"].all():
        raise ValueError("Core intervention checks failed; see manipulation_checks.csv before interpreting models")
    return frame


def prediction_metrics(predictions: pd.DataFrame) -> dict:
    truth, predicted = predictions["y_true"].astype(int), predictions["y_pred"].astype(int)
    classes = truth.nunique() == 2
    negatives = truth == 0
    return {"n_samples": len(truth), "positive_ratio": truth.mean(),
            "macro_f1": f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0),
            "pr_auc": average_precision_score(truth, predictions["pred_score"]) if classes else np.nan,
            "pr_auc_defined": classes,
            "false_stable": (predicted[negatives] == 1).mean() if negatives.any() else np.nan}


def event_metrics(predictions: pd.DataFrame, metadata: dict, links: dict[int, set]) -> dict:
    events = [metadata.get(key) for key in ("partition_start_step", "group_separation_step",
                                           "partition_end_step", "reconnect_step")]
    events = [int(time) for time in events if time is not None and int(time) > 0]
    if metadata["scenario_mode"] not in ("partition-reconnect", "crossing-swarms"):
        events = []
    horizon = int(predictions["horizon"].iloc[0])
    near_event = np.zeros(len(predictions), dtype=bool)
    for event in events:
        near_event |= predictions["time"].sub(event).abs().le(horizon).to_numpy()
    selected = predictions.loc[near_event]
    groups = metadata.get("partition_groups", {})
    group_a, group_b = set(groups.get("group_a", [])), set(groups.get("group_b", []))
    critical = {}
    for time, edges in links.items():
        graph = nx.Graph(list(edges))
        cut = set(nx.bridges(graph))
        cut = {tuple(sorted(edge)) for edge in cut}
        cut |= {edge for edge in edges if (edge[0] in group_a and edge[1] in group_b) or
                                         (edge[1] in group_a and edge[0] in group_b)}
        critical[time] = cut
    stable_total, stable_detected, failures, warned, lead_times = 0, 0, 0, 0, []
    for row in predictions.itertuples():
        edge = tuple(sorted((int(row.src), int(row.dst))))
        if edge not in critical.get(int(row.time), set()):
            continue
        if row.y_true == 1:
            stable_total += 1
            stable_detected += int(row.y_pred == 1)
        breaks = [offset for offset in range(1, horizon + 1) if edge not in links.get(int(row.time) + offset, set())]
        if breaks:
            failures += 1
            if row.y_pred == 0:
                warned += 1
                lead_times.append(min(breaks))
    return {"transition_samples": len(selected),
            "transition_f1": f1_score(selected["y_true"], selected["y_pred"], labels=[0, 1],
                                      average="macro", zero_division=0) if len(selected) else np.nan,
            "bridge_stable_samples": stable_total,
            "bridge_recall": stable_detected / stable_total if stable_total else np.nan,
            "bridge_failure_samples": failures,
            "bridge_failure_recall": warned / failures if failures else np.nan,
            "bridge_lead_time_steps": float(np.mean(lead_times)) if lead_times else np.nan}


def routing_metrics(predictions: pd.DataFrame, metadata: dict, raw: dict) -> dict:
    horizon = int(predictions["horizon"].iloc[0])
    source, destination = int(metadata["source_id"]), int(metadata["dest_id"])
    scores = {(int(row.time), min(int(row.src), int(row.dst)), max(int(row.src), int(row.dst))): row.pred_score
              for row in predictions.itertuples()}
    found, lifetimes, physically_found, predicted_found = [], [], [], []
    for time in sorted(predictions["time"].unique()):
        time = int(time)
        if any(time + offset not in raw for offset in range(horizon + 1)):
            raise ValueError("Incomplete future routing window")
        edges = raw[time]
        physical = shortest_path(build_strategy_graph(edges, "hop", time, {}, 0.0, "neglog"), source, destination)
        path = shortest_path(build_strategy_graph(edges, "gnn", time, {"gnn": scores}, 0.5, "neglog"), source, destination)
        found.append(int(path is not None))
        lifetime = 0
        if physical is not None:
            physically_found.append(time)
        if path is not None:
            predicted_found.append(time)
            for offset in range(1, horizon + 1):
                if not path_valid(path, raw[time + offset]):
                    break
                lifetime += 1
        lifetimes.append(lifetime)
    result = {"route_sessions": len(found), "found_rate": float(np.mean(found)),
              "route_lifetime": float(np.mean(lifetimes))}
    if metadata["scenario_mode"] == "partition-reconnect":
        returning = int(metadata["partition_end_step"])
        physical = next((time for time in physically_found if time >= returning), None)
        prediction = next((time for time in predicted_found if physical is not None and time >= physical), None)
        result.update(physical_recovery_step=physical, predicted_recovery_step=prediction,
                      recovery_success=int(prediction is not None),
                      recovery_delay_steps=prediction - physical if prediction is not None else np.nan)
    return result


def aggregate_runs(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["family", "condition", "model_id", "target", "horizon"]
    ignored = {*keys, "seed", "run_name", "protocol", "pr_auc_defined"}
    metrics = [name for name in frame.select_dtypes(include="number") if name not in ignored]
    rows = []
    for coordinate, group in frame.groupby(keys, dropna=False):
        if group["seed"].duplicated().any():
            raise ValueError("Duplicate condition/model/coordinate/seed metrics")
        result = dict(zip(keys, coordinate, strict=True))
        result["n_seeds"] = group["seed"].nunique()
        for name in metrics:
            values = group[name].dropna().to_numpy()
            low, high = bootstrap_mean_ci(values)
            result.update({f"{name}_n_seeds": len(values), f"{name}_mean": float(values.mean()) if len(values) else np.nan,
                           f"{name}_ci95_low": low, f"{name}_ci95_high": high})
        rows.append(result)
    return pd.DataFrame(rows)


def generate_report(jobs, raw_root: Path, data_root: Path, output_root: Path, report: Path, models: list[str]) -> None:
    from src.training.baselines.common import evaluate_split
    from src.training.baselines.persistence_baseline import PersistenceModel

    manipulation = manipulation_report(jobs, raw_root, report)
    rows = []
    for job in jobs:
        directory = raw_root / job.run_name
        metadata = json.loads((directory / "scenario.json").read_text())
        _, links = topology_series(pd.read_csv(directory / "edges.csv"), int(metadata["num_uavs"]))
        raw = load_raw_edges(job.run_name, raw_root)
        for target in ("qos", "survival"):
            for horizon in (1, 2, 3, 5):
                run_root = data_root / target / f"k{horizon}" / job.run_name
                labels = pd.read_csv(run_root / "features" / "edges_labeled.csv")
                for model_id in models:
                    fold = output_root / model_id / target / f"k{horizon}" / f"seed{job.seed}"
                    path = fold / f"test_predictions_{job.run_name}.csv"
                    if model_id == "persistence":
                        model = PersistenceModel(target, float(labels["tau_snr"].iloc[0]),
                                                 float(labels["tau_loss"].iloc[0]), float(labels["tau_delay"].iloc[0]))
                        _, predictions = evaluate_split(model, model_id, "Last-state Persistence", labels, "test")
                        fold.mkdir(parents=True, exist_ok=True)
                        predictions.to_csv(path, index=False)
                    else:
                        model_metadata = json.loads((fold / "metadata.json").read_text())
                        from src.utils.experiment import seed_group
                        if any(seed_group(run) == seed_group(job.run_name) for run in model_metadata["train_runs"]):
                            raise ValueError(f"Seed leakage in {fold}")
                    path = validate_test_predictions(run_root, path, test_splits=("train", "val", "test", "purged"))
                    predictions = pd.read_csv(path).rename(columns={"label": "y_true", "pred_label": "y_pred"})
                    predictions["horizon"] = horizon
                    rows.append({"family": job.family, "condition": job.condition, "seed": job.seed,
                                 "run_name": job.run_name, "model_id": model_id, "target": target,
                                 "horizon": horizon, "protocol": "held-seed-full-run",
                                 **prediction_metrics(predictions), **event_metrics(predictions, metadata, links),
                                 **routing_metrics(predictions, metadata, raw)})
    detailed = pd.DataFrame(rows).merge(manipulation.drop(columns=["seed", "family", "condition"]),
                                        on="run_name", validate="many_to_one")
    detailed.to_csv(report / "metrics_runs.csv", index=False)
    aggregate = aggregate_runs(detailed)
    aggregate.to_csv(report / "metrics_aggregate.csv", index=False)
    if not aggregate["n_seeds"].ge(10).all():
        raise ValueError("Incomplete experimental seed coverage")
    undefined = detailed.loc[~detailed["pr_auc_defined"], ["run_name", "model_id", "target", "horizon", "positive_ratio"]]
    undefined.to_csv(report / "undefined_pr_auc.csv", index=False)
    (report / "REPORT.md").write_text(
        f"# Core matrix results\n\nStatus: COMPLETE\n\n{len(jobs)} runs; {len(jobs) // len({job.seed for job in jobs})} conditions; "
        f"{len({job.seed for job in jobs})} paired seeds/condition.\n\n"
        "Evaluation: held seed; all conditions sharing a test seed are excluded from training. "
        "Purge-boundary times are evaluated only on wholly held-out seeds.\n\n"
        "CI: bootstrap over paired seed units, never edges or sessions. "
        "PR-AUC is undefined for one-class runs; see undefined_pr_auc.csv. "
        "Bridge metrics concern observed structural bridges and intergroup cut edges; novel unseen links are outside support. "
        "Queue delay is MAC queue residence including backoff/retries/service; it is not IP end-to-end delay. "
        "Offered load is aggregate payload divided by nominal 6 Mbps PHY rate, not achieved goodput.\n",
        encoding="utf-8")
    print(f"[OK] measured core matrix: {report}", flush=True)

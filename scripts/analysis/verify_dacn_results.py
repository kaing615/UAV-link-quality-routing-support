"""Read-only verification of the core-matrix evidence; write a separate audit.

Run from the repository root with a Python environment containing the research
dependencies. No original dataset, checkpoint, prediction or report is changed.
Missing prepared graphs are reconstructed in memory from raw snapshots, not
silently treated as available historical training inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.gnn.edge_gnn import EdgeAwareSAGEEdgeClassifier

IDENTITY = ["time", "src", "dst"]
EDGE_FEATURES = ["distance", "rssi", "snr", "delay", "packet_loss", "relative_speed", "throughput"]
NODE_FEATURES = ["x", "y", "z", "vx", "vy", "vz", "degree", "load"]
SPLIT_ORDER = ("train", "val", "test", "purged")
MODELS = ("persistence", "logreg", "xgb", "edge-sage")


def digest(path: Path, normalize: bool = False) -> str:
    if normalize:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(recorded: str) -> Path | None:
    name = recorded.replace("\\", "/")
    for marker in ("/UAV-link-quality-routing-support/", "/repo/"):
        if marker in name:
            return Path(name.split(marker, 1)[1])
    return None


def verify_manifests(paths: list[Path], destination: Path) -> tuple[dict, pd.DataFrame]:
    expected = defaultdict(set)
    integrity = []
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        payload = {"config": manifest["config"], "files": manifest["files"]}
        calculated = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        integrity.append({"path": str(path), "fingerprint_valid": calculated == manifest["fingerprint"]})
        for recorded, value in manifest["files"].items():
            local = local_path(recorded)
            expected[str(local) if local is not None else recorded].add(value)
    rows = []
    for name, values in sorted(expected.items()):
        path = Path(name)
        if not path.is_file():
            status = "missing_local" if not name.startswith("/") else "external_binary_unavailable"
            current = ""
        else:
            current = digest(path)
            if current in values and len(values) == 1:
                status = "exact"
            elif path.suffix in (".py", ".md", ".sh", ".txt", ".cc") and digest(path, True) in values:
                status = "text_matches_after_crlf_normalization"
            else:
                status = "different_bytes"
        rows.append({"path": name, "status": status, "recorded_versions": len(values),
                     "current_sha256": current, "recorded_sha256": "|".join(sorted(values))})
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "file_hash_checks.csv", index=False)
    pd.DataFrame(integrity).to_csv(destination / "manifest_integrity.csv", index=False)
    return {"manifests": len(paths), "valid_fingerprints": sum(r["fingerprint_valid"] for r in integrity),
            "file_status_counts": dict(Counter(frame.status))}, frame


def reconstruct(edges: pd.DataFrame, nodes: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Independent label calculation using raw time/edge lookups."""
    current = edges[(edges.connected == 1) & (edges.time <= edges.time.max() - 5)].copy()
    lookup = edges.set_index(IDENTITY)
    index = pd.MultiIndex.from_arrays([current.time + horizon, current.src, current.dst], names=IDENTITY)
    future = lookup.reindex(index)
    if target == "qos":
        labels = ((future.connected.to_numpy() == 1) & (future.snr.to_numpy() >= 18)
                  & (future.packet_loss.to_numpy() <= .1) & (future.delay.to_numpy() <= 10))
    else:
        labels = np.ones(len(current), dtype=bool)
        for step in range(1, horizon + 1):
            index = pd.MultiIndex.from_arrays([current.time + step, current.src, current.dst], names=IDENTITY)
            labels &= lookup.connected.reindex(index).fillna(0).to_numpy() == 1
    current["label"] = labels.astype(int)
    return current.sort_values(IDENTITY).reset_index(drop=True)


def split_times(times: list[int], horizon: int) -> dict[str, list[int]]:
    n_train, n_val = int(len(times) * .7), int(len(times) * .15)
    return {"train": times[:n_train - horizon],
            "val": times[n_train:n_train + n_val - horizon],
            "test": times[n_train + n_val:],
            "purged": times[n_train - horizon:n_train] + times[n_train + n_val - horizon:n_train + n_val]}


def make_graphs(labels: pd.DataFrame, nodes: pd.DataFrame, horizon: int) -> tuple[list[dict], list[Data]]:
    node_groups = {int(t): g.sort_values("node_id") for t, g in nodes.groupby("time")}
    edge_groups = {int(t): g for t, g in labels.groupby("time", sort=True)}
    split = split_times(sorted(edge_groups), horizon)
    records, graphs = [], []
    for split_name in SPLIT_ORDER:
        for time in split[split_name]:
            n, e = node_groups[time], edge_groups[time]
            ids = n.node_id.tolist()
            positions = {node: i for i, node in enumerate(ids)}
            pairs = torch.tensor([[positions[s], positions[d]] for s, d in zip(e.src, e.dst)], dtype=torch.long).T.contiguous()
            attributes = torch.tensor(e[EDGE_FEATURES].to_numpy(), dtype=torch.float32)
            indices = torch.stack((pairs, pairs.flip(0)), dim=2).reshape(2, -1)
            attrs = attributes.repeat_interleave(2, dim=0)
            record = {"time": time, "split": split_name, "node_ids": ids,
                      "x": torch.tensor(n[NODE_FEATURES].to_numpy(), dtype=torch.float32),
                      "edge_index": indices, "edge_attr": attrs, "edge_label_index": pairs,
                      "edge_label": torch.tensor(e.label.to_numpy(), dtype=torch.long)}
            records.append(record)
            graphs.append(Data(x=record["x"], edge_index=indices, edge_attr=attrs,
                               edge_label_index=pairs, edge_label=record["edge_label"],
                               labeled_edge_attr=attributes))
    return records, graphs


def check_prepared(directory: Path, labels: pd.DataFrame, records: list[dict], horizon: int) -> dict:
    available = (directory / "features/edges_labeled.csv").is_file()
    result = {"prepared_available": available, "csv_matches_raw": None, "graph_matches_raw": None,
              "split_matches": None, "purge_no_overlap": None, "error": ""}
    if not available:
        return result
    try:
        actual = pd.read_csv(directory / "features/edges_labeled.csv").sort_values(IDENTITY).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[[*IDENTITY, "label", *EDGE_FEATURES]],
                                      labels[[*IDENTITY, "label", *EDGE_FEATURES]], check_dtype=False,
                                      atol=1e-10, rtol=1e-10)
        result["csv_matches_raw"] = True
        splits = pd.read_csv(directory / "splits/time_splits.csv")
        expected = split_times(sorted(labels.time.unique().tolist()), horizon)
        result["split_matches"] = all(sorted(splits.loc[splits.split == name, "time"].tolist()) == values
                                      for name, values in expected.items())
        result["purge_no_overlap"] = max(expected["train"]) + horizon < min(expected["val"]) and max(expected["val"]) + horizon < min(expected["test"])
        for name in SPLIT_ORDER:
            saved = torch.load(directory / "graph_dataset" / f"{name}.pt", weights_only=False, map_location="cpu")
            rebuilt = [r for r in records if r["split"] == name]
            if len(saved) != len(rebuilt):
                raise ValueError("graph count mismatch")
            for a, b in zip(saved, rebuilt, strict=True):
                if a["time"] != b["time"] or list(a["node_ids"]) != b["node_ids"]:
                    raise ValueError("graph identity mismatch")
                for key in ("x", "edge_index", "edge_attr", "edge_label_index", "edge_label"):
                    torch.testing.assert_close(a[key], b[key], rtol=1e-6, atol=1e-6)
        result["graph_matches_raw"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {str(error)[:250]}"
    return result


def normalize_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.rename(columns={"y_true": "label", "y_pred": "pred_label"})


def metric_values(frame: pd.DataFrame) -> dict:
    y, pred = frame.label.astype(int), frame.pred_label.astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n_samples": len(frame), "positive_ratio": y.mean(),
            "macro_f1": f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0),
            "pr_auc": average_precision_score(y, frame.pred_score) if y.nunique() == 2 else np.nan,
            "false_stable": fp / (tn + fp) if tn + fp else np.nan,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def compare_predictions(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    if actual.duplicated(IDENTITY).any():
        raise ValueError("duplicate prediction identity")
    actual = actual.sort_values(IDENTITY).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[[*IDENTITY, "label"]], expected[[*IDENTITY, "label"]], check_dtype=False)
    if not np.isfinite(actual.pred_score).all() or not actual.pred_score.between(0, 1).all():
        raise ValueError("invalid probability")


def holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    result = np.empty_like(values)
    result[order] = np.minimum(adjusted, 1)
    return result


def paired_effects(runs: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    signs = np.array(list(itertools.product((-1, 1), repeat=10)))
    for metric in ("macro_f1", "false_stable", "route_lifetime", "found_rate"):
        pivot = runs.pivot(index=["target", "horizon", "seed", "condition"], columns="model_id", values=metric)
        for (target, horizon), group in pivot.groupby(["target", "horizon"]):
            delta = group["edge-sage"] - group["xgb"]
            by_seed = delta.groupby("seed").mean().to_numpy()
            intervals = np.quantile(rng.choice(by_seed, (20000, len(by_seed)), replace=True).mean(axis=1), [.025, .975])
            p = float((np.abs((signs * by_seed).mean(axis=1)) >= abs(by_seed.mean()) - 1e-15).mean())
            rows.append({"metric": metric, "target": target, "horizon": horizon,
                         "delta_edge_sage_minus_xgb": by_seed.mean(), "ci95_low": intervals[0], "ci95_high": intervals[1],
                         "n_seed_units": len(by_seed), "exact_two_sided_sign_flip_p": p,
                         "conditions_won": int((delta.groupby("condition").mean() > 1e-12).sum())})
    frame = pd.DataFrame(rows)
    for metric, group in frame.groupby("metric"):
        frame.loc[group.index, "holm_p_within_metric_8_comparisons"] = holm(group.exact_two_sided_sign_flip_p.to_numpy())
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=Path("reports/core_matrix_bursty_fixed"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/core_matrix_bursty_fixed/raw"))
    parser.add_argument("--data-root", type=Path, default=Path("data/core_matrix_bursty_fixed/multihorizon"))
    parser.add_argument("--model-root", type=Path, default=Path("outputs/core_matrix_bursty_fixed"))
    parser.add_argument("--audit-root", type=Path, default=Path("reports/dacn_verification_2026-11-10"))
    parser.add_argument("--limit-runs", type=int, default=None, help="Smoke check only; no full-cohort conclusion")
    args = parser.parse_args()
    destination = args.audit_root
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    plan = pd.read_csv(args.report_root / "plan.csv")
    runs = pd.read_csv(args.report_root / "metrics_runs.csv")
    aggregate = pd.read_csv(args.report_root / "metrics_aggregate.csv")
    keys = ["run_name", "model_id", "target", "horizon"]
    if len(plan) != 210 or len(runs) != 6720 or runs.duplicated(keys).any():
        raise ValueError("Incomplete or duplicate core metric cohort")
    indexed = runs.set_index(keys)
    fold_metadata, cached_models, fold_checks = {}, {}, []
    manifest_paths = []
    for model in MODELS[1:]:
        for target in ("qos", "survival"):
            for horizon in (1, 2, 3, 5):
                for seed in sorted(plan.seed.unique()):
                    fold = args.model_root / model / target / f"k{horizon}" / f"seed{seed}"
                    metadata = json.loads((fold / "metadata.json").read_text(encoding="utf-8"))
                    manifest = json.loads((fold / "experiment.json").read_text(encoding="utf-8"))
                    command = manifest["config"]["command"]
                    expected_test = set(plan.loc[plan.seed == seed, "run_name"])
                    expected_train = set(plan.loc[plan.seed != seed, "run_name"])
                    valid = set(metadata["test_runs"]) == expected_test and set(metadata["train_runs"]) == expected_train
                    valid &= set(command[command.index("--test-runs") + 1].split(",")) == expected_test
                    valid &= set(command[command.index("--train-runs") + 1].split(",")) == expected_train
                    valid &= "--include-purged-test" in command
                    fold_metadata[(model, target, horizon, seed)] = metadata
                    checkpoint = fold / ("best_model.pt" if model == "edge-sage" else "model.pkl")
                    fold_checks.append({"model_id": model, "target": target, "horizon": horizon, "seed": seed,
                                        "n_train": len(expected_train), "n_test": len(expected_test),
                                        "metadata_and_command_cohort_match": bool(valid),
                                        "checkpoint_available": checkpoint.is_file(),
                                        "checkpoint_sha256": digest(checkpoint) if checkpoint.is_file() else ""})
                    manifest_paths.append(fold / "experiment.json")
    manifest_paths += sorted(args.raw_root.glob("*/experiment.json"))
    manifest_paths += sorted(args.data_root.glob("*/*/*/experiment.json"))
    manifest_summary, file_checks = verify_manifests(manifest_paths, destination)
    pd.DataFrame(fold_checks).to_csv(destination / "fold_checks.csv", index=False)
    print(f"[MANIFESTS] {manifest_summary}", flush=True)
    checks, prepared_checks, raw_checks, recomputed = [], [], [], []
    cohort = plan.head(args.limit_runs) if args.limit_runs else plan
    for position, job in enumerate(cohort.itertuples(), 1):
        raw_dir = args.raw_root / job.run_name
        edges, nodes = pd.read_csv(raw_dir / "edges.csv"), pd.read_csv(raw_dir / "nodes.csv")
        scenario = json.loads((raw_dir / "scenario.json").read_text())
        nodes["load"] = (nodes.degree / (nodes.groupby("time").node_id.transform("nunique") - 1).clip(lower=1)).round(6)
        raw_checks.append({"run_name": job.run_name, "seed": job.seed,
                           "scenario_plan_match": scenario["core_matrix"] == {k: getattr(job, k) for k in scenario["core_matrix"]},
                           "complete_times": set(edges.time) == set(range(scenario["time_steps"])) == set(nodes.time),
                           "duplicate_edges": int(edges.duplicated(IDENTITY).sum()),
                           "finite_input_features": bool(np.isfinite(edges[EDGE_FEATURES]).all().all() and np.isfinite(nodes[NODE_FEATURES]).all().all())})
        for target in ("qos", "survival"):
            for horizon in (1, 2, 3, 5):
                expected = reconstruct(edges, nodes, target, horizon)
                records, graphs = make_graphs(expected, nodes, horizon)
                prepared = check_prepared(args.data_root / target / f"k{horizon}" / job.run_name, expected, records, horizon)
                prepared_checks.append({"run_name": job.run_name, "target": target, "horizon": horizon, **prepared})
                graph_identity = pd.DataFrame([
                    {"time": g["time"], "src": g["node_ids"][s], "dst": g["node_ids"][d]}
                    for g in records for s, d in g["edge_label_index"].T.tolist()])
                for model in MODELS:
                    check = {"run_name": job.run_name, "model_id": model, "target": target, "horizon": horizon,
                             "cohort_and_labels_match_raw": False, "metrics_match": False,
                             "inference_match": None, "error": ""}
                    try:
                        fold = args.model_root / model / target / f"k{horizon}" / f"seed{job.seed}"
                        path = fold / f"test_predictions_{job.run_name}.csv"
                        original = normalize_predictions(path)
                        actual = original
                        if not set(IDENTITY) <= set(actual):
                            actual = normalize_predictions(path.with_name(path.stem + "_identified.csv"))
                            if not np.allclose(original.pred_score, actual.pred_score, atol=0, rtol=0):
                                raise ValueError("identified prediction scores differ from original")
                            if not np.array_equal(original.label, actual.label) or not np.array_equal(original.pred_label, actual.pred_label):
                                raise ValueError("identified labels differ from original")
                        compare_predictions(expected, actual)
                        check["cohort_and_labels_match_raw"] = True
                        values = metric_values(actual)
                        saved = indexed.loc[(job.run_name, model, target, horizon)]
                        check["metrics_match"] = all(np.isclose(values[name], saved[name], rtol=1e-8, atol=1e-10, equal_nan=True)
                                                     for name in ("n_samples", "positive_ratio", "macro_f1", "pr_auc", "false_stable"))
                        recomputed.append({**{k: check[k] for k in keys}, **values})
                        ordered = actual.sort_values(IDENTITY).reset_index(drop=True)
                        if model == "persistence":
                            predicted = np.ones(len(expected), dtype=int) if target == "survival" else ((expected.snr >= 18) & (expected.packet_loss <= .1) & (expected.delay <= 10)).astype(int).to_numpy()
                            check["inference_match"] = bool(np.array_equal(predicted, ordered.pred_label) and np.array_equal(predicted, ordered.pred_score))
                        else:
                            metadata = fold_metadata[(model, target, horizon, job.seed)]
                            check["threshold_labels_match"] = bool(np.array_equal((actual.pred_score >= metadata["threshold"]).astype(int), actual.pred_label))
                            if model == "edge-sage":
                                coordinate = (target, horizon, job.seed)
                                if coordinate not in cached_models:
                                    network = EdgeAwareSAGEEdgeClassifier(node_in_channels=8, edge_in_channels=7,
                                        hidden_channels=metadata["hidden_channels"], num_layers=metadata["num_layers"], dropout=metadata["dropout"],
                                        use_message_edge_features=metadata["use_message_edge_features"],
                                        use_decoder_edge_features=metadata["use_decoder_edge_features"])
                                    network.load_state_dict(torch.load(fold / "best_model.pt", weights_only=True, map_location="cpu"))
                                    cached_models[coordinate] = network.eval()
                                scores = []
                                with torch.inference_mode():
                                    for batch in DataLoader(graphs, batch_size=16, shuffle=False):
                                        scores.append(cached_models[coordinate](batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, batch.labeled_edge_attr).sigmoid().numpy())
                                replay = graph_identity.copy()
                                replay["score"] = np.concatenate(scores)
                                replay = replay.sort_values(IDENTITY).reset_index(drop=True)
                                difference = np.abs(replay.score.to_numpy() - ordered.pred_score.to_numpy())
                                check["max_abs_probability_error"] = float(difference.max())
                                check["label_disagreements"] = int(((replay.score >= metadata["threshold"]).astype(int) != ordered.pred_label).sum())
                                check["inference_match"] = bool(difference.max() <= 1e-4 and check["label_disagreements"] == 0)
                    except Exception as error:
                        check["error"] = f"{type(error).__name__}: {str(error)[:300]}"
                    checks.append(check)
        if position % 10 == 0 or position == len(cohort):
            print(f"[VERIFY] {position}/{len(cohort)} raw runs; {len(checks)} prediction cohorts", flush=True)
            pd.DataFrame(checks).to_csv(destination / "prediction_checks.csv", index=False)
    pd.DataFrame(raw_checks).to_csv(destination / "raw_checks.csv", index=False)
    pd.DataFrame(prepared_checks).to_csv(destination / "prepared_checks.csv", index=False)
    pd.DataFrame(recomputed).to_csv(destination / "recomputed_prediction_metrics.csv", index=False)
    frame = pd.DataFrame(checks)
    group_keys = ["family", "condition", "model_id", "target", "horizon"]
    numeric = [name for name in runs.select_dtypes(include="number") if f"{name}_mean" in aggregate.columns]
    rebuilt = runs.groupby(group_keys)[numeric].mean().sort_index()
    recorded = aggregate.set_index(group_keys).sort_index()
    aggregate_checks = [{"metric": name, "all_means_match": bool(np.allclose(rebuilt[name], recorded[f"{name}_mean"], equal_nan=True, atol=1e-10, rtol=1e-8))} for name in numeric]
    pd.DataFrame(aggregate_checks).to_csv(destination / "aggregate_checks.csv", index=False)
    effects = paired_effects(runs)
    effects.to_csv(destination / "paired_effects.csv", index=False)
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "full_cohort": args.limit_runs is None, "raw_runs_checked": len(raw_checks),
               "prediction_cohorts_checked": len(checks), "cohorts_match_raw": int(frame.cohort_and_labels_match_raw.sum()),
               "prediction_metrics_match": int(frame.metrics_match.sum()), "errors": int(frame.error.ne("").sum()),
               "inference": {model: {"checked": int(group.inference_match.notna().sum()), "matched": int(group.inference_match.eq(True).sum())}
                             for model, group in frame.groupby("model_id")},
               "folds_checked": len(fold_checks), "fold_cohorts_match": sum(c["metadata_and_command_cohort_match"] for c in fold_checks),
               "prepared_available": sum(c["prepared_available"] for c in prepared_checks),
               "prepared_csv_match": sum(c["csv_matches_raw"] is True for c in prepared_checks),
               "prepared_graph_match": sum(c["graph_matches_raw"] is True for c in prepared_checks),
               "aggregate_metrics_checked": len(aggregate_checks), "aggregate_metrics_match": sum(c["all_means_match"] for c in aggregate_checks),
               "manifests": manifest_summary,
               "limitations": ["Baseline training did not save fitted XGBoost/Logistic Regression models; their saved predictions can be verified, but historical model inference cannot be replayed.",
                               "Missing prepared datasets are reconstructed in memory from raw inputs; this does not recover historical serialized training files or their hashes.",
                               "Original simulator binary and source.sha256 from the simulation container are unavailable locally.",
                               "Routing aggregate means are checked against per-run tables; this audit does not replay packet traffic or independently recompute every routing session.",
                               "Ablation and temporal-model results are not validated by this core-matrix audit."]}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    input_paths = [args.report_root / name for name in ("plan.csv", "metrics_runs.csv", "metrics_aggregate.csv")]
    input_paths += [Path(__file__)]
    (destination / "audit_input_hashes.json").write_text(json.dumps({str(p): digest(p) for p in input_paths}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

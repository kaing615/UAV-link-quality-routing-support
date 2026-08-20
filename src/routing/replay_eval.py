from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, cast

import networkx as nx
import pandas as pd

TAU_SNR = 18.0
TAU_LOSS = 0.1
TAU_DELAY = 10.0

EPS = 1e-6


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def load_raw_edges(
    run_name: str,
    raw_root: Path = Path("data/raw_snapshots"),
) -> dict[int, dict[tuple[int, int], dict]]:
    """time -> {(u, v) -> row dict} for connected edges; includes validity flag."""
    edges = pd.read_csv(raw_root / run_name / "edges.csv")
    by_time: dict[int, dict[tuple[int, int], dict]] = {}
    for r in edges.itertuples(index=False):
        row: Any = r
        if int(row.connected) != 1:
            continue
        valid = row.snr > TAU_SNR and row.packet_loss < TAU_LOSS and (row.delay < TAU_DELAY)
        by_time.setdefault(int(row.time), {})[canonical(int(row.src), int(row.dst))] = {
            "delay": float(row.delay),
            "packet_loss": float(row.packet_loss),
            "valid": bool(valid),
        }
    return by_time


def load_prediction_scores(csv_path: Path) -> dict[tuple[int, int, int], float]:
    df = pd.read_csv(csv_path)
    required = {"time", "src", "dst", "pred_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction file {csv_path} is missing columns: {sorted(missing)}")
    if df["pred_score"].isna().any() or not df["pred_score"].between(0.0, 1.0).all():
        raise ValueError(f"Prediction scores must be finite values in [0, 1]: {csv_path}")
    scores = {}
    for r in df.itertuples(index=False):
        row: Any = r
        u, v = canonical(int(row.src), int(row.dst))
        scores[int(row.time), u, v] = float(row.pred_score)
    return scores


def load_test_times(run_name: str, split_csv: Path | None = None) -> list[int]:
    split_csv = split_csv or (Path("data/graph_dataset") / run_name / "splits" / "time_splits.csv")
    splits = pd.read_csv(split_csv)
    return sorted(int(t) for t in splits[splits["split"] == "test"]["time"])


def load_olsr_routes(
    run_name: str,
    raw_root: Path = Path("data/raw_snapshots"),
) -> tuple[tuple[int, int] | None, dict[int, list[int] | None]]:
    """Actual routes chosen by ns-3's OLSR (traffic_log.csv) for the single
    recorded (src, dst) pair: time -> node path, None when unreachable."""
    log_csv = raw_root / run_name / "traffic_log.csv"
    if not log_csv.exists():
        return (None, {})
    df = pd.read_csv(log_csv)
    if df.empty:
        return (None, {})
    pair = (int(df.iloc[0]["source"]), int(df.iloc[0]["destination"]))
    routes: dict[int, list[int] | None] = {}
    for r in df.itertuples(index=False):
        row: Any = r
        if int(row.reachable) == 1 and isinstance(row.route_path, str) and row.route_path:
            routes[int(row.time)] = [int(n) for n in row.route_path.split("->")]
        else:
            routes[int(row.time)] = None
    return (pair, routes)


def build_strategy_graph(
    edges_t: dict[tuple[int, int], dict],
    strategy: str,
    t: int,
    scores: dict[str, dict[tuple[int, int, int], float]],
    p_th: float,
    weight_mode: str = "one-minus",
) -> nx.Graph:
    if weight_mode not in {"one-minus", "neglog"}:
        raise ValueError("weight_mode must be 'one-minus' or 'neglog'")
    g = nx.Graph()
    for (u, v), attrs in edges_t.items():
        if strategy == "hop":
            w = 1.0
        elif strategy == "delay":
            w = attrs["delay"] + EPS
        else:
            score = scores[strategy].get((t, u, v), 0.5)
            if score < p_th:
                continue
            w = -math.log(max(score, EPS)) + EPS if weight_mode == "neglog" else (1.0 - score) + EPS
        g.add_edge(u, v, weight=w)
    return g


def shortest_path(g: nx.Graph, src: int, dst: int) -> list[int] | None:
    try:
        return cast(list[int], nx.dijkstra_path(g, src, dst, weight="weight"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def path_edges(path: list[int]) -> list[tuple[int, int]]:
    return [canonical(u, v) for u, v in zip(path[:-1], path[1:])]


def path_valid(path: list[int], edges_t: dict[tuple[int, int], dict]) -> bool:
    return all(e in edges_t and edges_t[e]["valid"] for e in path_edges(path))


def path_pdr(path: list[int], edges_t: dict[tuple[int, int], dict]) -> float:
    pdr = 1.0
    for e in path_edges(path):
        if e not in edges_t:
            return 0.0
        pdr *= 1.0 - edges_t[e]["packet_loss"]
    return pdr


def evaluate_run(
    run_name: str,
    gnn_predictions_csv: Path | None,
    horizon: int = 5,
    p_th: float = 0.0,
    output_dir: Path | None = None,
    strict: bool = False,
    *,
    prediction_csvs: dict[str, Path] | None = None,
    split_csv: Path | None = None,
    raw_root: Path = Path("data/raw_snapshots"),
    weight_mode: str = "one-minus",
    include_olsr: bool = True,
    target: str | None = None,
    prediction_horizon: int | None = None,
) -> tuple[Path, Path]:
    raw = load_raw_edges(run_name, raw_root)
    if not raw:
        raise ValueError(f"No connected raw edges found for {run_name}")
    max_time = max(raw.keys())
    test_times = [t for t in load_test_times(run_name, split_csv) if t + 1 <= max_time]

    if prediction_csvs is None:
        if gnn_predictions_csv is None:
            raise ValueError("gnn_predictions_csv or prediction_csvs is required")
        prediction_csvs = {"gnn": gnn_predictions_csv}
        xgb_csv = Path("outputs/baselines/xgb") / run_name / "test_predictions.csv"
        if xgb_csv.exists():
            prediction_csvs["xgb"] = xgb_csv
    scores = {strategy: load_prediction_scores(path) for strategy, path in prediction_csvs.items()}
    prediction_strategies = set(scores)
    # hop/delay ignore the p_th filter — only evaluate them on the unfiltered
    # pass (p_th == 0) and reuse those rows as the reference in sweeps.
    strategies = (["hop", "delay"] if p_th == 0.0 else []) + list(scores)

    # Routes ns-3's OLSR actually chose, for its single recorded pair.
    olsr_pair, olsr_routes = (None, {}) if p_th != 0.0 or not include_olsr else load_olsr_routes(run_name, raw_root)

    context = {
        "target": target,
        "prediction_horizon": prediction_horizon,
        "weight_mode": weight_mode,
    }

    records = []
    for t in test_times:
        edges_t = raw.get(t, {})
        if not edges_t:
            continue
        h_t = min(horizon, max_time - t)
        base_graph = nx.Graph(list(edges_t.keys()))
        nodes = sorted(base_graph.nodes())
        pairs = [(s, d) for i, s in enumerate(nodes) for d in nodes[i + 1 :] if nx.has_path(base_graph, s, d)]
        if olsr_pair is not None and olsr_pair not in set(pairs):
            pairs.append(olsr_pair)

        graphs = {st: build_strategy_graph(edges_t, st, t, scores, p_th, weight_mode) for st in strategies}

        for s, d in pairs:
            for st in strategies:
                path = shortest_path(graphs[st], s, d)
                if path is None and st in prediction_strategies and p_th > 0.0 and not strict:
                    # p_th filter disconnected the pair: fall back to unfiltered
                    path = shortest_path(build_strategy_graph(edges_t, st, t, scores, 0.0, weight_mode), s, d)
                if path is None:
                    records.append(
                        {
                            "time": t,
                            "src": s,
                            "dst": d,
                            "strategy": st,
                            "route_found": 0,
                            **context,
                        }
                    )
                    continue
                lifetime = 0
                for k in range(1, h_t + 1):
                    if path_valid(path, raw.get(t + k, {})):
                        lifetime += 1
                    else:
                        break
                cur = path
                changes = 0
                disconnected = 0
                for k in range(1, h_t + 1):
                    tk = t + k
                    edges_tk = raw.get(tk, {})
                    if path_valid(cur, edges_tk):
                        continue
                    changes += 1
                    new_path = shortest_path(build_strategy_graph(edges_tk, st, tk, scores, p_th, weight_mode), s, d)
                    if new_path is None:
                        disconnected = 1
                        break
                    cur = new_path
                records.append(
                    {
                        "time": t,
                        "src": s,
                        "dst": d,
                        "strategy": st,
                        "route_found": 1,
                        "hops": len(path) - 1,
                        "e2e_delay_ms": sum(edges_t[e]["delay"] for e in path_edges(path)),
                        "est_pdr": path_pdr(path, edges_t),
                        "horizon": h_t,
                        "route_lifetime": lifetime,
                        "survival_at_1": int(lifetime >= 1),
                        "survival_at_horizon": int(lifetime >= h_t),
                        "realized_pdr_t1": path_pdr(path, raw.get(t + 1, {}))
                        if path_valid(path, raw.get(t + 1, {}))
                        else 0.0,
                        "realized_pdr_at_horizon": path_pdr(path, raw.get(t + h_t, {}))
                        if path_valid(path, raw.get(t + h_t, {}))
                        else 0.0,
                        "route_changes": changes,
                        "disconnected": disconnected,
                        **context,
                    }
                )
    if olsr_pair is not None:
        strategies.append("olsr")
        s, d = olsr_pair
        for t in test_times:
            edges_t = raw.get(t, {})
            if not edges_t:
                continue
            h_t = min(horizon, max_time - t)
            path = olsr_routes.get(t)
            if not path:
                records.append(
                    {
                        "time": t,
                        "src": s,
                        "dst": d,
                        "strategy": "olsr",
                        "route_found": 0,
                        **context,
                    }
                )
                continue
            lifetime = 0
            for k in range(1, h_t + 1):
                if path_valid(path, raw.get(t + k, {})):
                    lifetime += 1
                else:
                    break
            changes = 0
            disconnected = 0
            prev = path
            for k in range(1, h_t + 1):
                nxt = olsr_routes.get(t + k)
                if not nxt:
                    disconnected = 1
                    break
                if nxt != prev:
                    changes += 1
                prev = nxt
            present = [e for e in path_edges(path) if e in edges_t]
            records.append(
                {
                    "time": t,
                    "src": s,
                    "dst": d,
                    "strategy": "olsr",
                    "route_found": 1,
                    "hops": len(path) - 1,
                    "e2e_delay_ms": sum(edges_t[e]["delay"] for e in present)
                    if len(present) == len(path) - 1
                    else float("nan"),
                    "est_pdr": path_pdr(path, edges_t),
                    "horizon": h_t,
                    "route_lifetime": lifetime,
                    "survival_at_1": int(lifetime >= 1),
                    "survival_at_horizon": int(lifetime >= h_t),
                    "realized_pdr_t1": path_pdr(path, raw.get(t + 1, {}))
                    if path_valid(path, raw.get(t + 1, {}))
                    else 0.0,
                    "realized_pdr_at_horizon": path_pdr(path, raw.get(t + h_t, {}))
                    if path_valid(path, raw.get(t + h_t, {}))
                    else 0.0,
                    "route_changes": changes,
                    "disconnected": disconnected,
                    **context,
                }
            )
    details = pd.DataFrame(records)
    output_dir = output_dir or Path("outputs/routing") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if p_th == 0.0 else f"_pth{p_th:.2f}"
    details_csv = output_dir / f"details{suffix}.csv"
    details.to_csv(details_csv, index=False)

    def summarize(det: pd.DataFrame) -> pd.DataFrame:
        found = det[det["route_found"] == 1]
        rows = []
        for st in strategies:
            sub = found[found["strategy"] == st]
            all_st = det[det["strategy"] == st]
            if all_st.empty:
                continue

            def m(col: str) -> float:
                return float(sub[col].mean()) if not sub.empty else float("nan")

            rows.append(
                {
                    "run_name": run_name,
                    "strategy": st,
                    "p_th": p_th,
                    "n_sessions": int(len(all_st)),
                    "route_found_rate": float(len(sub) / len(all_st)),
                    "mean_hops": m("hops"),
                    "mean_e2e_delay_ms": m("e2e_delay_ms"),
                    "mean_est_pdr": m("est_pdr"),
                    "mean_route_lifetime": m("route_lifetime"),
                    "survival_at_1": m("survival_at_1"),
                    "survival_at_horizon": m("survival_at_horizon"),
                    "mean_realized_pdr_t1": m("realized_pdr_t1"),
                    "mean_realized_pdr_at_horizon": m("realized_pdr_at_horizon"),
                    "mean_route_changes": m("route_changes"),
                    "disconnected_rate": m("disconnected"),
                    "mean_horizon": m("horizon"),
                    **context,
                }
            )
        return pd.DataFrame(rows)

    summary = summarize(details)
    summary_csv = output_dir / f"summary{suffix}.csv"
    summary.to_csv(summary_csv, index=False)
    if olsr_pair is not None:
        s0, d0 = olsr_pair
        same_pair = details[(details["src"] == s0) & (details["dst"] == d0)]
        summarize(same_pair).to_csv(output_dir / "summary_olsr_pair.csv", index=False)
    return (summary_csv, details_csv)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay-based routing evaluation on test snapshots.")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument(
        "--gnn-predictions",
        type=Path,
        default=None,
        help="edge_predictions CSV (default: outputs/routing/<RUN>/edge_predictions_edge-sage.csv)",
    )
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument(
        "--p-th",
        type=str,
        default="0.0",
        help="Threshold(s) excluding links with predicted stability below the value; comma-separated for a sweep, e.g. '0.3,0.5,0.7'",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="No fallback to the unfiltered graph when p_th disconnects a pair (route_found_rate then measures the connectivity cost)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gnn_csv = args.gnn_predictions or Path("outputs/routing") / args.run_name / "edge_predictions_edge-sage.csv"
    for p_th in [float(v) for v in args.p_th.split(",")]:
        summary_csv, details_csv = evaluate_run(
            args.run_name, gnn_csv, horizon=args.horizon, p_th=p_th, strict=args.strict
        )
        print(f"[OK] summary: {summary_csv}")
        print(pd.read_csv(summary_csv).to_string(index=False))

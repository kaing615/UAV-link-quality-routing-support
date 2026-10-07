"""Read-only audit reproductions; generated inputs/results live in a temporary directory."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.preprocessing.gnn.build_labels import assign_label
from src.routing.replay_eval import evaluate_run, load_raw_edges


def run_case(root, name, rows, horizon, threshold=0.0):
    raw = root / name
    raw.mkdir()
    pd.DataFrame(rows).to_csv(raw / "edges.csv", index=False)
    split = root / f"{name}_split.csv"
    pd.DataFrame({"time": [0], "split": ["test"]}).to_csv(split, index=False)
    predictions = root / f"{name}_scores.csv"
    pd.DataFrame(
        [{"time": row["time"], "src": 0, "dst": 1, "pred_score": 0.1} for row in rows]
    ).to_csv(predictions, index=False)
    _, detail_path = evaluate_run(
        name,
        None,
        horizon=horizon,
        p_th=threshold,
        output_dir=root / f"{name}_out",
        prediction_csvs={"model": predictions},
        split_csv=split,
        raw_root=root,
        include_olsr=False,
    )
    details = pd.read_csv(detail_path)
    return details[details["strategy"] == "model"].iloc[0]


def main():
    with TemporaryDirectory(prefix="uav-routing-audit-") as temporary:
        root = Path(temporary)
        def edge(time, loss=0.01, connected=1):
            return dict(time=time, src=0, dst=1, connected=connected, snr=25.0,
                        packet_loss=loss, delay=1.0)

        boundary_dir = root / "boundary"
        boundary_dir.mkdir()
        pd.DataFrame([edge(0, loss=0.1)]).to_csv(boundary_dir / "edges.csv", index=False)
        boundary = {
            "training_label": assign_label(1, 25.0, 0.1, 1.0, 18.0, 0.1, 10.0),
            "replay_valid": load_raw_edges("boundary", root)[0][(0, 1)]["valid"],
        }
        assert boundary == {"training_label": 1, "replay_valid": False}

        # All raw times exist, but final disconnected time disappears from max(raw).
        tail = run_case(root, "tail", [edge(0), edge(1), edge(2, connected=0)], 2)
        assert tail["horizon"] == 1 and tail["survival_at_horizon"] == 1

        # Connected topology remains available; non-strict fallback is lost on replanning.
        rows = [edge(time, loss=0.2) for time in range(3)]
        baseline = run_case(root, "unfiltered", rows, 2)
        filtered = run_case(root, "fallback", rows, 2, threshold=0.3)
        assert baseline["disconnected"] == 0 and baseline["route_changes"] == 2
        assert filtered["route_found"] == 1 and filtered["disconnected"] == 1
        assert filtered["route_changes"] == 1

        print(json.dumps({
            "boundary": boundary,
            "tail": {"requested_horizon": 2, "reported_horizon": int(tail["horizon"]),
                     "reported_survival_at_horizon": int(tail["survival_at_horizon"]),
                     "actual_t2_connected": 0},
            "non_strict_threshold": {
                "unfiltered_route_changes": int(baseline["route_changes"]),
                "filtered_route_changes": int(filtered["route_changes"]),
                "filtered_disconnected": int(filtered["disconnected"]),
                "topology_always_connected": True,
            },
        }, indent=2))


if __name__ == "__main__":
    main()

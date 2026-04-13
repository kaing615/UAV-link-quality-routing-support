from __future__ import annotations

import json

import config
import mobility
from config_utils import validate_config
from entities import initialize_uavs
from io_utils import make_edge_rows, make_node_rows
from pipeline_io import ensure_output_dir, save_all_outputs
from routing import dijkstra_shortest_path
from topology import build_topology
from visualization import (
    draw_live_scene,
    finalize_live_plot,
    save_topology_snapshot,
    setup_live_plot,
    snapshot_steps,
)


def save_scenario_metadata() -> None:
    scenario = {
        "run_name": config.RUN_NAME,
        "seed": config.SEED,
        "mobility_model": config.MOBILITY_MODEL,
        "num_uavs": config.NUM_UAVS,
        "time_steps": config.TIME_STEPS,
        "dt": config.DT,
        "x_limit": list(config.X_LIMIT),
        "y_limit": list(config.Y_LIMIT),
        "z_limit": list(config.Z_LIMIT),
        "velocity_component_range": list(config.VELOCITY_COMPONENT_RANGE),
        "comm_range": config.COMM_RANGE,
        "source_id": config.SOURCE_ID,
        "dest_id": config.DEST_ID,
        "gauss_markov_alpha": config.GAUSS_MARKOV_ALPHA,
        "gauss_markov_mean_velocity": list(config.GAUSS_MARKOV_MEAN_VELOCITY),
        "gauss_markov_stddev": config.GAUSS_MARKOV_STDDEV,
        "rwp_speed_range": list(config.RWP_SPEED_RANGE),
        "rwp_pause_steps": config.RWP_PAUSE_STEPS,
        "rwp_reach_threshold": config.RWP_REACH_THRESHOLD,
        "tx_power_dbm": config.TX_POWER_DBM,
        "reference_path_loss_db": config.REFERENCE_PATH_LOSS_DB,
        "path_loss_exponent": config.PATH_LOSS_EXPONENT,
        "noise_floor_dbm": config.NOISE_FLOOR_DBM,
        "base_delay_ms": config.BASE_DELAY_MS,
        "disconnected_delay_ms": config.DISCONNECTED_DELAY_MS,
        "max_throughput_mbps": config.MAX_THROUGHPUT_MBPS,
        "output_dir": str(config.OUTPUT_DIR),
        "plots_dir": str(config.PLOTS_DIR),
    }

    scenario_path = config.OUTPUT_DIR / "scenario.json"
    scenario_path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    print(f"- Scenario JSON  : {scenario_path}")


def main() -> None:
    ensure_output_dir()
    validate_config()
    save_scenario_metadata()

    print(
        f"[RUN] name={config.RUN_NAME}, seed={config.SEED}, "
        f"mobility={config.MOBILITY_MODEL}, output={config.OUTPUT_DIR}"
    )

    uavs = initialize_uavs()

    all_node_rows: list[dict] = []
    all_edge_rows: list[dict] = []
    traffic_rows: list[dict] = []

    fig, ax = setup_live_plot()

    try:
        for t in range(config.TIME_STEPS):
            edges, degree_map, adjacency, weighted_adjacency = build_topology(uavs)
            connected_edge_count = sum(edge["connected"] for edge in edges)

            route_path = dijkstra_shortest_path(
                weighted_adjacency,
                config.SOURCE_ID,
                config.DEST_ID,
            )
            reachable = route_path is not None

            node_rows = make_node_rows(t, uavs, degree_map)
            edge_rows = make_edge_rows(t, edges)

            all_node_rows.extend(node_rows)
            all_edge_rows.extend(edge_rows)

            traffic_rows.append(
                {
                    "time": t,
                    "source": config.SOURCE_ID,
                    "destination": config.DEST_ID,
                    "reachable": int(reachable),
                    "route_path": "" if route_path is None else "->".join(map(str, route_path)),
                    "num_edges": connected_edge_count,
                }
            )

            if t % config.PRINT_EVERY == 0 or t == config.TIME_STEPS - 1:
                route_text = "None" if route_path is None else " -> ".join(map(str, route_path))
                print(
                    f"[t={t:03d}] edges={connected_edge_count:02d}, "
                    f"reachable={int(reachable)}, route={route_text}"
                )

            draw_live_scene(ax, uavs, edges, reachable, t, route_path)

            if getattr(config, "SAVE_PLOTS", False) and t in snapshot_steps:
                save_topology_snapshot(uavs, edges, reachable, t, route_path)

            mobility.update_positions(uavs, config.DT)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    finally:
        save_all_outputs(all_node_rows, all_edge_rows, traffic_rows)
        finalize_live_plot(ax)

        import matplotlib.pyplot as plt
        plt.close(fig)
        plt.close("all")


if __name__ == "__main__":
    main()

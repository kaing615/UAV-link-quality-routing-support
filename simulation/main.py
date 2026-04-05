from __future__ import annotations

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


def main() -> None:
    ensure_output_dir()
    validate_config()

    uavs = initialize_uavs()

    all_node_rows: list[dict] = []
    all_edge_rows: list[dict] = []
    traffic_rows: list[dict] = []

    _, ax = setup_live_plot()

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


if __name__ == "__main__":
    main()

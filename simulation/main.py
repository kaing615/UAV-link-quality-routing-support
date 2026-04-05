from __future__ import annotations

import csv
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons, RadioButtons, Slider

import config
import mobility


@dataclass
class UAV:
    node_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)


def ensure_output_dir() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if getattr(config, "SAVE_PLOTS", False):
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def random_nonzero_velocity() -> tuple[float, float, float]:
    low, high = config.VELOCITY_COMPONENT_RANGE
    while True:
        vx = random.uniform(low, high)
        vy = random.uniform(low, high)
        vz = random.uniform(low, high)
        if abs(vx) + abs(vy) + abs(vz) > 1e-9:
            return vx, vy, vz


def initialize_uavs() -> list[UAV]:
    random.seed(config.SEED)
    uavs: list[UAV] = []

    for i in range(config.NUM_UAVS):
        x = random.uniform(*config.X_LIMIT)
        y = random.uniform(*config.Y_LIMIT)
        z = random.uniform(*config.Z_LIMIT)
        vx, vy, vz = random_nonzero_velocity()

        uavs.append(
            UAV(
                node_id=i,
                x=x,
                y=y,
                z=z,
                vx=vx,
                vy=vy,
                vz=vz,
            )
        )

    return uavs


def euclidean_distance_3d(a: UAV, b: UAV) -> float:
    return math.sqrt(
        (a.x - b.x) ** 2
        + (a.y - b.y) ** 2
        + (a.z - b.z) ** 2
    )


def compute_relative_speed(a: UAV, b: UAV) -> float:
    dx = b.x - a.x
    dy = b.y - a.y
    dz = b.z - a.z
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

    if distance <= 1e-9:
        return 0.0

    ux = dx / distance
    uy = dy / distance
    uz = dz / distance

    dvx = b.vx - a.vx
    dvy = b.vy - a.vy
    dvz = b.vz - a.vz

    radial_speed = dvx * ux + dvy * uy + dvz * uz
    return abs(radial_speed)


def estimate_rssi(distance: float) -> float:
    tx_power_dbm = float(getattr(config, "TX_POWER_DBM", 20.0))
    reference_loss_db = float(getattr(config, "REFERENCE_PATH_LOSS_DB", 40.0))
    path_loss_exponent = float(getattr(config, "PATH_LOSS_EXPONENT", 2.2))

    d = max(distance, 1.0)
    path_loss_db = reference_loss_db + 10.0 * path_loss_exponent * math.log10(d)
    return tx_power_dbm - path_loss_db


def estimate_snr(rssi: float) -> float:
    noise_floor_dbm = float(getattr(config, "NOISE_FLOOR_DBM", -90.0))
    return rssi - noise_floor_dbm


def estimate_delay(
    distance: float,
    relative_speed: float,
    connected: int,
    load_factor: float,
) -> float:
    propagation_ms = distance / 300_000_000.0 * 1000.0
    base_processing_ms = float(getattr(config, "BASE_DELAY_MS", 2.0))
    mobility_penalty_ms = 0.08 * relative_speed
    contention_penalty_ms = 4.0 * load_factor
    disconnected_penalty_ms = (
        float(getattr(config, "DISCONNECTED_DELAY_MS", 50.0)) if not connected else 0.0
    )

    delay = (
        base_processing_ms
        + propagation_ms
        + mobility_penalty_ms
        + contention_penalty_ms
        + disconnected_penalty_ms
    )
    return round(delay, 4)


def estimate_packet_loss(snr: float, connected: int, load_factor: float) -> float:
    if not connected:
        return 1.0

    if snr >= 25.0:
        loss = 0.01
    elif snr >= 18.0:
        loss = 0.03
    elif snr >= 12.0:
        loss = 0.08
    elif snr >= 8.0:
        loss = 0.18
    else:
        loss = 0.35

    loss += 0.10 * load_factor
    return round(min(max(loss, 0.0), 1.0), 4)


def estimate_throughput(
    snr: float,
    packet_loss: float,
    connected: int,
    load_factor: float,
) -> float:
    max_rate_mbps = float(getattr(config, "MAX_THROUGHPUT_MBPS", 100.0))

    if not connected:
        return 0.0

    snr_eff = min(max(snr, 0.0), 30.0) / 30.0
    throughput = max_rate_mbps * snr_eff * (1.0 - packet_loss) * (1.0 - 0.35 * load_factor)
    return round(max(throughput, 0.0), 4)


def build_topology(uavs: list[UAV]) -> tuple[list[dict], dict[int, int], dict[int, list[int]]]:
    edges: list[dict] = []
    degree_map = {uav.node_id: 0 for uav in uavs}
    adjacency = {uav.node_id: [] for uav in uavs}

    for i in range(len(uavs)):
        for j in range(i + 1, len(uavs)):
            a = uavs[i]
            b = uavs[j]

            distance = euclidean_distance_3d(a, b)
            connected = int(distance <= config.COMM_RANGE)

            if connected:
                degree_map[a.node_id] += 1
                degree_map[b.node_id] += 1
                adjacency[a.node_id].append(b.node_id)
                adjacency[b.node_id].append(a.node_id)

            edges.append(
                {
                    "src": a.node_id,
                    "dst": b.node_id,
                    "distance": distance,
                    "connected": connected,
                }
            )

    max_degree = max(config.NUM_UAVS - 1, 1)
    uav_map = {uav.node_id: uav for uav in uavs}

    for edge in edges:
        a = uav_map[edge["src"]]
        b = uav_map[edge["dst"]]

        relative_speed = compute_relative_speed(a, b)
        rssi = estimate_rssi(edge["distance"])
        snr = estimate_snr(rssi)

        load_factor = (degree_map[a.node_id] + degree_map[b.node_id]) / (2.0 * max_degree)

        delay = estimate_delay(edge["distance"], relative_speed, edge["connected"], load_factor)
        packet_loss = estimate_packet_loss(snr, edge["connected"], load_factor)
        throughput = estimate_throughput(snr, packet_loss, edge["connected"], load_factor)

        edge["distance"] = round(edge["distance"], 4)
        edge["relative_speed"] = round(relative_speed, 4)
        edge["rssi"] = round(rssi, 4)
        edge["snr"] = round(snr, 4)
        edge["delay"] = delay
        edge["packet_loss"] = packet_loss
        edge["throughput"] = throughput

    return edges, degree_map, adjacency


def shortest_path(adjacency: dict[int, list[int]], source: int, destination: int) -> list[int] | None:
    if source == destination:
        return [source]

    queue = deque([source])
    parent: dict[int, int | None] = {source: None}

    while queue:
        current = queue.popleft()

        for neighbor in adjacency.get(current, []):
            if neighbor not in parent:
                parent[neighbor] = current
                if neighbor == destination:
                    path = [destination]
                    node = destination
                    while parent[node] is not None:
                        node = parent[node]
                        path.append(node)
                    path.reverse()
                    return path
                queue.append(neighbor)

    return None


def make_node_rows(time_step: int, uavs: list[UAV], degree_map: dict[int, int]) -> list[dict]:
    rows: list[dict] = []

    for uav in uavs:
        rows.append(
            {
                "time": time_step,
                "node_id": uav.node_id,
                "x": round(uav.x, 4),
                "y": round(uav.y, 4),
                "z": round(uav.z, 4),
                "vx": round(uav.vx, 4),
                "vy": round(uav.vy, 4),
                "vz": round(uav.vz, 4),
                "speed": round(uav.speed, 4),
                "degree": degree_map[uav.node_id],
            }
        )

    return rows


def make_edge_rows(time_step: int, edges: list[dict]) -> list[dict]:
    rows: list[dict] = []

    for edge in edges:
        rows.append(
            {
                "time": time_step,
                "src": edge["src"],
                "dst": edge["dst"],
                "distance": edge["distance"],
                "connected": edge["connected"],
                "relative_speed": edge["relative_speed"],
                "rssi": edge["rssi"],
                "snr": edge["snr"],
                "delay": edge["delay"],
                "packet_loss": edge["packet_loss"],
                "throughput": edge["throughput"],
            }
        )

    return rows


def write_csv(path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


live_view_state = {
    "mode": getattr(config, "VIEW_MODE", "iso").lower(),
    "rotate": getattr(config, "ROTATE_CAMERA", True),
    "azim_step": float(getattr(config, "CAMERA_AZIM_STEP", 0.2)),
}

_ui_refs: dict[str, object] = {}


def _set_rotate_state(value: bool) -> None:
    current = live_view_state["rotate"]
    if current == value:
        return

    live_view_state["rotate"] = value

    check = _ui_refs.get("check")
    if check is not None:
        status = check.get_status()[0]
        if status != value:
            check.set_active(0)


def set_view_mode(label: str) -> None:
    live_view_state["mode"] = label


def toggle_rotate(_label: str) -> None:
    check = _ui_refs.get("check")
    if check is not None:
        live_view_state["rotate"] = check.get_status()[0]
    else:
        live_view_state["rotate"] = not live_view_state["rotate"]


def update_rotation_speed(value: float) -> None:
    live_view_state["azim_step"] = float(value)


def on_key_press(event) -> None:
    radio = _ui_refs.get("radio")

    if event.key == "1":
        live_view_state["mode"] = "iso"
        if radio is not None:
            radio.set_active(0)
    elif event.key == "2":
        live_view_state["mode"] = "top_z"
        if radio is not None:
            radio.set_active(1)
    elif event.key == "3":
        live_view_state["mode"] = "view_x"
        if radio is not None:
            radio.set_active(2)
    elif event.key == "4":
        live_view_state["mode"] = "view_y"
        if radio is not None:
            radio.set_active(3)
    elif event.key == "r":
        _set_rotate_state(not live_view_state["rotate"])


def setup_live_plot():
    if not config.LIVE_SIMULATION:
        return None, None

    plt.ion()
    fig = plt.figure(figsize=config.FIGSIZE)

    ax = fig.add_axes([0.05, 0.08, 0.72, 0.84], projection="3d")

    ax_radio = fig.add_axes([0.80, 0.62, 0.17, 0.20])
    ax_check = fig.add_axes([0.80, 0.50, 0.17, 0.08])
    ax_slider = fig.add_axes([0.82, 0.42, 0.12, 0.03])

    view_options = ("iso", "top_z", "view_x", "view_y")
    active_index = view_options.index(live_view_state["mode"]) if live_view_state["mode"] in view_options else 0

    radio = RadioButtons(ax_radio, view_options, active=active_index)
    check = CheckButtons(ax_check, ["rotate"], [live_view_state["rotate"]])
    slider = Slider(
        ax=ax_slider,
        label="rot speed",
        valmin=0.0,
        valmax=1.0,
        valinit=live_view_state["azim_step"],
        valstep=0.05,
    )

    radio.on_clicked(set_view_mode)
    check.on_clicked(toggle_rotate)
    slider.on_changed(update_rotation_speed)

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    _ui_refs["radio"] = radio
    _ui_refs["check"] = check
    _ui_refs["slider"] = slider

    ax_radio.set_title("View", fontsize=10)
    ax_check.set_title("Options", fontsize=10)

    return fig, ax


snapshot_steps = {
    step for step in getattr(config, "TOPOLOGY_SNAPSHOT_STEPS", [])
    if 0 <= step < config.TIME_STEPS
}


trail_history: dict[int, deque[tuple[float, float, float]]] = defaultdict(
    lambda: deque(maxlen=getattr(config, "TRAIL_LENGTH", 12))
)


def update_trails(uavs: list[UAV]) -> None:
    if not getattr(config, "SHOW_TRAILS", True):
        return

    for uav in uavs:
        trail_history[uav.node_id].append((uav.x, uav.y, uav.z))


def apply_axes_style(
    ax,
    time_step: int,
    force_view_mode: str | None = None,
    force_rotate: bool | None = None,
) -> None:
    if getattr(config, "DARK_THEME", False):
        ax.set_facecolor("#0f172a")
        try:
            ax.xaxis.pane.set_facecolor((0.07, 0.10, 0.18, 1.0))
            ax.yaxis.pane.set_facecolor((0.07, 0.10, 0.18, 1.0))
            ax.zaxis.pane.set_facecolor((0.07, 0.10, 0.18, 1.0))
        except Exception:
            pass
        label_color = "white"
        grid_color = "#64748b"
    else:
        ax.set_facecolor("#f8fafc")
        try:
            ax.xaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
            ax.yaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
            ax.zaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
        except Exception:
            pass
        label_color = "#0f172a"
        grid_color = "#94a3b8"

    ax.set_xlim(config.X_LIMIT)
    ax.set_ylim(config.Y_LIMIT)
    ax.set_zlim(config.Z_LIMIT)
    ax.set_box_aspect(
        (
            config.X_LIMIT[1] - config.X_LIMIT[0],
            config.Y_LIMIT[1] - config.Y_LIMIT[0],
            config.Z_LIMIT[1] - config.Z_LIMIT[0],
        )
    )

    ax.set_xlabel("X (m)", color=label_color, labelpad=10)
    ax.set_ylabel("Y (m)", color=label_color, labelpad=10)
    ax.set_zlabel("Z (m)", color=label_color, labelpad=10)
    ax.tick_params(colors=label_color)

    if getattr(config, "SHOW_GRID", True):
        ax.grid(True, alpha=0.35, color=grid_color)
    else:
        ax.grid(False)

    view_mode = force_view_mode if force_view_mode is not None else live_view_state["mode"]
    rotate_enabled = force_rotate if force_rotate is not None else live_view_state["rotate"]

    if view_mode == "top_z":
        elev, azim = 90, -90
    elif view_mode == "view_x":
        elev, azim = 0, 0
    elif view_mode == "view_y":
        elev, azim = 0, 90
    else:
        elev = getattr(config, "CAMERA_ELEV", 24)
        azim = getattr(config, "CAMERA_AZIM", 38)

    if rotate_enabled:
        azim += time_step * live_view_state["azim_step"]

    ax.view_init(elev=elev, azim=azim)


def draw_trails(ax, uavs: list[UAV]) -> None:
    if not getattr(config, "SHOW_TRAILS", True):
        return

    for uav in uavs:
        trail = list(trail_history[uav.node_id])
        if len(trail) < 2:
            continue

        xs = [p[0] for p in trail]
        ys = [p[1] for p in trail]
        zs = [p[2] for p in trail]
        ax.plot(xs, ys, zs, linestyle="--", linewidth=1.4, alpha=0.35)


def edge_color_from_quality(edge: dict) -> str:
    if edge["connected"] == 0:
        return "#cbd5e1"

    snr = edge.get("snr", 0.0)
    if snr >= 20:
        return "#22c55e"
    if snr >= 10:
        return "#f59e0b"
    return "#ef4444"


def draw_edges(ax, uavs: list[UAV], edges: list[dict]) -> None:
    if not config.SHOW_EDGE_LINES:
        return

    uav_map = {u.node_id: u for u in uavs}
    connected_edges = [edge for edge in edges if edge["connected"] == 1]

    for edge in connected_edges:
        src = uav_map[edge["src"]]
        dst = uav_map[edge["dst"]]
        ax.plot(
            [src.x, dst.x],
            [src.y, dst.y],
            [src.z, dst.z],
            linewidth=getattr(config, "EDGE_LINE_WIDTH", 1.2),
            alpha=getattr(config, "EDGE_LINE_ALPHA", 0.35),
            color=edge_color_from_quality(edge),
        )


def draw_route(ax, uavs: list[UAV], route_path: list[int] | None) -> None:
    if not (config.HIGHLIGHT_ROUTE and route_path is not None and len(route_path) >= 2):
        return

    uav_map = {u.node_id: u for u in uavs}
    for i in range(len(route_path) - 1):
        a = uav_map[route_path[i]]
        b = uav_map[route_path[i + 1]]
        ax.plot(
            [a.x, b.x],
            [a.y, b.y],
            [a.z, b.z],
            linewidth=getattr(config, "ROUTE_LINE_WIDTH", 4.0),
            color="#2563eb",
            alpha=0.95,
        )


def draw_nodes(ax, uavs: list[UAV]) -> None:
    for uav in uavs:
        if uav.node_id == config.SOURCE_ID:
            color = "#16a34a"
            marker = "o"
            size = getattr(config, "NODE_SIZE_SOURCE", 180)
        elif uav.node_id == config.DEST_ID:
            color = "#dc2626"
            marker = "^"
            size = getattr(config, "NODE_SIZE_DEST", 180)
        else:
            color = "#0ea5e9"
            marker = "o"
            size = getattr(config, "NODE_SIZE_NORMAL", 90)

        ax.scatter(
            uav.x,
            uav.y,
            uav.z,
            s=size,
            color=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.6,
            depthshade=True,
        )

        if config.SHOW_NODE_LABELS:
            ax.text(
                uav.x,
                uav.y,
                uav.z + 3.0,
                f"{uav.node_id}",
                fontsize=9,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
            )

        if getattr(config, "SHOW_VELOCITY_ARROWS", True):
            ax.quiver(
                uav.x,
                uav.y,
                uav.z,
                uav.vx,
                uav.vy,
                uav.vz,
                length=getattr(config, "VELOCITY_ARROW_SCALE", 2.5),
                normalize=False,
                arrow_length_ratio=0.2,
                linewidth=0.8,
                alpha=0.6,
            )


def draw_status_box(
    ax,
    time_step: int,
    reachable: bool,
    route_path: list[int] | None,
    edges: list[dict],
) -> None:
    if not getattr(config, "SHOW_STATUS_BOX", True):
        return

    connected_edges = [e for e in edges if e["connected"] == 1]
    avg_snr = sum(e["snr"] for e in connected_edges) / len(connected_edges) if connected_edges else 0.0
    avg_delay = sum(e["delay"] for e in connected_edges) / len(connected_edges) if connected_edges else 0.0

    route_text = "None" if route_path is None else " -> ".join(map(str, route_path))
    text = (
        f"t = {time_step}\n"
        f"connected edges = {len(connected_edges)}\n"
        f"reachable = {int(reachable)}\n"
        f"avg SNR = {avg_snr:.2f} dB\n"
        f"avg delay = {avg_delay:.2f} ms\n"
        f"route = {route_text}"
    )

    ax.text2D(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92),
    )


def add_legend(ax) -> None:
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#16a34a", markeredgecolor="black", markersize=10, label="Source"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#dc2626", markeredgecolor="black", markersize=10, label="Destination"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0ea5e9", markeredgecolor="black", markersize=8, label="UAV"),
        Line2D([0], [0], color="#22c55e", lw=2, label="Good link"),
        Line2D([0], [0], color="#f59e0b", lw=2, label="Medium link"),
        Line2D([0], [0], color="#ef4444", lw=2, label="Weak link"),
        Line2D([0], [0], color="#2563eb", lw=3, label="Selected route"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.95)


def draw_live_scene(
    ax,
    uavs: list[UAV],
    edges: list[dict],
    reachable: bool,
    time_step: int,
    route_path: list[int] | None,
) -> None:
    if ax is None:
        return

    ax.cla()
    update_trails(uavs)
    apply_axes_style(ax, time_step)

    draw_trails(ax, uavs)
    draw_edges(ax, uavs, edges)
    draw_route(ax, uavs, route_path)
    draw_nodes(ax, uavs)
    draw_status_box(ax, time_step, reachable, route_path, edges)
    add_legend(ax)

    ax.set_title("UAV Network Simulation", fontsize=13, weight="bold", pad=14)
    ax.figure.canvas.draw_idle()
    plt.pause(config.LIVE_PAUSE)


def save_topology_snapshot(
    uavs: list[UAV],
    edges: list[dict],
    reachable: bool,
    time_step: int,
    route_path: list[int] | None,
) -> None:
    if not getattr(config, "SAVE_PLOTS", False):
        return

    fig = plt.figure(figsize=config.FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    apply_axes_style(ax, time_step, force_view_mode="iso", force_rotate=False)
    draw_trails(ax, uavs)
    draw_edges(ax, uavs, edges)
    draw_route(ax, uavs, route_path)
    draw_nodes(ax, uavs)
    draw_status_box(ax, time_step, reachable, route_path, edges)
    add_legend(ax)

    ax.set_title(f"Topology Snapshot @ t={time_step}", fontsize=13, weight="bold", pad=14)

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.06, top=0.92)
    output_path = config.PLOTS_DIR / f"topology_t{time_step}.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_output_dir()

    if config.SOURCE_ID >= config.NUM_UAVS or config.DEST_ID >= config.NUM_UAVS:
        raise ValueError("SOURCE_ID hoặc DEST_ID đang vượt quá NUM_UAVS")

    uavs = initialize_uavs()

    all_node_rows: list[dict] = []
    all_edge_rows: list[dict] = []
    traffic_rows: list[dict] = []

    fig, ax = setup_live_plot()

    try:
        for t in range(config.TIME_STEPS):
            edges, degree_map, adjacency = build_topology(uavs)
            connected_edge_count = sum(edge["connected"] for edge in edges)

            route_path = shortest_path(adjacency, config.SOURCE_ID, config.DEST_ID)
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
        write_csv(
            config.NODES_CSV,
            all_node_rows,
            ["time", "node_id", "x", "y", "z", "vx", "vy", "vz", "speed", "degree"],
        )

        write_csv(
            config.EDGES_CSV,
            all_edge_rows,
            [
                "time",
                "src",
                "dst",
                "distance",
                "connected",
                "relative_speed",
                "rssi",
                "snr",
                "delay",
                "packet_loss",
                "throughput",
            ],
        )

        write_csv(
            config.TRAFFIC_CSV,
            traffic_rows,
            ["time", "source", "destination", "reachable", "route_path", "num_edges"],
        )

        print("\nDone.")
        print(f"- Nodes CSV   : {config.NODES_CSV}")
        print(f"- Edges CSV   : {config.EDGES_CSV}")
        print(f"- Traffic CSV : {config.TRAFFIC_CSV}")

        if config.LIVE_SIMULATION:
            if getattr(config, "SAVE_PLOTS", False) and ax is not None:
                apply_axes_style(ax, config.TIME_STEPS - 1, force_view_mode="iso", force_rotate=False)
                ax.figure.canvas.draw_idle()
                plt.pause(0.001)
                ax.figure.savefig(config.FINAL_FRAME_PNG, dpi=200, bbox_inches="tight")

            plt.ioff()
            plt.show()


if __name__ == "__main__":
    main()

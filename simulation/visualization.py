from __future__ import annotations

from collections import defaultdict, deque

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons, RadioButtons, Slider

import config


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


def update_trails(uavs) -> None:
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


def draw_trails(ax, uavs) -> None:
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


def draw_edges(ax, uavs, edges: list[dict]) -> None:
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


def draw_route(ax, uavs, route_path: list[int] | None) -> None:
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


def draw_nodes(ax, uavs) -> None:
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
    uavs,
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
    uavs,
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


def finalize_live_plot(ax) -> None:
    if not config.LIVE_SIMULATION:
        return

    if getattr(config, "SAVE_PLOTS", False) and ax is not None:
        apply_axes_style(
            ax,
            config.TIME_STEPS - 1,
            force_view_mode="iso",
            force_rotate=False,
        )
        ax.figure.canvas.draw_idle()
        plt.pause(0.001)
        ax.figure.savefig(config.FINAL_FRAME_PNG, dpi=200, bbox_inches="tight")

    plt.ioff()
    plt.show()

# main.py

from __future__ import annotations

import csv
import math
import random
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt

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

# Lưu file kết quả vào thư mục output, tạo thư mục nếu chưa tồn tại
def ensure_output_dir() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if getattr(config, "SAVE_PLOTS", False):
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Tạo vận tốc ngẫu nhiên cho UAV, đảm bảo không phải là (0, 0, 0)
def random_nonzero_velocity() -> tuple[float, float, float]:
    low, high = config.VELOCITY_COMPONENT_RANGE
    while True:
        vx = random.uniform(low, high)
        vy = random.uniform(low, high)
        vz = random.uniform(low, high)
        if abs(vx) + abs(vy) + abs(vz) > 1e-9:
            return vx, vy, vz

# Khởi tạo UAV với vị trí và vận tốc ngẫu nhiên trong giới hạn cấu hình
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

"""
# Khi UAV chạm vào giới hạn không gian, phản xạ lại và đảo chiều vận tốc tương ứng
def reflect_position(pos: float, vel: float, lower: float, upper: float) -> tuple[float, float]:
    while pos < lower or pos > upper:
        if pos < lower:
            pos = lower + (lower - pos)
            vel = -vel
        elif pos > upper:
            pos = upper - (pos - upper)
            vel = -vel
    return pos, vel

# Cập nhật vị trí của tất cả UAV dựa trên vận tốc và thời gian dt, đồng thời xử lý phản xạ nếu chạm giới hạn
def update_positions(uavs: list[UAV], dt: float) -> None:
    for uav in uavs:
        uav.x += uav.vx * dt
        uav.y += uav.vy * dt
        uav.z += uav.vz * dt

        uav.x, uav.vx = reflect_position(uav.x, uav.vx, *config.X_LIMIT)
        uav.y, uav.vy = reflect_position(uav.y, uav.vy, *config.Y_LIMIT)
        uav.z, uav.vz = reflect_position(uav.z, uav.vz, *config.Z_LIMIT)
"""

# Tính khoảng cách Euclidean 3D giữa hai UAV
def euclidean_distance_3d(a: UAV, b: UAV) -> float:
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )

# Xây dựng topology dựa trên khoảng cách giữa các UAV, trả về danh sách cạnh, bậc của mỗi node và bảng kề
def build_topology(uavs: list[UAV]) -> tuple[list[dict], dict[int, int], dict[int, list[int]]]:
    edges: list[dict] = []
    degree_map = {uav.node_id: 0 for uav in uavs}
    adjacency = {uav.node_id: [] for uav in uavs}

    for i in range(len(uavs)):
        for j in range(i + 1, len(uavs)):
            d = euclidean_distance_3d(uavs[i], uavs[j])
            if d <= config.COMM_RANGE:
                src = uavs[i].node_id
                dst = uavs[j].node_id

                edges.append(
                    {
                        "src": src,
                        "dst": dst,
                        "distance": round(d, 4),
                        "connected": 1,
                    }
                )

                degree_map[src] += 1
                degree_map[dst] += 1
                adjacency[src].append(dst)
                adjacency[dst].append(src)

    return edges, degree_map, adjacency

# Tìm đường đi ngắn nhất giữa source và destination trong đồ thị không trọng số sử dụng BFS, trả về danh sách node trên đường đi hoặc None nếu không thể đến được
def shortest_path(adjacency: dict[int, list[int]], source: int, destination: int) -> list[int] | None:
    """
    BFS shortest path in unweighted graph.
    Returns a path like [0, 2, 4], or None if unreachable.
    """
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

# Tạo các hàng dữ liệu cho nodes.csv, bao gồm thời gian, id node, vị trí, vận tốc, tốc độ và bậc của node
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

# Tạo các hàng dữ liệu cho edges.csv, bao gồm thời gian, node nguồn, node đích, khoảng cách và trạng thái kết nối
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
            }
        )

    return rows

# Ghi dữ liệu vào file CSV với đường dẫn, danh sách hàng và tên cột được chỉ định
def write_csv(path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Thiết lập đồ thị 3D cho mô phỏng trực tiếp, trả về đối tượng figure và axes hoặc None nếu không bật chế độ trực tiếp
def setup_live_plot():
    if not config.LIVE_SIMULATION:
        return None, None

    plt.ion()
    fig = plt.figure(figsize=config.FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax

# Tính toán các bước thời gian mà tại đó sẽ lưu ảnh chụp topology nếu bật tùy chọn SAVE_PLOTS, đảm bảo chỉ bao gồm các bước hợp lệ trong khoảng thời gian mô phỏng
snapshot_steps = {
    step for step in getattr(config, "TOPOLOGY_SNAPSHOT_STEPS", [])
    if 0 <= step < config.TIME_STEPS
}

# Vẽ cảnh mô phỏng trực tiếp, bao gồm các UAV, cạnh nối và đường đi được chọn nếu có, đồng thời cập nhật tiêu đề với thông tin thời gian, số cạnh, trạng thái kết nối và đường đi hiện tại
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

    uav_map = {u.node_id: u for u in uavs}

    # Draw all normal edges first
    if config.SHOW_EDGE_LINES:
        for edge in edges:
            src = uav_map[edge["src"]]
            dst = uav_map[edge["dst"]]
            ax.plot(
                [src.x, dst.x],
                [src.y, dst.y],
                [src.z, dst.z],
                linewidth=1.0,
                alpha=0.45,
            )

    # Highlight chosen route path
    if config.HIGHLIGHT_ROUTE and route_path is not None and len(route_path) >= 2:
        for i in range(len(route_path) - 1):
            a = uav_map[route_path[i]]
            b = uav_map[route_path[i + 1]]
            ax.plot(
                [a.x, b.x],
                [a.y, b.y],
                [a.z, b.z],
                linewidth=3.5,
                color="red",
            )

    # Draw nodes
    for uav in uavs:
        if uav.node_id == config.SOURCE_ID:
            ax.scatter(uav.x, uav.y, uav.z, s=120, color="green", marker="o")
        elif uav.node_id == config.DEST_ID:
            ax.scatter(uav.x, uav.y, uav.z, s=120, color="red", marker="^")
        else:
            ax.scatter(uav.x, uav.y, uav.z, s=60)

        if config.SHOW_NODE_LABELS:
            ax.text(uav.x, uav.y, uav.z, f"{uav.node_id}", fontsize=9)

    route_text = "None"
    if route_path is not None:
        route_text = " -> ".join(str(x) for x in route_path)

    ax.set_xlim(config.X_LIMIT)
    ax.set_ylim(config.Y_LIMIT)
    ax.set_zlim(config.Z_LIMIT)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"UAV 3D Live Simulation | t={time_step} | "
        f"edges={len(edges)} | reachable={int(reachable)}\n"
        f"route: {route_text}"
    )

    plt.tight_layout()
    plt.draw()
    plt.pause(config.LIVE_PAUSE)

# Lưu ảnh chụp topology hiện tại vào thư mục plots nếu bật tùy chọn SAVE_PLOTS, bao gồm các UAV, cạnh nối và đường đi được chọn nếu có, đồng thời cập nhật tiêu đề với thông tin thời gian, số cạnh, trạng thái kết nối và đường đi hiện tại
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

    uav_map = {u.node_id: u for u in uavs}

    # Draw normal edges
    if config.SHOW_EDGE_LINES:
        for edge in edges:
            src = uav_map[edge["src"]]
            dst = uav_map[edge["dst"]]
            ax.plot(
                [src.x, dst.x],
                [src.y, dst.y],
                [src.z, dst.z],
                linewidth=1.0,
                alpha=0.45,
            )

    # Highlight route path
    if config.HIGHLIGHT_ROUTE and route_path is not None and len(route_path) >= 2:
        for i in range(len(route_path) - 1):
            a = uav_map[route_path[i]]
            b = uav_map[route_path[i + 1]]
            ax.plot(
                [a.x, b.x],
                [a.y, b.y],
                [a.z, b.z],
                linewidth=3.5,
                color="red",
            )

    # Draw nodes
    for uav in uavs:
        if uav.node_id == config.SOURCE_ID:
            ax.scatter(uav.x, uav.y, uav.z, s=120, color="green", marker="o")
        elif uav.node_id == config.DEST_ID:
            ax.scatter(uav.x, uav.y, uav.z, s=120, color="red", marker="^")
        else:
            ax.scatter(uav.x, uav.y, uav.z, s=60)

        if config.SHOW_NODE_LABELS:
            ax.text(uav.x, uav.y, uav.z, f"{uav.node_id}", fontsize=9)

    route_text = "None" if route_path is None else " -> ".join(map(str, route_path))

    ax.set_xlim(config.X_LIMIT)
    ax.set_ylim(config.Y_LIMIT)
    ax.set_zlim(config.Z_LIMIT)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"Topology Snapshot | t={time_step} | "
        f"edges={len(edges)} | reachable={int(reachable)}\n"
        f"route: {route_text}"
    )

    plt.tight_layout()
    output_path = config.PLOTS_DIR / f"topology_t{time_step}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# Hàm main để chạy mô phỏng, bao gồm khởi tạo UAV, xây dựng topology, tìm đường đi, lưu dữ liệu và vẽ cảnh mô phỏng trực tiếp nếu được bật
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
                    "num_edges": len(edges),
                }
            )

            if t % config.PRINT_EVERY == 0 or t == config.TIME_STEPS - 1:
                route_text = "None" if route_path is None else " -> ".join(map(str, route_path))
                print(
                    f"[t={t:03d}] edges={len(edges):02d}, "
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
            ["time", "src", "dst", "distance", "connected"],
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
            if config.LIVE_SIMULATION and getattr(config, "SAVE_PLOTS", False):
                plt.savefig(config.FINAL_FRAME_PNG, dpi=200, bbox_inches="tight")
            plt.ioff()
            plt.show()

# Entry point của chương trình, gọi hàm main để bắt đầu mô phỏng
if __name__ == "__main__":
    main()

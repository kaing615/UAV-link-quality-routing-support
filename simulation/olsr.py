from __future__ import annotations

import heapq
from collections import deque


class OLSRNode:
    __slots__ = (
        "node_id",
        "neighbor_table",
        "two_hop_table",
        "mpr_set",
        "mpr_selector_set",
        "topology_table",
        "routing_table",
    )

    def __init__(self, node_id: int) -> None:
        self.node_id = node_id
        self.neighbor_table: dict[int, dict] = {}
        self.two_hop_table: dict[int, set[int]] = {}
        self.mpr_set: set[int] = set()
        self.mpr_selector_set: set[int] = set()
        self.topology_table: dict[tuple[int, int], float] = {}
        self.routing_table: dict[int, tuple[int, int, float]] = {}

    def reset(self) -> None:
        self.neighbor_table.clear()
        self.two_hop_table.clear()
        self.mpr_set.clear()
        self.mpr_selector_set.clear()
        self.topology_table.clear()
        self.routing_table.clear()


class OLSRProtocol:
    def __init__(self, node_ids: list[int], metric: str = "link_quality") -> None:
        self.nodes: dict[int, OLSRNode] = {nid: OLSRNode(nid) for nid in node_ids}
        self.metric = metric

    def update(
        self,
        adjacency: dict[int, list[int]],
        weighted_adjacency: dict[int, list[tuple[int, float]]],
    ) -> None:
        for node in self.nodes.values():
            node.reset()

        self._process_hello(adjacency, weighted_adjacency)
        self._select_mprs()
        self._process_tc()
        self._compute_routing_tables()

    def find_route(self, source: int, destination: int) -> list[int] | None:
        if source == destination:
            return [source]

        source_node = self.nodes.get(source)
        if source_node is None or destination not in source_node.routing_table:
            return None

        path = [source]
        current = source
        visited = {source}

        while current != destination:
            current_node = self.nodes[current]
            if destination not in current_node.routing_table:
                return None

            next_hop = current_node.routing_table[destination][0]

            if next_hop in visited:
                return None

            path.append(next_hop)
            visited.add(next_hop)
            current = next_hop

        return path

    def get_stats(self) -> dict:
        n = max(len(self.nodes), 1)
        mpr_count = sum(1 for nd in self.nodes.values() if nd.mpr_selector_set)
        avg_rt = sum(len(nd.routing_table) for nd in self.nodes.values()) / n
        avg_tt = sum(len(nd.topology_table) for nd in self.nodes.values()) / n
        avg_mpr = sum(len(nd.mpr_set) for nd in self.nodes.values()) / n

        return {
            "mpr_nodes": mpr_count,
            "total_nodes": len(self.nodes),
            "avg_routing_table_size": round(avg_rt, 1),
            "avg_topology_entries": round(avg_tt, 1),
            "avg_mpr_set_size": round(avg_mpr, 1),
        }

    def _process_hello(
        self,
        adjacency: dict[int, list[int]],
        weighted_adjacency: dict[int, list[tuple[int, float]]],
    ) -> None:
        weight_of: dict[tuple[int, int], float] = {}
        for nid, neighbors in weighted_adjacency.items():
            for nbr_id, w in neighbors:
                weight_of[(nid, nbr_id)] = w

        for nid, neighbors in adjacency.items():
            node = self.nodes[nid]
            for nbr_id in neighbors:
                node.neighbor_table[nbr_id] = {
                    "weight": weight_of.get((nid, nbr_id), 1.0),
                }

        for nid, node in self.nodes.items():
            for one_hop_id in node.neighbor_table:
                one_hop_node = self.nodes[one_hop_id]
                for two_hop_id in one_hop_node.neighbor_table:
                    if two_hop_id == nid or two_hop_id in node.neighbor_table:
                        continue
                    node.two_hop_table.setdefault(two_hop_id, set()).add(one_hop_id)

    def _select_mprs(self) -> None:
        for node in self.nodes.values():
            node.mpr_set = self._compute_mpr_for_node(node)

        for node in self.nodes.values():
            for mpr_id in node.mpr_set:
                self.nodes[mpr_id].mpr_selector_set.add(node.node_id)

    @staticmethod
    def _compute_mpr_for_node(node: OLSRNode) -> set[int]:
        if not node.two_hop_table:
            return set()

        uncovered: set[int] = set(node.two_hop_table.keys())
        mpr: set[int] = set()

        for two_hop_id, via_set in node.two_hop_table.items():
            if len(via_set) == 1:
                mpr.add(next(iter(via_set)))

        for mpr_id in mpr:
            for reachable in node.two_hop_table:
                if mpr_id in node.two_hop_table[reachable]:
                    uncovered.discard(reachable)

        while uncovered:
            best, best_cov = None, 0
            for one_hop_id in node.neighbor_table:
                if one_hop_id in mpr:
                    continue
                coverage = sum(
                    1 for th in uncovered
                    if one_hop_id in node.two_hop_table.get(th, set())
                )
                if coverage > best_cov:
                    best_cov = coverage
                    best = one_hop_id

            if best is None:
                break

            mpr.add(best)
            for th in list(uncovered):
                if best in node.two_hop_table.get(th, set()):
                    uncovered.discard(th)

        return mpr

    def _process_tc(self) -> None:
        components = self._connected_components()

        for component in components:
            tc_entries: list[tuple[int, int, float]] = []

            for nid in component:
                node = self.nodes[nid]
                if not node.mpr_selector_set:
                    continue
                for selector_id in node.mpr_selector_set:
                    weight = node.neighbor_table.get(selector_id, {}).get("weight", 1.0)
                    tc_entries.append((nid, selector_id, weight))

            for recv_id in component:
                recv_node = self.nodes[recv_id]
                for adv, reach, w in tc_entries:
                    if adv == recv_id:
                        continue
                    recv_node.topology_table[(adv, reach)] = w

    def _connected_components(self) -> list[set[int]]:
        visited: set[int] = set()
        components: list[set[int]] = []

        for nid in self.nodes:
            if nid in visited:
                continue
            component: set[int] = set()
            queue: deque[int] = deque([nid])
            while queue:
                cur = queue.popleft()
                if cur in visited:
                    continue
                visited.add(cur)
                component.add(cur)
                for nbr in self.nodes[cur].neighbor_table:
                    if nbr not in visited:
                        queue.append(nbr)
            components.append(component)

        return components

    def _compute_routing_tables(self) -> None:
        for node in self.nodes.values():
            self._dijkstra_local(node)

    def _dijkstra_local(self, node: OLSRNode) -> None:
        use_lq = self.metric == "link_quality"

        graph: dict[int, list[tuple[int, float]]] = {}

        def _add_edge(a: int, b: int, w: float) -> None:
            graph.setdefault(a, []).append((b, w))
            graph.setdefault(b, []).append((a, w))

        for nbr_id, info in node.neighbor_table.items():
            _add_edge(node.node_id, nbr_id, info["weight"] if use_lq else 1.0)

        for (adv, reach), tc_w in node.topology_table.items():
            _add_edge(adv, reach, tc_w if use_lq else 1.0)

        dist: dict[int, float] = {node.node_id: 0.0}
        hops: dict[int, int] = {node.node_id: 0}
        parent: dict[int, int | None] = {node.node_id: None}
        heap: list[tuple[float, int]] = [(0.0, node.node_id)]

        while heap:
            d, cur = heapq.heappop(heap)
            if d > dist.get(cur, float("inf")):
                continue
            for nbr, w in graph.get(cur, []):
                nd = d + w
                if nd < dist.get(nbr, float("inf")):
                    dist[nbr] = nd
                    hops[nbr] = hops[cur] + 1
                    parent[nbr] = cur
                    heapq.heappush(heap, (nd, nbr))

        for dest_id, cost in dist.items():
            if dest_id == node.node_id or cost == float("inf"):
                continue

            trace = dest_id
            while parent.get(trace) not in (node.node_id, None):
                trace = parent[trace]

            if parent.get(trace) == node.node_id:
                node.routing_table[dest_id] = (trace, hops[dest_id], cost)

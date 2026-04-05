import heapq


def dijkstra_shortest_path(
    weighted_adjacency: dict[int, list[tuple[int, float]]],
    source: int,
    destination: int,
) -> list[int] | None:
    if source == destination:
        return [source]

    dist = {node: float("inf") for node in weighted_adjacency}
    parent: dict[int, int | None] = {source: None}
    dist[source] = 0.0

    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        current_dist, current = heapq.heappop(heap)

        if current_dist > dist[current]:
            continue

        if current == destination:
            break

        for neighbor, weight in weighted_adjacency.get(current, []):
            new_dist = current_dist + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                parent[neighbor] = current
                heapq.heappush(heap, (new_dist, neighbor))

    if destination not in parent:
        return None

    path = [destination]
    node = destination
    while parent[node] is not None:
        node = parent[node]
        path.append(node)
    path.reverse()
    return path

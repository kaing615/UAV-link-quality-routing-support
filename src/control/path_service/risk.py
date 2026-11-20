"""Observed path union-bound-shaped score; estimated marginals need not calibrate OOD."""
import heapq
import math
from ..prevention.features import measured_edges, bridge_edges
from ..prevention.prediction import HORIZONS


def assess_risk(frame, previous, config, predictor, method, no_service=False):
    risks = predictor.predict(frame, previous, config.comm_range)
    edges = set(measured_edges(frame))
    for h in HORIZONS:
        for e in edges:
            if e not in risks[h] or not math.isfinite(risks[h][e]) or not 0 <= risks[h][e] <= 1: raise ValueError('Missing/invalid edge risk')
    path, origin = [], 'unavailable'
    if no_service:
        selected = sorted(bridge_edges(frame)); origin = 'global_bridge_ablation'
    else:
        route = frame['service_observation']['route']
        if route['success'] and all(tuple(sorted(e)) in edges for e in zip(route['nodes'], route['nodes'][1:])):
            path, origin = route['nodes'], 'observed_olsr'
        else:
            neighbors = {i: [] for i in range(config.num_uavs)}
            for a,b in sorted(edges): neighbors[a].append(b); neighbors[b].append(a)
            queue, visited = [(0.,0,(config.source,))], set()
            while queue:
                cost, hops, candidate = heapq.heappop(queue); node=candidate[-1]
                if node in visited: continue
                visited.add(node)
                if node == config.destination: path, origin = list(candidate), 'measured_min_sum_H20'; break
                for other in neighbors[node]:
                    if other not in visited: heapq.heappush(queue,(cost+risks[20][tuple(sorted((node,other)))],hops+1,candidate+(other,)))
        selected = [tuple(sorted(e)) for e in zip(path,path[1:])]
    scores = {h: (max((risks[h][e] for e in selected),default=0.) if no_service else min(1.,sum(risks[h][e] for e in selected))) for h in HORIZONS}
    crossing = next((h for h in HORIZONS if selected and scores[h] >= method.risk_threshold),None)
    return dict(forecast_path=path, path_origin=origin, scores=scores, crossing_horizon_s=crossing,
                score_semantics='global bridge maximum' if no_service else 'min(1,sum estimated edge marginals); heuristic, not calibrated path/action probability',
                edge_risks=[dict(edge=e, risks={h:risks[h][e] for h in HORIZONS}) for e in selected])

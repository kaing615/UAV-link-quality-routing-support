"""Same bounded position law across reactive, graph and graphless triggers."""
import numpy as np


def widest(q, source, destination):
    q = np.minimum(np.asarray(q, float), np.asarray(q, float).T)
    n = len(q)
    scores = np.zeros(n)
    scores[source] = 1.
    done = set()
    for _ in range(n):
        node = max((i for i in range(n) if i not in done), key=lambda i: (scores[i], -i))
        done.add(node)
        if node == destination:
            return float(scores[node])
        for other in range(n):
            if other not in done:
                scores[other] = max(scores[other], min(scores[node], q[node, other]))
    return float(scores[destination])


def swept_safe(positions, velocities, minimum):
    n = len(positions)
    for a in range(n):
        for b in range(a + 1, n):
            delta = positions[a] - positions[b]
            speed = velocities[a] - velocities[b]
            denom = float(speed @ speed)
            t = float(np.clip(-(delta @ speed) / denom, 0, 1)) if denom else 0.
            if np.linalg.norm(delta + t * speed) < minimum - 1e-8:
                return False
    return True


def position_action(frame, config):
    positions = np.asarray(frame["positions"], float)
    velocities = np.asarray(frame["velocities"], float).copy()
    start, end = positions[config.source], positions[config.destination]
    axis = end - start
    ordered = sorted(config.relay_ids, key=lambda node: (float((positions[node] - start) @ axis), node))
    targets = {node: start + (rank + 1) / (len(ordered) + 1) * axis for rank, node in enumerate(ordered)}
    for node in config.relay_ids:
        delta = targets[node] - positions[node]
        distance = float(np.linalg.norm(delta))
        budget = max(0., config.budget - frame["travel"][node])
        speed = min(config.max_speed, budget, distance) if distance > 1. else 0.
        velocities[node] = delta / max(distance, 1e-9) * speed
    safe = swept_safe(positions, velocities, config.min_separation)
    action = np.concatenate([velocities[node] / config.max_speed for node in config.relay_ids])
    if not safe:
        action[:] = 0
    return action, safe, {str(node): target.tolist() for node, target in targets.items()}


class Motion:
    def __init__(self, config, forecast=False):
        self.config = config
        self.forecast = forecast
        self.low = 0
        self.recovered = 0
        self.remaining = 0
        self.cooldown = 0
        self.first = None

    def act(self, frame, current, predicted=None):
        c = self.config
        observed = widest(current, c.source, c.destination)
        future = widest(predicted, c.source, c.destination) if predicted is not None else None
        diagnostic = dict(current_widest=observed, predicted_widest=future, started=False,
                          reason="hold", safety_blocked=False, first_motion_step=self.first)
        zero = np.zeros(3 * len(c.relay_ids))
        if frame["step"] < 3:
            return zero, dict(diagnostic, reason="warmup")
        if self.cooldown:
            self.cooldown -= 1
            self.low = 0
            return zero, dict(diagnostic, reason="cooldown")
        if not self.remaining:
            weak = observed < .70 or (future is not None and future < .70)
            self.low = self.low + 1 if weak else 0
            if self.low < 2:
                return zero, dict(diagnostic, reason="confirm" if weak else "hold")
            self.remaining, self.low, self.recovered = 10, 0, 0
            diagnostic["started"] = True
        recovered = observed >= .80 and (future is None or future >= .80)
        self.recovered = self.recovered + 1 if recovered else 0
        if self.recovered >= 3:
            self.remaining, self.cooldown = 0, 5
            return zero, dict(diagnostic, reason="recovered")
        action, safe, targets = position_action(frame, c)
        diagnostic.update(targets=targets, safety_blocked=not safe, reason="positioning" if safe else "swept_separation")
        if np.any(action) and self.first is None:
            self.first = frame["step"]
        diagnostic["first_motion_step"] = self.first
        self.remaining -= 1
        if self.remaining == 0:
            self.cooldown = 5
        return action, diagnostic

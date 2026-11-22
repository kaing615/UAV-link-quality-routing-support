"""Route-controlled prevention environment: same simulator, --routeControl=external,
UAVACT 2 lines (velocities + pinned route). The v1 bridge/env stay untouched."""
import numpy as np

from ..bridge import ProtocolError, _finite_number
from ..prevention.environment import Ns3PreventionEnv


class RoutedAction(np.ndarray):
    """Velocity vector that also carries the route to pin (() = live OLSR)."""

    def __new__(cls, values, route=()):
        obj = np.array(values, dtype=float).view(cls)
        obj.route = tuple(int(n) for n in route)
        return obj

    def __array_finalize__(self, obj):
        self.route = getattr(obj, 'route', ())


def routed_line(step, values, route):
    return (f"UAVACT 2 {step} " + " ".join(format(v, ".9g") for v in values)
            + f" ROUTE {len(route)}" + "".join(f" {n}" for n in route) + "\n")


def send_routed_action(bridge, step, action, route):
    expected = 3 * len(bridge.config.relay_ids)
    if len(action) != expected:
        raise ValueError(f"Action must contain {expected} normalized velocity values")
    values = [_finite_number(float(v), "action") for v in action]
    if any(abs(v) > 1 for v in values):
        raise ValueError("Action values must lie in [-1,1]")
    n, config = bridge.config.num_uavs, bridge.config
    if route and (len(route) < 2 or route[0] != config.source or route[-1] != config.destination
                  or len(set(route)) != len(route) or any(not 0 <= v < n for v in route)):
        raise ValueError("Route must be a simple source..destination path")
    if bridge.process is None or bridge.process.poll() is not None or bridge.process.stdin is None:
        bridge.close(force=True)
        raise ProtocolError("Simulator is not running")
    try:
        bridge.process.stdin.write(routed_line(step, values, route))
        bridge.process.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        bridge.close(force=True)
        raise ProtocolError("Could not send routed action to simulator") from exc


class RouteControlEnv(Ns3PreventionEnv):
    def _command(self, output_dir):
        return super()._command(output_dir) + ['--routeControl=external']

    def step(self, action):
        route = tuple(getattr(action, 'route', ()))
        bridge = self.bridge
        if bridge is not None:
            # Parent step() is reused unchanged; only the line format differs.
            bridge.send_action = lambda step, values: send_routed_action(bridge, step, values, route)
        try:
            return super().step(np.asarray(action, dtype=float))
        finally:
            if bridge is not None and 'send_action' in vars(bridge):
                del bridge.send_action

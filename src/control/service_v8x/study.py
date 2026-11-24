"""Exploratory v8x: is the v8 GNN routing gain just extra smoothing?

Same collector, binary, flags and route rule as v8 lq-route; only the quality
smoothing (EWMA alpha) or the dwell changes. gnn-route reruns the locked v8
weights on the same fresh seeds so every comparison is paired.
Not confirmatory: no pre-registered hypothesis, one phase, descriptive CIs.
"""
from functools import partial
import json
from pathlib import Path

import numpy as np

from ..service_v2.config import ServiceV2Config
from ..service_v2.study import collect_case, make_controller_factory
from ..service_v3 import protocol as p3
from ..service_v4 import routing
from ..service_v4.routing import LinkQuality, RouteDecider
from ..service_v4.study import RouteController
from ..service_v7.study import ARP_FLAG, BACKGROUND
from ..service_v8.study import ForecastController

ARMS = ("lq-route", "gnn-route", "lq-ewma02", "lq-ewma01", "lq-dwell15")
ROUTE_ARMS = ARMS
ALPHAS = {"lq-ewma02": .2, "lq-ewma01": .1}
V8X = p3.Study("v8x", {"validation": (0, 20)}, {"validation": ARMS}, 3_800_000, 3_900_000,
               "src.control.service_v8x.cli", regimes=("n5", "n7"), predictor_scenarios=False)
load = partial(p3.load, study=V8X)


def arm_flags(policy):
    if policy not in ARMS:
        raise ValueError("Unknown v8x arm: " + policy)
    return [ARP_FLAG, *BACKGROUND]


def env_for(policy):
    from ..service_v4.env import RouteControlEnv

    class ArmEnv(RouteControlEnv):
        def _command(self, output_dir):
            return super()._command(output_dir) + arm_flags(policy)
    return ArmEnv


class SmoothedQuality(LinkQuality):
    """LinkQuality with a different EWMA weight; raw broken-link rule unchanged."""

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def update(self, rx):
        previous = self.q
        super().update(rx)
        if previous is not None:
            self.q = self.alpha * np.asarray(rx, float) + (1 - self.alpha) * previous


def make_factory(phase, new, old):
    selection = json.loads((Path(phase["_path"]).parent / "selection.json").read_text())
    base = make_controller_factory(new, old, ServiceV2Config(), phase["new_model"], phase["method_sha256"])

    def factory(policy, config, **kwargs):
        hold = base("hold", config, **kwargs)
        if policy in ("lq-route", "lq-dwell15"):
            # ponytail: one cell per process, so the module constant is safe to set here
            routing.DWELL_S = 15 if policy == "lq-dwell15" else 5
            return RouteController(hold, config, 1.)
        if policy in ALPHAS:
            routing.DWELL_S = 5
            controller = RouteController(hold, config, 1.)
            controller.decider = RouteDecider(config.num_uavs, config.source, config.destination, 1.,
                                              quality=SmoothedQuality(ALPHAS[policy]))
            return controller
        if policy == "gnn-route":
            routing.DWELL_S = 5
            return ForecastController(hold, config, selection["models"]["gnn"], phase["selection_sha256"])
        raise ValueError("Unknown v8x policy")
    return factory


def run_case(protocol_path, cell_id, attempt, output):
    phase = load(protocol_path)
    cell = next(cell for cell in phase["cells"] if cell["cell_id"] == cell_id)
    return collect_case(protocol_path, cell_id, attempt, output, loader=load, factory_builder=make_factory,
                        env_class=env_for(cell["policy"]), validate_new=False)

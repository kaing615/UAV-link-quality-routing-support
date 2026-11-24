"""Frozen v8 arms: current quality, causal GNN, graphless ablation, tuned ETX."""
from functools import partial
import json
from pathlib import Path

import numpy as np

from ..service_v2.config import ServiceV2Config
from ..service_v2.study import collect_case, make_controller_factory
from ..service_v3 import protocol as p3
from ..service_v4.env import RouteControlEnv, RoutedAction
from ..service_v4.routing import LinkQuality, RouteDecider
from ..service_v4.study import RouteController
from ..service_v7.study import ARP_FLAG, BACKGROUND, ETX_FLAG, hello_flags
from .predictor import Predictor

ARMS = ("lq-route", "gnn-route", "mlp-route", "olsr-etx-fast")
ROUTE_ARMS = ARMS[:3]
V8 = p3.Study("v8", {"validation": (2, 8), "test": (4, 20)},
              {"validation": ARMS, "test": ARMS}, 3_400_000, 3_500_000,
              "src.control.service_v8.cli", regimes=("n5", "n7"), predictor_scenarios=False)
load = partial(p3.load, study=V8)


def arm_flags(policy):
    if policy not in ARMS:
        raise ValueError("Unknown v8 arm: " + policy)
    return [ARP_FLAG, *BACKGROUND] + ([ETX_FLAG, *hello_flags(.5)] if policy == "olsr-etx-fast" else [])


def env_for(policy):
    class ArmEnv(RouteControlEnv):
        def _command(self, output_dir):
            return super()._command(output_dir) + arm_flags(policy)
    return ArmEnv


class ForecastQuality(LinkQuality):
    def update(self, rx):
        # Preserve the exact raw-link failover and warmup rules of v7.
        super().update(rx)
        self.q = self.forecast.copy()


class ForecastController:
    def __init__(self, hold, config, artifact, model_sha256):
        self.inner = hold
        self.predictor = Predictor(artifact)
        self.model_sha256 = model_sha256
        self.decider = RouteDecider(config.num_uavs, config.source, config.destination, 1., quality=ForecastQuality())

    def act(self, frame):
        action, decision = self.inner.act(frame)
        predicted = self.predictor.predict(frame)
        self.decider.quality.forecast = predicted
        live_route = frame["service_observation"]["route"]
        olsr = live_route["nodes"] if live_route["success"] else []
        route, diagnostic = self.decider.decide(frame["step"], frame["rx_ratio"], olsr)
        return RoutedAction(action, route), dict(decision, route_control=diagnostic,
                forecast_control=dict(model_sha256=self.model_sha256, origin_step=frame["step"],
                                      horizon=5, q=predicted.tolist()))


def make_factory(phase, new, old):
    selection = json.loads((Path(phase["_path"]).parent / "selection.json").read_text())
    base = make_controller_factory(new, old, ServiceV2Config(), phase["new_model"], phase["method_sha256"])

    def factory(policy, config, **kwargs):
        hold = base("hold", config, **kwargs)
        if policy == "lq-route":
            return RouteController(hold, config, 1.)
        if policy in ("gnn-route", "mlp-route"):
            return ForecastController(hold, config, selection["models"][policy.split("-")[0]], phase["selection_sha256"])
        if policy == "olsr-etx-fast":
            return hold
        raise ValueError("Unknown v8 policy")
    return factory


def run_case(protocol_path, cell_id, attempt, output):
    phase = load(protocol_path)
    cell = next(cell for cell in phase["cells"] if cell["cell_id"] == cell_id)
    return collect_case(protocol_path, cell_id, attempt, output, loader=load, factory_builder=make_factory,
                        env_class=env_for(cell["policy"]), validate_new=False)


def forecast_trace_ok(row, selection_sha256):
    if row["policy"] not in ("gnn-route", "mlp-route"):
        return True
    n = row["config"]["num_uavs"]
    for decision in row["decisions"]:
        item = decision.get("forecast_control", {})
        q = np.asarray(item.get("q", []))
        if (item.get("model_sha256") != selection_sha256 or item.get("origin_step") != decision["step"]
                or item.get("horizon") != 5 or q.shape != (n, n) or not np.isfinite(q).all()
                or np.any((q < 0) | (q > 1))):
            return False
    return True

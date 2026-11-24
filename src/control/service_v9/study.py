"""Common measured-quality routing, with reactive or forecast motion triggers."""
from dataclasses import replace
from functools import partial
import json
from pathlib import Path

import numpy as np

from ..service_v2.config import ServiceV2Config
from ..service_v2.protocol import scenario_config as base_config
from ..service_v2.study import collect_case, make_controller_factory
from ..service_v3.protocol import Study
from ..service_v4.env import RouteControlEnv, RoutedAction
from ..service_v4.routing import RouteDecider
from ..service_v7.study import ARP_FLAG, BACKGROUND
from ..service_v8.predictor import Predictor
from .motion import Motion
from . import protocol as p9

ARMS = ("route-only", "reactive-motion", "gnn-motion", "mlp-motion")
V9 = Study("v9", {"validation": (2, 8), "test": (4, 20)},
           {"validation": ARMS, "test": ARMS}, 3_600_000, 3_700_000,
           "src.control.service_v9.cli", regimes=("n5", "n7"), predictor_scenarios=False)
load = partial(p9.load, study=V9)


def scenario_config(scenario):
    return replace(base_config(scenario), mission_speed=2.)


def arm_flags(policy):
    if policy not in ARMS:
        raise ValueError("Unknown v9 policy")
    return [ARP_FLAG, *BACKGROUND]


def env_for(policy):
    class ArmEnv(RouteControlEnv):
        def _command(self, output_dir):
            return super()._command(output_dir) + arm_flags(policy)
    return ArmEnv


class Controller:
    def __init__(self, hold, config, policy, model, selection_sha256):
        self.hold, self.config, self.policy = hold, config, policy
        self.decider = RouteDecider(config.num_uavs, config.source, config.destination, 1.)
        self.predictor = Predictor(model) if model is not None else None
        self.motion = Motion(config, forecast=model is not None)
        self.selection_sha256 = selection_sha256

    def act(self, frame):
        action, decision = self.hold.act(frame)
        live = frame["service_observation"]["route"]
        route, diagnostic = self.decider.decide(frame["step"], frame["rx_ratio"], live["nodes"] if live["success"] else [])
        predicted = None
        forecast = None
        if self.predictor is not None:
            predicted = self.predictor.predict(frame)
            forecast = dict(model_sha256=self.selection_sha256, origin_step=frame["step"], horizon=5, q=predicted.tolist())
            predicted = np.where(np.asarray(frame["rx_ratio"]) >= .1, predicted, 0.)
        motion = dict(started=False, reason="route_only", safety_blocked=False)
        if self.policy != "route-only":
            action, motion = self.motion.act(frame, self.decider.quality.q, predicted)
        decision.update(policy=self.policy, route_control=diagnostic, motion_control=motion,
                        intervention_started=motion["started"], commanded_motion=bool(np.any(action)),
                        sent_action=action.tolist())
        if forecast is not None:
            decision["forecast_control"] = forecast
        return RoutedAction(action, route), decision


def make_factory(phase, new, old):
    selection = json.loads((Path(phase["_path"]).parent / "selection.json").read_text())
    base = make_controller_factory(new, old, ServiceV2Config(), phase["new_model"], phase["method_sha256"])

    def factory(policy, config, **kwargs):
        if policy not in ARMS:
            raise ValueError("Unknown v9 policy")
        model = selection["models"][policy.split("-")[0]] if policy in ("gnn-motion", "mlp-motion") else None
        return Controller(base("hold", config, **kwargs), config, policy, model, phase["selection_sha256"])
    return factory


def run_case(protocol_path, cell_id, attempt, output):
    phase = load(protocol_path)
    cell = next(c for c in phase["cells"] if c["cell_id"] == cell_id)
    return collect_case(protocol_path, cell_id, attempt, output, loader=load, factory_builder=make_factory,
                        env_class=env_for(cell["policy"]), validate_new=False)

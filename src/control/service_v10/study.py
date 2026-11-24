"""v10: tuned-smoothing EWMA vs tuned-smoothing GNN (docs/service_v10_protocol.md).

Two packages reuse the v3 freeze machinery:
  v10t  tuning grid (frozen as its "validation" phase, which needs a selection = v8 weights)
  v10   validation + test with alpha*/beta* chosen on v10t
"""
from functools import partial
import json
from pathlib import Path

from ..service_v2.config import ServiceV2Config
from ..service_v2.study import collect_case, make_controller_factory
from ..service_v3 import protocol as p3
from ..service_v4.routing import LinkQuality, RouteDecider
from ..service_v4.study import RouteController
from ..service_v8.study import ForecastController, ForecastQuality
from ..service_v8x.study import SmoothedQuality, arm_flags as _flags, env_for as _env

ALPHAS = {"ewma-a50": .5, "ewma-a30": .3, "ewma-a20": .2, "ewma-a10": .1, "ewma-a05": .05}
BETAS = {"gnn-b100": 1., "gnn-b50": .5, "gnn-b30": .3, "gnn-b20": .2, "gnn-b10": .1}
TUNE_ARMS = (*ALPHAS, *BETAS)
TEST_ARMS = ("lq-route", "ewma-tuned", "gnn-tuned")
V10T = p3.Study("v10t", {"validation": (0, 8)}, {"validation": TUNE_ARMS}, 4_000_000, 4_100_000,
                "src.control.service_v10.tune_cli", regimes=("n5", "n7"), predictor_scenarios=False)
V10 = p3.Study("v10", {"validation": (2, 8), "test": (4, 20)}, {"validation": TEST_ARMS, "test": TEST_ARMS},
               4_200_000, 4_300_000, "src.control.service_v10.cli", regimes=("n5", "n7"), predictor_scenarios=False)
load_tune = partial(p3.load, study=V10T)
load = partial(p3.load, study=V10)


class SmoothedForecastQuality(ForecastQuality):
    """EWMA over the GNN forecast; raw broken-link rule and warmup unchanged (beta=1: v8 gnn-route)."""

    def __init__(self, beta):
        super().__init__()
        self.beta, self.smoothed = beta, None

    def update(self, rx):
        LinkQuality.update(self, rx)
        f = self.forecast
        self.smoothed = f.copy() if self.smoothed is None else self.beta * f + (1 - self.beta) * self.smoothed
        self.q = self.smoothed.copy()


def ewma_controller(hold, config, alpha):
    controller = RouteController(hold, config, 1.)
    controller.decider = RouteDecider(config.num_uavs, config.source, config.destination, 1.,
                                      quality=SmoothedQuality(alpha))
    return controller


def gnn_controller(hold, config, artifact, sha, beta):
    controller = ForecastController(hold, config, artifact, sha)
    controller.decider = RouteDecider(config.num_uavs, config.source, config.destination, 1.,
                                      quality=SmoothedForecastQuality(beta))
    return controller


def make_factory(phase, new, old):
    selection = json.loads((Path(phase["_path"]).parent / "selection.json").read_text())
    base = make_controller_factory(new, old, ServiceV2Config(), phase["new_model"], phase["method_sha256"])
    gnn, sha = selection["models"]["gnn"], phase["selection_sha256"]

    def factory(policy, config, **kwargs):
        hold = base("hold", config, **kwargs)
        if policy == "lq-route":
            return RouteController(hold, config, 1.)
        if policy in ALPHAS:
            return ewma_controller(hold, config, ALPHAS[policy])
        if policy in BETAS:
            return gnn_controller(hold, config, gnn, sha, BETAS[policy])
        if policy == "ewma-tuned":
            return ewma_controller(hold, config, selection["alpha"])
        if policy == "gnn-tuned":
            return gnn_controller(hold, config, gnn, sha, selection["beta"])
        raise ValueError("Unknown v10 policy: " + policy)
    return factory


def arm_flags(policy):
    if policy not in (*TUNE_ARMS, *TEST_ARMS):
        raise ValueError("Unknown v10 arm: " + policy)
    return _flags("lq-route")


def _run(loader, protocol_path, cell_id, attempt, output):
    phase = loader(protocol_path)
    cell = next(cell for cell in phase["cells"] if cell["cell_id"] == cell_id)
    arm_flags(cell["policy"])
    return collect_case(protocol_path, cell_id, attempt, output, loader=loader, factory_builder=make_factory,
                        env_class=_env("lq-route"), validate_new=False)


run_case_tune = partial(_run, load_tune)
run_case = partial(_run, load)

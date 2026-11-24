"""v10 contract: selection rule, GNN smoothing semantics, seed plans."""
import numpy as np

from src.control.service_v3 import protocol as p3
from src.control.service_v10.analysis import choose
from src.control.service_v10.study import ALPHAS, BETAS, V10, V10T, SmoothedForecastQuality


def test_choose_takes_best_and_breaks_ties_towards_less_smoothing():
    means = {"ewma-a50": .90, "ewma-a30": .95, "ewma-a20": .95, "ewma-a10": .94, "ewma-a05": .80}
    assert choose(means, ALPHAS) == .3
    means["ewma-a10"] = .96
    assert choose(means, ALPHAS) == .1


def test_beta_one_is_raw_forecast_and_smaller_beta_smooths():
    rx = np.full((3, 3), .9)
    forecasts = [np.full((3, 3), .2), np.full((3, 3), .8)]
    for beta, expected in ((1., .8), (.5, .5)):
        quality = SmoothedForecastQuality(beta)
        for f in forecasts:
            quality.forecast = f
            quality.update(rx)
        assert np.allclose(quality.q, expected)
        assert np.allclose(quality.raw, rx) and quality.count == 2


def test_tuning_and_test_seed_plans_are_disjoint_and_sized():
    tune, main = p3.build_phases(V10T), p3.build_phases(V10)
    seeds = lambda phases: {(s["layout_seed"], s["radio_seed"]) for rows in phases.values() for s in rows}
    assert len(tune["validation"]) == 16 and not seeds(tune) & seeds(main)
    assert len(main["validation"]) == 20 and len(main["test"]) == 48
    assert set(BETAS.values()) >= {1.}

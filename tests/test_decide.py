"""The siting optimizer disperses capacity to smooth aggregate output."""

import numpy as np
import pandas as pd
import pytest

from solarfleet.compose import Site, CouplingKernel
from solarfleet import decide


def _candidates():
    # A cluster of three near London plus three well-separated sites.
    return [
        Site("london", 51.51, -0.13, 35, 180, 1.0),
        Site("london2", 51.52, -0.12, 35, 180, 1.0),
        Site("london3", 51.50, -0.14, 35, 180, 1.0),
        Site("bristol", 51.45, -2.59, 35, 180, 1.0),
        Site("leeds", 53.80, -1.55, 35, 180, 1.0),
        Site("glasgow", 55.86, -4.25, 35, 180, 1.0),
    ]


def _params(n):
    theta = np.full(n, 0.3)
    sigma = np.full(n, 0.8)
    mu = np.full(n, -0.2)
    return theta, sigma, mu


def _times():
    return pd.date_range("2023-06-21 00:00", "2023-06-21 23:30", freq="30min",
                         tz="UTC")


def test_optimised_weights_are_feasible():
    sites = _candidates()
    theta, sigma, mu = _params(len(sites))
    res = decide.optimise_siting(sites, CouplingKernel(), theta, sigma, mu,
                                 _times(), total_capacity=12.0)
    assert res.weights.min() >= -1e-9
    assert res.weights.sum() == pytest.approx(12.0, rel=1e-6)


def test_min_variance_beats_equal_weight_and_concentration():
    sites = _candidates()
    theta, sigma, mu = _params(len(sites))
    k, times, cap = CouplingKernel(), _times(), 12.0

    opt = decide.optimise_siting(sites, k, theta, sigma, mu, times, cap)
    eq = decide.equal_weight(sites, k, theta, sigma, mu, times, cap)
    conc = decide.concentrated(sites, k, theta, sigma, mu, times, cap)

    # Optimised variance is the lowest (it is the minimiser), and concentration is
    # the worst — dispersion smooths the aggregate.
    assert opt.output_std < eq.output_std
    assert eq.output_std < conc.output_std


def test_optimiser_avoids_piling_into_the_correlated_cluster():
    # The three London sites are mutually near-perfectly correlated; a
    # variance-minimiser should not concentrate there. Effective number of sites
    # used should comfortably exceed 1, and the dispersed sites should collectively
    # get more than their per-site share.
    sites = _candidates()
    theta, sigma, mu = _params(len(sites))
    res = decide.optimise_siting(sites, CouplingKernel(), theta, sigma, mu,
                                 _times(), total_capacity=12.0)
    assert res.concentration() > 2.5           # not all crammed into one/two sites
    cluster = res.weights[:3].sum()
    spread = res.weights[3:].sum()
    assert spread > cluster                    # decorrelated sites preferred


def test_output_floor_trades_variance_for_expected_output():
    # Raising the required expected-output floor cannot lower the achievable
    # minimum variance (the frontier is monotone).
    sites = _candidates()
    theta, sigma, mu = _params(len(sites))
    k, times, cap = CouplingKernel(), _times(), 12.0

    free = decide.optimise_siting(sites, k, theta, sigma, mu, times, cap)
    hi_target = 0.98 * decide.concentrated(sites, k, theta, sigma, mu, times, cap).expected_output
    constrained = decide.optimise_siting(sites, k, theta, sigma, mu, times, cap,
                                         target_output=hi_target)
    assert constrained.expected_output >= hi_target - 1e-6
    assert constrained.output_std >= free.output_std - 1e-9


def test_wider_kernel_decay_makes_dispersion_more_valuable():
    # With faster spatial decorrelation (larger c1), spreading buys more variance
    # reduction relative to equal weight.
    sites = _candidates()
    theta, sigma, mu = _params(len(sites))
    times, cap = _times(), 12.0

    def ratio(c1):
        k = CouplingKernel(c1=c1)
        opt = decide.optimise_siting(sites, k, theta, sigma, mu, times, cap)
        eq = decide.equal_weight(sites, k, theta, sigma, mu, times, cap)
        return opt.output_std / eq.output_std

    # More decorrelation (bigger c1) => optimiser gains more over equal weight.
    assert ratio(2e-4) < ratio(1e-6)

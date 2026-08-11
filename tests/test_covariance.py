"""Full-covariance form (b): the DSL expresses it, and it reproduces distance-decay.

This is the §9 headline. The plan pre-registered that form (b) would NOT be
expressible in the DSL. It is: the Cholesky matvec ``L @ xi`` is
``each(S, i, dot(slice(lflat, i*S, S), xi))``. These tests confirm the config
runs, keeps the hard invariants, and — the thing the single-factor form (a)
structurally cannot do — produces log-K correlation that decays with distance and
an aggregate that smooths as the fleet disperses.
"""

import numpy as np
import pandas as pd
import pytest

from solarfleet import geometry as geo
from solarfleet.compose import (
    Site, CouplingKernel, build_covariance_config, correlation_matrix,
    simulate_covariance,
)


def _spread_fleet():
    # London + Reading are ~60 km apart; Glasgow is ~550 km from both.
    return [
        Site("london", 51.51, -0.13, tilt=35, surface_azimuth=180, kwp=4.0),
        Site("reading", 51.46, -0.97, tilt=35, surface_azimuth=180, kwp=4.0),
        Site("glasgow", 55.86, -4.25, tilt=35, surface_azimuth=180, kwp=4.0),
    ]


def _day():
    return pd.date_range("2023-06-21 00:00", "2023-06-21 23:30", freq="30min",
                         tz="UTC")


def test_covariance_config_runs_and_keeps_invariants():
    sites, times = _spread_fleet(), _day()
    df = simulate_covariance(sites, times, seed=3)
    assert len(df) == len(times)
    t = times.tz_convert("UTC").tz_localize(None).values
    for i, s in enumerate(sites):
        gen = df[f"sites[{len(sites) + i}]"].values  # generation slots come after log K
        assert np.all(gen >= 0)
        alt, _ = geo.solar_position(s.latitude, s.longitude, t)
        assert np.all(gen[alt <= 0.0] == 0.0), s.name
    # Conservation: fleet == sum of site generation.
    site_gen = sum(df[f"sites[{len(sites) + i}]"] for i in range(len(sites)))
    assert np.allclose(df["fleet[0]"].values, site_gen.values, rtol=1e-9, atol=1e-6)


def test_empirical_logk_correlation_matches_the_kernel():
    # Run long enough to estimate correlation; log K dynamics are independent of
    # the irradiance driver, so a long synthetic clock is fine.
    sites = _spread_fleet()
    kernel = CouplingKernel(sigma=0.8, theta=0.3)
    times = pd.date_range("2023-06-21", periods=4001, freq="30min", tz="UTC")
    df = simulate_covariance(sites, times, kernel=kernel, seed=11)

    logk = np.column_stack([df[f"sites[{i}]"].values[500:] for i in range(len(sites))])
    emp = np.corrcoef(logk, rowvar=False)
    target = correlation_matrix(sites, kernel)

    # Near pair (london-reading) correlates much more than a far pair (london-glasgow).
    assert emp[0, 1] > emp[0, 2] + 0.15
    # Empirical correlation tracks the kernel within sampling error.
    assert emp[0, 1] == pytest.approx(target[0, 1], abs=0.08)
    assert emp[0, 2] == pytest.approx(target[0, 2], abs=0.08)


def test_dispersion_smooths_the_aggregate():
    # The flagship claim: at fixed total capacity, spreading sites apart lowers the
    # variability of aggregate log K. Compare a compact fleet (all near London) with
    # a dispersed one (London, Reading, Glasgow) under the same kernel and seed.
    kernel = CouplingKernel(sigma=0.8, theta=0.3)
    times = pd.date_range("2023-06-21", periods=4001, freq="30min", tz="UTC")

    compact = [
        Site("a", 51.51, -0.13, 35, 180, kwp=4.0),
        Site("b", 51.52, -0.12, 35, 180, kwp=4.0),   # ~1.4 km away
        Site("c", 51.50, -0.14, 35, 180, kwp=4.0),   # ~1.4 km away
    ]
    dispersed = _spread_fleet()

    def aggregate_logk_std(sites):
        df = simulate_covariance(sites, times, kernel=kernel, seed=11)
        agg = sum(df[f"sites[{i}]"] for i in range(len(sites)))
        return agg.values[500:].std()

    assert aggregate_logk_std(dispersed) < aggregate_logk_std(compact)

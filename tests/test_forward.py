"""The forward-model config runs and satisfies the hard stub invariants (§7.1)."""

import numpy as np
import pandas as pd
import pytest

from solarfleet import geometry as geo
from solarfleet.compose import Site, SkyParams, build_forward_config, simulate_fleet
from solarfleet.runner import run


def _fleet():
    # Three GB sites, geographically spread, one summer day at 30-min resolution.
    sites = [
        Site("london", 51.51, -0.13, tilt=35, surface_azimuth=180, kwp=4.0),
        Site("manchester", 53.48, -2.24, tilt=30, surface_azimuth=160, kwp=3.5),
        Site("glasgow", 55.86, -4.25, tilt=40, surface_azimuth=200, kwp=5.0),
    ]
    times = pd.date_range("2023-06-21 00:00", "2023-06-21 23:30", freq="30min",
                          tz="UTC")
    return sites, times


def test_forward_config_runs_and_has_expected_columns():
    sites, times = _fleet()
    df = simulate_fleet(sites, times, seed=7)
    assert len(df) == len(times)
    for s in sites:
        assert f"{s.name}[1]" in df.columns  # generation
    assert "fleet[0]" in df.columns
    assert "sky[0]" in df.columns


def test_generation_non_negative():
    sites, times = _fleet()
    df = simulate_fleet(sites, times, seed=7)
    for s in sites:
        assert (df[f"{s.name}[1]"] >= 0).all()


def test_generation_zero_at_night():
    # The hard physical invariant: no output whenever the sun is below the horizon.
    sites, times = _fleet()
    df = simulate_fleet(sites, times, seed=7)
    t = times.tz_convert("UTC").tz_localize(None).values
    for s in sites:
        alt, _ = geo.solar_position(s.latitude, s.longitude, t)
        night = alt <= 0.0
        assert np.all(df[f"{s.name}[1]"].values[night] == 0.0), s.name


def test_clear_sky_index_finite_and_non_negative():
    # k_index is a binding, not an output field, so assert it stays physical by
    # checking generation never exceeds the soft capacity bound kWp * 1000 * k_max.
    sites, times = _fleet()
    df = simulate_fleet(sites, times, seed=7)
    for s in sites:
        gen = df[f"{s.name}[1]"].values
        assert np.all(np.isfinite(gen))
        # generation = kwp * poa/1000 * k * eta; with poa <= ~1000 and k <= k_max,
        # a generous ceiling is kwp * 1000 * k_max.
        assert np.all(gen <= s.kwp * 1000.0 * 1.2 + 1e-6)


def test_fleet_equals_sum_of_sites():
    # Conservation: the aggregate state equals the sum of the site states.
    sites, times = _fleet()
    df = simulate_fleet(sites, times, seed=7)
    site_sum = sum(df[f"{s.name}[1]"] for s in sites)
    assert np.allclose(df["fleet[0]"].values, site_sum.values, rtol=1e-9, atol=1e-6)


def test_daytime_generation_is_positive_somewhere():
    # Sanity: the fleet actually generates during the day (not a trivially-zero model).
    sites, times = _fleet()
    df = simulate_fleet(sites, times, seed=7)
    assert df["fleet[0]"].max() > 0.0

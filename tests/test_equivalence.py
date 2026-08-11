"""Phase 4 — the declarative twin, verified against the numpy reference.

With no bespoke Go there is no shared-RNG 1e-12 oracle. So:
  1. deterministic (sigma=0): config == reference to floating-point tolerance;
  2. distributional (sigma>0): config and reference agree on statistics;
  3. dt scaling: the sqrt(dt) innovation term behaves;
  4. mutation sentinels: a wrong model is actually caught.

Never change the model to make the twin agree — a disagreement is a finding.
"""

import numpy as np
import pandas as pd
import pytest

from solarfleet import geometry as geo
from solarfleet.compose import (
    Site, CouplingKernel, build_covariance_config, correlation_matrix,
)
from solarfleet.reference import covariance_reference
from solarfleet.runner import run


def _sites():
    return [
        Site("london", 51.51, -0.13, 35, 180, 4.0),
        Site("reading", 51.46, -0.97, 30, 170, 3.5),
        Site("glasgow", 55.86, -4.25, 40, 200, 5.0),
    ]


def _config_arrays(df, s):
    logk = np.column_stack([df[f"sites[{i}]"].values for i in range(s)])
    gen = np.column_stack([df[f"sites[{s + i}]"].values for i in range(s)])
    return logk, gen, df["fleet[0]"].values


# --- 1. deterministic exactness (the strongest available oracle) ---------------

def test_deterministic_config_matches_reference_exactly():
    # sigma=0 => no draws => the config and the independent numpy reference must
    # agree to floating point. init_logk != mu so the OU relaxation is non-trivial
    # and the update expression is genuinely exercised, not a fixed point.
    sites = _sites()
    kernel = CouplingKernel(sigma=0.0, theta=0.3)
    times = pd.date_range("2023-06-21 00:00", "2023-06-21 23:30", freq="30min",
                          tz="UTC")
    init = [0.3, -0.4, 0.1]

    df = run(build_covariance_config(sites, times, kernel, seed=1, init_logk=init))
    c_logk, c_gen, c_fleet = _config_arrays(df, len(sites))
    r_logk, r_gen, r_fleet = covariance_reference(sites, times, kernel, seed=1,
                                                  init_logk=init)

    assert np.allclose(c_logk, r_logk, rtol=1e-9, atol=1e-9)
    assert np.allclose(c_gen, r_gen, rtol=1e-9, atol=1e-9)
    assert np.allclose(c_fleet, r_fleet, rtol=1e-9, atol=1e-6)


def test_deterministic_generation_zero_at_night_both_sides():
    sites = _sites()
    kernel = CouplingKernel(sigma=0.0, theta=0.3)
    times = pd.date_range("2023-06-21", "2023-06-21 23:30", freq="30min", tz="UTC")
    df = run(build_covariance_config(sites, times, kernel, seed=1))
    _, c_gen, _ = _config_arrays(df, len(sites))
    t = times.tz_convert("UTC").tz_localize(None).values
    for i, s in enumerate(sites):
        alt, _ = geo.solar_position(s.latitude, s.longitude, t)
        assert np.all(c_gen[:, i][alt <= 0] == 0.0)


# --- 2. distributional agreement on the stochastic parts -----------------------

def test_distributional_agreement_of_logk_statistics():
    # Validate BOTH the config and the independent reference against the analytic
    # stationary statistics of the discrete OU — stronger than run-vs-run, which
    # is only pinned to Monte-Carlo error. Stationary mean = mu; discrete Euler
    # stationary variance = sigma^2 / (theta (2 - theta dt)); correlation = rho.
    sites = _sites()
    kernel = CouplingKernel(sigma=0.7, theta=0.3)
    dt = 1.0
    times = pd.date_range("2023-06-21", periods=8001, freq="30min", tz="UTC")

    df = run(build_covariance_config(sites, times, kernel, seed=2))
    c_logk, _, _ = _config_arrays(df, len(sites))
    r_logk, _, _ = covariance_reference(sites, times, kernel, seed=999)

    mu = np.array([s.mu for s in sites])
    sd = kernel.sigma / np.sqrt(kernel.theta * (2.0 - kernel.theta * dt))
    rho = correlation_matrix(sites, kernel)
    burn = 500

    for logk in (c_logk[burn:], r_logk[burn:]):
        assert np.allclose(logk.mean(0), mu, atol=0.1)
        assert np.allclose(logk.std(0), sd, rtol=0.12)
        assert np.allclose(np.corrcoef(logk, rowvar=False), rho, atol=0.07)


# --- 3. dt scaling: sqrt(dt) on the innovation ---------------------------------

def test_innovation_variance_scales_with_dt():
    # One-step innovation variance of log K is sigma^2 * dt (drift negligible over
    # one step from the stationary mean). Check it moves the right way with dt.
    sites = _sites()[:1]
    kernel = CouplingKernel(sigma=0.5, theta=0.1)

    def one_step_var(dt):
        times = pd.date_range("2023-06-21", periods=3001, freq="30min", tz="UTC")
        _, _, _ = covariance_reference(sites, times, kernel, seed=4, dt=dt)
        logk, _, _ = covariance_reference(sites, times, kernel, seed=4, dt=dt)
        d = np.diff(logk[500:, 0])
        return d.var()

    v_half, v_one, v_two = one_step_var(0.5), one_step_var(1.0), one_step_var(2.0)
    assert v_half < v_one < v_two            # variance grows with dt
    assert v_two / v_half == pytest.approx(4.0, rel=0.2)  # ~ dt scaling


# --- 4. mutation sentinels: a wrong model is caught ----------------------------

def test_mutation_dropped_transposition_term_is_caught():
    # Drop the cos(zeta - gamma) azimuth term: a south vs east panel would then
    # look identical. The geometry test suite's south>north assertion catches it,
    # verified here by constructing the mutant and confirming it fails the check.
    times = pd.date_range("2023-06-21", "2023-06-21 23:30", freq="30min").values

    def poa_mutant(surface_azimuth):
        alt, az = geo.solar_position(51.5, 0.0, times)
        dni = geo.clear_sky_normal_irradiance(alt)
        # Mutant cos-incidence with the azimuth term dropped.
        h = np.radians(alt)
        tilt = np.radians(35.0)
        cos_inc = np.sin(tilt) * np.cos(h) + np.cos(tilt) * np.sin(h)
        return dni * np.clip(cos_inc, 0, None)

    south = poa_mutant(180.0).sum()
    east = poa_mutant(90.0).sum()
    assert south == east          # mutant is blind to orientation...
    # ...whereas the real model is not:
    real_south = geo.clear_sky_poa(51.5, 0.0, times, 35.0, 180.0).sum()
    real_east = geo.clear_sky_poa(51.5, 0.0, times, 35.0, 90.0).sum()
    assert real_south > real_east


def test_mutation_wrong_clear_sky_constant_is_caught():
    # Changing A from 1353 shifts the zenith clear-sky value off 1353*0.7.
    good = geo.clear_sky_normal_irradiance(90.0)
    mutant = geo.clear_sky_normal_irradiance(90.0, a=1000.0)
    assert good == pytest.approx(1353.0 * 0.7)
    assert abs(mutant - good) > 100.0     # the mutation is detectable


# --- branch coverage: the capacity clip actually engages -----------------------

def test_capacity_clip_engages_and_matches_reference():
    # init_logk = 1.0 => exp(1.0) = 2.72 > k_max 1.2, so the clip binds at step 0.
    sites = _sites()
    kernel = CouplingKernel(sigma=0.0, theta=0.3)
    times = pd.date_range("2023-06-21 11:00", "2023-06-21 13:00", freq="30min",
                          tz="UTC")
    init = [1.0, 1.0, 1.0]
    df = run(build_covariance_config(sites, times, kernel, seed=1, init_logk=init))
    c_logk, c_gen, _ = _config_arrays(df, len(sites))
    _, r_gen, _ = covariance_reference(sites, times, kernel, seed=1, init_logk=init)

    # The clip genuinely bound (raw exp exceeds k_max), and both sides clamp it.
    assert np.exp(c_logk[0, 0]) > 1.2
    midday = c_gen[0]                       # noon, sun up, clip active
    kwp = np.array([s.kwp for s in sites])
    poa0 = np.array([geo.clear_sky_poa(s.latitude, s.longitude, times[:1].tz_convert(
        "UTC").tz_localize(None).values, s.tilt, s.surface_azimuth)[0] for s in sites])
    eta = np.array([s.eta for s in sites])
    expected = kwp * poa0 / geo.I_STC * 1.2 * eta      # k clamped to k_max
    assert np.allclose(midday, expected, rtol=1e-9)
    assert np.allclose(c_gen, r_gen, rtol=1e-9, atol=1e-9)


# --- §6.4: the engine rejects silently-ignored config keys (plan assumed it wouldn't)

def test_unknown_config_key_is_rejected():
    # The plan warned yaml.v2 silently ignores unknown keys (the state_width trap).
    # v0.15.0 rejects them with an actionable message — verified here.
    cfg = {"main": {
        "partitions": [{"name": "p", "params": {"a": [1.0]},
                        "init_state_values": [0.0], "state_history_depth": 1,
                        "seed": 1, "totally_bogus_key": [1, 2, 3]}],
        "expressions": [{"partition": "p", "fields": [{"name": "x"}],
                         "outputs": ["x + a"]}],
        "simulation": {"termination_condition": {"type": "number_of_steps",
                                                 "max_steps": 2},
                       "timestep_function": {"type": "constant", "stepsize": 1.0},
                       "init_time_value": 0.0}}}
    with pytest.raises(RuntimeError) as exc:
        run(cfg)
    assert "bogus_key" in str(exc.value)

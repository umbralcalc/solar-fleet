"""Inference recovers known parameters and the particle filter tracks the latent.

Calibration is validated by recovery: generate data with known parameters, fit,
recover them. The particle filter is validated by tracking a known latent path
and by the ESS diagnostic behaving correctly.
"""

import numpy as np
import pandas as pd
import pytest

from solarfleet import geometry as geo
from solarfleet import infer
from solarfleet.compose import Site, CouplingKernel, correlation_matrix


def _ou_path(theta, mu, sigma, n, dt=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = (x[t - 1] + theta * (mu - x[t - 1]) * dt
                + sigma * np.sqrt(dt) * rng.standard_normal())
    return x


# --- per-site OU calibration ---------------------------------------------------

def test_ou_recovery_closed_form():
    theta, mu, sigma = 0.25, -0.2, 0.3
    x = _ou_path(theta, mu, sigma, n=20000, seed=1)
    fit = infer.calibrate_site_ou(x, dt=1.0)
    assert fit.theta == pytest.approx(theta, rel=0.12)
    assert fit.mu == pytest.approx(mu, abs=0.05)
    assert fit.sigma == pytest.approx(sigma, rel=0.08)


def test_ou_level_absorbs_efficiency():
    # eta and mu are confounded: fitting log(k_eff) recovers mu + log(eta).
    theta, mu, sigma, eta = 0.3, -0.15, 0.25, 0.85
    x = _ou_path(theta, mu, sigma, n=20000, seed=2)
    logk_eff = x + np.log(eta)
    fit = infer.calibrate_site_ou(logk_eff, dt=1.0)
    assert fit.mu == pytest.approx(mu + np.log(eta), abs=0.05)
    assert fit.theta == pytest.approx(theta, rel=0.15)


def test_invert_generation_marks_night_invalid():
    site = Site("s", 51.5, -0.1, 35, 180, kwp=4.0)
    times = pd.date_range("2023-06-21", periods=48, freq="30min").values
    poa = geo.clear_sky_poa(site.latitude, site.longitude, times, site.tilt,
                            site.surface_azimuth)
    logk_true = _ou_path(0.3, -0.2, 0.2, n=48, seed=3)
    gen = np.clip(site.kwp * poa / geo.I_STC * np.exp(logk_true) * 0.85, 0, None)
    logk, valid = infer.invert_to_logk(gen, poa, site.kwp)
    assert not valid[poa < 20.0].any()          # night marked invalid
    assert np.all(np.isnan(logk[~valid]))        # NaN where invalid


# --- fleet coupling-kernel calibration -----------------------------------------

def test_coupling_decay_recovered_from_correlations():
    from solarfleet.compose import build_covariance_config
    from solarfleet.runner import run

    sites = [
        Site("london", 51.51, -0.13, 35, 180, 4.0),
        Site("reading", 51.46, -0.97, 35, 180, 4.0),
        Site("bristol", 51.45, -2.59, 35, 180, 4.0),
        Site("glasgow", 55.86, -4.25, 35, 180, 4.0),
    ]
    kernel = CouplingKernel(sigma=0.8, theta=0.3, c1=6.0e-5)
    times = pd.date_range("2023-06-21", periods=6001, freq="30min", tz="UTC")
    df = run(build_covariance_config(sites, times, kernel, seed=5))
    logk = np.column_stack([df[f"sites[{i}]"].values[500:] for i in range(len(sites))])

    fit = infer.calibrate_coupling(sites, logk, cloud_speed=kernel.cloud_speed)
    assert fit.c1 == pytest.approx(kernel.c1, rel=0.35)


# --- particle filter -----------------------------------------------------------

def _synthetic_fleet_observations(sites, kernel, ou, times, obs_sigma, seed=7):
    """Generate a known latent log-K path and noisy generation observations."""
    rng = np.random.default_rng(seed)
    t = pd.DatetimeIndex(times).tz_convert("UTC").tz_localize(None).values
    s = len(sites)
    cov = (np.outer([kernel.sigma] * s, [kernel.sigma] * s)
           * correlation_matrix(sites, kernel) + kernel.nugget * np.eye(s))
    chol = np.linalg.cholesky(cov)

    poa = np.column_stack([
        geo.clear_sky_poa(si.latitude, si.longitude, t, si.tilt, si.surface_azimuth)
        for si in sites])
    kwp = np.array([si.kwp for si in sites])

    logk = np.empty((len(t), s))
    logk[0] = ou.mu
    for k in range(1, len(t)):
        eps = rng.standard_normal(s) @ chol.T
        logk[k] = logk[k - 1] + ou.theta * (ou.mu - logk[k - 1]) + eps
    clean = kwp * poa / geo.I_STC * np.exp(logk)
    gen = np.clip(clean + obs_sigma * rng.standard_normal(clean.shape), 0, None)
    return gen, poa, logk


def test_particle_filter_tracks_latent_and_reports_ess():
    sites = [
        Site("london", 51.51, -0.13, 35, 180, 4.0),
        Site("reading", 51.46, -0.97, 35, 180, 4.0),
        Site("glasgow", 55.86, -4.25, 35, 180, 4.0),
    ]
    kernel = CouplingKernel(sigma=0.6, theta=0.3)
    ou = infer.OUFit(theta=0.3, mu=-0.2, sigma=0.6)
    times = pd.date_range("2023-06-21", periods=6 * 48, freq="30min", tz="UTC")
    # Generation is in kW (peak ~3 kW); a realistic ~0.1 kW obs noise makes the
    # likelihood informative, so weights concentrate and ESS actually degrades.
    obs_sigma = 0.1
    gen, poa, logk_true = _synthetic_fleet_observations(sites, kernel, ou, times,
                                                        obs_sigma, seed=7)

    pf = infer.ParticleFilter(sites, kernel, ou, obs_sigma, n_particles=3000, seed=1)
    res = pf.filter(gen, poa, dt=1.0)

    # ESS equals the closed-form 1/sum(w^2) and stays within [1, N].
    assert np.all(res.ess >= 1.0)
    assert np.all(res.ess <= pf.n + 1e-6)
    # Degeneracy triggers resampling at least sometimes.
    assert res.resampled.any()

    # The filter tracks the latent during daytime (where observations inform it).
    day = poa >= 20.0
    err = np.abs(res.logk_mean - logk_true)[day]
    # Prior stationary sd ~ sigma/sqrt(2 theta) ~ 0.77; filtered error is well below.
    assert np.median(err) < 0.25


def test_ess_formula_matches_definition():
    w = np.array([0.7, 0.2, 0.1])
    w = w / w.sum()
    assert infer.effective_sample_size(w) == pytest.approx(1.0 / np.sum(w ** 2))
    # Uniform weights give ESS == N.
    assert infer.effective_sample_size(np.full(50, 1 / 50)) == pytest.approx(50.0)

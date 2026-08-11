"""The decision layer: capacity siting under output-variability risk.

Invariant A territory — a downstream decision built on the calibrated model, not
in the engine. The question: given candidate rooftop sites and a total capacity
budget, how much capacity to install at each so the *aggregate* fleet output is
as steady as possible (optionally subject to an expected-output floor)?

This is where the dispersion-smoothing effect becomes an actionable lever. The
aggregate output variance is a quadratic form ``w' C w`` in the capacity
allocation ``w``, where

    C_ij = Cov[K_i, K_j] * sum_t a_i(t) a_j(t)

combines the **weather coupling** ``Cov[K_i, K_j]`` (distance-decaying, from the
calibrated kernel — form (b)) with the **co-illumination** ``sum_t a_i a_j`` (how
much two sites are sunlit at the same time, from geometry). Minimising ``w' C w``
therefore spreads capacity toward sites that are both geographically decorrelated
*and* illuminated at complementary times — the dispersion decision, made
quantitatively.

``C`` is positive semidefinite (a Hadamard product of two PSD matrices, by the
Schur product theorem), so this is a convex QP solved with SLSQP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import geometry as geo
from .compose import Site, CouplingKernel, correlation_matrix


def output_moments(sites: list[Site], kernel: CouplingKernel, theta, sigma, mu,
                   times, dt: float = 1.0):
    """Return ``(m, C)``: expected output and output-covariance per unit capacity.

    ``m_i`` is expected total output of site ``i`` per kW installed over the
    horizon; ``C_ij`` is the covariance of site outputs per unit capacity. Both
    are per-unit-capacity, so a capacity allocation ``w`` (kW) gives expected
    output ``m . w`` and output variance ``w' C w``.
    """
    theta = np.asarray(theta, float)
    sigma = np.asarray(sigma, float)
    mu = np.asarray(mu, float)
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True)) \
        .tz_convert("UTC").tz_localize(None).values

    # Per-unit-capacity clear-sky output a_i(t) = poa_i/I_STC * eta_i.
    a = np.column_stack([
        geo.clear_sky_poa(s.latitude, s.longitude, t, s.tilt, s.surface_azimuth)
        / geo.I_STC * s.eta
        for s in sites])                                   # (T, n)

    # Stationary statistics of log K (discrete Euler OU), lognormal K moments.
    v = sigma ** 2 / (theta * (2.0 - theta * dt))          # stationary var of log K
    rho = correlation_matrix(sites, kernel)
    s_logk = rho * np.sqrt(np.outer(v, v))                 # cov of log K
    ek = np.exp(mu + 0.5 * v)                              # E[K_i] (lognormal mean)
    cov_k = np.outer(ek, ek) * (np.exp(s_logk) - 1.0)      # Cov[K_i, K_j]

    co_illum = a.T @ a                                     # sum_t a_i(t) a_j(t)
    m = a.sum(axis=0) * ek                                # expected total output
    c = cov_k * co_illum                                  # output covariance
    return m, c


@dataclass
class SitingResult:
    weights: np.ndarray          # capacity (kW) per site
    expected_output: float       # m . w
    output_std: float            # sqrt(w' C w)
    m: np.ndarray
    C: np.ndarray

    def concentration(self) -> float:
        """Herfindahl 1/sum(share^2): the effective number of sites used."""
        share = self.weights / self.weights.sum()
        return float(1.0 / np.sum(share ** 2))


def optimise_siting(sites, kernel, theta, sigma, mu, times, total_capacity: float,
                    per_site_max=None, target_output: float | None = None,
                    dt: float = 1.0) -> SitingResult:
    """Minimise aggregate output variance for a capacity budget.

    ``w >= 0``, ``sum(w) = total_capacity``, ``w_i <= per_site_max_i`` (default the
    whole budget), and optionally ``m . w >= target_output``.
    """
    m, c = output_moments(sites, kernel, theta, sigma, mu, times, dt=dt)
    n = len(sites)
    if per_site_max is None:
        per_site_max = np.full(n, total_capacity)
    per_site_max = np.asarray(per_site_max, float)

    def variance(w):
        return w @ c @ w

    def variance_grad(w):
        return 2.0 * c @ w

    constraints = [{"type": "eq",
                    "fun": lambda w: w.sum() - total_capacity,
                    "jac": lambda w: np.ones(n)}]
    if target_output is not None:
        constraints.append({"type": "ineq",
                            "fun": lambda w: m @ w - target_output,
                            "jac": lambda w: m})

    bounds = [(0.0, per_site_max[i]) for i in range(n)]
    w0 = np.full(n, total_capacity / n)
    res = minimize(variance, w0, jac=variance_grad, bounds=bounds,
                   constraints=constraints, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, None)
    w *= total_capacity / w.sum()                          # renormalise tiny drift
    return SitingResult(weights=w, expected_output=float(m @ w),
                        output_std=float(np.sqrt(max(w @ c @ w, 0.0))), m=m, C=c)


def equal_weight(sites, kernel, theta, sigma, mu, times, total_capacity, dt=1.0
                 ) -> SitingResult:
    """The naive baseline: split capacity equally across all candidate sites."""
    m, c = output_moments(sites, kernel, theta, sigma, mu, times, dt=dt)
    n = len(sites)
    w = np.full(n, total_capacity / n)
    return SitingResult(w, float(m @ w), float(np.sqrt(w @ c @ w)), m, c)


def concentrated(sites, kernel, theta, sigma, mu, times, total_capacity, dt=1.0
                 ) -> SitingResult:
    """The all-in-one-site baseline (put the whole budget on the highest-yield site)."""
    m, c = output_moments(sites, kernel, theta, sigma, mu, times, dt=dt)
    w = np.zeros(len(sites))
    w[int(np.argmax(m))] = total_capacity
    return SitingResult(w, float(m @ w), float(np.sqrt(w @ c @ w)), m, c)


if __name__ == "__main__":
    # Capstone demo: calibrate against real uk_pv, then site a capacity budget
    # across the calibrated sites to minimise aggregate output variability.
    import pandas as pd
    from . import fetch as fetchmod, ingest, analysis

    years, months, root = [2024], [5, 6, 7], "data/uk_pv"
    fetchmod.fetch([(2024, m) for m in months], dest=root)
    meta = ingest.load_metadata(root + "/metadata.csv")
    ss_ids = analysis.select_sites(root, meta, years, months, n=10)
    res = analysis.calibrate_fleet(root, ss_ids, meta, years, months)

    # Keep only sites that calibrated, and carry their fitted OU parameters.
    sites, theta, sigma, mu = [], [], [], []
    for s in res["sites"]:
        f = res["ou_fits"][s.name]
        if f is None:
            continue
        sites.append(s)
        theta.append(f.theta); sigma.append(f.sigma); mu.append(f.mu)
    kernel = CouplingKernel(c1=res["kernel_fit"].c1, cloud_speed=6.0)
    day = pd.date_range("2024-06-21 00:00", "2024-06-21 23:30", freq="30min", tz="UTC")
    budget = 10.0  # kW to allocate across the calibrated sites

    eq = equal_weight(sites, kernel, theta, sigma, mu, day, budget)
    # Honest apples-to-apples: minimise variance while holding expected output at
    # the equal-weight level, so the gain is dispersion/timing, not just picking
    # low-yield (low-variance) sites.
    opt = optimise_siting(sites, kernel, theta, sigma, mu, day, budget,
                          target_output=eq.expected_output)
    print("Capacity siting on the REAL-calibrated model "
          f"({len(sites)} sites, {budget} kW budget, expected output held fixed):")
    print(f"  {'site':>8} {'lat':>5}  optimal_kW  equal_kW")
    for s, w in zip(sites, opt.weights):
        print(f"  {s.name:>8} {s.latitude:5.1f}   {w:8.2f}   {budget/len(sites):7.2f}")
    red = 100 * (1 - opt.output_std / eq.output_std)
    print(f"  same expected output ({opt.expected_output:.1f}); output std "
          f"{opt.output_std:.2f} vs equal-weight {eq.output_std:.2f} "
          f"({red:.0f}% lower variability), effective sites {opt.concentration():.1f}/{len(sites)}")

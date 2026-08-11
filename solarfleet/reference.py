"""An independent numpy reference for the full-covariance forward model.

This is the twin's oracle. With no bespoke Go, the config cannot be checked
step-for-step at 1e-12 against a shared RNG stream (Python and Go PCG differ). So
verification splits:

* **deterministic** (``sigma = 0``): the config must match this reference to
  floating-point tolerance — same OU update, same clamps, same generation
  formula, no draws;
* **distributional** (``sigma > 0``): config and reference are independent draws,
  compared on summary statistics within Monte-Carlo error.

Keeping this reference deliberately separate from ``compose.py`` (which emits the
config) is what makes the comparison meaningful — two implementations of the same
equations, not one calling the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geometry as geo
from .compose import Site, CouplingKernel, cholesky_factor, K_MAX_DEFAULT


def covariance_reference(sites: list[Site], times, kernel: CouplingKernel,
                         *, seed: int = 12345, k_max: float = K_MAX_DEFAULT,
                         dt: float = 1.0, init_logk=None):
    """Run the reference form (b) and return ``(logk, gen, fleet)`` arrays.

    ``logk`` and ``gen`` are ``(T, S)``; ``fleet`` is ``(T,)``. The RNG here is
    numpy's, unrelated to the engine's — so this matches the config exactly only
    when ``sigma == 0`` (no draws), and distributionally otherwise.
    """
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    t_naive = times.tz_convert("UTC").tz_localize(None).values
    s = len(sites)
    n = len(times)

    mu = np.array([site.mu for site in sites])
    kwp = np.array([site.kwp for site in sites])
    eta = np.array([site.eta for site in sites])
    lower = cholesky_factor(sites, kernel)

    poa = np.column_stack([
        geo.clear_sky_poa(site.latitude, site.longitude, t_naive,
                          site.tilt, site.surface_azimuth)
        for site in sites])

    logk = np.empty((n, s))
    gen = np.empty((n, s))
    logk[0] = mu if init_logk is None else np.broadcast_to(
        np.asarray(init_logk, float), (s,))
    gen[0] = np.clip(kwp * poa[0] / geo.I_STC
                     * np.minimum(np.exp(logk[0]), k_max) * eta, 0.0, None)

    rng = np.random.default_rng(seed)
    for t in range(1, n):
        innov = rng.standard_normal(s) @ lower.T
        logk[t] = logk[t - 1] + kernel.theta * (mu - logk[t - 1]) * dt \
            + np.sqrt(dt) * innov
        k = np.clip(np.exp(logk[t]), 0.0, k_max)
        gen[t] = np.clip(kwp * poa[t] / geo.I_STC * k * eta, 0.0, 1e12)

    return logk, gen, gen.sum(axis=1)

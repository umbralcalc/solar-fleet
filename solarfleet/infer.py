"""Inference: calibration and filtering. Invariant A — this all lives here, in
Python, never in the engine.

The clear-sky backbone is **not** fitted — it is physics. What is fitted:

* per-site Ornstein-Uhlenbeck parameters of the log clear-sky index (reversion
  speed, volatility, and the effective level), by closed-form AR(1) regression;
* the fleet coupling kernel (decay of cross-site correlation with separation),
  from the empirical correlation-vs-distance relationship.

And what is estimated online: the latent clear-sky-index field given observed
generation, by a bootstrap **particle filter** that tracks its own effective
sample size — the degeneracy diagnostic the engine's ``posterior_estimation``
does not provide (see ``STOCHADEX_GAPS.md``).

Identifiability note (recorded in FINDINGS.md): from generation and clear-sky POA
alone, the system efficiency ``eta`` and the OU mean of ``log K`` are confounded —
only ``eta * exp(mu)`` sets the generation level. So calibration recovers an
*effective* level ``mu_eff = mu + log(eta)``, plus the dynamics (``theta``,
``sigma``) which are identified independently of the level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from .compose import Site, CouplingKernel, correlation_matrix, haversine_km


# ------------------------------------------------------------------------------
# Per-site OU calibration (closed-form AR(1))
# ------------------------------------------------------------------------------

@dataclass
class OUFit:
    """Ornstein-Uhlenbeck parameters of log K (level is the effective mu)."""

    theta: float   # reversion speed
    mu: float      # effective long-run mean (mu_true + log eta)
    sigma: float   # innovation volatility


def calibrate_site_ou(logk: np.ndarray, dt: float = 1.0) -> OUFit:
    """Fit OU parameters from a consecutive log-K series by AR(1) OLS.

    The Euler OU step ``x_{t+1} = x_t + theta (mu - x_t) dt + sigma sqrt(dt) e``
    is the AR(1) ``x_{t+1} = a + b x_t + noise`` with ``b = 1 - theta dt``,
    ``a = theta mu dt`` and ``noise sd = sigma sqrt(dt)``. So OLS of ``x_{t+1}``
    on ``x_t`` recovers all three in closed form.
    """
    x, y = logk[:-1], logk[1:]
    return _ou_from_pairs(x, y, dt)


def calibrate_site_ou_daytime(logk: np.ndarray, valid: np.ndarray, dt: float = 1.0
                             ) -> OUFit:
    """Fit OU from a gapped series, using only consecutive valid (daytime) pairs."""
    both = valid[:-1] & valid[1:]
    x, y = logk[:-1][both], logk[1:][both]
    return _ou_from_pairs(x, y, dt)


def _ou_from_pairs(x: np.ndarray, y: np.ndarray, dt: float) -> OUFit:
    if len(x) < 3:
        raise ValueError("need at least 3 consecutive pairs to fit OU")
    b, a = np.polyfit(x, y, 1)          # y = b x + a
    resid = y - (b * x + a)
    b = min(max(b, 1e-6), 1 - 1e-9)      # keep 0 < b < 1 so theta > 0
    theta = (1.0 - b) / dt
    mu = a / (1.0 - b)
    sigma = resid.std(ddof=2) / np.sqrt(dt)
    return OUFit(theta=float(theta), mu=float(mu), sigma=float(sigma))


def invert_to_logk(generation: np.ndarray, poa: np.ndarray, kwp: float,
                   i_stc: float = geo.I_STC, poa_floor: float = 20.0):
    """Invert generation to the effective log clear-sky index, per timestamp.

    ``k_eff = generation / (kWp * poa / I_STC) = eta * K``; ``log k_eff`` is what
    the OU calibration consumes. Returns ``(logk, valid)`` where ``valid`` marks
    daytime timestamps (``poa >= poa_floor``) with positive generation; ``logk``
    is NaN elsewhere. The floor avoids the near-horizon regime where a tiny POA
    denominator makes ``k_eff`` explode.
    """
    generation = np.asarray(generation, dtype=float)
    poa = np.asarray(poa, dtype=float)
    valid = (poa >= poa_floor) & (generation > 0.0)
    k_eff = np.full_like(generation, np.nan)
    denom = kwp * poa / i_stc
    k_eff[valid] = generation[valid] / denom[valid]
    logk = np.full_like(generation, np.nan)
    logk[valid] = np.log(k_eff[valid])
    return logk, valid


# ------------------------------------------------------------------------------
# Fleet coupling-kernel calibration
# ------------------------------------------------------------------------------

@dataclass
class KernelFit:
    c1: float
    clearness_power: float
    cloud_speed: float


def empirical_correlation(logk_matrix: np.ndarray) -> np.ndarray:
    """Cross-site correlation matrix of the (T, S) effective log-K matrix."""
    return np.corrcoef(logk_matrix, rowvar=False)


def calibrate_coupling(sites: list[Site], logk_matrix: np.ndarray,
                       cloud_speed: float = 6.0,
                       clearness_power: float = 1.0) -> KernelFit:
    """Fit the decay rate ``c1`` from empirical correlation vs temporal distance.

    With ``clearness_power`` fixed, ``rho = exp(-c1 * td) ** p`` gives
    ``log(rho) = -c1 p * td``, so a no-intercept least-squares of ``log(rho_ij)``
    on ``td_ij`` over site pairs recovers ``c1``. (``cloud_speed`` and ``c1`` are
    degenerate through ``td = dist / cloud_speed``; ``cloud_speed`` is held.)
    """
    emp = empirical_correlation(logk_matrix)
    tds, logs = [], []
    n = len(sites)
    for i in range(n):
        for j in range(i + 1, n):
            r = emp[i, j]
            if r <= 0:
                continue
            d_m = haversine_km(sites[i].latitude, sites[i].longitude,
                               sites[j].latitude, sites[j].longitude) * 1000.0
            tds.append(d_m / cloud_speed)
            logs.append(np.log(r))
    tds = np.asarray(tds)
    logs = np.asarray(logs)
    # No-intercept fit: slope = sum(td*log)/sum(td^2) = -c1 * p.
    slope = float(np.sum(tds * logs) / np.sum(tds * tds))
    c1 = -slope / clearness_power
    return KernelFit(c1=c1, clearness_power=clearness_power, cloud_speed=cloud_speed)


# ------------------------------------------------------------------------------
# Bootstrap particle filter for the latent clear-sky-index field
# ------------------------------------------------------------------------------

def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = 1 / sum(w_i^2) for normalised weights — the degeneracy diagnostic."""
    w = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(w * w))


def _systematic_resample(weights: np.ndarray, rng) -> np.ndarray:
    """Low-variance systematic resampling; returns particle indices to keep."""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


@dataclass
class PFResult:
    logk_mean: np.ndarray   # (T, S) posterior mean of the latent log K
    ess: np.ndarray         # (T,) effective sample size per step
    resampled: np.ndarray   # (T,) bool, whether resampling fired that step


class ParticleFilter:
    """Bootstrap PF for the latent log clear-sky index of a fleet.

    State: ``log K`` per site, evolving by the correlated OU of form (b).
    Observation: ``gen = kWp * poa/I_STC * exp(log K)`` with Gaussian noise; at
    night (``poa`` below the floor) the observation is uninformative and weights
    are left unchanged. Resamples systematically when ESS falls below
    ``ess_threshold * n_particles``.
    """

    def __init__(self, sites: list[Site], kernel: CouplingKernel, ou: OUFit,
                 obs_sigma: float, n_particles: int = 2000, seed: int = 0,
                 poa_floor: float = 20.0, ess_threshold: float = 0.5):
        self.sites = sites
        self.s = len(sites)
        self.ou = ou
        self.obs_sigma = obs_sigma
        self.n = n_particles
        self.rng = np.random.default_rng(seed)
        self.poa_floor = poa_floor
        self.ess_threshold = ess_threshold
        self.kwp = np.array([site.kwp for site in sites])

        cov = (np.outer([kernel.sigma] * self.s, [kernel.sigma] * self.s)
               * correlation_matrix(sites, kernel) + kernel.nugget * np.eye(self.s))
        self.chol = np.linalg.cholesky(cov)

    def filter(self, generation: np.ndarray, poa: np.ndarray, dt: float = 1.0
              ) -> PFResult:
        generation = np.asarray(generation, dtype=float)  # (T, S)
        poa = np.asarray(poa, dtype=float)
        t_steps = generation.shape[0]

        # Initialise particles at the OU stationary distribution.
        stat_sd = self.ou.sigma / np.sqrt(2.0 * self.ou.theta)
        particles = self.ou.mu + stat_sd * self.rng.standard_normal((self.n, self.s))

        logk_mean = np.zeros((t_steps, self.s))
        ess = np.zeros(t_steps)
        resampled = np.zeros(t_steps, dtype=bool)

        for t in range(t_steps):
            # Propagate: correlated OU step.
            eps = self.rng.standard_normal((self.n, self.s)) @ self.chol.T
            particles = (particles
                         + self.ou.theta * (self.ou.mu - particles) * dt
                         + np.sqrt(dt) * eps)

            # Weight by the observation likelihood, per informative (daytime) site.
            day = poa[t] >= self.poa_floor
            if np.any(day):
                pred = (self.kwp[day] * poa[t][day] / geo.I_STC
                        * np.exp(particles[:, day]))
                resid = generation[t][day] - pred
                loglik = -0.5 * np.sum((resid / self.obs_sigma) ** 2, axis=1)
                loglik -= loglik.max()
                w = np.exp(loglik)
                w /= w.sum()
            else:
                w = np.full(self.n, 1.0 / self.n)

            logk_mean[t] = w @ particles
            ess[t] = effective_sample_size(w)

            if ess[t] < self.ess_threshold * self.n:
                idx = _systematic_resample(w, self.rng)
                particles = particles[idx]
                resampled[t] = True

        return PFResult(logk_mean=logk_mean, ess=ess, resampled=resampled)

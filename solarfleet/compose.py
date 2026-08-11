"""Assemble the fleet forward-model as a stochadex config (factor form).

This is the partition-graph assembly. It mirrors the shared-latent-drives-many-
sites structure of ``bathing-water-forecaster``'s ``regional.go``, but in pure
config:

    sky   : one shared regional cloud-anomaly OU latent  z(t)
    site_i: per-site idiosyncratic OU noise u_i(t), loaded on z, times the
            numpy-computed clear-sky plane-of-array driver -> generation (W)
    fleet : sum of site generation (the aggregate is a first-class state)

**Factor form (a).** The clear-sky index is modelled on log K:

    log K_i(t) = mu_i + lambda_i * z(t) + u_i(t)

with ``z`` a mean-reverting regional factor (shared) and ``u_i`` an independent
per-site mean-zero OU noise. ``lambda_i`` is the site's loading on the regional
factor. This is cheap, allocation-free and config-expressible. Its known limit —
a single factor gives pairwise correlation ``~ lambda_i * lambda_j`` with **no
distance dependence** — is exactly why the full-covariance form (b) exists; see
``FINDINGS.md`` and the dispersion claim.

The deterministic clear-sky irradiance is computed in :mod:`solarfleet.geometry`
and injected as a per-site param series read by step (``slice(series, step, 1)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import geometry as geo


@dataclass
class Site:
    """A single rooftop PV system: physical config + factor-model parameters."""

    name: str
    latitude: float
    longitude: float
    tilt: float             # degrees from horizontal
    surface_azimuth: float  # degrees clockwise from north (180 = due south)
    kwp: float              # nameplate capacity, kW at STC

    # Factor-model parameters for log clear-sky index.
    mu: float = -0.2        # stationary mean of log K (K ~ 0.8 on a typical day)
    lam: float = 0.6        # loading on the shared regional cloud factor z
    theta_u: float = 0.4    # reversion speed of the idiosyncratic OU noise
    sigma_u: float = 0.25   # volatility of the idiosyncratic OU noise
    eta: float = 0.85       # effective system efficiency (derate)


@dataclass
class SkyParams:
    """The shared regional cloud-anomaly OU latent z(t)."""

    theta: float = 0.3      # reversion speed
    sigma: float = 0.8      # volatility (drives fleet-wide covariance)
    mu: float = 0.0         # long-run mean (0 => neutral cloud anomaly)
    init: float = 0.0


K_MAX_DEFAULT = 1.2  # soft cap on clear-sky index (systems can briefly exceed 1)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two lat/lon points (degrees)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def build_forward_config(
    sites: list[Site],
    times,
    sky: SkyParams | None = None,
    *,
    seed: int = 12345,
    k_max: float = K_MAX_DEFAULT,
    stepsize: float = 1.0,
) -> dict:
    """Build the factor-form forward-model config for a fleet over ``times``.

    ``times`` is a UTC datetime-like sequence of length N; the run has N steps
    (one per timestamp, with row 0 the init row).
    """
    sky = sky or SkyParams()
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    n_steps = len(times) - 1  # row 0 is the init row
    times_naive = times.tz_convert("UTC").tz_localize(None).values

    partitions: list[dict] = []
    expressions: list[dict] = []

    # --- sky: shared regional cloud-anomaly latent -----------------------------
    partitions.append({
        "name": "sky",
        "iteration": {"type": "ornstein_uhlenbeck"},
        "params": {"thetas": [sky.theta], "mus": [sky.mu], "sigmas": [sky.sigma]},
        "init_state_values": [sky.init],
        "state_history_depth": 1,
        "seed": seed,
    })

    # --- one partition per site ------------------------------------------------
    fleet_upstreams: dict[str, dict] = {}
    fleet_terms: list[str] = []
    for k, s in enumerate(sites):
        poa = geo.clear_sky_poa(
            s.latitude, s.longitude, times_naive, s.tilt, s.surface_azimuth
        )
        poa_series = [float(v) for v in poa]
        # Generation at the init row (step 0), with u=0 and z=sky.init.
        k0 = min(np.exp(s.mu + s.lam * sky.init), k_max)
        gen0 = max(s.kwp * poa_series[0] / geo.I_STC * k0 * s.eta, 0.0)

        partitions.append({
            "name": s.name,
            "params": {
                "mu": [s.mu],
                "lam": [s.lam],
                "theta_u": [s.theta_u],
                "sigma_u": [s.sigma_u],
                "kwp": [s.kwp],
                "eta": [s.eta],
                "i_stc": [geo.I_STC],
                "k_max": [k_max],
                "poa": poa_series,
            },
            "params_from_upstream": {"z": {"upstream": "sky"}},
            "init_state_values": [0.0, gen0],
            "state_history_depth": 1,
            "seed": seed + 1 + k,
        })
        expressions.append({
            "partition": s.name,
            "fields": [{"name": "u"}, {"name": "gen"}],
            "bindings": [
                {"name": "u_next",
                 "expr": "u + theta_u * (0 - u) * dt + sigma_u * sqrt(dt) * shared(normal(0, 1))"},
                {"name": "logk", "expr": "mu + lam * z[0] + u_next"},
                {"name": "kidx", "expr": "clamp(exp(logk), 0, k_max)"},
                {"name": "poa_now", "expr": "slice(poa, step, 1)"},
            ],
            "outputs": [
                "u_next",
                "clamp(kwp * poa_now / i_stc * kidx * eta, 0, 1e12)",
            ],
        })
        alias = f"g{k}"
        fleet_upstreams[alias] = {"upstream": s.name}
        fleet_terms.append(f"{alias}[1]")

    # --- fleet: aggregate generation (first-class state) -----------------------
    partitions.append({
        "name": "fleet",
        "params": {},
        "params_from_upstream": fleet_upstreams,
        "init_state_values": [0.0],
        "state_history_depth": 1,
        "seed": seed,
    })
    expressions.append({
        "partition": "fleet",
        "fields": [{"name": "total"}],
        "outputs": [" + ".join(fleet_terms) if fleet_terms else "0"],
    })

    return {
        "main": {
            "partitions": partitions,
            "expressions": expressions,
            "simulation": {
                "output_condition": {"type": "every_step"},
                "termination_condition": {
                    "type": "number_of_steps", "max_steps": n_steps,
                },
                "timestep_function": {"type": "constant", "stepsize": stepsize},
                "init_time_value": 0.0,
            },
        }
    }


@dataclass
class CouplingKernel:
    """Distance-decaying spatial coupling for the full-covariance form (b).

    ``rho(i,j) = exp(-c1 * temporal_distance(i,j)) ** clearness_power`` with
    ``temporal_distance = distance_m / cloud_speed`` (the PV-fleet-literature
    form: physical separation divided by regional cloud speed). ``sigma`` and
    ``theta`` are the (shared) innovation volatility and OU reversion speed of
    ``log K``. Because the drift is isotropic, the stationary correlation of
    ``log K`` equals ``rho`` exactly — which is what makes distance-decay real,
    unlike the single-factor form (a).
    """

    cloud_speed: float = 6.0        # m/s, regional cloud advection speed
    c1: float = 6.0e-5              # 1/s decay; ~half-correlation near 2 hours' travel
    clearness_power: float = 1.0
    sigma: float = 0.8              # per-site log K innovation volatility
    theta: float = 0.3             # OU reversion speed of log K
    nugget: float = 1e-9            # diagonal jitter to keep Cholesky PSD


def correlation_matrix(sites: list[Site], kernel: CouplingKernel) -> np.ndarray:
    """The distance-decay correlation matrix rho(i,j) for a fleet."""
    n = len(sites)
    rho = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            d_m = haversine_km(sites[i].latitude, sites[i].longitude,
                               sites[j].latitude, sites[j].longitude) * 1000.0
            td = d_m / kernel.cloud_speed
            r = np.exp(-kernel.c1 * td) ** kernel.clearness_power
            rho[i, j] = rho[j, i] = r
    return rho


def cholesky_factor(sites: list[Site], kernel: CouplingKernel) -> np.ndarray:
    """Lower-triangular ``L`` with ``L @ L.T == Sigma`` (zeros when sigma == 0)."""
    s = len(sites)
    if kernel.sigma == 0.0:
        return np.zeros((s, s))
    sig = np.full(s, kernel.sigma)
    cov = np.outer(sig, sig) * correlation_matrix(sites, kernel) + kernel.nugget * np.eye(s)
    return np.linalg.cholesky(cov)


def build_covariance_config(
    sites: list[Site],
    times,
    kernel: CouplingKernel | None = None,
    *,
    seed: int = 12345,
    k_max: float = K_MAX_DEFAULT,
    stepsize: float = 1.0,
    init_logk=None,
) -> dict:
    """Full-covariance form (b): one wide partition, Cholesky-correlated draws.

    The per-step innovation is ``L @ xi`` with ``xi ~ iid N(0,1)^S`` drawn once
    and ``L`` the Cholesky factor of ``Sigma_ij = sigma_i sigma_j rho_ij``,
    applied in the DSL as ``each(S, i, dot(slice(lflat, i*S, S), xi))`` — the
    matvec verified to work against the engine. This reproduces true
    distance-dependent coupling, which form (a) structurally cannot.
    """
    kernel = kernel or CouplingKernel()
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    n_steps = len(times) - 1
    times_naive = times.tz_convert("UTC").tz_localize(None).values
    s = len(sites)

    lower = cholesky_factor(sites, kernel)   # lower @ lower.T == cov (zeros if sigma=0)
    lflat = [float(v) for v in lower.reshape(-1)]  # row-major

    mu = np.array([site.mu for site in sites])
    kwp = np.array([site.kwp for site in sites])
    eta = np.array([site.eta for site in sites])
    logk0 = mu.copy() if init_logk is None else np.broadcast_to(
        np.asarray(init_logk, dtype=float), (s,)).copy()

    # Per-site clear-sky POA driver, laid out step-major: poa_flat[step*S + i].
    poa_cols = [
        geo.clear_sky_poa(site.latitude, site.longitude, times_naive,
                          site.tilt, site.surface_azimuth)
        for site in sites
    ]
    poa_stack = np.column_stack(poa_cols)     # (N, S)
    poa_flat = [float(v) for v in poa_stack.reshape(-1)]

    # Init row: log K at logk0 (default the stationary mean), generation there.
    k0 = np.minimum(np.exp(logk0), k_max)
    gen0 = np.clip(kwp * poa_stack[0] / geo.I_STC * k0 * eta, 0.0, None)
    init_state = [float(v) for v in logk0] + [float(v) for v in gen0]

    # 2S per-field outputs: slice the vectorised log K / generation bindings.
    outputs = [f"slice(logk_next, {i}, 1)" for i in range(s)]
    outputs += [f"slice(gen, {i}, 1)" for i in range(s)]

    partitions = [
        {
            "name": "sites",
            "params": {
                "theta": [kernel.theta],
                "mu": [float(v) for v in mu],
                "kwp": [float(v) for v in kwp],
                "eta": [float(v) for v in eta],
                "i_stc": [geo.I_STC],
                "k_max": [k_max],
                "lflat": lflat,
                "poa_flat": poa_flat,
            },
            "init_state_values": init_state,
            "state_history_depth": 1,
            "seed": seed,
        },
        {
            "name": "fleet",
            "params": {},
            "params_from_upstream": {"s": {"upstream": "sites"}},
            "init_state_values": [float(gen0.sum())],
            "state_history_depth": 1,
            "seed": seed,
        },
    ]
    expressions = [
        {
            "partition": "sites",
            "fields": [{"name": f"lk{i}"} for i in range(s)]
                      + [{"name": f"g{i}"} for i in range(s)],
            "upstreams": {"me": "sites"},
            "bindings": [
                {"name": "logk_prev", "expr": f"slice(me, 0, {s})"},
                {"name": "xi", "expr": f"iid({s}, normal(0, 1))"},
                {"name": "innov",
                 "expr": f"each({s}, i, dot(slice(lflat, i * {s}, {s}), xi))"},
                {"name": "logk_next",
                 "expr": "logk_prev + theta * (mu - logk_prev) * dt + sqrt(dt) * innov"},
                {"name": "kidx", "expr": "clamp(exp(logk_next), 0, k_max)"},
                {"name": "poa_now", "expr": f"slice(poa_flat, step * {s}, {s})"},
                {"name": "gen",
                 "expr": "clamp(kwp * poa_now / i_stc * kidx * eta, 0, 1e12)"},
            ],
            "outputs": outputs,
        },
        {
            "partition": "fleet",
            "fields": [{"name": "total"}],
            "outputs": [f"sum(slice(s, {s}, {s}))"],
        },
    ]

    return {
        "main": {
            "partitions": partitions,
            "expressions": expressions,
            "simulation": {
                "output_condition": {"type": "every_step"},
                "termination_condition": {
                    "type": "number_of_steps", "max_steps": n_steps,
                },
                "timestep_function": {"type": "constant", "stepsize": stepsize},
                "init_time_value": 0.0,
            },
        }
    }


def simulate_covariance(sites: list[Site], times,
                        kernel: CouplingKernel | None = None,
                        **kwargs) -> pd.DataFrame:
    """Build and run the full-covariance form (b); return a time-indexed frame."""
    from .runner import run

    config = build_covariance_config(sites, times, kernel, **kwargs)
    df = run(config)
    df.index = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return df


def simulate_fleet(sites: list[Site], times, sky: SkyParams | None = None,
                   **kwargs) -> pd.DataFrame:
    """Build and run the forward model; return a DataFrame indexed by UTC time.

    Columns include ``sky[0]``, ``<site>[0]`` (idiosyncratic noise u),
    ``<site>[1]`` (generation W), and ``fleet[0]`` (aggregate W).
    """
    from .runner import run

    config = build_forward_config(sites, times, sky, **kwargs)
    df = run(config)
    df.index = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return df

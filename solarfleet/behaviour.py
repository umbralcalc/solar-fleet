"""Observed structural-driver behaviour of the fleet model.

The Python analog of a stochadex catalogue entry's ``ObservedBehaviour()``: a list
of monotonic claims, each a statement + a direction + the two measured points that
witness it. These are the sign-and-monotonicity claims the model is *for* — not
calibrated magnitudes.

Three are stochastic (they run the full-covariance forward model): cloud
volatility, geographic dispersion, and mean-reversion speed all act on the
variability of the aggregate fleet clear-sky index. Three are pure geometry
(deterministic): tilt, orientation and latitude act on output shares.

The flagship is ``wider_geographic_dispersion_lowers_fleet_variability`` — it is
non-vacuous here precisely because coupling derives from geography (form (b)),
which the single-factor form (a) cannot deliver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import geometry as geo
from .compose import Site, CouplingKernel, build_covariance_config
from .runner import run


@dataclass
class Claim:
    id: str
    statement: str
    monotone: int                       # +1 rises with the driver, -1 falls
    observations: list = field(default_factory=list)  # [(label, value), ...]

    def holds(self) -> bool:
        vals = [v for _, v in self.observations]
        diffs = np.diff(vals)
        return bool(np.all(diffs > 0)) if self.monotone > 0 else bool(np.all(diffs < 0))


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------

def _base_fleet():
    return [
        Site("london", 51.51, -0.13, 35, 180, 4.0),
        Site("reading", 51.46, -0.97, 35, 180, 4.0),
        Site("bristol", 51.45, -2.59, 35, 180, 4.0),
        Site("glasgow", 55.86, -4.25, 35, 180, 4.0),
    ]


def fleet_index_variability(sites, kernel, *, periods=4001, seed=17,
                            burn=400) -> float:
    """Std of the capacity-weighted aggregate fleet clear-sky index (daytime)."""
    times = pd.date_range("2023-06-21", periods=periods, freq="30min", tz="UTC")
    df = run(build_covariance_config(sites, times, kernel, seed=seed))
    s = len(sites)
    t = times.tz_convert("UTC").tz_localize(None).values

    gen = np.column_stack([df[f"sites[{s + i}]"].values for i in range(s)])
    clearsky = np.column_stack([
        si.kwp * geo.clear_sky_poa(si.latitude, si.longitude, t, si.tilt,
                                   si.surface_azimuth) / geo.I_STC * si.eta
        for si in sites])
    fleet_gen = gen.sum(1)
    fleet_clear = clearsky.sum(1)
    day = fleet_clear > 0.5                      # daytime, non-trivial sun
    agg_k = fleet_gen[day] / fleet_clear[day]
    return float(agg_k[burn:].std())


def _seasonal_output(latitude, tilt, surface_azimuth, date):
    """Integrated clear-sky POA over a day (a proxy for daily energy)."""
    times = pd.date_range(f"{date} 00:00", f"{date} 23:30", freq="30min").values
    poa = geo.clear_sky_poa(latitude, 0.0, times, tilt, surface_azimuth)
    return float(poa.sum())


def winter_output_share(tilt) -> float:
    winter = _seasonal_output(51.5, tilt, 180.0, "2023-12-21")
    summer = _seasonal_output(51.5, tilt, 180.0, "2023-06-21")
    return winter / (winter + summer)


def annual_output(surface_azimuth) -> float:
    # A coarse annual proxy: sum over one representative day per month (the 21st).
    return sum(_seasonal_output(51.5, 35.0, surface_azimuth, f"2023-{m:02d}-21")
               for m in range(1, 13))


def summer_winter_ratio(latitude) -> float:
    summer = _seasonal_output(latitude, 35.0, 180.0, "2023-06-21")
    winter = _seasonal_output(latitude, 35.0, 180.0, "2023-12-21")
    return summer / max(winter, 1e-6)


# ------------------------------------------------------------------------------
# The claims
# ------------------------------------------------------------------------------

def observed_behaviour() -> list[Claim]:
    sites = _base_fleet()

    # 1) cloud volatility -> fleet variability (up).
    lo = fleet_index_variability(sites, CouplingKernel(sigma=0.5, theta=0.3))
    hi = fleet_index_variability(sites, CouplingKernel(sigma=1.0, theta=0.3))
    claim_vol = Claim(
        "higher_cloud_volatility_raises_fleet_variability",
        "Higher sky volatility raises the variability of aggregate fleet output",
        +1, [("sigma 0.5", lo), ("sigma 1.0", hi)])

    # 2) dispersion -> fleet variability (down) — the flagship.
    compact = [Site("a", 51.50, -0.12, 35, 180, 4.0),
               Site("b", 51.51, -0.13, 35, 180, 4.0),
               Site("c", 51.52, -0.14, 35, 180, 4.0),
               Site("d", 51.49, -0.11, 35, 180, 4.0)]
    kern = CouplingKernel(sigma=0.8, theta=0.3)
    disp_v = fleet_index_variability(sites, kern)
    comp_v = fleet_index_variability(compact, kern)
    claim_disp = Claim(
        "wider_geographic_dispersion_lowers_fleet_variability",
        "Spreading sites apart lowers aggregate variability at fixed capacity",
        -1, [("compact", comp_v), ("dispersed", disp_v)])

    # 3) mean-reversion speed -> variability (down).
    slow = fleet_index_variability(sites, CouplingKernel(sigma=0.8, theta=0.15))
    fast = fleet_index_variability(sites, CouplingKernel(sigma=0.8, theta=0.6))
    claim_rev = Claim(
        "faster_cloud_reversion_lowers_output_variability",
        "Faster mean-reversion lowers output variability",
        -1, [("theta 0.15", slow), ("theta 0.6", fast)])

    # 4) tilt -> winter output share (up).
    claim_tilt = Claim(
        "steeper_tilt_shifts_output_toward_winter",
        "Increasing tilt raises the winter-season output share",
        +1, [("tilt 20", winter_output_share(20.0)),
             ("tilt 50", winter_output_share(50.0))])

    # 5) orientation toward south -> annual output (up).
    claim_orient = Claim(
        "southward_orientation_raises_annual_output",
        "Orientation closer to due south raises annual output",
        +1, [("azimuth 120", annual_output(120.0)),
             ("azimuth 180", annual_output(180.0))])

    # 6) latitude -> summer/winter ratio (up; the plan's statement, see FINDINGS).
    claim_lat = Claim(
        "higher_latitude_widens_summer_winter_ratio",
        "Higher latitude widens the summer/winter output ratio",
        +1, [("lat 45", summer_winter_ratio(45.0)),
             ("lat 60", summer_winter_ratio(60.0))])

    return [claim_vol, claim_disp, claim_rev, claim_tilt, claim_orient, claim_lat]

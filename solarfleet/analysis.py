"""Calibrate the fleet model against real OCF uk_pv generation.

Ties the pieces together on real data: read selected sites, clean, invert
generation to the effective log clear-sky index using the numpy geometry, fit
per-site OU parameters, and fit the distance-decay coupling kernel from the
cross-site correlations. Everything here is Python (Invariant A).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geometry as geo
from . import ingest, infer
from .compose import Site, CouplingKernel, haversine_km


def build_sites(meta: pd.DataFrame, ss_ids) -> list[Site]:
    sites = []
    for sid in ss_ids:
        row = meta.loc[sid]
        sites.append(Site(
            name=str(sid),
            latitude=float(row["latitude_rounded"]),
            longitude=float(row["longitude_rounded"]),
            tilt=float(row["tilt"]),
            surface_azimuth=float(row["orientation"]),
            kwp=float(row["kWp"]),
        ))
    return sites


def _site_grid(site: Site, series: pd.DataFrame, grid: pd.DatetimeIndex,
               period_minutes=30):
    """Reindex a site's generation to the full 30-min grid; return logk + valid."""
    g_wh = (series.set_index("datetime_GMT")["generation_Wh"].reindex(grid))
    # generation_Wh is energy per period; convert to average power in kW to match
    # the clear-sky reference kWp * poa/I_STC (a power). kW = Wh / (period_h) / 1000.
    period_h = period_minutes / 60.0
    g_power_kw = g_wh.to_numpy(dtype=float) / period_h / 1000.0
    midpoint = grid - pd.Timedelta(minutes=period_minutes / 2)
    poa = geo.clear_sky_poa(site.latitude, site.longitude,
                            midpoint.tz_convert("UTC").tz_localize(None).values,
                            site.tilt, site.surface_azimuth)
    logk, valid = infer.invert_to_logk(g_power_kw, poa, site.kwp)
    valid = valid & np.isfinite(logk)
    return logk, valid


def calibrate_fleet(root, ss_ids, meta, years, months, dt: float = 1.0):
    """Return per-site OU fits, the coupling-kernel fit, and diagnostics."""
    sites = build_sites(meta, ss_ids)
    df = ingest.read_uk_pv_sites(root, ss_ids, years=years, months=months)
    bad = ingest.load_bad_data(root + "/bad_data.csv" if isinstance(root, str)
                               else str(root) + "/bad_data.csv")
    clean, report = ingest.clean(df, meta, bad)

    grid = pd.date_range(clean.datetime_GMT.min(), clean.datetime_GMT.max(),
                         freq="30min", tz="UTC")

    logk_cols, valid_cols, ou_fits = [], [], {}
    for site in sites:
        series = clean[clean.ss_id == int(site.name)]
        logk, valid = _site_grid(site, series, grid)
        logk_cols.append(logk)
        valid_cols.append(valid)
        try:
            ou_fits[site.name] = infer.calibrate_site_ou_daytime(logk, valid, dt)
        except ValueError:
            ou_fits[site.name] = None

    logk_mat = np.column_stack(logk_cols)
    valid_mat = np.column_stack(valid_cols)

    # Pairwise empirical correlation on jointly-valid (daytime) timestamps, and
    # the fitted decay c1 from log(corr) vs temporal distance.
    n = len(sites)
    emp = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            both = valid_mat[:, i] & valid_mat[:, j]
            if both.sum() > 50:
                emp[i, j] = np.corrcoef(logk_mat[both, i], logk_mat[both, j])[0, 1]

    tds, logs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            r = emp[i, j]
            if np.isfinite(r) and r > 0:
                d_m = haversine_km(sites[i].latitude, sites[i].longitude,
                                   sites[j].latitude, sites[j].longitude) * 1000.0
                tds.append(d_m / 6.0)
                logs.append(np.log(r))
    tds, logs = np.asarray(tds), np.asarray(logs)
    c1 = float(-np.sum(tds * logs) / np.sum(tds * tds)) if len(tds) else np.nan
    kernel_fit = KernelFitReal(c1=c1, cloud_speed=6.0, n_pairs=len(tds))

    return {
        "sites": sites,
        "report": report,
        "ou_fits": ou_fits,
        "empirical_corr": emp,
        "kernel_fit": kernel_fit,
        "logk": logk_mat,
        "valid": valid_mat,
    }


class KernelFitReal:
    def __init__(self, c1, cloud_speed, n_pairs):
        self.c1, self.cloud_speed, self.n_pairs = c1, cloud_speed, n_pairs

    def __repr__(self):
        return f"KernelFitReal(c1={self.c1:.3e}, cloud_speed={self.cloud_speed}, n_pairs={self.n_pairs})"


def select_sites(root, meta, years, months, n=10, *, orientation_range=(135, 225),
                 tilt_range=(15, 55), kwp_range=(1.5, 6.0), coverage_q=0.95):
    """Pick ``n`` well-covered, south-ish, geographically spread systems.

    Scans only the ``ss_id`` column of the slice to measure per-system coverage,
    joins metadata, filters to domestic south-facing systems with near-full
    coverage, and spreads the picks across the latitude range.
    """
    import pyarrow.dataset as pads

    ds = _dataset_root(root)
    expr = ingest._partition_filter(years, months)
    ss = ds.to_table(columns=["ss_id"], filter=expr).column("ss_id").to_numpy()
    uniq, counts = np.unique(ss, return_counts=True)
    cov = pd.Series(counts, index=uniq, name="rows")

    m = meta.join(cov, how="inner")
    full = m["rows"].quantile(coverage_q)
    sel = m[(m.orientation.between(*orientation_range))
            & (m.tilt.between(*tilt_range))
            & (m.kWp.between(*kwp_range))
            & (m["rows"] >= coverage_q * full)].sort_values("latitude_rounded")
    idx = np.linspace(0, len(sel) - 1, n).astype(int)
    return sel.iloc[idx].index.tolist()


def _dataset_root(root):
    import pyarrow.dataset as pads
    import pathlib
    base = pathlib.Path(root)
    if (base / "30_minutely").exists():
        base = base / "30_minutely"
    return pads.dataset(str(base), format="parquet", partitioning="hive")


def format_report(res) -> str:
    """A human-readable summary of a calibrate_fleet result."""
    lines = []
    r = res["report"]
    lines.append(f"Cleaning: kept {r.kept:,}/{r.total_in:,} "
                 f"({100 * r.kept / max(r.total_in, 1):.1f}%); "
                 f"night-days dropped {r.dropped_night_days:,}")
    lines.append("Per-site OU (dt = one 30-min period):")
    lines.append(f"  {'site':>8} {'lat':>5} {'theta':>6} {'mu_eff':>7} "
                 f"{'sigma':>6}  half-life  meanK")
    for s in res["sites"]:
        f = res["ou_fits"][s.name]
        if f is None:
            lines.append(f"  {s.name:>8} {s.latitude:5.1f}  (insufficient daytime data)")
            continue
        hl = np.log(2) / f.theta * 0.5
        lines.append(f"  {s.name:>8} {s.latitude:5.1f} {f.theta:6.3f} {f.mu:7.3f} "
                     f"{f.sigma:6.3f}   {hl:4.1f}h   {np.exp(f.mu):.2f}")
    k = res["kernel_fit"]
    lines.append(f"Coupling kernel: c1={k.c1:.3e} over {k.n_pairs} pairs "
                 f"-> corr {np.exp(-k.c1 * 50000 / 6):.2f}@50km, "
                 f"{np.exp(-k.c1 * 150000 / 6):.2f}@150km, "
                 f"{np.exp(-k.c1 * 300000 / 6):.2f}@300km")
    return "\n".join(lines)


if __name__ == "__main__":
    # One-command real-data calibration: fetch summer 2024, select, calibrate.
    from . import fetch as fetchmod

    years, months = [2024], [5, 6, 7]
    root = "data/uk_pv"
    fetchmod.fetch([(2024, m) for m in months], dest=root)
    meta = ingest.load_metadata(root + "/metadata.csv")
    ss_ids = select_sites(root, meta, years, months, n=10)
    res = calibrate_fleet(root, ss_ids, meta, years, months)
    print(format_report(res))

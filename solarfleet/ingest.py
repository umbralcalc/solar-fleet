"""Ingest and clean OCF ``uk_pv`` generation data.

Two things live here:

1. **Partition-pruned Parquet reading.** The engine has no native Parquet source
   and no Hive-partition awareness (see ``STOCHADEX_GAPS.md``), so reading is done
   in pandas/pyarrow — pruning on the Hive ``year=/month=`` partitions *before*
   the datetime range — and the result is handed to the model as a driver or,
   when the engine must see it, written to the canonical CSV shape.

2. **The cleaning contract (plan §4.3).** OCF's ``bad_data.csv`` plus their
   recommended steps are a real data-agreement artifact, so they are implemented
   as an explicit, testable contract that returns a :class:`CleaningReport`
   counting exactly what each clause dropped — not ad-hoc filtering.

Timestamps in ``uk_pv`` are **period-ending** (a row stamped 12:00 in the 30-min
data covers 11:30-12:00) and ``generation_Wh`` is **energy per period** (average
power W = Wh / period-hours, i.e. x2 for 30-min data). Both are handled here so a
half-period phase error cannot masquerade as a geometry error downstream.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from . import geometry as geo


# ------------------------------------------------------------------------------
# Reading (partition-pruned)
# ------------------------------------------------------------------------------

def _dataset(root: str | pathlib.Path) -> pads.Dataset:
    base = pathlib.Path(root)
    if (base / "30_minutely").exists():
        base = base / "30_minutely"
    return pads.dataset(str(base), format="parquet", partitioning="hive")


def pruned_fragment_paths(root, years=None, months=None) -> list[str]:
    """The Parquet files a (year, month) filter actually touches — proof of pruning."""
    dataset = _dataset(root)
    expr = _partition_filter(years, months)
    return [f.path for f in dataset.get_fragments(filter=expr)]


def _partition_filter(years, months):
    expr = None
    if years is not None:
        expr = pads.field("year").isin(list(years))
    if months is not None:
        m = pads.field("month").isin(list(months))
        expr = m if expr is None else (expr & m)
    return expr


def read_uk_pv(root, years=None, months=None) -> pd.DataFrame:
    """Read the Hive-partitioned generation Parquet, pruning by year/month first.

    Returns columns ``ss_id, datetime_GMT, generation_Wh`` sorted by ss_id then
    time. ``years``/``months`` prune the Hive partitions before any row is read.
    """
    dataset = _dataset(root)
    table = dataset.to_table(filter=_partition_filter(years, months))
    df = table.to_pandas()
    df = df[["ss_id", "datetime_GMT", "generation_Wh"]].copy()
    df["datetime_GMT"] = pd.to_datetime(df["datetime_GMT"], utc=True)
    return df.sort_values(["ss_id", "datetime_GMT"]).reset_index(drop=True)


def read_uk_pv_sites(root, ss_ids, years=None, months=None) -> pd.DataFrame:
    """Read only the given systems, pushing the ss_id filter into the scan.

    For real data (100M+ rows across 25k systems) reading everything is
    infeasible; selecting a handful of sites first keeps the working set small.
    """
    dataset = _dataset(root)
    expr = _partition_filter(years, months)
    id_expr = pads.field("ss_id").isin(list(ss_ids))
    expr = id_expr if expr is None else (expr & id_expr)
    df = dataset.to_table(filter=expr).to_pandas()
    df = df[["ss_id", "datetime_GMT", "generation_Wh"]].copy()
    df["datetime_GMT"] = pd.to_datetime(df["datetime_GMT"], utc=True)
    return df.sort_values(["ss_id", "datetime_GMT"]).reset_index(drop=True)


def load_metadata(path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    return meta.set_index("ss_id")


def load_bad_data(path) -> pd.DataFrame:
    bad = pd.read_csv(path)
    bad["start_datetime_GMT"] = pd.to_datetime(bad["start_datetime_GMT"], utc=True)
    # A blank end means "to the end of the series" -> NaT, handled in clean().
    bad["end_datetime_GMT"] = pd.to_datetime(bad["end_datetime_GMT"], errors="coerce", utc=True)
    return bad


# ------------------------------------------------------------------------------
# The cleaning contract
# ------------------------------------------------------------------------------

@dataclass
class CleaningReport:
    """Row counts dropped by each contract clause, applied in order."""

    total_in: int = 0
    dropped_bad_data: int = 0
    dropped_negative: int = 0
    dropped_over_capacity: int = 0
    dropped_night_days: int = 0
    kept: int = 0
    night_day_diagnostics: list = field(default_factory=list)

    @property
    def total_dropped(self) -> int:
        return (self.dropped_bad_data + self.dropped_negative
                + self.dropped_over_capacity + self.dropped_night_days)


# OCF widen the physical bound kWp*500 Wh (30-min at STC) to kWp*750: systems do
# sometimes exceed nominal capacity.
CAP_FACTOR_WH = 750.0


def clean(df: pd.DataFrame, metadata: pd.DataFrame,
          bad_data: pd.DataFrame | None = None, *,
          period_minutes: int = 30, cap_factor_wh: float = CAP_FACTOR_WH,
          night_tol_wh: float = 0.0) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the cleaning contract; return ``(clean_df, report)``.

    Clauses, in order (so the report's counts are disjoint):
      1. drop the periods listed in ``bad_data`` (blank end = to end of series);
      2. drop negative ``generation_Wh``;
      3. drop ``generation_Wh > kWp * cap_factor_wh``;
      4. drop whole (site, day) blocks with non-zero generation at night — this is
         also the invariant the forward model asserts, kept as a diagnostic.
    """
    report = CleaningReport(total_in=len(df))
    df = df.copy()

    # 1) bad_data windows.
    if bad_data is not None and len(bad_data):
        mask = pd.Series(False, index=df.index)
        for _, row in bad_data.iterrows():
            sel = (df.ss_id == row.ss_id) & (df.datetime_GMT >= row.start_datetime_GMT)
            if pd.notna(row.end_datetime_GMT):
                sel &= df.datetime_GMT <= row.end_datetime_GMT
            mask |= sel
        report.dropped_bad_data = int(mask.sum())
        df = df[~mask]

    # 2) negatives.
    neg = df.generation_Wh < 0
    report.dropped_negative = int(neg.sum())
    df = df[~neg]

    # 3) over-capacity.
    kwp = df.ss_id.map(metadata["kWp"])
    over = df.generation_Wh > kwp * cap_factor_wh
    report.dropped_over_capacity = int(over.sum())
    df = df[~over]

    # 4) days with non-zero night generation -> drop the whole (site, day).
    df = df.reset_index(drop=True)
    period_start = df.datetime_GMT - pd.Timedelta(minutes=period_minutes)
    midpoint = period_start + pd.Timedelta(minutes=period_minutes / 2)
    lat = df.ss_id.map(metadata["latitude_rounded"]).to_numpy(float)
    lon = df.ss_id.map(metadata["longitude_rounded"]).to_numpy(float)
    midpoint_naive = midpoint.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    altitude, _ = geo.solar_position(lat, lon, midpoint_naive)
    is_night = altitude <= 0.0
    night_generating = is_night & (df.generation_Wh.to_numpy() > night_tol_wh)

    # Group by the stamp's calendar day (night detection above stays on the
    # period midpoint); a night spike then drops the whole stamped day. Keyed
    # vectorially via a (site|day) string so this stays fast on real data
    # (100M+ rows) rather than looping in Python.
    day = df.datetime_GMT.dt.date.astype(str)
    key = df.ss_id.astype("int64").astype(str) + "|" + day
    bad_keys = pd.unique(key[night_generating])
    report.night_day_diagnostics = sorted(
        (int(k.split("|")[0]), k.split("|")[1]) for k in bad_keys)
    if len(bad_keys):
        drop = key.isin(set(bad_keys))
        report.dropped_night_days = int(drop.sum())
        df = df[~drop.to_numpy()]

    df = df.reset_index(drop=True)
    report.kept = len(df)
    return df, report


# ------------------------------------------------------------------------------
# Unit / phase conversions and the engine canonical shape
# ------------------------------------------------------------------------------

def add_power_and_period_start(df: pd.DataFrame, period_minutes: int = 30
                              ) -> pd.DataFrame:
    """Add ``period_start`` (from period-ending stamps) and average ``power_W``."""
    out = df.copy()
    out["period_start"] = out.datetime_GMT - pd.Timedelta(minutes=period_minutes)
    out["power_W"] = out.generation_Wh / (period_minutes / 60.0)
    return out


def to_canonical_csv(df: pd.DataFrame, path, value_col: str = "generation_Wh",
                     partition_col: str = "ss_id",
                     dense_fill: float | None = None) -> pathlib.Path:
    """Write the engine's canonical wide CSV: ``time`` then one column per site.

    The engine's ``csv`` ``data.source`` maps columns to partitions by integer
    index (no header-name mapping) and requires a **dense float64 matrix** — it
    has no missing-value concept and rejects an empty cell with
    ``ParseFloat: parsing ""``. After cleaning, different sites lose different
    periods, so the wide pivot is ragged; pass ``dense_fill`` (e.g. 0.0) to fill
    the holes. That is a lossy, explicit choice the caller must make — imputing a
    gap as generation is wrong in general — and it is the practical face of the
    engine's no-nullability data model (see ``STOCHADEX_GAPS.md``).
    """
    wide = df.pivot_table(index="datetime_GMT", columns=partition_col,
                          values=value_col)
    wide = wide.sort_index(axis=1)
    if dense_fill is not None:
        wide = wide.fillna(dense_fill)
    time_seconds = (wide.index.view("int64") / 1e9)
    out = pd.DataFrame({"time": time_seconds})
    for col in wide.columns:
        out[str(col)] = wide[col].to_numpy()
    path = pathlib.Path(path)
    out.to_csv(path, index=False)
    return path

"""Generate a small, deterministic uk_pv-shaped fixture for offline CI.

The real Open Climate Fix ``uk_pv`` dataset is gated (needs an ``HF_TOKEN``) and
~6.25 GB, so tests run against this synthetic stand-in instead. It mimics the
real shape exactly:

* Hive-partitioned Parquet ``30_minutely/year=YYYY/month=MM/data.parquet`` with
  columns ``ss_id, datetime_GMT, generation_Wh``, sorted by ss_id then time.
* ``metadata.csv`` with ``ss_id, latitude_rounded, longitude_rounded,
  orientation, tilt, kWp``.
* ``bad_data.csv`` with ``ss_id, start_datetime_GMT, end_datetime_GMT`` (a blank
  end means "drop to the end of the series").

Deliberately injected bad rows exercise every clause of the cleaning contract:
a negative value, an over-capacity value, and a night-time generation spike.

Run: ``python testdata/make_fixture.py`` (writes into ``testdata/uk_pv/``).
Timestamps are **period-ending** (a row stamped 12:00 covers 11:30-12:00), as in
the real dataset.
"""

import pathlib

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from solarfleet import geometry as geo  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
UK_PV = ROOT / "uk_pv"

SITES = [
    # ss_id, lat, lon, orientation, tilt, kWp
    (1001, 51.51, -0.13, 180.0, 35.0, 4.0),
    (1002, 53.48, -2.24, 160.0, 30.0, 3.0),
]


def _clear_generation_wh(lat, lon, orient, tilt, kwp, times):
    """A plausible clean 30-min generation series in Wh (period-ending)."""
    # Average power over the period ~ clear-sky POA at the period midpoint * derate.
    mid = times - pd.Timedelta(minutes=15)
    poa = geo.clear_sky_poa(lat, lon, mid.values, tilt, orient)
    power_w = kwp * poa / geo.I_STC * 0.85 * 1000.0  # kWp is kW -> W
    return power_w * 0.5  # Wh over a half-hour period


def _day_rows(day, sites):
    times = pd.date_range(f"{day} 00:00", f"{day} 23:30", freq="30min")
    frames = []
    for ss_id, lat, lon, orient, tilt, kwp in sites:
        wh = _clear_generation_wh(lat, lon, orient, tilt, kwp, times)
        frames.append(pd.DataFrame({
            "ss_id": ss_id,
            "datetime_GMT": times,
            "generation_Wh": np.round(wh, 3),
        }))
    return pd.concat(frames, ignore_index=True)


def _inject_bad(df):
    """Corrupt a few rows to exercise the cleaning contract; return the frame."""
    df = df.copy()
    # 1) a negative reading (sensor glitch) for ss_id 1001.
    neg = (df.ss_id == 1001) & (df.datetime_GMT == pd.Timestamp("2023-06-15 03:00"))
    df.loc[neg, "generation_Wh"] = -5.0
    # 2) an over-capacity reading: 900 * kWp Wh >> kWp*750 bound for ss_id 1002.
    over = (df.ss_id == 1002) & (df.datetime_GMT == pd.Timestamp("2023-06-15 12:00"))
    df.loc[over, "generation_Wh"] = 900.0 * 3.0
    # 3) a night-time generation spike for ss_id 1001 on 2023-06-16 (whole day bad).
    night = (df.ss_id == 1001) & (df.datetime_GMT == pd.Timestamp("2023-06-16 01:30"))
    df.loc[night, "generation_Wh"] = 250.0
    return df


def main():
    if UK_PV.exists():
        import shutil
        shutil.rmtree(UK_PV)

    # June partition: two days, both sites, with injected bad rows.
    june = pd.concat([
        _day_rows("2023-06-15", SITES),
        _day_rows("2023-06-16", SITES),
    ], ignore_index=True)
    june = _inject_bad(june).sort_values(["ss_id", "datetime_GMT"]).reset_index(drop=True)

    # July partition: one clean day, one site — used to test month partition pruning.
    july = _day_rows("2023-07-10", SITES[:1]).sort_values(
        ["ss_id", "datetime_GMT"]).reset_index(drop=True)

    for year, month, frame in [(2023, 6, june), (2023, 7, july)]:
        d = UK_PV / "30_minutely" / f"year={year}" / f"month={month:02d}"
        d.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(d / "data.parquet", index=False)

    meta = pd.DataFrame(
        [(s[0], s[1], s[2], s[3], s[4], s[5]) for s in SITES],
        columns=["ss_id", "latitude_rounded", "longitude_rounded",
                 "orientation", "tilt", "kWp"],
    )
    meta.to_csv(ROOT / "metadata.csv", index=False)

    # bad_data.csv: one bounded window, and one open-ended (blank end) drop.
    bad = pd.DataFrame([
        {"ss_id": 1002, "start_datetime_GMT": "2023-06-15 08:00",
         "end_datetime_GMT": "2023-06-15 09:00"},
        {"ss_id": 1001, "start_datetime_GMT": "2023-07-10 00:00",
         "end_datetime_GMT": ""},  # blank end => drop to end of series
    ])
    bad.to_csv(ROOT / "bad_data.csv", index=False)

    print(f"wrote fixture under {UK_PV}")
    print(f"  june rows: {len(june)}, july rows: {len(july)}")


if __name__ == "__main__":
    main()

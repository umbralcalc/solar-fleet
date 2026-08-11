"""Real-data calibration path, validated offline against the committed fixture.

The fixture's clean generation was built as clear-sky * eta (eta=0.85) with no
weather noise, so inverting it must recover an effective clear-sky index of
exactly eta during daylight. This pins the Wh->power unit conversion — the
gotcha the plan's §4.2 caveat 4 warned about and that a first real-data run hit.
"""

import pathlib

import numpy as np
import pandas as pd

from solarfleet import ingest
from solarfleet.analysis import build_sites, _site_grid

TESTDATA = pathlib.Path(__file__).resolve().parent.parent / "testdata"


def test_build_sites_from_metadata():
    meta = ingest.load_metadata(TESTDATA / "metadata.csv")
    sites = build_sites(meta, [1001, 1002])
    assert [s.name for s in sites] == ["1001", "1002"]
    assert sites[0].surface_azimuth == 180.0   # orientation -> surface azimuth
    assert sites[0].kwp == 4.0


def test_inversion_recovers_effective_index_units():
    # Fixture eta = 0.85, no weather noise => k_eff == 0.85 in daylight.
    meta = ingest.load_metadata(TESTDATA / "metadata.csv")
    bad = ingest.load_bad_data(TESTDATA / "bad_data.csv")
    df = ingest.read_uk_pv(TESTDATA / "uk_pv", years=[2023], months=[6])
    clean, _ = ingest.clean(df, meta, bad)

    site = build_sites(meta, [1001])[0]
    grid = pd.date_range(clean.datetime_GMT.min(), clean.datetime_GMT.max(),
                         freq="30min", tz="UTC")
    series = clean[clean.ss_id == 1001]
    logk, valid = _site_grid(site, series, grid)

    k_eff = np.exp(logk[valid])
    assert len(k_eff) > 5
    # Recovered effective index is eta (0.85) to a few percent, NOT ~500x off.
    assert np.allclose(k_eff, 0.85, atol=0.05)

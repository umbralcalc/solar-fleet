"""Validate solarfleet.geometry against published astronomical values.

The strategy is to pin the ephemeris to real astronomy (declination extremes,
equinox declination, solar-noon altitude = 90 - |lat - decl|) plus the hard
physical invariants the whole fleet model depends on (night => zero output,
south-facing beats north, air mass monotonicity). These are values one can look
up, not golden numbers this code emitted.
"""

import numpy as np
import pandas as pd
import pytest

from solarfleet import geometry as geo


def _day(date, freq="1min", tz_naive_utc=True):
    """A full UTC day of timestamps at the given resolution."""
    return pd.date_range(f"{date} 00:00", f"{date} 23:59", freq=freq).values


# --- Declination: anchors the ephemeris to published astronomy -----------------

def test_declination_extremes_match_axial_tilt():
    # Over a year the declination should reach +/- the Earth's axial tilt, 23.44 deg,
    # at the solstices, and pass through ~0 at the equinoxes.
    times = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="1h").values
    decl = geo.solar_declination(times)
    assert decl.max() == pytest.approx(23.44, abs=0.05)
    assert decl.min() == pytest.approx(-23.44, abs=0.05)


def test_equinox_declination_near_zero():
    # March equinox 2023 was 2023-03-20 21:24 UTC; declination crosses zero there.
    decl = geo.solar_declination(np.datetime64("2023-03-20T21:24"))
    assert abs(float(decl)) < 0.02


# --- Solar-noon altitude: 90 - |latitude - declination| ------------------------

@pytest.mark.parametrize("lat,date,expected", [
    (51.5, "2023-06-21", 90 - (51.5 - 23.44)),   # London, summer solstice ~= 61.9
    (51.5, "2023-12-21", 90 - (51.5 + 23.44)),   # London, winter solstice ~= 15.1
    (23.44, "2023-06-21", 90.0),                  # Tropic of Cancer, sun overhead
    (0.0, "2023-03-20", 90.0),                    # Equator, equinox
])
def test_solar_noon_altitude(lat, date, expected):
    # Scan the day at 1-min resolution at lon=0; the peak altitude is solar noon.
    times = _day(date)
    alt, az = geo.solar_position(lat, 0.0, times)
    assert alt.max() == pytest.approx(expected, abs=0.4)


def test_solar_noon_azimuth_is_due_south_northern_hemisphere():
    times = _day("2023-06-21")
    alt, az = geo.solar_position(51.5, 0.0, times)
    noon = np.argmax(alt)
    assert az[noon] == pytest.approx(180.0, abs=1.5)


def test_azimuth_sweeps_east_to_west_through_the_day():
    # Morning sun in the east (az < 180), afternoon in the west (az > 180).
    times = _day("2023-06-21")
    alt, az = geo.solar_position(51.5, 0.0, times)
    hours = pd.to_datetime(times).hour + pd.to_datetime(times).minute / 60.0
    up = alt > 5.0  # only meaningful while the sun is well up
    morning = up & (hours < 11.5)
    afternoon = up & (hours > 12.5)
    assert np.all(az[morning] < 180.0)
    assert np.all(az[afternoon] > 180.0)


# --- Clear-sky normal irradiance (Meinel) -------------------------------------

def test_clear_sky_zero_below_horizon():
    assert geo.clear_sky_normal_irradiance(-5.0) == 0.0
    assert geo.clear_sky_normal_irradiance(0.0) == 0.0
    assert geo.clear_sky_normal_irradiance(30.0) > 0.0


def test_clear_sky_monotone_increasing_with_altitude():
    alts = np.array([5.0, 15.0, 30.0, 60.0, 90.0])
    dni = geo.clear_sky_normal_irradiance(alts)
    assert np.all(np.diff(dni) > 0)


def test_clear_sky_zenith_value_is_physical():
    # Sun overhead: air mass 1, I = 1353 * 0.7 ** 1 = 947.1 W/m^2.
    assert geo.clear_sky_normal_irradiance(90.0) == pytest.approx(1353.0 * 0.7, abs=1e-6)


# --- Plane-of-array transposition ---------------------------------------------

def test_poa_zero_at_night():
    poa = geo.clear_sky_poa(51.5, 0.0, _day("2023-06-21"), tilt_deg=35.0,
                            surface_azimuth_deg=180.0)
    times = pd.to_datetime(_day("2023-06-21"))
    alt, _ = geo.solar_position(51.5, 0.0, times.values)
    # The hard invariant: no plane-of-array irradiance whenever the sun is down.
    assert np.all(poa[alt <= 0.0] == 0.0)
    assert np.any(poa > 0.0)


def test_south_facing_beats_north_facing_northern_hemisphere():
    times = _day("2023-06-21")
    south = geo.clear_sky_poa(51.5, 0.0, times, 35.0, 180.0)
    north = geo.clear_sky_poa(51.5, 0.0, times, 35.0, 0.0)
    assert south.sum() > north.sum()


def test_normal_incidence_recovers_dni():
    # A panel pointed straight at the sun receives the full normal irradiance:
    # tilt = 90 - altitude, surface azimuth = solar azimuth => cos(incidence)=1.
    times = _day("2023-06-21")
    alt, az = geo.solar_position(51.5, 0.0, times)
    noon = np.argmax(alt)
    tilt = 90.0 - alt[noon]
    poa = geo.clear_sky_poa(51.5, 0.0, times[noon], tilt, az[noon])
    dni = geo.clear_sky_normal_irradiance(alt[noon])
    assert float(poa) == pytest.approx(float(dni), rel=1e-9)


def test_flat_panel_poa_equals_dni_times_sin_altitude():
    # Horizontal surface (tilt 0): POA beam = DNI * sin(altitude) = GHI beam term.
    times = _day("2023-06-21")
    alt, az = geo.solar_position(51.5, 0.0, times)
    up = alt > 0
    poa = geo.clear_sky_poa(51.5, 0.0, times, 0.0, 180.0)
    dni = geo.clear_sky_normal_irradiance(alt)
    expected = dni * np.sin(np.radians(alt))
    assert np.allclose(poa[up], expected[up], rtol=1e-9)

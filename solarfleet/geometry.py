"""Deterministic solar geometry: position, clear-sky irradiance, plane-of-array.

Pure functions of ``(time, site metadata)``. No state, no random draws, no
external data. This is the physical backbone of the fleet model and everything
downstream inherits its errors, so it is kept dependency-free (numpy only) and
validated against published astronomical values in ``tests/test_geometry.py``
before anything depends on it.

Conventions
-----------
* Times are UTC, supplied as anything ``numpy.datetime64`` can represent
  (``datetime64`` array, pandas ``DatetimeIndex``, list of ``datetime``).
* Angles in the public API are **degrees**. Azimuths are measured **clockwise
  from north** (north=0, east=90, south=180, west=270), matching the NOAA
  convention and the ``orientation`` column of the OCF ``uk_pv`` metadata.
* Solar *altitude* (a.k.a. elevation) is the geometric angle above the horizon,
  with no atmospheric-refraction correction: the clear-sky model below is only
  evaluated for altitude > 0 and refraction is a horizon-grazing effect that the
  Meinel form does not claim to resolve.

Solar position follows the NOAA Solar Calculator algorithm (General Solar
Position Calculations, NOAA GML), accurate to ~0.01 deg over 1900-2100, which
is far finer than any error the stochastic layer introduces.
"""

from __future__ import annotations

import numpy as np

# Apparent extraterrestrial (solar-constant-like) normal irradiance used by the
# Meinel clear-sky form, W/m^2. Deliberately the Meinel A = 1353, not the modern
# solar constant 1361 -- the 0.7 transmittance and the 1353 are a matched pair.
A_EXTRATERRESTRIAL = 1353.0

# Standard Test Conditions reference irradiance, W/m^2. Panel nameplate kWp is
# defined at this plane-of-array irradiance, so generation scales as I_poa/I_STC.
I_STC = 1000.0

# Julian Date of the Unix epoch (1970-01-01T00:00:00 UTC). JD = unix_s/86400 + this.
_JD_UNIX_EPOCH = 2440587.5


def _unix_seconds(times) -> np.ndarray:
    """Return float seconds since the Unix epoch for any datetime-like input."""
    t = np.asarray(times, dtype="datetime64[ns]")
    return t.astype("datetime64[ns]").astype("int64") / 1e9


def _solar_params(times):
    """NOAA intermediate quantities shared by declination / eq-of-time / position.

    Returns ``(decl_rad, eqtime_min, seconds_since_utc_midnight)`` as arrays.
    """
    unix_s = _unix_seconds(times)
    jd = unix_s / 86400.0 + _JD_UNIX_EPOCH
    jc = (jd - 2451545.0) / 36525.0  # Julian centuries since J2000.0

    # Geometric mean longitude and anomaly of the sun (degrees).
    l0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    m_rad = np.radians(m)

    # Eccentricity of Earth's orbit and the equation of centre.
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    c = (
        np.sin(m_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + np.sin(2 * m_rad) * (0.019993 - 0.000101 * jc)
        + np.sin(3 * m_rad) * 0.000289
    )
    true_long = l0 + c
    # Apparent longitude (nutation + aberration correction).
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))

    # Obliquity of the ecliptic, with the small correction term.
    seconds = 21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))
    mean_obliq = 23.0 + (26.0 + seconds / 60.0) / 60.0
    obliq_corr = mean_obliq + 0.00256 * np.cos(np.radians(omega))
    obliq_rad = np.radians(obliq_corr)

    decl_rad = np.arcsin(np.sin(obliq_rad) * np.sin(np.radians(app_long)))

    # Equation of time (minutes).
    y = np.tan(obliq_rad / 2.0) ** 2
    l0_rad = np.radians(l0)
    eqtime = 4.0 * np.degrees(
        y * np.sin(2 * l0_rad)
        - 2 * e * np.sin(m_rad)
        + 4 * e * y * np.sin(m_rad) * np.cos(2 * l0_rad)
        - 0.5 * y * y * np.sin(4 * l0_rad)
        - 1.25 * e * e * np.sin(2 * m_rad)
    )

    secs_since_midnight = np.mod(unix_s, 86400.0)
    return decl_rad, eqtime, secs_since_midnight


def solar_declination(times) -> np.ndarray:
    """Solar declination in degrees (positive north). Vectorised over ``times``."""
    decl_rad, _, _ = _solar_params(times)
    return np.degrees(decl_rad)


def solar_position(latitude, longitude, times):
    """Solar altitude and azimuth in degrees.

    Parameters
    ----------
    latitude, longitude : float or array
        Site latitude (deg, north positive) and longitude (deg, east positive).
        Broadcast against ``times``.
    times : datetime-like
        UTC timestamps.

    Returns
    -------
    (altitude_deg, azimuth_deg) : tuple of ndarray
        Altitude above the horizon (geometric, no refraction; may be negative at
        night). Azimuth clockwise from north in [0, 360).
    """
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    decl_rad, eqtime, secs = _solar_params(times)

    lat_rad = np.radians(lat)

    # True solar time (minutes) and hour angle (degrees). UTC => timezone 0.
    minutes_utc = secs / 60.0
    true_solar_time = np.mod(minutes_utc + eqtime + 4.0 * lon, 1440.0)
    hour_angle = true_solar_time / 4.0 - 180.0  # degrees, 0 at local solar noon
    ha_rad = np.radians(hour_angle)

    cos_zenith = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(
        decl_rad
    ) * np.cos(ha_rad)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith_rad = np.arccos(cos_zenith)
    altitude = 90.0 - np.degrees(zenith_rad)

    # Azimuth via the NOAA branch on the hour angle sign.
    sin_zenith = np.sin(zenith_rad)
    # Guard the poles / exact zenith where sin_zenith -> 0.
    safe = sin_zenith > 1e-9
    cos_az = np.where(
        safe,
        (np.sin(lat_rad) * cos_zenith - np.sin(decl_rad))
        / np.where(safe, np.cos(lat_rad) * sin_zenith, 1.0),
        0.0,
    )
    cos_az = np.clip(cos_az, -1.0, 1.0)
    az_core = np.degrees(np.arccos(cos_az))
    azimuth = np.where(hour_angle > 0.0, np.mod(az_core + 180.0, 360.0),
                       np.mod(540.0 - az_core, 360.0))

    return altitude, azimuth


def clear_sky_normal_irradiance(altitude_deg, a=A_EXTRATERRESTRIAL) -> np.ndarray:
    """Clear-sky *normal* (beam) irradiance, W/m^2 (Meinel / Heliodon form).

    ``I_csn(h) = A * 0.7 ** ((1/sin h) ** 0.678)`` for ``0 < h < 90``, else 0.
    This is the deterministic direct-normal irradiance under a cloudless sky; the
    stochastic clear-sky index multiplies it downstream.
    """
    altitude = np.asarray(altitude_deg, dtype=float)
    sin_h = np.sin(np.radians(altitude))
    # Only defined for the sun above the horizon; air mass 1/sin h blows up at h=0.
    positive = sin_h > 0.0
    air_mass = np.where(positive, 1.0 / np.where(positive, sin_h, 1.0), np.inf)
    dni = a * np.power(0.7, np.power(air_mass, 0.678))
    return np.where(positive, dni, 0.0)


def poa_cos_incidence(altitude_deg, azimuth_deg, tilt_deg, surface_azimuth_deg):
    """Cosine of the beam incidence angle on a tilted plane (clipped at 0).

    ``cos(theta) = sin(tilt) cos(h) cos(zeta - gamma) + cos(tilt) sin(h)``
    with ``h`` = solar altitude, ``gamma`` = solar azimuth, ``zeta`` = surface
    azimuth, ``tilt`` = surface tilt from horizontal. Values below zero mean the
    sun is behind the panel plane and are clipped to zero.
    """
    h = np.radians(np.asarray(altitude_deg, dtype=float))
    gamma = np.radians(np.asarray(azimuth_deg, dtype=float))
    tilt = np.radians(np.asarray(tilt_deg, dtype=float))
    zeta = np.radians(np.asarray(surface_azimuth_deg, dtype=float))
    cos_inc = np.sin(tilt) * np.cos(h) * np.cos(zeta - gamma) + np.cos(tilt) * np.sin(h)
    return np.clip(cos_inc, 0.0, None)


def clear_sky_poa(latitude, longitude, times, tilt_deg, surface_azimuth_deg,
                  a=A_EXTRATERRESTRIAL):
    """Clear-sky plane-of-array (beam) irradiance for a site, W/m^2.

    Composes :func:`solar_position`, :func:`clear_sky_normal_irradiance` and
    :func:`poa_cos_incidence`. Zero whenever the sun is below the horizon (the
    hard night-time invariant the whole model asserts) or behind the panel.

    ``tilt_deg`` and ``surface_azimuth_deg`` may be scalars or arrays broadcast
    against ``times``.
    """
    altitude, azimuth = solar_position(latitude, longitude, times)
    dni = clear_sky_normal_irradiance(altitude, a=a)
    cos_inc = poa_cos_incidence(altitude, azimuth, tilt_deg, surface_azimuth_deg)
    return dni * cos_inc

"""The learned residual — the honest ONNX slot (plan §5.3).

After the physical clear-sky backbone and the stochastic OU clear-sky index, what
remains is soiling, shading, snow, inverter clipping and temperature derating:
per-site, structured, and genuinely non-analytic. That is what a small learned
correction is for, and it is where ONNX belongs in this ecosystem.

Two hard constraints from the plan:

* **Fitting stays here (Invariant A).** Training is Python/sklearn against real
  data; the engine only ever *evaluates* a frozen ONNX artifact.
* **It is a correction, not the model.** The physical + OU layers are validated
  without it (Phases 1-4); the residual is added last so we know which layer does
  the work.

The model is a gradient-boosted tree ensemble, exported via ``skl2onnx``. That
exercises the ``ai.onnx.ml`` ``TreeEnsembleRegressor`` operator specifically — the
ONNX-ML operator set, as opposed to the standard neural-net ops. It predicts the
structured mean of ``log K`` from conditions the OU cannot express (time of day,
season, irradiance level), leaving the OU to model the stochastic remainder.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from . import geometry as geo

FEATURE_NAMES = ["altitude", "sin_hour", "cos_hour", "sin_doy", "cos_doy", "poa_norm"]


def build_features(latitude, longitude, tilt, surface_azimuth, times) -> np.ndarray:
    """Feature matrix (T, 6) for a site: geometry + calendar + irradiance level.

    Deliberately uses only quantities available without external weather data —
    the structured, learnable part of the residual.
    """
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    t_naive = t.tz_convert("UTC").tz_localize(None).values
    alt, az = geo.solar_position(latitude, longitude, t_naive)
    poa = geo.clear_sky_poa(latitude, longitude, t_naive, tilt, surface_azimuth)

    hour = t.hour + t.minute / 60.0
    doy = t.dayofyear.to_numpy(dtype=float)
    return np.column_stack([
        np.clip(alt, 0.0, None),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
        poa / geo.I_STC,
    ]).astype(np.float32)


def train_residual(x: np.ndarray, y: np.ndarray, *, n_estimators: int = 60,
                   max_depth: int = 3, random_state: int = 0):
    """Fit a gradient-boosted tree ensemble predicting log K from features."""
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=random_state)
    model.fit(x.astype(np.float32), y.astype(np.float64))
    return model


def export_onnx(model, n_features: int, path) -> pathlib.Path:
    """Export a fitted sklearn model to a frozen ONNX artifact (float32 I/O)."""
    from skl2onnx import to_onnx

    onx = to_onnx(model, np.zeros((1, n_features), dtype=np.float32),
                  target_opset=None)
    path = pathlib.Path(path)
    path.write_bytes(onx.SerializeToString())
    return path


def onnx_predict(path, x: np.ndarray) -> np.ndarray:
    """Evaluate an ONNX model with onnxruntime (for verification against sklearn)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: x.astype(np.float32)})[0]
    return np.asarray(out).reshape(-1)


def assemble_training_data(site, cleaned_generation: pd.DataFrame,
                           period_minutes: int = 30):
    """From a site's cleaned generation, build ``(X, y)`` for residual training.

    ``y`` is ``log K_obs`` on valid daytime rows; ``X`` the matching features.
    """
    from . import infer

    g_wh = cleaned_generation.set_index("datetime_GMT")["generation_Wh"].sort_index()
    times = g_wh.index
    period_h = period_minutes / 60.0
    power_kw = g_wh.to_numpy(dtype=float) / period_h / 1000.0

    midpoint = times - pd.Timedelta(minutes=period_minutes / 2)
    poa = geo.clear_sky_poa(site.latitude, site.longitude,
                            midpoint.tz_convert("UTC").tz_localize(None).values,
                            site.tilt, site.surface_azimuth)
    logk, valid = infer.invert_to_logk(power_kw, poa, site.kwp)

    x = build_features(site.latitude, site.longitude, site.tilt,
                       site.surface_azimuth, midpoint)
    return x[valid], logk[valid]

"""The residual trains, exports to ONNX, and evaluates identically off-engine.

Offline (no real data): synthesise a target with structure the model should
learn, train, export to ONNX, and confirm onnxruntime reproduces the sklearn
prediction. This validates the fit->export->evaluate path independent of the
engine's cgo ONNX build.
"""

import numpy as np
import pandas as pd
import pytest

from solarfleet import geometry as geo
from solarfleet.residual import (
    FEATURE_NAMES, build_features, train_residual, export_onnx, onnx_predict,
)


def test_build_features_shape_and_daytime():
    times = pd.date_range("2023-06-21", periods=48, freq="30min", tz="UTC")
    x = build_features(51.5, -0.1, 35, 180, times)
    assert x.shape == (48, len(FEATURE_NAMES))
    assert x.dtype == np.float32
    assert (x[:, 0] >= 0).all()          # altitude clipped at 0


def _synthetic_dataset(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2023-05-01", periods=n, freq="30min", tz="UTC")
    x = build_features(52.0, -1.0, 35, 180, times)
    day = x[:, 0] > 0
    x, times = x[day], times[day]
    # Structured target: inverter clipping at high POA + morning shading, + noise.
    poa = x[:, 5]
    morning = x[:, 2] > 0                 # cos_hour > 0 => around midday-ish; use sin
    y = (-0.15 * np.clip(poa - 0.8, 0, None) / 0.2       # clip losses at high POA
         - 0.10 * np.clip(0.3 - x[:, 0] / 60.0, 0, None)  # low-sun shading
         - 0.2 + 0.05 * rng.standard_normal(len(x)))
    return x, y


def test_onnx_matches_sklearn(tmp_path):
    x, y = _synthetic_dataset()
    model = train_residual(x, y, n_estimators=60, max_depth=3)
    path = export_onnx(model, x.shape[1], tmp_path / "residual.onnx")

    skl = model.predict(x.astype(np.float32))
    onx = onnx_predict(path, x)
    # float32 round-trip: agreement to a few 1e-5.
    assert np.allclose(skl, onx, atol=1e-4), np.abs(skl - onx).max()


def test_residual_learns_the_structure():
    # The model should capture the injected high-POA clipping: predictions at high
    # POA are lower than at moderate POA, all else equal.
    x, y = _synthetic_dataset()
    model = train_residual(x, y)
    pred = model.predict(x)
    hi = pred[x[:, 5] > 0.9]
    mid = pred[(x[:, 5] > 0.4) & (x[:, 5] < 0.6)]
    assert hi.mean() < mid.mean()

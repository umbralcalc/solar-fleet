"""In-loop ONNX: the engine evaluates the residual identically to onnxruntime.

Requires the cgo `onnx`-tagged stochadex binary and the onnxruntime shared
library, so it SKIPS in a pure-Go/CI environment. Where both are present it proves
the frozen residual runs inside the forward loop and matches onnxruntime exactly.
"""

import glob
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("skl2onnx")
pytest.importorskip("sklearn")

REPO = pathlib.Path(__file__).resolve().parent.parent
ONNX_BIN = REPO / ".bin" / "stochadex-onnx"


def _onnxruntime_lib():
    import onnxruntime
    d = pathlib.Path(onnxruntime.__file__).parent / "capi"
    libs = glob.glob(str(d / "libonnxruntime*.dylib")) + glob.glob(str(d / "libonnxruntime*.so"))
    return libs[0] if libs else None


pytestmark = pytest.mark.skipif(
    not ONNX_BIN.exists() or _onnxruntime_lib() is None,
    reason="needs the cgo onnx-tagged stochadex binary and onnxruntime shared library",
)


def test_engine_in_loop_onnx_matches_onnxruntime(tmp_path):
    from solarfleet.residual import build_features, train_residual, export_onnx, onnx_predict
    from solarfleet.runner import run

    os.environ["ONNXRUNTIME_LIB_PATH"] = _onnxruntime_lib()

    times = pd.date_range("2023-05-01", periods=3000, freq="30min", tz="UTC")
    x = build_features(52.0, -1.0, 35, 180, times)
    x = x[x[:, 0] > 0]
    y = (-0.2 - 0.15 * np.clip(x[:, 5] - 0.8, 0, None) / 0.2
         + 0.03 * np.random.default_rng(0).standard_normal(len(x)))
    model = train_residual(x, y)
    onnx_path = export_onnx(model, x.shape[1], tmp_path / "residual.onnx")

    n, f = 20, x.shape[1]
    feat = x[:n]
    cfg = {"main": {
        "partitions": [
            {"name": "feats", "params": {"ff": feat.reshape(-1).tolist()},
             "init_state_values": [float(v) for v in feat[0]],
             "state_history_depth": 1, "seed": 0},
            {"name": "residual",
             "iteration": {"type": "onnx_inference", "model_path": str(onnx_path),
                           "input_param": "features"},
             "params_from_upstream": {"features": {"upstream": "feats"}},
             "init_state_values": [0.0], "state_history_depth": 1, "seed": 0},
        ],
        "expressions": [
            {"partition": "feats", "fields": [{"name": f"f{i}"} for i in range(f)],
             "outputs": [f"slice(ff, step*{f}+{i}, 1)" for i in range(f)]},
        ],
        "simulation": {"termination_condition": {"type": "number_of_steps",
                                                 "max_steps": n - 1},
                       "timestep_function": {"type": "constant", "stepsize": 1.0},
                       "init_time_value": 0.0}}}

    df = run(cfg, cli=str(ONNX_BIN))
    engine = df["residual[0]"].to_numpy()[1:]
    ort = onnx_predict(onnx_path, feat)[1:len(engine) + 1]
    assert np.allclose(engine, ort, atol=1e-4), np.abs(engine - ort).max()

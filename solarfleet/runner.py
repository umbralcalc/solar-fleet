"""The Python <-> stochadex bridge.

Python writes a config, invokes the ``stochadex`` CLI, and reads the run back
into a pandas DataFrame. This loop is the thing the project exists to validate:
whether stochadex sits comfortably inside a Python data-science workflow.

A ``main:`` simulation run is egressed as an Arrow IPC file
(``output_function: {type: arrow}``), which pyarrow reads directly. The Arrow
table is ``time`` plus one ``FixedSizeList<float64>`` column per partition; we
expand each partition's list into ``<partition>[i]`` columns so the result is a
flat, tidy frame.
"""

from __future__ import annotations

import copy
import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _to_native(obj):
    """Recursively coerce numpy scalars/arrays to native Python types.

    A DS workflow produces numpy values everywhere; PyYAML's SafeDumper cannot
    represent ``np.float64`` etc., so sanitise the whole config before dumping.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    return obj


def stochadex_cli() -> str:
    """Locate the stochadex binary: ``$STOCHADEX_BIN``, then ``.bin/``, then PATH."""
    env = os.environ.get("STOCHADEX_BIN")
    if env:
        return env
    local = _REPO_ROOT / ".bin" / "stochadex"
    if local.exists():
        return str(local)
    found = shutil.which("stochadex")
    if found:
        return found
    raise FileNotFoundError(
        "stochadex CLI not found. Set STOCHADEX_BIN, build it into .bin/stochadex, "
        "or put it on PATH (see README)."
    )


def read_arrow(path: str | pathlib.Path) -> pd.DataFrame:
    """Read a stochadex Arrow IPC run file into a flat DataFrame.

    Each partition column (a fixed-size list) is expanded into ``name[0]``,
    ``name[1]``, ... alongside the shared ``time`` column.
    """
    with ipc.open_file(str(path)) as reader:
        table = reader.read_all()

    out = {"time": np.asarray(table.column("time").to_pylist(), dtype=float)}
    for name in table.schema.names:
        if name == "time":
            continue
        values = np.array(table.column(name).to_pylist(), dtype=float)
        if values.ndim == 1:  # width-1 partition
            out[f"{name}[0]"] = values
        else:
            for i in range(values.shape[1]):
                out[f"{name}[{i}]"] = values[:, i]
    return pd.DataFrame(out)


def run(config: dict, *, cli: str | None = None, workdir: str | None = None,
        keep: bool = False) -> pd.DataFrame:
    """Run a ``main:`` simulation config and return its output as a DataFrame.

    The config is deep-copied and its egress is forced to an Arrow file with
    every-step output (the layout ``read_arrow`` expects). The caller's
    ``simulation`` block is otherwise respected.
    """
    cli = cli or stochadex_cli()
    cfg = copy.deepcopy(config)
    sim = cfg.setdefault("main", {}).setdefault("simulation", {})
    sim.setdefault("output_condition", {"type": "every_step"})

    tmp = workdir or tempfile.mkdtemp(prefix="solarfleet-run-")
    tmp_path = pathlib.Path(tmp)
    tmp_path.mkdir(parents=True, exist_ok=True)
    arrow_path = tmp_path / "run.arrow"
    sim["output_function"] = {"type": "arrow", "path": str(arrow_path)}

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_to_native(cfg), sort_keys=False))

    proc = subprocess.run(
        [cli, "--config", str(config_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not arrow_path.exists():
        raise RuntimeError(
            f"stochadex run failed (exit {proc.returncode}).\n"
            f"config: {config_path}\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )

    df = read_arrow(arrow_path)
    if not keep and workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return df


def run_raw(config: dict, *, cli: str | None = None) -> subprocess.CompletedProcess:
    """Run a config (e.g. an analysis ``data:``/``macros:`` config) capturing stdout.

    Analysis macros write their results to stdout rather than the Arrow sink, so
    this returns the completed process for the caller to parse.
    """
    cli = cli or stochadex_cli()
    with tempfile.TemporaryDirectory(prefix="solarfleet-raw-") as tmp:
        config_path = pathlib.Path(tmp) / "config.yaml"
        config_path.write_text(yaml.safe_dump(_to_native(config), sort_keys=False))
        return subprocess.run(
            [cli, "--config", str(config_path)], capture_output=True, text=True,
        )

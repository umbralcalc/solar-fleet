"""Fetch a slice of the Open Climate Fix ``uk_pv`` dataset.

The dataset is **gated**: it requires accepting the conditions at
``huggingface.co/datasets/openclimatefix/uk_pv`` and an ``HF_TOKEN`` in the
environment. This module pulls only the requested month partitions (download-time
partition pruning) plus ``metadata.csv`` and ``bad_data.csv`` into a local Hive
tree that :func:`solarfleet.ingest.read_uk_pv` reads directly.

Nothing here is committed — ``data/`` is gitignored. CI uses the synthetic
``testdata/`` fixture instead.
"""

from __future__ import annotations

import pathlib

REPO_ID = "openclimatefix/uk_pv"


def fetch(months: list[tuple[int, int]], dest: str = "data/uk_pv",
          resolution: str = "30_minutely") -> pathlib.Path:
    """Download the given ``(year, month)`` partitions + metadata into ``dest``.

    Returns the local dataset root. Requires ``HF_TOKEN`` (or a cached login).
    """
    from huggingface_hub import hf_hub_download

    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    patterns = ["metadata.csv", "bad_data.csv"] + [
        f"{resolution}/year={y}/month={m:02d}/data.parquet" for y, m in months
    ]
    for rel in patterns:
        hf_hub_download(REPO_ID, rel, repo_type="dataset",
                        local_dir=str(dest))
    return dest


if __name__ == "__main__":
    import sys

    # Default: summer 2024, the window with the strongest generation signal.
    window = [(2024, 5), (2024, 6), (2024, 7)]
    root = fetch(window)
    print(f"fetched {len(window)} month(s) + metadata into {root}")

"""solarfleet — aggregate domestic solar PV fleet output.

A downstream stochadex project built entirely in Python + stochadex YAML
configs. The deterministic physical backbone (solar position + clear-sky
plane-of-array irradiance) lives in pure numpy (:mod:`solarfleet.geometry`);
the stochastic clear-sky-index field and fleet aggregation live in a stochadex
config that Python writes, runs via the CLI, and reads back.
"""

__all__ = ["geometry"]

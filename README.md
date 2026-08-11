# solar-fleet

Aggregate domestic solar-PV fleet output across Great Britain, as a downstream
[stochadex](https://github.com/umbralcalc/stochadex) project — built **entirely in
Python + stochadex YAML configs**, with no bespoke Go.

**The question it answers.** Given a fleet of dispersed rooftop PV systems, what
aggregate output does the fleet produce, how variable is it, and how does that
variability respond to geographic dispersion, weather volatility, and the fleet's
physical configuration?

This is structurally the same family as `bathing-water-forecaster` and
`measles-risk-forecaster` — a shared latent coupling many sites. What
distinguishes it is a **deterministic physical backbone** (solar geometry is known
physics, not a fitted term) and **distance-dependent coupling** derived from
geography rather than a scalar per-site loading.

## Why Python + configs

The deliberate constraint is to test whether stochadex sits comfortably in a
Python data-science workflow. It does. The split:

- **Deterministic physics in numpy** (`solarfleet/geometry.py`): NOAA solar
  position, Meinel clear-sky irradiance, plane-of-array transposition. Pure,
  dependency-free, validated against published astronomy.
- **Stochastic core as a stochadex config** (`solarfleet/compose.py`): the
  clear-sky-index field and fleet aggregation, written as YAML, run by the CLI,
  read back into pandas (`solarfleet/runner.py`).

Because geometry lives in numpy, the config's stochastic core needs only
primitives the DSL already has (`sin`, `cos`, `pow`) — the absent inverse-trig
primitives never come up. See `FINDINGS.md`.

## Layout

```
solarfleet/
  geometry.py   # solar position + clear-sky POA (pure numpy)
  compose.py    # partition-graph assembly: factor form (a) and full-covariance form (b)
  runner.py     # the Python <-> stochadex CLI bridge (config -> Arrow -> pandas)
  ingest.py     # partition-pruned Parquet reading + the cleaning contract
tests/          # geometry, forward model, covariance, ingestion
testdata/       # committed synthetic uk_pv fixture (real dataset is gated)
FINDINGS.md         # design decisions + the headline result
STOCHADEX_GAPS.md   # engine capability gaps, recorded as hit
```

## The two model forms

- **Factor form (a)** — one shared regional cloud OU latent plus per-site
  idiosyncratic OU noise. Cheap and config-expressible, but a single factor gives
  pairwise correlation `~ lambda_i * lambda_j` with no distance dependence, so it
  cannot produce the dispersion-smoothing effect.
- **Full-covariance form (b)** — a Cholesky-correlated innovation vector with a
  distance-decaying kernel. It reproduces true distance-dependent coupling and the
  flagship **dispersion-smoothing** effect (spreading sites apart lowers aggregate
  variability at fixed capacity).

**Headline result:** form (b) *is* expressible in the config DSL —
`each(S, i, dot(slice(L, i*S, S), xi))` — contrary to the design plan's
prediction. Draw the whole innovation vector once and transform it; no per-lane
draw control is needed. Details in `FINDINGS.md`.

## Data

The forcing dataset is Open Climate Fix
[`uk_pv`](https://huggingface.co/datasets/openclimatefix/uk_pv) (CC-BY-4.0, DOI
10.57967/hf/0878). Honest caveats:

1. **Gated.** Public but requires accepting conditions and an `HF_TOKEN`. Not
   freely fetchable; CI runs against the committed `testdata/` fixture instead.
2. **HuggingFace, not S3.** Exercising an S3 path means re-hosting into your own
   bucket, at which point the provenance story is yours to restate.
3. **Period-ending timestamps.** A row stamped 12:00 in the 30-minute data covers
   11:30–12:00. Handled in `ingest.py`; getting it wrong looks like a geometry
   phase error.
4. **Units.** `generation_Wh` is energy per period; average power is `×2` for
   30-minute data. Default to the 30-minutely subset (OCF's recommendation; the
   5-minutely subset is smaller, instantaneous and noisier).

The cleaning contract (`bad_data.csv` + OCF's recommended steps) is implemented as
an explicit, testable contract returning a drop-count report, not ad-hoc filtering.

## Running

Requires the `stochadex` CLI. Build the pure-Go binary (no toolchain needed to
*run* configs) into `.bin/`, or set `STOCHADEX_BIN`:

```bash
python -m venv --system-site-packages .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

## Status

Phases 0–2 complete (geometry, both model forms, ingestion). Inference
(calibration + a Python particle filter with ESS) and the declarative-twin
verification are in progress; the plan's Phase 5 (extraction into the stochadex Go
catalogue) is out of scope under the Python-only constraint. See `FINDINGS.md`.

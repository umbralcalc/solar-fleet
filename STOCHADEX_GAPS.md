# stochadex gaps

Capability gaps in the [stochadex](https://github.com/umbralcalc/stochadex) engine
found while building `solar-fleet`, recorded as they are hit.

**Why a file rather than issues.** This repo is a downstream application. Its
standing test — from stochadex's own `models/CONVENTIONS.md` — is whether the
domain model can be written as data (a config): if it can, the engine is
sufficient and any bespoke code is a convenience; if it cannot, the engine has a
real gap and one project is enough to prove it. Each entry below is one thing
this project needed and the engine could or could not do.

**What belongs here.** Only gaps **verified against the code or by a failing
example**, with **what they blocked in this project**. Not wishes. An entry that
cannot name what it blocked is not a gap, it is a preference — those are recorded
as design notes in `FINDINGS.md` instead. This discipline is borrowed from the
sibling `cryptobook/STOCHADEX_GAPS.md`, which also notes that a gap asserting the
engine *cannot* do something deserves the same adversarial check as a claim that
a model *can*.

Checked against **v0.15.0** (working checkout `/Users/roberth/Code/stochadex`,
HEAD `v0.15.0-1-g43454d6`). This project is built entirely in Python + configs,
so "could the engine do it" is asked of the **config surface**, not of Go.

---

## Verified absent, but did NOT block this project (recorded so the next reader doesn't re-verify)

### A. `atan2`, `asin`, `acos` are absent from the expression DSL

**Severity: none for this project — the reframe routes around it.**

The DSL registry (`pkg/general/expression.go`, the two `switch name` blocks at
505-842) has `sin`, `cos`, `pow`, `sqrt`, `exp`, `log`, `erf`, `erfc` but **no
inverse trig and no `atan2`**. Solar azimuth needs `atan2`; solar altitude needs
`asin`/`acos`. Each is a 1-2 line addition to the second switch (the evaluator is
a hand-rolled `go/ast` walker, no grammar to change).

**What it blocked: nothing.** This project computes solar position in numpy
(`solarfleet/geometry.py`) and feeds the result into the config as a driver, so
the DSL never computes geometry. Recorded because the plan pre-registered these
as expected category-2 findings; they are genuinely absent, but only a project
that computes solar position *inside* the DSL would be blocked. If such a project
appears, this is a cheap, safe promotion (pure, elementwise, no draw-width
question).

**Note the plan's third predicted primitive is NOT a gap:** nested non-constant
`pow` is present (`expression.go:707`, `math.Pow` over two evaluated args), so
`pow(0.7, pow(1/sin(h), 0.678))` works today.

### B. No config-level way to replay a precomputed external series into a live `main:` simulation partition

**Severity: low — a clean workaround exists and is in use.**

The config catalogue has no `from_storage`-style iteration that steps through a
loaded time series inside a `main:` run. The `data.source` mechanism
(`csv`/`json_log`/`postgres`) is **analysis-tier only** — it builds a
`StateTimeStorage` consumed by `macros:`, not by a simulation partition
(confirmed: `DataConfig.Source` at `pkg/api/macros.go` feeds `buildStorage()`,
which the macro tier consumes; no simulation iteration reads it).

**What it blocked: nothing — it shaped a decision.** The forward model needs a
per-site deterministic clear-sky-irradiance driver, precomputed in numpy. It is
fed in as a param vector read by step number:
`slice(series, step, 1)` with `init_state_values[0] = series[0]` (verified: `step`
equals the output row index for computed rows). This is clean and allocation-free
at our series lengths. Recorded as a non-gap so the next reader does not go
looking for a streaming/from-storage source that is not there — matching
`cryptobook/STOCHADEX_GAPS.md` entry 3, independently.

---

## Assessed and found NOT to be gaps (the plan predicted otherwise)

### The full-covariance form (b) is expressible in the DSL

The plan pre-registered this as the likely headline gap: that a Cholesky-correlated
multivariate innovation could be written in bespoke Go but **not** in the config
DSL, failing on "per-lane control of draws". Verified against the v0.15.0 binary:
it **is** expressible. Draw the whole innovation vector once with `iid(S, normal(0,1))`
and apply the constant Cholesky factor as a row-wise matvec
`each(S, i, dot(slice(lflat, i*S, S), xi))`; a partition reads its own previous
state vector via a lag-1 `upstreams: {me: sites}` self-reference. No per-lane draw
control is needed because you transform a single vector draw rather than drawing
per lane. Full write-up and evidence in `FINDINGS.md`; runnable in
`solarfleet/compose.py` + `tests/test_covariance.py`. Recorded here so the next
reader does not re-file it as a gap.

### C. The `csv` `data.source` rejects empty cells — no missing-value concept

**Severity: low — it forces an explicit imputation choice, does not block.**

Feeding cleaned `uk_pv` data (Phase 2) the engine dies on the first empty cell
with `strconv.ParseFloat: parsing "": invalid syntax`. Everything is `float64`
with no nullability, so a ragged fleet matrix (sites lose different periods to
cleaning) must be densified before the engine sees it. Handled by
`to_canonical_csv(..., dense_fill=0.0)`, a lossy choice the caller makes. Verified
by `tests/test_ingest.py::test_engine_consumes_cleaned_data_via_csv_source`
(fails without the fill, exit 0 with it). Recorded because a downstream author
will hit it the moment real data has gaps.

## Resolved in Phase 2 — assessed, not blockers

- **Parquet / Hive-partitioned ingress.** Confirmed absent from the engine, but
  it does **not block**: reading is a Python/pyarrow step (`solarfleet/ingest.py`)
  that prunes `year=/month=` before any row is read and feeds the engine a
  canonical dense CSV. The full loop (prune → clean → CSV → `csv` source → macro)
  runs green. The gap shapes *where* ingest lives (Python), not *whether* it works.
- **Postgres external-schema mapping.** Still confirmed canonical-only, but not
  exercised here (needs a running instance). Same conclusion as above: ETL to the
  canonical shape in Python, then read. No negotiation layer to use even if one
  wanted to.

## Open / to-assess as later phases hit them

- **ESS / resampling particle filter (Phase 3).** `posterior_estimation` is a
  loglike-weighted rolling average with no effective-sample-size or resampling
  diagnostic; `smc_inference` surfaces none either. Whether this blocks the
  latent-index filter or is simply done in Python is a Phase 3 question.

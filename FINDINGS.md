# Findings

Engine capability findings and design decisions, recorded as they are hit —
the record the plan's §9 asks for. Checked against **stochadex v0.15.0**
(working checkout `/Users/roberth/Code/stochadex`, HEAD `v0.15.0-1-g43454d6`).

This project is built **entirely in Python + stochadex YAML configs** — no
bespoke Go. The deterministic solar physics lives in numpy; the stochastic core
and aggregation live in a config that Python writes, runs via the CLI, and reads
back. Whether that loop is comfortable is itself the thing under test.

---

## The Python <-> stochadex bridge works cleanly (validated)

`solarfleet/runner.py`. Python writes a config dict as YAML, invokes
`stochadex --config file.yaml`, and reads the run back into a pandas DataFrame.

- **Egress:** `output_function: {type: arrow, path: run.arrow}` writes one Arrow
  IPC file — `time` (float64) plus one `FixedSizeList<float64>` column per
  partition — read directly by pyarrow. Requires an every-step output condition
  (differing per-partition row counts are not a rectangular table).
- **The Arrow sink is CLI-registered, not in the engine module.** The pure-Go
  build (`CGO_ENABLED=0 go build`) ships it; only DuckDB/BLAS need cgo. So the
  bridge needs no toolchain to *run*, only the prebuilt binary.
- **Invocation:** `-c/--config <path>` (argparse), optional `-s/--socket`.

Verified end-to-end: a 20-step OU config round-trips to a 21-row DataFrame.

## Exogenous time-varying drivers enter as a step-indexed param (validated)

There is **no in-config way to replay a precomputed external series into a
`main:` simulation partition** — no `from_storage` iteration type exists in the
config catalogue (the CSV/Postgres `data.source` mechanism is analysis-tier
only, consumed by macros, not by a live simulation partition).

The idiom that works: put the whole series in a param and read the current value
by step number.

```yaml
partitions:
- name: driver
  params: {series: [<v0>, <v1>, ..., <vN>]}
  init_state_values: [<v0>]        # row 0 is the init row; set it to series[0]
  state_history_depth: 1
expressions:
- partition: driver
  fields: [{name: d}]
  outputs: ["slice(series, step, 1)"]
```

`step` equals the output row index for every *computed* row (row 0 is the init
row, rows 1..N use `step` = 1..N), so with `init_state_values[0] = series[0]`
the emitted column equals `series` exactly. This is how numpy-computed clear-sky
plane-of-array irradiance is fed per site into the forward-model config.

---

## HEADLINE (§9): the full-covariance form (b) IS expressible in the DSL

The plan's single most valuable pre-registered question — *can the full-covariance
Cholesky form be written in the config DSL?* — was predicted to be **no**, with
the failure "rhyming with per-lane control of draws". **The prediction is
refuted.** Form (b) is expressible, verified end-to-end against the v0.15.0 binary
(`solarfleet/compose.py:build_covariance_config`, `tests/test_covariance.py`).

**The mechanism.** The per-step innovation is `L @ xi` with `xi ~ iid N(0,1)^S`
drawn once and `L` the Cholesky factor of `Sigma_ij = sigma_i sigma_j rho_ij`.
In the DSL:

```
xi    = iid(S, normal(0, 1))                          # one draw of the whole vector
innov = each(S, i, dot(slice(lflat, i*S, S), xi))     # row-wise matvec  L @ xi
```

**Why the predicted failure does not arise.** The prediction assumed form (b)
needs *per-lane control of draws* (each site drawing its own correlated noise),
which the DSL cannot do. But you never need per-lane draws: draw the **full**
innovation vector once with `iid(S, ...)` and **transform** it with a constant
matrix. `each` + `dot` + `slice` express the matvec, and `xi` (an outer binding)
is visible unchanged inside every `each` lane — verified with an identity-`L`
probe (`L @ xi == xi` exactly). Two further engine mechanisms made it clean:

- a partition reads its **own previous row as a vector** via a lag-1
  `upstreams: {me: sites}` self-reference (verified) — so the OU update
  vectorises over all sites without reconstructing state from named scalars;
- the whole fleet lives in **one wide partition** (state `[logK_0..S-1,
  gen_0..S-1]`), matching the `measles-risk-forecaster` exemplar's shape.

**It works, not just runs.** The empirical log-K correlation of the config output
tracks the distance kernel within sampling error, the near pair correlates far
more than the far pair, and the flagship **dispersion-smoothing** effect holds:
at fixed total capacity a dispersed fleet has lower aggregate log-K variability
than a compact one. Form (a), the single shared factor, structurally cannot do
this (its pairwise correlation is `~ lambda_i * lambda_j`, distance-free) — so the
gap between (a) and (b) is real and is the reason (b) exists, but the gap is
*modelling*, not *engine capability*.

**Consequence for the ecosystem.** This is the opposite of a category-2 finding:
the DSL is *more* capable than the plan (and the four prior structural-gap
entries) assumed. The `scan`/`each`/`dot` additions that closed the LOB project's
folds also, as a side effect, made correlated multivariate innovations
config-expressible. Worth surfacing upstream as a worked pattern, not a gap.

---

## Phase 0 deltas — plan assumptions vs the live engine

**Engine capability gaps live in `STOCHADEX_GAPS.md`** (the running tracker).
In brief: nested non-constant `pow` is present so the "critical" §6.1 gap is not
one; `atan2`/`asin`/`acos` are absent but moot under the reframe (gap A); no
config-level exogenous-series replay into a live simulation (gap B); Parquet/Hive
ingress, Postgres external-schema mapping, and ESS diagnostics are open/to-assess
in Phases 2-3. The non-gap plan corrections and conventions notes are below.

### Conventions / exemplars (Phase 5 target)

- `measles-risk-forecaster` is the shared-latent+sites exemplar; it packs sites
  into **one wide partition** (state-width 2N) rather than N separate `site_i`
  partitions — a live §3.4 choice. `anglersim` is the claim-binding reference.
- Exact **1e-12** equivalence oracle is standard, preserved by `pkg/rng`.
  Category 1 = model-side standardisation debt (`math/rand` vs `pkg/rng`);
  category 2 = missing engine primitive.
- Minor stale docs: the CONVENTIONS folder-layout tree omits `behaviour.go`
  (required in practice); `CONVENTIONS.md:379` lists `pi` as a missing primitive
  but it is wired (`expression.go:269`).
- **Version:** pin **v0.15.0** (has `scan`, zero-width `slice`, likelihood and
  nil-map fixes), not the siblings' v0.13.0.

## Phase 2 — what the ingestion "schema negotiation" actually required (§4.4)

The plan expected a schema-negotiation step and asked for a record of it. There
is no negotiation layer; what the engine actually requires, discovered by feeding
it real cleaned data (`tests/test_ingest.py::test_engine_consumes_cleaned_data_via_csv_source`):

- **Reading Parquet is a Python job, not an engine one.** No native Parquet
  source, no Hive pruning (`STOCHADEX_GAPS.md`). `solarfleet/ingest.py` reads the
  Hive tree with pyarrow.dataset, pruning `year=/month=` before any row is read
  (verified: a `months=[6]` filter touches exactly one fragment), and hands the
  result on. This is clean and fast; it simply lives in pandas.
- **The engine's `csv` source maps by integer column index, not header name.**
  The canonical shape is `time` (float seconds) + one column per site, in a
  stable sorted order the writer documents; the config references them
  positionally (`state_columns: {gen: [1, 2]}`). There is no header-name mapping,
  no declared types, no unit or timezone semantics — time is a bare float64.
- **The engine has no missing-value concept.** After cleaning, sites lose
  different periods, so the wide matrix is ragged; the `csv` source rejects an
  empty cell outright (`strconv.ParseFloat: parsing "": invalid syntax`). A dense
  float64 matrix is mandatory, so gaps must be imputed *before* the engine sees
  them — a lossy choice the Python layer must make explicitly
  (`to_canonical_csv(..., dense_fill=0.0)`). This is the practical face of the
  "everything is float64, no nullability" data model.
- **Net:** the data-agreement contract lives **entirely in this repo's Python**
  (the cleaning contract + the canonical writer). The engine consumes a dense,
  positionally-mapped float matrix and nothing more. That is simpler than the
  plan assumed, and worth stating plainly rather than looking for a negotiation
  layer that was never designed.

Postgres round-trip (source == sink through the canonical `(partition_name, time,
state[])` table) is not exercised here — it needs a running instance — but the
shape requirement is the same canonical one.

## Phase 3 — inference (all in Python, Invariant A)

- **eta and the OU mean are not separately identifiable** from generation +
  clear-sky POA. Generation depends on `eta * exp(log K)`, so only
  `eta * exp(mu)` is pinned; a brighter panel and a higher clear-sky-index mean
  are indistinguishable. Calibration therefore recovers an *effective* level
  `mu_eff = mu + log(eta)`, plus the dynamics (`theta`, `sigma`) which are
  identified independently. The plan listed `eta` and `mu` as separate fit
  targets; they are confounded, and `solarfleet/infer.py` says so. (Recovered
  cleanly in `tests/test_infer.py`.)
- **OU calibration is closed-form.** The Euler OU step is an AR(1), so OLS of
  `logk_{t+1}` on `logk_t` recovers `(theta, mu_eff, sigma)` — no optimiser. The
  coupling kernel decay `c1` is a no-intercept linear fit of `log(rho_ij)` on
  temporal distance.
- **The particle filter and ESS live here, and that is the right boundary.** The
  engine's `posterior_estimation` is a loglike-weighted average with no ESS; the
  bootstrap PF in `infer.py` tracks the latent log-K field, computes
  `ESS = 1/sum(w^2)` every step, and resamples systematically when it degrades.
  Recovery + tracking + ESS behaviour are tested. This is exactly the
  forget-friendly streaming inference the plan wanted, and keeping it in Python
  is Invariant A working as intended, not a workaround.
- **ONNX residual (§5.3) deferred.** Needs `scikit-learn` + `skl2onnx` +
  `onnxruntime`, not installed here. It is explicitly the last, optional step
  (only after the physical + OU layers validate without it), so it is left for
  when those deps are added; the allocation-profile question it raises is noted
  for then.

## Phase 4 — the declarative twin, verified against a numpy reference

With no bespoke Go, the 1e-12 shared-RNG oracle is unavailable, so the twin is
the config vs an **independent** numpy reference (`solarfleet/reference.py`),
verified in `tests/test_equivalence.py`:

- **Which oracle was achieved.** *Exact to 1e-9* on the deterministic model
  (`sigma = 0`): the config and the reference — two separate implementations of
  the same OU update, clamps and generation formula — agree to floating point,
  with `init_logk != mu` so the OU relaxation and the capacity clip are genuinely
  exercised, not a fixed point. This is the strongest oracle possible here and it
  is stronger than a fat-tolerance statistical check would be.
- **Stochastic parts: distributional.** Config and reference are independent
  draws; both are validated against the *analytic* discrete-OU stationary
  statistics (mean `mu`, variance `sigma^2/(theta(2 - theta dt))`, correlation
  `rho`) — stronger than run-vs-run, which is only pinned to Monte-Carlo error.
- **dt scaling, mutation sentinels, branch coverage** all pass: `sqrt(dt)` moves
  innovation variance the right way; a dropped `cos(zeta-gamma)` term makes a
  south panel look like an east one (caught); a wrong clear-sky `A` is caught;
  the night and capacity-clip branches are both hit and, for the clip,
  cross-checked against the reference.
- **Plan correction (§6.4): unknown config keys are REJECTED, not ignored.** The
  plan warned that `yaml.v2` silently accepts unknown keys (the `state_width`
  trap). v0.15.0 rejects `totally_bogus_key` with an actionable message —
  *"...which nothing reads — check for a typo, or a key from an older schema."*
  The trap the plan guarded against no longer exists.

## §7.2 — the structural-driver claims (behaviour.py)

All six monotonic claims hold (`tests/test_behaviour.py`, one subtest per claim,
mirroring the catalogue's claim<->test binding):

- **The flagship dispersion claim is non-vacuous.** `wider_geographic_dispersion
  _lowers_fleet_variability` bites because coupling derives from geography in form
  (b): a compact fleet's aggregate clear-sky index is markedly more variable than
  a dispersed fleet's at fixed capacity. The plan's worry (§7.2) — that the claim
  passes without meaning under the single-factor form (a) — is avoided by using
  (b) as the model. Under (a) it would indeed be vacuous, which is the argument
  for (b).
- **Cloud volatility ↑ and mean-reversion speed ↑** move aggregate variability up
  and down respectively; **tilt, orientation, latitude** move winter share,
  annual output and the summer/winter ratio as expected — the last three are pure
  geometry, so deterministic.
- **Plan correction:** the claim table lists ID
  `higher_latitude_lowers_summer_winter_ratio` against the statement "Higher
  latitude *widens* the summer/winter output ratio" — the ID and statement
  disagree. The physics is that higher latitude *widens* the ratio (more extreme
  seasonality), so the claim is named `higher_latitude_widens_summer_winter_ratio`
  here to match the statement and the measurement.

## Real data — calibrated against OCF uk_pv (summer 2024)

The model is calibrated against the **real** gated dataset (10 GB-spread sites,
May–July 2024, `python -m solarfleet.analysis`). This is no longer synthetic-only.

- **The core structure validates against real cross-site correlations.** The
  fitted coupling kernel gives correlation **0.76 at 50 km, 0.44 at 150 km, 0.20
  at 300 km** — a textbook cloud-field decorrelation, fit from 22 real site-pairs.
  This is direct evidence that the distance-dependent coupling of form (b) is the
  right shape for real GB solar, not just an assumption.
- **Per-site OU parameters are physical:** reversion half-lives 1.5–5.4 h (cloud
  persistence), log-K volatility 0.38–0.49, and a mean effective clear-sky index
  of **0.57** across sites — exactly right for a cloudy GB summer.
- **The cleaning contract on real data** kept 91.1% of rows; the dominant drop was
  3,786 rows in whole (site,day) blocks flagged by the night-generation clause —
  real sensor noise, and the same invariant the forward model asserts.
- **Plan §4.2 caveat 4 is a real trap, and it was hit.** `generation_Wh` is energy
  per period, not power; inverting it directly against the `kWp * POA/I_STC` power
  reference gave a clear-sky index ~500x too large. The ×(60/period)/1000
  conversion fixes it. θ, σ and the kernel were unaffected (they come from AR(1)
  slopes/residuals and correlations, invariant to an additive log shift) — only
  the level μ was corrupted. Pinned by `tests/test_analysis.py` against the
  fixture (where the true effective index is known = 0.85).
- **Scale forced two real-data engineering choices:** the 3-month slice is **105.5M
  rows across 25,550 systems**, so ingestion selects a handful of sites first
  (ss_id pushdown) rather than cleaning everything, and the night-day clause was
  vectorised (a per-row Python membership test stalled at that scale).
- **Honest limitation:** 2 of the 10 selected sites had too little clean daytime
  data to fit (real-world messiness); they are reported as such, not silently
  dropped.

The dataset is gated, so none of it is committed (`data/` is gitignored); CI
still runs entirely on the synthetic `testdata/` fixture.

## Consequences of the python+configs reframe (flagged, not blockers)

- **The Phase 4 twin inverts.** With no bespoke Go there is no Go oracle for the
  exact 1e-12 twin — that oracle is a Go↔Go construct. Python↔Go cannot share
  the `pkg/rng` PCG stream, so equivalence drops to **exact on the deterministic
  parts / distributional on the stochastic parts**. That downgrade is itself a
  finding about stochadex-in-Python.
- **Phase 5 (Go catalogue extraction) is the one inherently-Go step.** Treated
  as out of scope unless revisited — the deliverable is the standalone
  python+configs project.

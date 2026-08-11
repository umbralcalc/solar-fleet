"""The cleaning contract drops exactly the injected bad rows, and pruning prunes."""

import pathlib

import pandas as pd
import pytest

from solarfleet import ingest

TESTDATA = pathlib.Path(__file__).resolve().parent.parent / "testdata"


@pytest.fixture
def metadata():
    return ingest.load_metadata(TESTDATA / "metadata.csv")


@pytest.fixture
def bad_data():
    return ingest.load_bad_data(TESTDATA / "bad_data.csv")


# --- partition-pruned reading --------------------------------------------------

def test_month_pruning_touches_only_the_requested_partition():
    paths = ingest.pruned_fragment_paths(TESTDATA / "uk_pv", years=[2023], months=[6])
    assert len(paths) == 1
    assert "month=06" in paths[0]
    # And the July file is genuinely excluded.
    assert not any("month=07" in p for p in paths)


def test_read_uk_pv_returns_expected_columns_and_order():
    df = ingest.read_uk_pv(TESTDATA / "uk_pv", years=[2023], months=[6])
    assert list(df.columns) == ["ss_id", "datetime_GMT", "generation_Wh"]
    assert df.equals(df.sort_values(["ss_id", "datetime_GMT"]).reset_index(drop=True))
    assert set(df.ss_id.unique()) == {1001, 1002}


# --- the cleaning contract, clause by clause -----------------------------------

def test_bad_data_window_and_blank_end(metadata, bad_data):
    # July has the blank-end drop for ss_id 1001 (whole series from 2023-07-10 00:00).
    df = ingest.read_uk_pv(TESTDATA / "uk_pv", years=[2023], months=[7])
    _, report = ingest.clean(df, metadata, bad_data)
    # All 48 ss_id-1001 July rows are dropped by the open-ended bad_data entry.
    assert report.dropped_bad_data == 48


def test_june_contract_drops_exactly_the_injected_rows(metadata, bad_data):
    df = ingest.read_uk_pv(TESTDATA / "uk_pv", years=[2023], months=[6])
    clean, report = ingest.clean(df, metadata, bad_data)

    assert report.total_in == 192
    # bad_data.csv drops the 08:00-09:00 window for ss_id 1002 (3 inclusive periods).
    assert report.dropped_bad_data == 3
    # one injected negative reading.
    assert report.dropped_negative == 1
    # one injected over-capacity reading (2700 Wh > 3.0 * 750).
    assert report.dropped_over_capacity == 1
    # one injected night spike flags the whole (1001, 2023-06-16) day = 48 rows.
    assert report.dropped_night_days == 48
    assert (1001, "2023-06-16") in report.night_day_diagnostics
    assert report.kept == 192 - 3 - 1 - 1 - 48
    assert len(clean) == report.kept


def test_clean_data_has_no_negatives_or_night_generation(metadata, bad_data):
    df = ingest.read_uk_pv(TESTDATA / "uk_pv", years=[2023], months=[6])
    clean, _ = ingest.clean(df, metadata, bad_data)
    assert (clean.generation_Wh >= 0).all()
    # Re-running the contract on already-clean data drops nothing.
    _, report2 = ingest.clean(clean, metadata, bad_data)
    assert report2.total_dropped == 0


# --- unit and period-ending handling -------------------------------------------

def test_power_conversion_is_times_two_for_half_hour(metadata, bad_data):
    df = ingest.read_uk_pv(TESTDATA / "uk_pv", years=[2023], months=[6])
    clean, _ = ingest.clean(df, metadata, bad_data)
    withp = ingest.add_power_and_period_start(clean, period_minutes=30)
    # average power W = Wh / 0.5 h = 2 * Wh.
    assert (withp.power_W == 2.0 * withp.generation_Wh).all()
    # period_start precedes the (period-ending) stamp by one period.
    assert (withp.datetime_GMT - withp.period_start == pd.Timedelta(minutes=30)).all()


def test_canonical_csv_is_time_plus_one_column_per_site(tmp_path, metadata, bad_data):
    df = ingest.read_uk_pv(TESTDATA / "uk_pv", years=[2023], months=[6])
    clean, _ = ingest.clean(df, metadata, bad_data)
    out = ingest.to_canonical_csv(clean, tmp_path / "canonical.csv")
    written = pd.read_csv(out)
    assert written.columns[0] == "time"
    assert set(written.columns[1:]) == {"1001", "1002"}


def test_engine_consumes_cleaned_data_via_csv_source(tmp_path, metadata, bad_data):
    # The full §4.4 ingestion loop: prune -> clean -> dense canonical CSV -> the
    # engine's csv data.source -> a macro. dense_fill is required because the
    # engine has no missing-value concept (empty cell => ParseFloat error).
    from solarfleet.runner import run_raw

    df = ingest.read_uk_pv(TESTDATA / "uk_pv", years=[2023], months=[6])
    clean, _ = ingest.clean(df, metadata, bad_data)
    csv = ingest.to_canonical_csv(clean, tmp_path / "canonical.csv", dense_fill=0.0)

    cfg = {
        "data": {"source": {"csv": {"path": str(csv), "time_column": 0,
                 "state_columns": {"gen": [1, 2]}, "skip_header": True}}},
        "macros": [{"type": "vector_mean", "name": "mean_gen",
                    "data": {"partition_name": "gen"},
                    "kernel": {"type": "exponential"},
                    "params": {"exponential_weighting_timescale": [50.0]},
                    "window": 20}],
    }
    proc = run_raw(cfg)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert any("mean_gen" in line for line in proc.stdout.splitlines())

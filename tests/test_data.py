import numpy as np
import pandas as pd
import pytest

from vol_forecast.data import align_sessions, clean_close, log_returns


def test_log_return_formula_and_ordering():
    idx = pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-06"])  # deliberately unsorted
    close, _ = clean_close(pd.Series([110.0, 100.0, 99.0], index=idx))
    r = log_returns(close)
    assert list(r.index) == sorted(idx)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(110 / 100))
    assert r.iloc[2] == pytest.approx(np.log(99 / 110))


def test_duplicates_and_nonpositive_prices():
    idx = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    close, report = clean_close(pd.Series([100.0, 500.0, 0.0, 102.0, 103.0], index=idx))
    assert report["duplicate_dates_dropped"] == 1
    assert report["nonpositive_prices_blanked"] == 1
    assert close.loc["2020-01-02"] == 100.0  # first of the duplicates kept
    r = log_returns(close)
    # The bad price poisons both returns that touch it; no bridging 100 -> 102 return is invented.
    assert np.isnan(r.loc["2020-01-03"]) and np.isnan(r.loc["2020-01-06"])
    assert r.loc["2020-01-07"] == pytest.approx(np.log(103 / 102))


def test_missing_bar_is_nan_not_filled():
    a = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    b = a.drop(pd.Timestamp("2020-01-03"))
    prices = align_sessions({"A": a, "B": b})
    assert np.isnan(prices.loc["2020-01-03", "B"])
    assert log_returns(prices["B"]).isna().all()

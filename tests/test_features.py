import numpy as np
import pandas as pd
import pytest

from vol_forecast.features import rv_features


def test_values_and_first_valid_indices(returns):
    r = returns.copy()
    r.iloc[0] = np.nan  # like real data: the first session has no return
    f = rv_features(r)
    assert f["rv_1"].first_valid_index() == r.index[1]
    assert f["rv_5"].first_valid_index() == r.index[5]
    assert f["rv_22"].first_valid_index() == r.index[22]
    t = 100
    assert f["rv_1"].iloc[t] == pytest.approx(r.iloc[t] ** 2)
    assert f["rv_5"].iloc[t] == pytest.approx((r.iloc[t - 4 : t + 1] ** 2).mean())
    assert f["rv_22"].iloc[t] == pytest.approx((r.iloc[t - 21 : t + 1] ** 2).mean())


def test_future_invariance(returns):
    """Features at origin t must not change when later data is appended or altered."""
    t = 300
    past_only = rv_features(returns.iloc[: t + 1])
    tampered = returns.copy()
    tampered.iloc[t + 1 :] = 9.9
    pd.testing.assert_frame_equal(past_only, rv_features(tampered).iloc[: t + 1])


def test_missing_return_blanks_window(returns):
    r = returns.copy()
    r.iloc[50] = np.nan
    f = rv_features(r)
    assert f["rv_5"].iloc[50:55].isna().all() and f["rv_5"].iloc[55:60].notna().all()
    assert f["rv_22"].iloc[50:72].isna().all() and not np.isnan(f["rv_22"].iloc[72])

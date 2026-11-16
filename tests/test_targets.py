import numpy as np
import pytest

from vol_forecast.targets import forward_variance_target


def test_targets_use_future_sessions(returns):
    t = 100
    daily = forward_variance_target(returns, 1)
    weekly = forward_variance_target(returns, 5)
    assert daily["target"].iloc[t] == pytest.approx(returns.iloc[t + 1] ** 2)
    assert weekly["target"].iloc[t] == pytest.approx((returns.iloc[t + 1 : t + 6] ** 2).mean())
    assert daily["maturity"].iloc[t] == returns.index[t + 1]
    assert weekly["maturity"].iloc[t] == returns.index[t + 5]


def test_target_ignores_the_origin_and_the_past(returns):
    t = 100
    tampered = returns.copy()
    tampered.iloc[: t + 1] = 9.9
    assert forward_variance_target(tampered, 5)["target"].iloc[t] == pytest.approx(
        forward_variance_target(returns, 5)["target"].iloc[t]
    )


def test_incomplete_target_is_missing(returns):
    weekly = forward_variance_target(returns, 5)
    assert weekly["target"].iloc[-5:].isna().all() and weekly["maturity"].iloc[-5:].isna().all()
    assert not np.isnan(weekly["target"].iloc[-6])
    r = returns.copy()
    r.iloc[105] = np.nan  # a hole must not shift the window to "the next 5 available returns"
    assert forward_variance_target(r, 5)["target"].iloc[100:105].isna().all()

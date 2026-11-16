"""Predictors known at forecast origin t (after the close of session t)."""

from __future__ import annotations

import pandas as pd

WINDOWS = (1, 5, 22)  # daily, weekly, monthly, in trading sessions (the HAR inputs)


def rv_features(returns: pd.Series) -> pd.DataFrame:
    """Backward-looking mean squared returns, in daily variance units.

    rv_1[t]  = r[t]^2
    rv_5[t]  = mean(r[t-4 .. t]^2)
    rv_22[t] = mean(r[t-21 .. t]^2)

    Windows end at t inclusive, so they only use returns through session t.
    A window containing any missing return is NaN (min_periods = full window).

    Note: with daily closes, a squared return is a very noisy proxy for the day's true
    variance (the literature uses intraday "realized variance"). Averages over 5 and 22
    sessions are much less noisy.
    """
    r2 = returns**2
    return pd.DataFrame(
        {f"rv_{w}": r2.rolling(window=w, min_periods=w).mean() for w in WINDOWS}
    )

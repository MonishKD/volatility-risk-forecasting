"""Forecast targets aligned to forecast origin t."""

from __future__ import annotations

import pandas as pd


def forward_variance_target(returns: pd.Series, horizon: int) -> pd.DataFrame:
    """Average squared return over the next `horizon` observed sessions.

    target[t]   = (1/h) * sum_{j=1..h} r[t+j]^2      (h=1: tomorrow's squared return)
    maturity[t] = date of session t+h, the first close at which target[t] is fully known.

    Dividing by h keeps every horizon in the same *daily* variance units.
    If any of the h future returns is missing or not yet in the data, target[t] is NaN:
    the definition never shifts to "the next h returns that happen to exist".
    """
    r2 = returns**2
    target = r2.rolling(window=horizon, min_periods=horizon).mean().shift(-horizon)
    maturity = pd.Series(returns.index, index=returns.index).shift(-horizon)
    return pd.DataFrame({"target": target, "maturity": maturity})

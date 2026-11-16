"""Glue: from raw snapshots to per-asset modelling frames."""

from __future__ import annotations

import pandas as pd

from vol_forecast.config import Config
from vol_forecast.data import align_sessions, clean_close, load_close, log_returns
from vol_forecast.features import rv_features
from vol_forecast.targets import forward_variance_target


def load_returns(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Cleaned log returns for all configured symbols on a shared session calendar."""
    closes, report = {}, {}
    for symbol in cfg.symbols:
        path = cfg.raw_dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/download_data.py first")
        closes[symbol], report[symbol] = clean_close(load_close(path))
    prices = align_sessions(closes)
    for symbol in cfg.symbols:
        report[symbol]["missing_bars_vs_shared_calendar"] = int(
            prices[symbol].isna().sum() - report[symbol]["missing_prices"]
        )
        report[symbol]["sessions"] = int(prices[symbol].notna().sum())
    returns = prices.apply(log_returns)
    return returns, report


def modelling_frame(returns: pd.Series, horizon: int) -> pd.DataFrame:
    """Features at origin t next to the horizon-h target and its maturity date."""
    return rv_features(returns).join(forward_variance_target(returns, horizon))

"""Interface every forecasting model plugs into (naive now; HAR, GARCH, LightGBM later)."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class ForecastModel(Protocol):
    name: str

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Estimate parameters. Receives ONLY rows whose labels have matured by the fit cutoff."""

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Variance forecasts, one per row, each using only that row's own features."""

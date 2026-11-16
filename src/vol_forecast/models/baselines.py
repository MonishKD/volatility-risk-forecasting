"""Naive baselines: the forecast is simply a past average of squared returns."""

from __future__ import annotations

import pandas as pd


class NaiveModel:
    """Predicts future variance = one backward-looking feature. Nothing is estimated."""

    def __init__(self, name: str, column: str):
        self.name = name
        self.column = column

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        return None

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return features[self.column]


def make_model(name: str):
    """Model registry. New models (har, garch, lightgbm) get added here."""
    if name == "naive_1":  # "tomorrow looks like today"
        return NaiveModel("naive_1", "rv_1")
    if name == "naive_22":  # "tomorrow looks like the last month"
        return NaiveModel("naive_22", "rv_22")
    raise ValueError(f"unknown model: {name}")

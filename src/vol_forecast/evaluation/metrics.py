"""Loss functions for variance forecasts, plus the positive floor."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_floor(prediction: pd.Series, floor: float) -> pd.Series:
    """Raise forecasts below `floor` up to `floor`. QLIKE needs strictly positive forecasts."""
    if not floor > 0:
        raise ValueError("floor must be strictly positive")
    return prediction.clip(lower=floor)


def qlike(realized, forecast) -> float:
    """Mean of log(v) + y/v. Lower is better.

    The primary loss: it stays a valid way to rank variance forecasts even though the
    realized target is a noisy proxy, and it penalises under-forecasting risk more
    heavily than over-forecasting.
    """
    y, v = np.asarray(realized, dtype=float), np.asarray(forecast, dtype=float)
    if (v <= 0).any():
        raise ValueError("QLIKE requires strictly positive forecasts; apply the floor first")
    return float(np.mean(np.log(v) + y / v))


def rmse(realized, forecast) -> float:
    """Root mean squared error in variance units. Dominated by a few extreme days."""
    y, v = np.asarray(realized, dtype=float), np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean((y - v) ** 2)))


def summarize(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Mean QLIKE and RMSE per asset / horizon / model, with sample and clip counts.

    Extension point: paired significance tests (Diebold-Mariano with HAC errors) will
    take the same `forecasts` table once the model set is final.
    """
    rows = []
    for (asset, horizon, model), g in forecasts.groupby(["asset", "horizon_sessions", "model"]):
        clipped = g["prediction_used"] != g["prediction_raw"]
        rows.append(
            {
                "asset": asset,
                "horizon_sessions": horizon,
                "model": model,
                "n": len(g),
                "first_origin": g["forecast_origin"].min(),
                "last_origin": g["forecast_origin"].max(),
                "qlike": qlike(g["realized_target"], g["prediction_used"]),
                "rmse": rmse(g["realized_target"], g["prediction_used"]),
                "n_clipped": int(clipped.sum()),
                "frac_clipped": float(clipped.mean()),
            }
        )
    return pd.DataFrame(rows)

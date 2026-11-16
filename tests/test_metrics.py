import numpy as np
import pandas as pd
import pytest

from vol_forecast.evaluation.metrics import apply_floor, qlike, rmse, summarize


def test_qlike_and_rmse_hand_calculated():
    y, v = [1.0, 4.0], [2.0, 2.0]
    assert qlike(y, v) == pytest.approx(((np.log(2) + 0.5) + (np.log(2) + 2.0)) / 2)
    assert rmse(y, v) == pytest.approx(np.sqrt((1 + 4) / 2))
    assert qlike([3.0], [3.0]) == pytest.approx(np.log(3) + 1)  # minimised when forecast = truth


def test_qlike_rejects_nonpositive_forecasts():
    with pytest.raises(ValueError):
        qlike([1.0], [0.0])


def test_floor_is_applied_and_counted():
    raw = pd.Series([-1.0, 0.0, 1e-9, 0.5])
    used = apply_floor(raw, 1e-6)
    assert list(used) == [1e-6, 1e-6, 1e-6, 0.5]
    with pytest.raises(ValueError):
        apply_floor(raw, 0.0)
    forecasts = pd.DataFrame({
        "asset": "A", "horizon_sessions": 1, "model": "m",
        "forecast_origin": pd.bdate_range("2020-01-01", periods=4),
        "prediction_raw": raw, "prediction_used": used, "realized_target": 0.5,
    })
    row = summarize(forecasts).iloc[0]
    assert row["n"] == 4 and row["n_clipped"] == 3 and row["frac_clipped"] == 0.75
    assert np.isfinite(row["qlike"])

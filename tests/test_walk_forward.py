import numpy as np
import pandas as pd
import pytest

from vol_forecast.evaluation.walk_forward import training_rows, variance_floor, walk_forward
from vol_forecast.models.baselines import make_model
from vol_forecast.pipeline import modelling_frame

FEATURES = ["rv_1", "rv_5", "rv_22"]
DEV_START, DEV_END = pd.Timestamp("2018-01-01"), pd.Timestamp("2020-12-31")


class SpyMeanModel:
    """Predicts the training-target mean and records exactly what it was allowed to see."""

    name = "spy_mean"

    def __init__(self):
        self.fits = []

    def fit(self, features, target):
        self.fits.append((features.index.max(), len(target)))
        self.mean = target.mean()

    def predict(self, features):
        return pd.Series(self.mean, index=features.index)


def run(returns, models, horizon=5):
    return walk_forward(modelling_frame(returns, horizon), models, FEATURES, horizon,
                        DEV_START, DEV_END, 1e-8, "TEST", "cfg")


def test_weekly_label_not_eligible_before_fifth_future_return(returns):
    frame = modelling_frame(returns, 5)
    s = 400
    for k in range(5):  # cutoffs s .. s+4: the label for origin s has not matured
        assert frame.index[s] not in training_rows(frame, FEATURES, frame.index[s + k]).index
    assert frame.index[s] in training_rows(frame, FEATURES, frame.index[s + 5]).index


def test_refit_never_sees_immature_labels_leakage_trap(returns):
    """Poison every return after a cutoff; fits at or before the cutoff must not change."""
    poisoned = returns.copy()
    cutoff = pd.Timestamp("2019-06-28")
    poisoned.loc[poisoned.index > cutoff] = 5.0  # absurd values: any leak is obvious
    clean, trap = run(returns, [SpyMeanModel()]), run(poisoned, [SpyMeanModel()])
    before = clean["fit_cutoff"] <= cutoff
    assert before.sum() > 100
    np.testing.assert_array_equal(clean.loc[before, "prediction_raw"], trap.loc[before, "prediction_raw"])
    assert trap.loc[before, "prediction_raw"].max() < 1.0


def test_fit_provenance_and_monthly_schedule(returns):
    spy = SpyMeanModel()
    out = run(returns, [spy])
    assert (out["label_maturity_cutoff"] <= out["fit_cutoff"]).all()
    assert (out["fit_cutoff"] <= out["forecast_origin"]).all()
    assert out["fit_cutoff"].nunique() == 36 == len(spy.fits)  # one refit per month, 2018-2020
    # A model fit at cutoff t trained on origins at least 5 sessions before t.
    idx = returns.index
    for (last_origin, _), cutoff in zip(spy.fits, sorted(out["fit_cutoff"].unique())):
        assert idx.get_loc(cutoff) - idx.get_loc(last_origin) == 5
    assert out.groupby("fit_cutoff")["n_train"].first().is_monotonic_increasing  # expanding window


def test_development_only_no_holdout(returns):
    """The fixture runs into 2021; no forecast, target, or training label may touch it."""
    assert returns.index.max() > DEV_END
    for horizon in (1, 5):
        out = run(returns, [SpyMeanModel()], horizon)
        assert out["forecast_origin"].min() >= DEV_START
        assert out["target_maturity"].max() <= DEV_END
        assert out["label_maturity_cutoff"].max() <= DEV_END
    # Changing holdout returns changes nothing at all.
    tampered = returns.copy()
    tampered.loc[tampered.index > DEV_END] = 5.0
    models = [make_model("naive_1"), make_model("naive_22"), SpyMeanModel()]
    pd.testing.assert_frame_equal(run(returns, models), run(tampered, [make_model("naive_1"), make_model("naive_22"), SpyMeanModel()]))


def test_models_share_origins_targets_and_units(returns):
    out = run(returns, [make_model("naive_1"), make_model("naive_22")])
    a, b = (out[out["model"] == m].reset_index(drop=True) for m in ("naive_1", "naive_22"))
    for col in ("forecast_origin", "realized_target", "target_maturity", "horizon_sessions"):
        pd.testing.assert_series_equal(a[col], b[col])
    frame = modelling_frame(returns, 5)
    origin = a["forecast_origin"].iloc[10]
    assert a["prediction_raw"].iloc[10] == pytest.approx(returns.loc[origin] ** 2)
    assert b["prediction_raw"].iloc[10] == pytest.approx(frame.loc[origin, "rv_22"])
    assert a["realized_target"].iloc[10] == pytest.approx(frame.loc[origin, "target"])


def test_floor_from_training_period_only(returns):
    end = pd.Timestamp("2017-12-31")
    floor = variance_floor(returns, end, 0.01)
    assert floor == pytest.approx(0.01 * (returns.loc[:end] ** 2).mean())
    tampered = returns.copy()
    tampered.loc[tampered.index > end] = 5.0
    assert variance_floor(tampered, end, 0.01) == floor

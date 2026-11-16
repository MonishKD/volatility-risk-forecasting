"""Walk-forward (expanding window) forecasting for one asset and one horizon.

At each forecast origin t in the development period we only use what was known after
the close of t. Models are refit on the first eligible origin of each calendar month and
reused, unchanged, for the rest of that month.

Label maturity: a training row with origin s has a target built from returns s+1 .. s+h.
It may be used in a fit at cutoff t only if session s+h <= t, i.e. its whole target has
already happened. For a 5-session target this excludes the last 5 origins before t.
"""

from __future__ import annotations

import pandas as pd

from vol_forecast.evaluation.metrics import apply_floor


def training_rows(frame: pd.DataFrame, feature_cols: list[str], fit_cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows usable for a fit at `fit_cutoff`: complete features and a fully matured label."""
    usable = frame[feature_cols].notna().all(axis=1) & frame["target"].notna()
    matured = frame["maturity"] <= fit_cutoff  # NaT (target not in the data yet) compares False
    return frame[usable & matured]


def variance_floor(returns: pd.Series, initial_train_end: pd.Timestamp, fraction: float) -> float:
    """Floor = fraction * mean squared return over the initial training period only."""
    floor = fraction * float((returns.loc[:initial_train_end] ** 2).mean())
    if not floor > 0:
        raise ValueError("could not derive a positive variance floor from the training period")
    return floor


def walk_forward(
    frame: pd.DataFrame,
    models: list,
    feature_cols: list[str],
    horizon: int,
    dev_start: pd.Timestamp,
    dev_end: pd.Timestamp,
    floor: float,
    asset: str,
    config_id: str,
) -> pd.DataFrame:
    """Forecast every eligible origin in [dev_start, dev_end] with every model.

    `frame` is indexed by session date with the feature columns plus `target` and `maturity`
    (see targets.forward_variance_target).

    An origin is eligible when its features are complete and its target is complete and
    matures on or before dev_end. That last condition keeps every scored return inside the
    development period, so nothing from the final holdout is touched even if the data extends
    past dev_end. Eligibility does not depend on the model, so all models are scored on
    identical origins and targets.
    """
    features_ok = frame[feature_cols].notna().all(axis=1)
    in_dev = (frame.index >= dev_start) & (frame["maturity"] <= dev_end)
    eligible = frame[features_ok & frame["target"].notna() & in_dev]
    if eligible.empty:
        raise ValueError(f"no eligible forecast origins for {asset} h={horizon}")

    out = []
    # Monthly refit schedule: one block of origins per calendar month.
    for _, block in eligible.groupby(eligible.index.to_period("M")):
        fit_cutoff = block.index[0]
        train = training_rows(frame, feature_cols, fit_cutoff)
        for model in models:
            model.fit(train[feature_cols], train["target"])
            raw = model.predict(block[feature_cols])
            out.append(
                pd.DataFrame(
                    {
                        "asset": asset,
                        "forecast_origin": block.index,
                        "horizon_sessions": horizon,
                        "model": model.name,
                        "prediction_raw": raw.to_numpy(),
                        "prediction_used": apply_floor(raw, floor).to_numpy(),
                        "realized_target": block["target"].to_numpy(),
                        "target_maturity": block["maturity"].to_numpy(),
                        "fit_cutoff": fit_cutoff,
                        # Latest date any training label depended on; always <= fit_cutoff.
                        "label_maturity_cutoff": train["maturity"].max(),
                        "n_train": len(train),
                        "floor": floor,
                        "config_id": config_id,
                    }
                )
            )
    return pd.concat(out, ignore_index=True)

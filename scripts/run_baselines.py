"""Days 1-5 experiment: download (if needed), plot realized volatility, walk-forward naive baselines.

Everything written to the output directory covers the development period or earlier only.
"""

import argparse
import json
import platform
from importlib.metadata import version

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vol_forecast.config import load_config
from vol_forecast.data import download_snapshot
from vol_forecast.evaluation.metrics import summarize
from vol_forecast.evaluation.walk_forward import variance_floor, walk_forward
from vol_forecast.features import WINDOWS, rv_features
from vol_forecast.models.baselines import make_model
from vol_forecast.pipeline import load_returns, modelling_frame

FEATURE_COLS = [f"rv_{w}" for w in WINDOWS]


def plot_realized_vol(returns: pd.DataFrame, dev_end: pd.Timestamp, path) -> None:
    """Annualized 5- and 22-session realized volatility per asset, up to dev_end only."""
    fig, axes = plt.subplots(len(returns.columns), 1, figsize=(10, 3 * len(returns.columns)), sharex=True)
    for ax, symbol in zip(np.atleast_1d(axes), returns.columns):
        rv = rv_features(returns[symbol]).loc[:dev_end]
        # variance per day -> volatility per year, in percent
        for col, label in (("rv_5", "5-session"), ("rv_22", "22-session")):
            ax.plot(rv.index, 100 * np.sqrt(252 * rv[col]), label=label, linewidth=0.8)
        ax.set_title(f"{symbol}: annualized realized volatility")
        ax.set_ylabel("% per year")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in cfg.symbols:
        download_snapshot(symbol, cfg.data_start, cfg.data_end, cfg.auto_adjust, cfg.raw_dir)
    returns, data_report = load_returns(cfg)

    last_session = returns.index.max()
    if last_session > cfg.dev_end:
        print(f"note: data extends to {last_session.date()}, beyond dev_end; "
              "holdout sessions are loaded but never forecast, scored, or plotted")

    plot_realized_vol(returns, cfg.dev_end, cfg.output_dir / "realized_vol.png")

    models = [make_model(name) for name in cfg.models]
    parts = []
    for symbol in cfg.symbols:
        floor = variance_floor(returns[symbol], cfg.initial_train_end, cfg.floor_fraction)
        for horizon in cfg.horizons:
            parts.append(
                walk_forward(
                    modelling_frame(returns[symbol], horizon), models, FEATURE_COLS, horizon,
                    cfg.dev_start, cfg.dev_end, floor, symbol, cfg.config_id,
                )
            )
    forecasts = pd.concat(parts, ignore_index=True)

    # Guards: nothing from the holdout, and all models share origins and targets.
    assert forecasts["target_maturity"].max() <= cfg.dev_end
    assert (forecasts["label_maturity_cutoff"] <= forecasts["fit_cutoff"]).all()
    wide = forecasts.pivot(index=["asset", "horizon_sessions", "forecast_origin"],
                           columns="model", values="realized_target")
    assert wide.notna().all().all() and wide.nunique(axis=1).eq(1).all()

    metrics = summarize(forecasts)
    forecasts.to_csv(cfg.output_dir / "forecasts_dev.csv", index=False)
    metrics.to_csv(cfg.output_dir / "metrics_dev.csv", index=False)
    metadata = {
        "config_id": cfg.config_id,
        "config": cfg.raw,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "pandas", "yfinance", "matplotlib", "pyyaml")},
        "data_report": data_report,
        "data_manifest": json.loads((cfg.raw_dir / "manifest.json").read_text()),
    }
    (cfg.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    with pd.option_context("display.width", 200, "display.float_format", "{:.6g}".format):
        print(metrics.to_string(index=False))
    print(f"\nwrote outputs to {cfg.output_dir}/ (config_id {cfg.config_id})")


if __name__ == "__main__":
    main()

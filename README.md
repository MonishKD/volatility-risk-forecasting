# Volatility and Risk Forecasting (QUARCC)

Does machine learning forecast volatility better than the standard baselines (HAR, GARCH),
and when? `START_HERE_CODING_AGENT.md` defines the scope and research protocol. Reference papers are in `risk-volatility-documentation/`.

## Status

Done so far: data pipeline, realized-volatility features, naive baselines,
and a leakage-tested walk-forward evaluation loop on SPY, TLT, and GLD.
Results and method notes: [`reports/days1_5/README.md`](reports/days1_5/README.md).
Next: HAR, which completes Milestone 1 of `START_HERE_CODING_AGENT.md`.

## Run it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # create .venv and install dependencies
uv run pytest                            # offline tests (synthetic data, no internet)
uv run python scripts/run_baselines.py   # download if needed, plot, evaluate
```

## Layout

```text
configs/baselines.yaml      symbols, dates, horizons, refit schedule, floor policy
src/vol_forecast/
  data.py                   download snapshots, clean prices, log returns
  features.py               rv_1, rv_5, rv_22 at forecast origin t
  targets.py                next-1 and next-5 session variance targets + maturity dates
  models/                   model interface and naive baselines (HAR/GARCH/LightGBM go here)
  evaluation/               walk-forward loop, QLIKE/RMSE, positive floor
scripts/                    download_data.py, run_baselines.py
tests/                      alignment and leakage tests
data/raw/                   local price snapshots (untracked) + manifest.json
reports/                    experiment outputs
```

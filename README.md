# Can Machine Learning Predict Market Risk? (QUARCC)

We're testing whether machine learning can predict how much markets will move,
which is called volatility, better than the standard models finance has used for decades.

**In plain terms:** predicting whether a stock goes up or down is close to impossible,
but predicting *how much* it will swing is not. Calm days tend to follow calm days,
and turbulent days follow turbulent ones. We build models that forecast those swings
for SPY, TLT and GLD, then check:

1. **Does ML actually win?** We compare LightGBM against the two classic models,
   HAR and GARCH, giving each the same data so the comparison is fair.
2. **When does it win?** We break the results down by asset, forecast horizon
   (next day vs. next week), and calm vs. stressed markets.
3. **Does it matter?** We test whether better forecasts lead to better risk
   estimates (Value-at-Risk) and a better-performing portfolio.

A result showing that ML adds little is still a valid finding.

**Getting started:** `START_HERE_CODING_AGENT.md` has the full scope and research
protocol, and the reference papers are in `risk-volatility-documentation/`.

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

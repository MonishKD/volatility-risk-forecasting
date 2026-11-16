# Days 1-5: data, realized volatility, naive baselines, walk-forward evaluation

Run with `uv run python scripts/run_baselines.py` (config: `configs/baselines.yaml`).
Results below are from the run recorded in `run_metadata.json` (config_id `8eb84d70e856`,
yfinance data for SPY/TLT/GLD, 2010-01-04 to 2020-12-31, 2769 sessions each, no duplicate,
nonpositive, or missing bars found).

## Timing convention

- `r[t] = ln(P[t]/P[t-1])` from dividend- and split-adjusted closes (`auto_adjust=True`).
- A forecast with **origin t** is made after the close of session t, using returns through
  `r[t]` only, and predicts sessions `t+1 .. t+h`. It could not be traded at that same close.
- Everything is in **daily variance units** (squared log returns). The plot converts to
  annualized volatility (`sqrt(252 * variance)`) only for readability.

| Quantity | Definition |
|---|---|
| `rv_1[t]`, `rv_5[t]`, `rv_22[t]` | mean of `r^2` over the last 1 / 5 / 22 sessions ending at t |
| daily target `y[t,1]` | `r[t+1]^2` |
| weekly target `y[t,5]` | mean of `r[t+1..t+5]^2` (next five trading sessions) |
| `naive_1` | forecast = `rv_1[t]` |
| `naive_22` | forecast = `rv_22[t]` |

## Evaluation protocol

- 2010-2017: initial training only. 2018-2020: walk-forward forecasts, the only scored period.
- **2021 onward is a reserved final holdout.** The config stops the download at 2020-12-31, and
  the loop additionally refuses any origin whose target ends after 2020-12-31. That is why the
  weekly horizon has 751 origins and the daily one 755.
- Expanding window, refit on the first eligible origin of each month. A training row is used
  only if its whole target has already happened by the refit date (label maturity). The naive
  models have nothing to fit, but they run through the same loop so HAR, GARCH, and LightGBM
  inherit the tested leakage protection.
- **Positive floor:** forecasts below `0.01 x mean(r^2 over 2010-2017)` (per asset) are raised to
  that value before scoring. Fixed before seeing development results. QLIKE divides by the
  forecast, so a zero forecast would give an infinite loss.
- Losses: QLIKE `= log(v) + y/v` (primary), RMSE (secondary). Lower is better for both.

## Results (development period, 2018-2020)

| asset | horizon | model | n | QLIKE | RMSE | clipped |
|---|---|---|---|---|---|---|
| SPY | 1 | naive_1 | 755 | 6.829 | 8.50e-4 | 82 (10.9%) |
| SPY | 1 | naive_22 | 755 | -7.979 | 7.87e-4 | 0 |
| SPY | 5 | naive_1 | 751 | 9.196 | 7.18e-4 | 82 (10.9%) |
| SPY | 5 | naive_22 | 751 | -7.765 | 6.21e-4 | 0 |
| TLT | 1 | naive_1 | 755 | -1.022 | 3.31e-4 | 69 (9.1%) |
| TLT | 1 | naive_22 | 755 | -8.719 | 3.55e-4 | 0 |
| TLT | 5 | naive_1 | 751 | -0.025 | 2.99e-4 | 68 (9.1%) |
| TLT | 5 | naive_22 | 751 | -8.657 | 2.95e-4 | 0 |
| GLD | 1 | naive_1 | 755 | 2.631 | 2.88e-4 | 97 (12.8%) |
| GLD | 1 | naive_22 | 755 | -8.579 | 2.13e-4 | 0 |
| GLD | 5 | naive_1 | 751 | 2.197 | 2.24e-4 | 97 (12.9%) |
| GLD | 5 | naive_22 | 751 | -8.545 | 1.19e-4 | 0 |

How to read this:

- `naive_22` beats `naive_1` on QLIKE everywhere, and on RMSE everywhere except TLT at the daily
  horizon. One day's squared return is too noisy to be a forecast; a month's average is the
  baseline that HAR and the later models have to beat.
- **The `naive_1` QLIKE values depend heavily on the floor.** About 10% of its forecasts are
  near-zero (a flat day) and get clipped; when the next day is volatile, `y/v` explodes. A
  different floor would change those numbers a lot, so treat `naive_1` QLIKE as "very bad",
  not as a precise figure. `naive_22` is never clipped, so its numbers do not depend on the floor.
- RMSE is dominated by March 2020.

## Files

- `forecasts_dev.csv`: one row per asset / origin / horizon / model, with raw and floored
  predictions, realized target, `fit_cutoff`, `label_maturity_cutoff`, `target_maturity`, `config_id`.
- `metrics_dev.csv`: the table above, full precision.
- `realized_vol.png`: annualized 5- and 22-session realized volatility, 2010-2020.
- `run_metadata.json`: config, package versions, data-cleaning report, data manifest.

## Checks

`uv run pytest`: 18 offline tests on synthetic data, covering the return formula and bad-price
handling, feature windows and future-invariance, target alignment, label maturity, a poisoned-future
leakage trap, the monthly expanding refit schedule, holdout exclusion, identical origins/targets
across models, and hand-calculated QLIKE/RMSE with clip counting.

## Caveats and decisions to review

- **Scope.** `START_HERE_CODING_AGENT.md` is the only scope document. Its Milestone 1 also
  requires HAR, which is not implemented yet; this run covers the data pipeline, naive
  baselines, and evaluation loop only.
- Daily squared returns are a noisy stand-in for true variance; the papers in
  `risk-volatility-documentation/` use intraday realized variance. This is an adaptation.
- Missing-bar detection compares the three ETFs against each other (same exchange calendar),
  not against an official calendar. FX will need per-asset calendars when it is added.
- yfinance restates adjusted prices after each dividend, so a fresh download can differ in
  late decimals from this snapshot. `data/raw/manifest.json` holds the file hashes for this run.

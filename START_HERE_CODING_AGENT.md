# Coding Agent Kickoff — QUARCC Volatility Forecasting Project

You are working in the existing repository for a university quantitative-finance research project. Implement **Milestone 1 only**: a reliable, reproducible daily-data pipeline and leakage-safe walk-forward comparison of naive variance baselines versus a HAR-style linear model on SPY, TLT, and GLD.

## Before coding

1. Inspect the repository first (README, existing files, dependency manager, tests, git status). **Preserve existing work**; do not overwrite, reset, or restructure without need.
2. If `PROJECT_SPEC.md`, the QUARCC project proposal, or other project docs are present, read them and flag any conflicts. This prompt sets the implementation scope for Milestone 1.
3. Briefly state your implementation plan, assumptions, and files you expect to create/change. Then implement, run tests, and report what actually worked. Do not claim success for commands you did not run.
4. If repository access, network access, or package installation is unavailable, implement the pipeline with test fixtures and state exactly which live-data steps you could not verify. Never invent financial data, results, or test pass reports.

## Research context

**Question:** Can machine learning improve out-of-sample forecasts of future variance relative to HAR and GARCH using daily market data, and do forecast gains improve risk calibration or volatility-targeting performance?

This is an **adaptation**, not a strict replication, of intraday HAR-RV literature. With close-to-close daily prices, squared log returns are noisy proxies for conditional variance.

**Priority order:** P0 reproducible forecasting research; P1 VaR/strategy/live forecast logging; P2 Alpaca/dashboard/intraday/pooled models. Do not implement P1 or P2 yet.

## Scope of this milestone

**Assets:** `SPY`, `TLT`, `GLD`.

**Data:** Yahoo Finance via `yfinance`, requested from `2010-01-01` through the latest fully available session for these instruments. For reproducibility, allow a configurable **fixed end date** and persist raw snapshots with download metadata. Explicitly configure and document adjusted price handling (`auto_adjust` behavior); derive research returns from properly adjusted prices. Handle missing bars, duplicates, nonpositive prices, time ordering, and trading-session calendars without inventing observations. Fail clearly when downloads are unavailable. No VIX or FX in this milestone.

**Returns:** `r_t = ln(P_t/P_{t-1})`, with the close at session `t` understood to be available only after the forecast cutoff. Define and document one forecast-origin convention and apply it everywhere: a forecast issued **after the close of session t, once the necessary closing data are available**, predicts future session returns. Never treat that forecast as executable at the same close.

**Targets (variance units, not standard deviation):**

- Daily: `y[t,1] = r[t+1]^2`.
- Weekly: `y[t,5] = (1/5) * sum_{j=1..5}(r[t+j]^2)` — next five **observed trading sessions**, not calendar days.

**Predictors at forecast origin t:**

- `rv_1[t] = r[t]^2`
- `rv_5[t] = mean(r[t-4:t]^2)` (five sessions inclusive)
- `rv_22[t] = mean(r[t-21:t]^2)` (22 sessions inclusive)

Explicitly document any convention changes. Rolling windows must use only returns through session t. Missing returns must not silently shift target definitions.

**Models in Milestone 1:**

1. `naive_1`: predicts `rv_1[t]`.
2. `naive_22`: predicts `rv_22[t]`.
3. `HAR`: OLS regression on `[rv_1, rv_5, rv_22]` with an intercept, fit separately for each horizon and asset.

HAR predictions can be negative; specify a strictly positive variance floor for **all** forecasts before QLIKE evaluation, record the number/fraction clipped per model, and document how the floor is chosen without looking at final evaluation outcomes. Keep both raw and floored forecasts if helpful. The positive floor should be a documented, fixed or training-derived policy—not tuned on test performance.

## Exact evaluation protocol

- **2010-01-01 to 2017-12-31:** initial training observations, subject to predictor availability and **label maturity**.
- **2018-01-01 to 2020-12-31:** development-period walk-forward forecasts and evaluation. This is the only period to score in this milestone.
- **2021-01-01 onward:** reserved final holdout. **Do not generate, inspect, tune against, plot, or print metrics from this period** in this milestone. Retrieving data beyond 2020 for a cached snapshot is acceptable only if explicitly necessary, but prefer requesting data ending in 2020 for milestone evaluation to avoid accidental holdout access.

Generate a forecast for each eligible session in 2018–2020, separately for each asset and horizon. Use an **expanding training window** and a **predefined monthly parameter re-estimation schedule** for HAR; generate daily forecasts between refits with the most recently fitted parameters. On every refit, a training row is eligible only if **all returns required for its target have actually occurred by that refit cutoff**. For example, a five-day label for origin t cannot be used before session t+5 closes. Use prior data through the immediately preceding eligible session or through the forecast-origin cutoff, consistently documented. No shuffling or random cross-validation.

Before scoring, confirm that naive and HAR predictions for a given asset/horizon refer to identical forecast origin dates and targets. Do not manufacture results from missing observations or silently exclude difficult periods.

**Losses:**

- Primary: `QLIKE(y,vhat) = log(vhat) + y/vhat`, for `vhat > 0`.
- Secondary: `RMSE = sqrt(mean((y-vhat)^2))`.

Keep variance units consistent in all models, forecasts, and loss calculations. Report mean QLIKE and RMSE by asset, horizon, and model, plus sample counts and clip counts. Don't use the final holdout. Diebold–Mariano / HAC comparisons belong to the next milestone, once models and a prespecified comparison protocol are finalized; create an appropriate extension point, but don't implement ML, GARCH, VaR, or strategy code yet.

## Architecture

Adapt to the repository's current conventions. Prefer small, testable modules rather than a monolithic notebook. Suggested organization **only if the repo has no existing structure**:

```text
src/vol_forecast/
  data.py
  features.py
  targets.py
  models/
    baselines.py
    har.py
  evaluation/
    walk_forward.py
    metrics.py
  config.py
scripts/
  run_milestone1.py
tests/
  test_features.py
  test_targets.py
  test_label_maturity.py
  test_walk_forward.py
  test_metrics.py
configs/
  milestone1.yaml
reports/
  milestone1/
```

Use Python, pandas, NumPy, statsmodels or scikit-learn OLS (choose one), matplotlib only if useful, and yfinance. Respect the existing dependency/tooling setup. Use a configuration file for symbols, date ranges, horizon definitions, refit frequency, storage paths and prediction-floor policy. Do not introduce Docker, cloud deployment, databases, MLflow, or extra infrastructure yet.

Provide **one documented command** to run the experiment from a clean environment, with a deterministic fixture-based test suite requiring no internet access. Record package versions and the exact configuration used. Raw provider snapshots can stay untracked; commit or describe a data manifest and avoid committing bulky copyrighted/proprietary data.

## Required tests (small synthetic price/return series)

1. Log-return formula, ordering, and missing/nonpositive price handling.
2. `rv_1`, `rv_5`, `rv_22` use no information after forecast origin t; inspect first valid indices.
3. Both target formulas use **future** sessions and are aligned to origin t.
4. Weekly target for origin t is **not** eligible at any cutoff before its fifth future return is observed.
5. Refitting at cutoff t never accesses future data/immature labels. Test a deliberate fake leakage trap.
6. Forecasts and realized targets have matching dates, horizons, and units across models.
7. QLIKE and RMSE match hand-calculated values; positive-floor handling is stable and counted.
8. Development-only evaluation never trains on, scores against, or reports holdout observations.
9. Re-running feature creation for historical origin t with additional future data produces identical historical features (future-invariance test).

## Output and definition of done

Produce:

- Working, documented data/feature/target modules and baseline/HAR models.
- A reusable walk-forward forecast interface, suitable for adding GARCH and LightGBM later.
- A **development-only** forecasts file (CSV or Parquet) with at least `asset`, `forecast_origin`, `horizon_sessions`, `model`, `prediction_raw`, `prediction_used`, `realized_target`, `fit_cutoff`, `label_maturity_cutoff` (or an equivalent explicit provenance field), and `config_id`.
- A table of development-period QLIKE/RMSE, sample counts, and clip counts for each asset/horizon/model, if market data were actually retrievable.
- Automated tests verifying alignment and absence of leakage.
- A brief `reports/milestone1/README.md` explaining timing assumptions, data caveats, commands, checks, and what could not be verified.

**Stop after Milestone 1.** Do not start GARCH, LightGBM, VaR, paper trading, dashboards, or other extensions until this milestone is inspected and approved.

## How to communicate with me

I'm a software engineering student learning ML for finance. Briefly explain financial/statistical concepts as they arise, but use sound professional software practices. Avoid overengineering. Whenever you make an important methodological choice, document why. If an ambiguity materially changes research validity, select a conservative default, record it, and flag it for review. At completion, summarize files changed, commands/tests run and their outcomes, actual evaluation artifacts, methodological caveats, and the next suggested milestone.

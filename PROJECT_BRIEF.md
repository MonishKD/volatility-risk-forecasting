# Project Brief: Live Volatility and Risk Forecasting System

Context file for continuing this project in Claude Code. Read this first.

## Who and why

- Lead: Monish, research lead at QUARCC (Concordia's quantitative research club). Team of 1 to 3 members.
- Purpose: a club research project that also serves as a CV project for quant firms, banks, and data companies (IBM, Snowflake, Databricks).
- Monish is starting solo and is a beginner in ML for finance. Explain concepts briefly as they come up.

## Goal

Forecast volatility for a small multi-asset basket with ML models, test whether they beat the standard baselines (GARCH and HAR), then turn the forecasts into a live system that computes daily risk numbers and runs a volatility-targeted paper trading portfolio.

Research question: does ML beat HAR, and when (by asset class, horizon, and regime)? This is a replication and extension of existing literature with a live track record, not a claimed novel result.

## Assets

All available through `yfinance`, daily data from 2010-01-01.

| Asset | Ticker | Role |
|---|---|---|
| S&P 500 | `SPY` | Benchmark; build everything on this first |
| Nasdaq 100 | `QQQ` | Target |
| Small caps | `IWM` | Target |
| Long Treasuries | `TLT` | Target |
| Gold | `GLD` | Target |
| Euro | `EURUSD=X` | Target |
| Yen | `USDJPY=X` | Target |
| Pound | `GBPUSD=X` | Target |
| VIX | `^VIX` | Feature only, not a target |

Data download:

```python
import yfinance as yf

tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD",
           "EURUSD=X", "USDJPY=X", "GBPUSD=X", "^VIX"]

prices = yf.download(tickers, start="2010-01-01", auto_adjust=True)["Close"]
prices.to_csv("data/prices.csv")
```

Data notes:

- FX trades on days US markets are closed, so ETF columns have gaps. Compute returns per asset; do not forward-fill blindly.
- yfinance FX data has no volume and occasional bad prints. Check for gaps and outliers.
- yfinance is unofficial and can rate-limit. Download once and work from the CSV.

## Method

**Target:** next-day and next-week realized volatility, computed from daily log returns.

**Features:** lagged realized volatility over 1, 5, and 22 days (the HAR inputs), lagged returns, VIX level and change. Every feature must use only data available at the forecast date.

**Models, in order:**

1. Naive baselines: tomorrow's volatility equals today's; 22-day rolling average.
2. GARCH(1,1) with the `arch` package.
3. HAR: linear regression on daily, weekly, and monthly past volatility (Corsi, 2009).
4. Gradient boosting with LightGBM.

**Validation:** walk-forward only (expanding or rolling window). Never shuffle or use a random split.

**Metrics:** RMSE and QLIKE, reported per asset and compared against every baseline.

**Risk layer:** daily Value-at-Risk from the forecasts, backtested by counting exceedances and running a Kupiec test.

**Strategy:** volatility-targeted portfolio (scale exposure down when forecast volatility rises), compared to buy-and-hold on Sharpe ratio and max drawdown, net of transaction costs.

**Live system:** daily scheduled job that ingests data, scores the models, logs timestamped forecasts, places paper orders, and updates a dashboard of forecast versus realized volatility.

## Paper trading

- Alpaca paper account for the five ETFs (it does not trade spot FX).
- Forecast all eight assets; trade only the ETFs. An OANDA practice account is the option if FX legs are wanted later.
- Start the live job early (around week 4 or 5) with baselines only. The length of the live record cannot be made up later.

## Plan

**First two weeks (solo, SPY only):**

1. Days 1 to 2: download SPY, compute log returns and 5-day and 22-day realized volatility, plot them.
2. Days 3 to 5: naive baselines and the walk-forward evaluation loop (RMSE, QLIKE).
3. Days 6 to 9: HAR and GARCH(1,1), added to the same comparison table.
4. Days 10 to 14: LightGBM on the same features, plus a one-page summary of results.

**After that (with members):** extend to the full basket, add the VaR layer, build the strategy backtest, then the live pipeline and dashboard.

**Team roles for three people:** data and pipeline; models and research; risk, strategy, and dashboard.

## Engineering conventions

- Git repo from day one.
- The walk-forward evaluation loop is a reusable function that every model plugs into.
- Suggested layout: `data/`, `src/` (data loading, features, models, evaluation), `notebooks/`, `reports/`.
- Packages: `yfinance`, `pandas`, `numpy`, `matplotlib`, `arch`, `scikit-learn`, `lightgbm`.
- Later: experiment tracking with MLflow.

## Reading

- Corsi (2009), "A Simple Approximate Long-Memory Model of Realized Volatility" (HAR).
- Christensen, Siggaard and Veliyev (2023), "A Machine Learning Approach to Volatility Forecasting," Journal of Financial Econometrics. Closest template for this project.
- Hull, *Options, Futures, and Other Derivatives*: volatility and VaR chapters.
- `arch` package documentation for GARCH; scikit-learn `TimeSeriesSplit` docs for validation.

## Expectations to keep honest

- With only HAR's own inputs, ML gains are usually small. Gains come from extra predictors (such as VIX), pooling across assets, and longer horizons.
- A few weeks of paper trading tests the pipeline, not the strategy. Paper fills are optimistic about slippage.
- Results that look too good usually mean look-ahead leakage. Check for it first.

## Next step

Set up the repo and build the SPY baseline: data download, realized volatility, naive baselines, and the walk-forward evaluation loop (days 1 to 5 of the plan).

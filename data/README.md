# Data

`data/raw/<SYMBOL>.csv` are raw yfinance snapshots (untracked: unofficial provider data, and
re-downloads can differ slightly because adjusted prices are restated after each dividend).
`data/raw/manifest.json` records when, how, and with which settings each snapshot was taken,
plus a SHA-256 of each file, so a result can be tied to the exact data it used.

Create or refresh snapshots with `uv run python scripts/download_data.py` (add `--force` to overwrite).

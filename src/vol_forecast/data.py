"""Price download, snapshotting, cleaning, and log returns.

Nothing here invents observations: bad or missing prices become NaN, and every return,
feature, or target that would need them becomes NaN too (and is then simply not used).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class DataUnavailableError(RuntimeError):
    """Raised when a download fails or returns nothing. We never fall back to made-up data."""


def download_snapshot(
    symbol: str, start: str, end: str, auto_adjust: bool, raw_dir: Path, force: bool = False
) -> Path:
    """Download daily bars for one symbol and persist them as a raw CSV snapshot.

    `end` is inclusive here (yfinance's own `end` is exclusive, so one day is added).
    An existing snapshot is reused unless `force` is set: yfinance is unofficial and
    rate-limited, and adjusted prices get restated, so we download once and work from the file.
    """
    import yfinance as yf

    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{symbol}.csv"
    if path.exists() and not force:
        return path

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = yf.Ticker(symbol).history(
            start=start, end=end_exclusive, interval="1d", auto_adjust=auto_adjust
        )
    except Exception as exc:  # network errors, rate limits, provider changes
        raise DataUnavailableError(f"download failed for {symbol}: {exc}") from exc
    if df is None or df.empty or "Close" not in df:
        raise DataUnavailableError(f"no data returned for {symbol} ({start}..{end})")

    # Keep the exchange-local session date and drop the timezone.
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df.to_csv(path)

    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[symbol] = {
        "provider": "yfinance",
        "yfinance_version": yf.__version__,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_start": start,
        "requested_end_inclusive": end,
        "auto_adjust": auto_adjust,
        "rows": int(len(df)),
        "first_session": str(df.index.min().date()),
        "last_session": str(df.index.max().date()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_close(path: Path) -> pd.Series:
    """Read the Close column of a raw snapshot (adjusted close when auto_adjust was true)."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df["Close"]


def clean_close(close: pd.Series) -> tuple[pd.Series, dict]:
    """Sort by date, drop duplicate dates, and blank out unusable prices.

    Returns the cleaned series and a report of what was changed. Nonpositive prices are
    set to NaN rather than dropped, so the session stays on the calendar and anything
    that needs its price is marked missing instead of silently skipping over it.
    """
    close = close.astype(float).sort_index()
    duplicated = close.index.duplicated(keep="first")
    close = close[~duplicated]
    nonpositive = close <= 0
    close = close.mask(nonpositive)
    report = {
        "duplicate_dates_dropped": int(duplicated.sum()),
        "nonpositive_prices_blanked": int(nonpositive.sum()),
        "missing_prices": int(close.isna().sum()),
    }
    return close, report


def align_sessions(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """Put all assets on the union of their session dates.

    The assets here trade on the same exchange calendar, so a date present for one asset
    but absent for another is a missing bar. It shows up as NaN (never forward-filled).
    Only use this for assets that share a trading calendar (not for FX next to ETFs).
    """
    return pd.DataFrame(closes).sort_index()


def log_returns(close: pd.Series) -> pd.Series:
    """r[t] = ln(P[t] / P[t-1]) over consecutive sessions.

    If P[t] or P[t-1] is missing, r[t] is NaN: we do not bridge a gap with a multi-day return.
    """
    return np.log(close / close.shift(1)).rename("r")

"""Layer 3 — Technical indicators: EMA, RSI, MACD, VWAP, ATR.

All functions take/return ``pandas.Series`` indexed by timestamp and are
computed causally (rolling/ewm windows only look backwards).

Conventions
-----------
- EMA uses ``adjust=False`` (recursive, trader-terminal style).
- RSI and ATR use Wilder's smoothing (``alpha = 1/period``).
- VWAP resets every session for intraday data; for daily bars it falls
  back to a rolling window because a "session" is a single bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (Section 5: EMA 9 / EMA 21)."""
    if period < 1:
        raise ValueError("EMA period must be >= 1")
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    - RSI > 70 → potentially overbought
    - RSI < 30 → potentially oversold
    (We never trade these thresholds blindly — the strategy decides.)
    """
    if period < 1:
        raise ValueError("RSI period must be >= 1")
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # avg_loss == 0 with avg_gain > 0  -> RSI = 100 (all gains).
    # both zero (flat series)          -> RSI = 50 (neutral).
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss > 0, other=np.where(avg_gain > 0, 100.0, 50.0))
    return out.astype("float64")


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9
) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    if fast >= slow:
        raise ValueError("MACD fast period must be < slow period")
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume Weighted Average Price.

    Intraday data (timestamps with a time-of-day component) gets a VWAP
    that resets every session — the benchmark intraday traders use.
    Daily data gets a rolling VWAP because each bar is its own session.
    """
    for col in ("high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"VWAP needs a '{col}' column")

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]

    if _is_intraday(df.index):
        sessions = df.index.date
        cum_pv = pv.groupby(sessions).cumsum()
        cum_v = df["volume"].groupby(sessions).cumsum()
    else:
        if window < 1:
            raise ValueError("VWAP window must be >= 1")
        cum_pv = pv.rolling(window).sum()
        cum_v = df["volume"].rolling(window).sum()

    out = cum_pv / cum_v.where(cum_v > 0)
    return out.astype("float64")


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder). Feeds volatility-based stops and the
    regime detector — never say "stop = ₹5", say "stop = k × ATR"."""
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"ATR needs a '{col}' column")
    if period < 1:
        raise ValueError("ATR period must be >= 1")

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _is_intraday(index: pd.DatetimeIndex) -> bool:
    """True when timestamps carry a time-of-day component."""
    if not isinstance(index, pd.DatetimeIndex):
        return False
    midnight = pd.Timestamp("2000-01-01").time()
    return any(t != midnight for t in index.time)

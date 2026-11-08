"""Layer 4 — Market Regime Detection (Section 6).

Classifies every bar as one of:

    TRENDING        price is moving directionally over the lookback window
    SIDEWAYS        price is going nowhere
    HIGH_VOLATILITY ATR/close is extreme — risk matters more than entries

The rules are deliberately transparent. The regime label *gates* the AI
analysis later; it never issues orders itself.

Rule details
------------
- Volatility:  atr_pct = ATR(14)/close  >  threshold (default 3%)  -> HIGH_VOLATILITY
- Trend:       |EMA21 slope| over `slope_lookback` bars > threshold (default 2%) -> TRENDING
- Otherwise:   SIDEWAYS
"""

from __future__ import annotations

from enum import Enum

import pandas as pd

from sujeeth_trading_ai.config import RegimeConfig
from sujeeth_trading_ai.indicators.technical import atr, ema


class Regime(str, Enum):
    TRENDING = "TRENDING"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


def detect_regime(df: pd.DataFrame, config: RegimeConfig | None = None) -> pd.Series:
    """Return a per-bar Regime label series aligned to ``df``.

    Bars before the indicators warm up are labelled SIDEWAYS (the
    conservative "no edge" default).
    """
    cfg = config or RegimeConfig()
    close = df["close"]

    atr_pct = atr(df, period=14) / close
    ema21 = ema(close, 21)
    slope = (ema21 - ema21.shift(cfg.slope_lookback)) / ema21.shift(
        cfg.slope_lookback
    )

    high_vol = atr_pct > cfg.atr_pct_high_vol
    trending = slope.abs() > cfg.slope_trend_threshold

    labels = pd.Series(Regime.SIDEWAYS, index=df.index, dtype="object")
    labels[trending.fillna(False)] = Regime.TRENDING
    labels[high_vol.fillna(False)] = Regime.HIGH_VOLATILITY  # volatility wins ties
    return labels


def regime_summary(regimes: pd.Series) -> dict[str, float]:
    """Share of bars spent in each regime — shown in the backtest report."""
    counts = regimes.value_counts(normalize=True)
    return {regime.value: round(float(counts.get(regime, 0.0)), 4) for regime in Regime}

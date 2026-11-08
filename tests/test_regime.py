"""Tests for Layer 4 — market regime detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sujeeth_trading_ai.config import RegimeConfig
from sujeeth_trading_ai.regime import Regime, detect_regime, regime_summary


def _df_from_closes(closes: np.ndarray) -> pd.DataFrame:
    """Wrap closes in calm OHLC bars (range = 0.4% of price)."""
    closes = np.asarray(closes, dtype="float64")
    pad = closes * 0.002
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + pad,
            "low": closes - pad,
            "close": closes,
            "volume": 1_000_000.0,
        },
        index=pd.bdate_range("2025-01-01", periods=len(closes)),
    )


def test_strong_uptrend_is_trending():
    closes = np.linspace(100.0, 160.0, 120)  # +0.5/bar ≈ +5% per 10 bars
    labels = detect_regime(_df_from_closes(closes), RegimeConfig())
    assert labels.iloc[-1] == Regime.TRENDING
    assert (labels.iloc[60:] == Regime.TRENDING).mean() > 0.8


def test_quiet_flat_market_is_sideways():
    rng = np.random.default_rng(11)
    closes = 100 + rng.normal(0, 0.15, 150).cumsum() * 0.3  # barely moves
    labels = detect_regime(_df_from_closes(closes), RegimeConfig())
    assert labels.iloc[-1] == Regime.SIDEWAYS


def test_wild_swinging_market_is_high_volatility():
    closes = np.array(
        [100, 112, 96, 115, 90, 118, 92, 116, 94, 117] * 9, dtype="float64"
    )
    pad = closes * 0.01
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + pad,
            "low": closes - pad,
            "close": closes,
            "volume": 1_000_000.0,
        },
        index=pd.bdate_range("2025-01-01", periods=len(closes)),
    )
    labels = detect_regime(df, RegimeConfig())
    assert (labels == Regime.HIGH_VOLATILITY).any()


def test_volatility_beats_trend_on_ties():
    """A violently trending market should still be flagged HIGH_VOLATILITY."""
    closes = np.array([100, 130, 105, 140, 108, 150, 110, 160, 115, 170] * 5, "float64")
    pad = closes * 0.015
    df = pd.DataFrame(
        {
            "open": closes, "high": closes + pad, "low": closes - pad,
            "close": closes, "volume": 1e6,
        },
        index=pd.bdate_range("2025-01-01", periods=len(closes)),
    )
    labels = detect_regime(df, RegimeConfig())
    assert (labels == Regime.HIGH_VOLATILITY).any()


def test_regime_summary_covers_all_three_labels():
    summary = regime_summary(
        pd.Series(
            [Regime.TRENDING, Regime.SIDEWAYS, Regime.HIGH_VOLATILITY],
            dtype="object",
        )
    )
    # Shares are rounded to 4 decimals in the summary.
    assert summary["TRENDING"] == pytest.approx(1 / 3, abs=1e-3)
    assert summary["SIDEWAYS"] == pytest.approx(1 / 3, abs=1e-3)
    assert summary["HIGH_VOLATILITY"] == pytest.approx(1 / 3, abs=1e-3)

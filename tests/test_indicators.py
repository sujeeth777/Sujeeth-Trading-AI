"""Tests for Layer 3 — technical indicators (causality and known values)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sujeeth_trading_ai.indicators import atr, ema, macd, rsi, vwap


def _idx(n):
    return pd.bdate_range("2025-01-01", periods=n)


class TestEma:
    def test_constant_series(self):
        out = ema(pd.Series([100.0] * 30, index=_idx(30)), 9)
        assert np.allclose(out, 100.0)

    def test_recursive_definition_adjust_false(self):
        s = pd.Series([10.0, 20.0, 30.0], index=_idx(3))
        out = ema(s, 3)  # alpha = 2/(3+1) = 0.5
        assert out.iloc[0] == pytest.approx(10.0)
        assert out.iloc[1] == pytest.approx(10.0 + 0.5 * 10.0)
        assert out.iloc[2] == pytest.approx(15.0 + 0.5 * 15.0)

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            ema(pd.Series([1.0, 2.0]), 0)


class TestRsi:
    def test_flat_series_is_neutral_50(self):
        out = rsi(pd.Series([100.0] * 40), 14)
        assert out.dropna().nunique() == 1
        assert out.dropna().iloc[0] == pytest.approx(50.0)

    def test_strictly_increasing_is_100(self):
        out = rsi(pd.Series(np.arange(1.0, 41.0)), 14)
        assert out.iloc[-1] == pytest.approx(100.0)

    def test_strictly_decreasing_is_0(self):
        out = rsi(pd.Series(np.arange(40.0, 0.0, -1.0)), 14)
        assert out.iloc[-1] == pytest.approx(0.0)

    def test_bounded_0_100(self):
        rng = np.random.default_rng(0)
        out = rsi(pd.Series(100 + rng.normal(0, 2, 300).cumsum()), 14)
        assert out.dropna().between(0, 100).all()


class TestMacd:
    def test_macd_line_is_ema_difference(self):
        s = pd.Series(100 + np.random.default_rng(1).normal(0, 1, 200).cumsum())
        m = macd(s, fast=12, slow=26, signal_period=9)
        expected = ema(s, 12) - ema(s, 26)
        assert np.allclose(m["macd"], expected)

    def test_constant_series_gives_zero(self):
        m = macd(pd.Series([50.0] * 100), 12, 26, 9)
        assert np.allclose(m[["macd", "signal", "histogram"]], 0.0)

    def test_fast_must_be_smaller_than_slow(self):
        with pytest.raises(ValueError):
            macd(pd.Series([1.0, 2.0, 3.0]), fast=26, slow=12)


class TestVwap:
    def test_daily_rolling_vwap_matches_manual(self):
        n = 30
        idx = _idx(n)
        df = pd.DataFrame(
            {
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
                "volume": np.linspace(100, 300, n),
            },
            index=idx,
        )
        out = vwap(df, window=5)
        typical = (df["high"] + df["low"] + df["close"]) / 3
        manual = (typical * df["volume"]).rolling(5).sum() / df["volume"].rolling(5).sum()
        # NOTE: np.allclose mis-handles NaN rows on pandas Series (ufunc
        # dispatch quirk) — use pandas' own strict equality instead.
        pd.testing.assert_series_equal(out, manual)

    def test_intraday_vwap_resets_each_session(self):
        stamps = (
            list(pd.DatetimeIndex(["2025-01-01 09:15", "2025-01-01 09:20"]))
            + list(pd.DatetimeIndex(["2025-01-02 09:15", "2025-01-02 09:20"]))
        )
        df = pd.DataFrame(
            {
                "open": [100, 102, 200, 204],
                "high": [101, 103, 201, 205],
                "low": [99, 101, 199, 203],
                "close": [100.5, 102.5, 200.5, 204.5],
                "volume": [10.0, 30.0, 5.0, 5.0],
            },
            index=pd.DatetimeIndex(stamps),
        )
        out = vwap(df)
        # First bar of each session: cumulative PV/V over one bar = typical price.
        assert out.iloc[0] == pytest.approx((101 + 99 + 100.5) / 3)
        assert out.iloc[2] == pytest.approx((201 + 199 + 200.5) / 3)
        # Second bar of session 1 blends the first two bars of that session only.
        typical = (df["high"] + df["low"] + df["close"]) / 3
        expected = (typical.iloc[:2] * df["volume"].iloc[:2]).sum() / df["volume"].iloc[:2].sum()
        assert out.iloc[1] == pytest.approx(expected)


class TestAtr:
    def test_constant_range(self):
        n = 40
        df = pd.DataFrame(
            {
                "open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0,
                "volume": 1000.0,
            },
            index=_idx(n),
        )
        out = atr(df, 14)
        assert out.dropna().iloc[-1] == pytest.approx(4.0, rel=1e-6)  # TR = high - low

    def test_flat_series_is_zero(self):
        df = pd.DataFrame(
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0},
            index=_idx(30),
        )
        assert atr(df, 14).dropna().abs().max() == pytest.approx(0.0)

    def test_non_negative(self):
        rng = np.random.default_rng(2)
        n = 100
        close = pd.Series(100 + rng.normal(0, 1, n).cumsum())
        df = pd.DataFrame(
            {
                "open": close, "high": close + 1, "low": close - 1,
                "close": close, "volume": 1000.0,
            },
            index=_idx(n),
        )
        assert (atr(df, 14).dropna() >= 0).all()

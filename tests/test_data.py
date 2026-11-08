"""Tests for Layer 1+2 — data loading, validation and synthetic generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sujeeth_trading_ai.data import generate_synthetic_ohlcv, load_csv, validate_ohlcv


class TestValidateOhlcv:
    def test_drops_rows_with_high_below_close(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [101.0, 99.0],   # second row: high < close -> invalid
                "low": [99.0, 97.0],
                "close": [100.5, 100.0],
                "volume": [1000, 1000],
            },
            index=pd.bdate_range("2025-01-01", periods=2),
        )
        out = validate_ohlcv(df, verbose=False)
        assert len(out) == 1

    def test_drops_non_positive_prices_and_volume(self):
        df = pd.DataFrame(
            {
                "open": [100.0, -5.0, 100.0],
                "high": [101.0, 4.0, 102.0],
                "low": [99.0, -6.0, 99.0],
                "close": [100.5, 3.0, 101.0],
                "volume": [1000, 1000, 0],  # zero volume invalid
            },
            index=pd.bdate_range("2025-01-01", periods=3),
        )
        out = validate_ohlcv(df, verbose=False)
        assert len(out) == 1

    def test_missing_column_raises(self):
        df = pd.DataFrame({"open": [1.0], "high": [1.0], "close": [1.0], "volume": [1]})
        with pytest.raises(ValueError, match="low"):
            validate_ohlcv(df)


class TestLoadCsv:
    def _write(self, path, rows):
        header = "timestamp,open,high,low,close,volume\n"
        path.write_text(header + "\n".join(rows), encoding="utf-8")

    def test_cleans_duplicates_unsorted_and_invalid(self, tmp_path):
        csv = tmp_path / "dirty.csv"
        # Row 2 is a duplicate of row 1; row 3 is invalid (high < close);
        # rows are intentionally out of order.
        self._write(
            csv,
            [
                "2025-01-02,100,102,99,101,1500",
                "2025-01-01,100,101,99,100.5,1000",
                "2025-01-01,100,101,99,100.5,1000",
                "2025-01-03,101,100.5,100,100.8,1200",
                "2025-01-04,101,103,100,102,1300",
            ],
        )
        df = load_csv(csv, symbol="DIRTY")
        assert list(df.index) == list(pd.DatetimeIndex(["2025-01-01", "2025-01-02", "2025-01-04"]))
        assert df.attrs["symbol"] == "DIRTY"
        assert df["close"].iloc[-1] == pytest.approx(102.0)

    def test_accepts_common_column_aliases(self, tmp_path):
        csv = tmp_path / "aliased.csv"
        csv.write_text(
            "Date,Open,High,Low,Close,Qty\n"
            "2025-01-01,10,11,9,10.5,100\n",
            encoding="utf-8",
        )
        df = load_csv(csv)
        assert len(df) == 1

    def test_missing_required_column_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("timestamp,open,high,low,volume\n2025-01-01,10,11,9,100\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            load_csv(csv)


class TestSynthetic:
    def test_deterministic_for_same_seed(self):
        a = generate_synthetic_ohlcv(bars=200, seed=7)
        b = generate_synthetic_ohlcv(bars=200, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_differs_across_seeds(self):
        a = generate_synthetic_ohlcv(bars=200, seed=1)
        b = generate_synthetic_ohlcv(bars=200, seed=2)
        assert not np.allclose(a["close"], b["close"])

    def test_output_is_valid_ohlcv(self):
        df = generate_synthetic_ohlcv(bars=400, seed=3)
        assert (df[["open", "high", "low", "close", "volume"]] > 0).all().all()
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        assert isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing

    def test_requested_bar_count_approximated(self):
        df = generate_synthetic_ohlcv(bars=500, seed=5)
        assert 450 <= len(df) <= 550

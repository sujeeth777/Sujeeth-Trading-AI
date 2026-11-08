"""Tests for the backtesting engine (Section 18) — honesty invariants.

The big one is NO LOOK-AHEAD: a decision made from bar i's close must
fill at bar i+1's open, never bar i's own close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sujeeth_trading_ai.backtest import BacktestEngine, compute_metrics
from sujeeth_trading_ai.config import AppConfig, BacktestConfig, RiskConfig
from sujeeth_trading_ai.data import generate_synthetic_ohlcv
from sujeeth_trading_ai.regime import detect_regime
from sujeeth_trading_ai.strategy import EmaRsiVwapStrategy


def run_backtest(df, risk: RiskConfig | None = None, bt: BacktestConfig | None = None):
    cfg = AppConfig()
    strategy = EmaRsiVwapStrategy(cfg.strategy)
    engine = BacktestEngine(strategy, risk or cfg.risk, bt or cfg.backtest, symbol="TEST")
    return engine.run(df, regimes=detect_regime(df, cfg.regime))


class TestEngineBasics:
    def test_deterministic_reruns(self):
        df = generate_synthetic_ohlcv(bars=400, seed=42)
        a = run_backtest(df)
        b = run_backtest(df)
        assert [(t.entry_time, t.exit_time, t.profit_loss) for t in a.trades] == [
            (t.entry_time, t.exit_time, t.profit_loss) for t in b.trades
        ]
        pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)

    def test_too_short_data_raises(self):
        df = generate_synthetic_ohlcv(bars=50, seed=1)  # < warmup(60)+2
        with pytest.raises(ValueError, match="warmup"):
            run_backtest(df)

    def test_equity_conservation(self):
        """Final equity = initial capital + sum of all trade P&L."""
        df = generate_synthetic_ohlcv(bars=500, seed=42)
        result = run_backtest(df)
        initial = result.equity_curve.iloc[0]
        expected = initial + sum(t.profit_loss for t in result.trades)
        assert result.final_equity == pytest.approx(expected, abs=1e-6)

    def test_no_overlapping_positions(self):
        df = generate_synthetic_ohlcv(bars=600, seed=9)
        result = run_backtest(df)
        for prev, nxt in zip(result.trades, result.trades[1:]):
            assert nxt.entry_time >= prev.exit_time

    def test_equity_curve_has_one_point_per_bar(self):
        df = generate_synthetic_ohlcv(bars=500, seed=42)
        result = run_backtest(df)
        assert len(result.equity_curve) == len(df)
        assert result.equity_curve.index.equals(df.index)


class TestNoLookAhead:
    def test_entry_fills_at_next_bar_open(self):
        df = generate_synthetic_ohlcv(bars=500, seed=42)
        result = run_backtest(df)
        assert result.trades, "expected at least one trade in this sample"
        for t in result.trades:
            bar_open = float(df.loc[t.entry_time, "open"])
            assert t.entry_price == pytest.approx(bar_open)

    def test_no_trade_before_warmup(self):
        df = generate_synthetic_ohlcv(bars=500, seed=42)
        cfg = AppConfig()
        result = run_backtest(df)
        first_allowed = df.index[cfg.backtest.warmup_bars]
        for t in result.trades:
            assert t.entry_time >= first_allowed


class TestExitBehaviour:
    def test_stop_loss_fills_at_stop_price(self):
        df = generate_synthetic_ohlcv(bars=500, seed=42)
        result = run_backtest(df)
        stops = [t for t in result.trades if t.exit_reason == "STOP_LOSS"]
        for t in stops:
            assert t.exit_price == pytest.approx(t.stop_loss)
            assert t.profit_loss < 0

    def test_target_fills_at_target_price(self):
        df = generate_synthetic_ohlcv(bars=600, seed=17)
        result = run_backtest(df)
        for t in result.trades:
            if t.exit_reason == "TARGET":
                assert t.exit_price == pytest.approx(t.target)
                assert t.profit_loss > 0
                break
        else:
            pytest.fail("no TARGET exit found in 600 bars — bracket logic suspect")

    def test_every_trade_has_known_exit_reason(self):
        df = generate_synthetic_ohlcv(bars=500, seed=42)
        result = run_backtest(df)
        valid = {"STOP_LOSS", "TARGET", "STRATEGY_EXIT", "END_OF_DATA"}
        assert {t.exit_reason for t in result.trades} <= valid


class TestMetrics:
    def test_hand_built_metrics(self):
        from sujeeth_trading_ai.backtest.engine import BacktestResult, Trade

        ts = pd.bdate_range("2025-01-01", periods=4)
        trades = [
            Trade(1, "X", "LONG", ts[0], ts[1], 100.0, 101.0, 100, 99.0, 102.0,
                  100.0, 1.0, "TARGET", "TRENDING", "s", 1),
            Trade(2, "X", "LONG", ts[2], ts[3], 100.0, 99.5, 100, 99.0, 102.0,
                  -50.0, -0.5, "STOP_LOSS", "SIDEWAYS", "s", 1),
        ]
        result = BacktestResult(
            trades=trades,
            equity_curve=pd.Series([10_000.0, 10_100.0, 10_050.0], index=ts[:3]),
        )
        m = compute_metrics(result)
        assert m.total_trades == 2
        assert m.win_rate_pct == 50.0
        assert m.profit_factor == 2.0
        assert m.average_win == 100.0
        assert m.average_loss == -50.0
        assert m.max_drawdown_pct == pytest.approx(-0.495, abs=0.01)
        assert m.final_capital == 10_050.0


class TestRiskGatesInLoop:
    def test_tight_daily_limit_causes_halt(self):
        """A 0.05% daily loss limit must halt sessions on any losing day."""
        df = generate_synthetic_ohlcv(bars=500, seed=42)
        risk = RiskConfig(max_daily_loss_pct=0.0005)
        result = run_backtest(df, risk=risk)
        assert result.halted_sessions, "expected at least one halted session"

    def test_kill_switch_stops_all_new_entries(self):
        df = generate_synthetic_ohlcv(bars=500, seed=42)
        strategy = EmaRsiVwapStrategy()
        engine = BacktestEngine(strategy, RiskConfig(), BacktestConfig(), symbol="K")
        engine.risk.engage_kill_switch()
        result = engine.run(df, regimes=detect_regime(df))
        assert result.trades == []
        assert result.final_equity == pytest.approx(RiskConfig().initial_capital)

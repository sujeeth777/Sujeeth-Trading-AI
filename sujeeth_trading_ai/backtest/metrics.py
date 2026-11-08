"""Backtest performance metrics (Section 19).

Produces exactly the numbers the design document's BACKTEST REPORT shows:
total trades, win rate, profit factor, max drawdown, average win/loss,
plus a couple of extras (Sharpe on daily equity, CAGR-ish return).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sujeeth_trading_ai.backtest.engine import BacktestResult, Trade


@dataclass(slots=True)
class BacktestMetrics:
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    average_win: float
    average_loss: float
    expectancy_per_trade: float
    avg_bars_held: float
    sharpe_daily: float
    halted_sessions: int
    rejected_signals: int
    by_regime: dict[str, dict[str, float]]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(result: BacktestResult) -> BacktestMetrics:
    """Derive the report numbers from a finished backtest run."""
    equity = result.equity_curve
    initial = float(equity.iloc[0]) if equity is not None and len(equity) else 0.0
    final = result.final_equity

    trades = result.trades
    wins = [t.profit_loss for t in trades if t.profit_loss > 0]
    losses = [t.profit_loss for t in trades if t.profit_loss <= 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (math.inf if gross_profit > 0 else 0.0)
    )

    max_dd_pct = _max_drawdown_pct(equity) if equity is not None else 0.0

    by_regime: dict[str, dict[str, float]] = {}
    for regime in sorted({t.market_regime for t in trades}):
        group = [t for t in trades if t.market_regime == regime]
        group_wins = [t.profit_loss for t in group if t.profit_loss > 0]
        group_losses = [t.profit_loss for t in group if t.profit_loss <= 0]
        by_regime[regime] = {
            "trades": len(group),
            "win_rate_pct": round(100.0 * len(group_wins) / len(group), 2) if group else 0.0,
            "net_pnl": round(sum(t.profit_loss for t in group), 2),
            "profit_factor": round(
                sum(group_wins) / abs(sum(group_losses)), 2
            ) if group_losses else (math.inf if group_wins else 0.0),
        }

    daily_returns = equity.pct_change().dropna() if equity is not None else pd.Series(dtype="float64")
    sharpe = (
        float(daily_returns.mean() / daily_returns.std(ddof=0) * math.sqrt(252))
        if len(daily_returns) > 1 and daily_returns.std(ddof=0) > 0
        else 0.0
    )

    return BacktestMetrics(
        initial_capital=round(initial, 2),
        final_capital=round(final, 2),
        total_return_pct=round((final / initial - 1.0) * 100.0, 2) if initial else 0.0,
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate_pct=round(100.0 * len(wins) / len(trades), 2) if trades else 0.0,
        profit_factor=round(profit_factor, 2) if math.isfinite(profit_factor) else float("inf"),
        max_drawdown_pct=max_dd_pct,
        average_win=round(float(np.mean(wins)), 2) if wins else 0.0,
        average_loss=round(float(np.mean(losses)), 2) if losses else 0.0,
        expectancy_per_trade=round(
            (final - initial) / len(trades), 2
        ) if trades else 0.0,
        avg_bars_held=round(
            float(np.mean([t.bars_held for t in trades])), 1
        ) if trades else 0.0,
        sharpe_daily=round(sharpe, 2),
        halted_sessions=len(result.halted_sessions),
        rejected_signals=len(result.rejected_signals),
        by_regime=by_regime,
    )


def _max_drawdown_pct(equity: pd.Series) -> float:
    """Worst peak-to-trough decline of the equity curve, in percent."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return round(float(drawdown.min()) * 100.0, 2)

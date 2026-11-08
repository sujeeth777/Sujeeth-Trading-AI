"""BACKTEST REPORT formatting (Section 19) and CSV/JSON export.

The report is deliberately plain text — readable in a terminal, in git,
and in a commit diff. Machine-readable copies (trades CSV, metrics JSON,
equity curve CSV) land next to it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sujeeth_trading_ai.backtest.engine import BacktestResult
from sujeeth_trading_ai.backtest.metrics import BacktestMetrics, compute_metrics

_LINE = "─" * 46


def format_report(
    result: BacktestResult,
    symbol: str,
    metrics: BacktestMetrics | None = None,
) -> str:
    """Render the human-readable BACKTEST REPORT."""
    m = metrics or compute_metrics(result)
    lines = [
        "BACKTEST REPORT",
        _LINE,
        f"Symbol                {symbol}",
        f"Initial Capital       ₹{m.initial_capital:,.2f}",
        f"Final Capital         ₹{m.final_capital:,.2f}",
        f"Total Return          {m.total_return_pct:+.2f}%",
        _LINE,
        f"Total Trades          {m.total_trades}",
        f"Winning Trades        {m.winning_trades}",
        f"Losing Trades         {m.losing_trades}",
        f"Win Rate              {m.win_rate_pct:.2f}%",
        f"Profit Factor         {m.profit_factor}",
        f"Max Drawdown          {m.max_drawdown_pct:.2f}%",
        f"Average Win           ₹{m.average_win:,.2f}",
        f"Average Loss          ₹{m.average_loss:,.2f}",
        f"Expectancy / trade    ₹{m.expectancy_per_trade:,.2f}",
        f"Avg Bars Held         {m.avg_bars_held}",
        f"Sharpe (daily, ann.)  {m.sharpe_daily}",
        _LINE,
        f"Risk gates:  {m.rejected_signals} signals vetoed by the risk engine",
        f"             {m.halted_sessions} session(s) halted by the daily loss limit",
        _LINE,
        "PERFORMANCE BY MARKET REGIME",
    ]
    if m.by_regime:
        lines.append(f"{'regime':<18}{'trades':>7}{'win%':>8}{'net P&L':>12}{'PF':>8}")
        for regime, stats in m.by_regime.items():
            lines.append(
                f"{regime:<18}{stats['trades']:>7}{stats['win_rate_pct']:>8.2f}"
                f"{stats['net_pnl']:>12,.2f}{stats['profit_factor']:>8}"
            )
    else:
        lines.append("(no trades taken)")

    lines += [_LINE, "TRADE LOG (last 15)"]
    if result.trades:
        for t in result.trades[-15:]:
            lines.append(
                f"#{t.trade_id:03d} {t.entry_time.date()} -> {t.exit_time.date()}  "
                f"{t.symbol:<9} {t.side:<5} qty={t.quantity:<5d} "
                f"entry ₹{t.entry_price:.2f} exit ₹{t.exit_price:.2f}  "
                f"P&L ₹{t.profit_loss:+,.2f} ({t.profit_loss_pct:+.2f}%)  "
                f"[{t.exit_reason} | {t.market_regime}]"
            )
    else:
        lines.append("(no trades)")
    return "\n".join(lines)


def save_outputs(
    result: BacktestResult,
    symbol: str,
    out_dir: str | Path,
    metrics: BacktestMetrics | None = None,
) -> dict[str, Path]:
    """Write report.txt, trades.csv, metrics.json and equity_curve.csv."""
    m = metrics or compute_metrics(result)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["report"] = out / f"{symbol}_report.txt"
    paths["report"].write_text(format_report(result, symbol, m), encoding="utf-8")

    trades_df = pd.DataFrame([asdict(t) for t in result.trades])
    paths["trades"] = out / f"{symbol}_trades.csv"
    trades_df.to_csv(paths["trades"], index=False)

    metrics_json = m.to_dict() | {"symbol": symbol}
    paths["metrics"] = out / f"{symbol}_metrics.json"
    paths["metrics"].write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    paths["equity"] = out / f"{symbol}_equity_curve.csv"
    result.equity_curve.to_csv(paths["equity"], index_label="timestamp")
    return paths

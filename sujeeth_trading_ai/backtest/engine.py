"""Backtesting Engine (Section 18).

Walks historical candles in strict time order. Two rules keep it honest:

1. Decisions use bar ``i`` information only (indicators are causal).
2. A signal confirmed at bar ``i``'s close fills at bar ``i+1``'s open —
   you can never trade a price you couldn't have traded in reality.

Stops and targets are bracket levels fixed when the order was placed
(derived from ATR at bar i), and intra-bar exits assume the WORST case:
if a bar touches both stop and target, the stop is taken.

Exits: stop-loss, take-profit, strategy exit signal (at close), or
end-of-data flat liquidation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from sujeeth_trading_ai.config import BacktestConfig, RiskConfig
from sujeeth_trading_ai.indicators.technical import atr, ema, macd, rsi, vwap
from sujeeth_trading_ai.regime.detector import Regime
from sujeeth_trading_ai.risk.engine import PositionProposal, RiskEngine
from sujeeth_trading_ai.strategy.base import Signal, Strategy, StrategyContext


@dataclass(slots=True)
class Trade:
    trade_id: int
    symbol: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    stop_loss: float
    target: float
    profit_loss: float
    profit_loss_pct: float
    exit_reason: str  # STOP_LOSS | TARGET | STRATEGY_EXIT | END_OF_DATA
    market_regime: str
    strategy: str
    bars_held: int


@dataclass(slots=True)
class OpenPosition:
    entry_time: pd.Timestamp
    entry_price: float
    quantity: int
    stop_loss: float
    target: float
    regime_at_entry: str
    entry_bar: int


@dataclass(slots=True)
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    halted_sessions: list[pd.Timestamp] = field(default_factory=list)
    rejected_signals: list[str] = field(default_factory=list)
    warmup_bars: int = 0

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if self.equity_curve is not None else 0.0


class BacktestEngine:
    """Event-driven, long-only, one-position-at-a-time backtester."""

    def __init__(
        self,
        strategy: Strategy,
        risk_config: RiskConfig,
        backtest_config: BacktestConfig | None = None,
        symbol: str = "SYNTH",
    ):
        self.strategy = strategy
        self.risk = RiskEngine(risk_config)
        self.bt_cfg = backtest_config or BacktestConfig()
        self.symbol = symbol

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, regimes: pd.Series | None = None) -> BacktestResult:
        """Run the backtest over an OHLCV DataFrame."""
        result = BacktestResult()
        result.warmup_bars = self.bt_cfg.warmup_bars
        if len(df) <= self.bt_cfg.warmup_bars + 2:
            raise ValueError(
                f"Need more than warmup_bars+2 rows (got {len(df)}, warmup={self.bt_cfg.warmup_bars})"
            )

        feat = self._features(df)
        if regimes is None:
            regimes = pd.Series(Regime.SIDEWAYS.value, index=df.index)

        initial_capital = self.risk.config.initial_capital
        cash = initial_capital
        position: OpenPosition | None = None
        equity_values: list[float] = []

        session_start_equity = initial_capital
        halted_dates: set[pd.Timestamp] = set()
        trade_id = 0
        pending_entry: tuple[int, OpenPosition] | None = None  # (fill_bar, position)

        for i in range(len(df)):
            ts = df.index[i]
            bar = df.iloc[i]
            row = feat.iloc[i]

            # A signal confirmed on the previous bar fills at THIS bar's open.
            if pending_entry is not None and pending_entry[0] == i:
                pos = pending_entry[1]
                pos.entry_price = float(bar["open"])  # actual fill price
                if pos.entry_price <= pos.stop_loss:
                    # Gapped through the stop at the open: fill, stop out instantly.
                    cash -= pos.quantity * pos.entry_price
                    cash += pos.quantity * pos.stop_loss
                    trade_id += 1
                    pnl = pos.quantity * (pos.stop_loss - pos.entry_price)
                    result.trades.append(self._make_trade(
                        trade_id, pos, exit_time=ts, exit_price=pos.stop_loss,
                        pnl=pnl, reason="STOP_LOSS", bars_held=0,
                    ))
                    pending_entry = None
                else:
                    cash -= pos.quantity * pos.entry_price
                    position = pos
                    pending_entry = None

            # --- manage an open position on this bar -----------------------
            if position is not None:
                exited = self._manage_position(position, bar, row, ts, i)
                if exited is not None:
                    trade_id += 1
                    cash += exited[0]  # proceeds
                    result.trades.append(self._make_trade(
                        trade_id, position,
                        exit_time=exited[1], exit_price=exited[2],
                        pnl=exited[0] - position.quantity * position.entry_price,
                        reason=exited[3], bars_held=i - position.entry_bar,
                    ))
                    position = None

            # --- mark-to-market equity -------------------------------------
            equity = cash + (
                position.quantity * float(bar["close"]) if position is not None else 0.0
            )
            equity_values.append(equity)

            # --- daily loss limit gate (Section 14) -------------------------
            # Session start = equity at the START of the day, i.e. the prior
            # bar's close equity (or initial capital on the very first bar).
            session_date = pd.Timestamp(ts.date())
            if i == 0:
                session_start_equity = initial_capital
            elif df.index[i - 1].date() != ts.date():
                session_start_equity = equity_values[i - 1]
            if (
                session_date not in halted_dates
                and self.risk.is_halted_for_day(session_start_equity, equity)
            ):
                halted_dates.add(session_date)
                result.halted_sessions.append(session_date)

            # --- look for a new entry (decided at close of bar i) -----------
            if (
                position is None
                and pending_entry is None
                and i >= self.bt_cfg.warmup_bars
                and i < len(df) - 1  # need a next bar to fill at
                and session_date not in halted_dates
            ):
                ctx = self._context(ts, bar, row)
                signal = self.strategy.entry_signal(ctx)
                if signal is Signal.BUY_CANDIDATE:
                    close_i = float(bar["close"])
                    stop = close_i - self._stop_distance(row)
                    target = close_i + self._target_distance(row)
                    decision = self.risk.evaluate(
                        PositionProposal("LONG", close_i, stop, target),
                        equity=equity,
                        cash=cash,
                        session_start_equity=session_start_equity,
                        current_equity=equity,
                    )
                    if decision.approved:
                        regime_i = regimes.iloc[i]
                        pending_entry = (
                            i + 1,
                            OpenPosition(
                                entry_time=df.index[i + 1],
                                entry_price=close_i,  # reference; refilled at open
                                quantity=decision.quantity,
                                stop_loss=stop,
                                target=target,
                                regime_at_entry=getattr(regime_i, "value", str(regime_i)),
                                entry_bar=i + 1,
                            ),
                        )
                    else:
                        result.rejected_signals.append(f"{ts} {decision}")

        # --- end of data: liquidate any open position at the last close ----
        if position is not None:
            last_ts = df.index[-1]
            last_close = float(df["close"].iloc[-1])
            proceeds = position.quantity * last_close
            trade_id += 1
            result.trades.append(self._make_trade(
                trade_id, position, exit_time=last_ts, exit_price=last_close,
                pnl=proceeds - position.quantity * position.entry_price,
                reason="END_OF_DATA", bars_held=len(df) - 1 - position.entry_bar,
            ))
            equity_values[-1] = cash + proceeds

        result.equity_curve = pd.Series(
            equity_values, index=df.index, name="equity"
        )
        result.halted_sessions = sorted(halted_dates)
        return result

    # ------------------------------------------------------------------
    def _manage_position(self, pos: OpenPosition, bar, row, ts, i):
        """Check stop/target/strategy-exit for the current bar.

        Returns (proceeds, exit_time, exit_price, reason) or None.
        Worst-case assumption: if the bar touches both stop and target,
        the stop fires first.
        """
        low, high, close = float(bar["low"]), float(bar["high"]), float(bar["close"])

        if low <= pos.stop_loss:
            return pos.quantity * pos.stop_loss, ts, pos.stop_loss, "STOP_LOSS"
        if high >= pos.target:
            return pos.quantity * pos.target, ts, pos.target, "TARGET"
        ctx = self._context(ts, bar, row)
        if self.strategy.exit_signal(ctx):
            return pos.quantity * close, ts, close, "STRATEGY_EXIT"
        return None

    def _stop_distance(self, row) -> float:
        return self._atr_stop_multiplier * float(row["atr"])

    def _target_distance(self, row) -> float:
        return self._atr_target_multiplier * float(row["atr"])

    # The strategy config carries the ATR multipliers (k*ATR stop, 2k*ATR target).
    @property
    def _atr_stop_multiplier(self) -> float:
        return getattr(self.strategy, "config").atr_stop_multiplier

    @property
    def _atr_target_multiplier(self) -> float:
        return getattr(self.strategy, "config").atr_target_multiplier

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Causal indicator frame — value at i uses bars 0..i only."""
        scfg = getattr(self.strategy, "config", None)
        fast = getattr(scfg, "ema_fast", 9)
        slow = getattr(scfg, "ema_slow", 21)
        rsi_p = getattr(scfg, "rsi_period", 14)
        vol_w = getattr(scfg, "volume_avg_window", 20)
        vwap_w = getattr(scfg, "vwap_window", 20)
        atr_p = getattr(scfg, "atr_period", 14)

        feat = pd.DataFrame(index=df.index)
        feat["ema_fast"] = ema(df["close"], fast)
        feat["ema_slow"] = ema(df["close"], slow)
        feat["rsi"] = rsi(df["close"], rsi_p)
        feat["vwap"] = vwap(df, vwap_w)
        feat["atr"] = atr(df, atr_p)
        feat["volume_avg"] = df["volume"].rolling(vol_w).mean()
        feat["macd_hist"] = macd(df["close"])["histogram"]
        return feat

    def _context(self, ts, bar, row) -> StrategyContext:
        return StrategyContext(
            timestamp=ts,
            open=float(bar["open"]), high=float(bar["high"]),
            low=float(bar["low"]), close=float(bar["close"]),
            volume=float(bar["volume"]),
            ema_fast=float(row["ema_fast"]), ema_slow=float(row["ema_slow"]),
            rsi=float(row["rsi"]), vwap=float(row["vwap"]), atr=float(row["atr"]),
            volume_avg=float(row["volume_avg"]),
            macd_histogram=float(row["macd_hist"]),
        )

    def _make_trade(self, trade_id, pos: OpenPosition, *, exit_time, exit_price,
                    pnl, reason, bars_held) -> Trade:
        cost = pos.quantity * pos.entry_price
        return Trade(
            trade_id=trade_id,
            symbol=self.symbol,
            side="LONG",
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            stop_loss=pos.stop_loss,
            target=pos.target,
            profit_loss=pnl,  # full precision; the equity curve must reconcile exactly
            profit_loss_pct=pnl / cost * 100.0 if cost else 0.0,
            exit_reason=reason,
            market_regime=pos.regime_at_entry,
            strategy=self.strategy.name,
            bars_held=bars_held,
        )

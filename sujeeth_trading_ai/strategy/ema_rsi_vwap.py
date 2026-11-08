"""The v0.1 reference strategy — EMA + RSI + VWAP + volume (Section 7).

    BUY CANDIDATE IF:
        EMA9 > EMA21              (short-term bullish momentum)
        AND RSI > 50              (momentum confirms, not overbought-required)
        AND close > VWAP          (price above the volume benchmark)
        AND volume > avg volume   (participation confirms the move)

    EXIT an open long IF:
        EMA9 crosses below EMA21  (momentum has flipped)

A SELL_CANDIDATE is the mirror image; in v0.1 (long-only) it is emitted
for analytics but not traded. Stops and targets are NOT decided here —
the strategy proposes direction, the risk engine sizes and bounds it.
"""

from __future__ import annotations

import math

from sujeeth_trading_ai.config import StrategyConfig
from sujeeth_trading_ai.strategy.base import Signal, Strategy, StrategyContext


class EmaRsiVwapStrategy(Strategy):
    name = "ema_rsi_vwap_v1"

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()

    def entry_signal(self, ctx: StrategyContext) -> Signal:
        if not all(math.isfinite(v) for v in (ctx.ema_fast, ctx.ema_slow, ctx.rsi, ctx.vwap, ctx.volume_avg)):
            return Signal.NEUTRAL  # indicators still warming up

        bull = (
            ctx.ema_fast > ctx.ema_slow
            and ctx.rsi > self.config.rsi_bull_threshold
            and ctx.close > ctx.vwap
            and ctx.volume > ctx.volume_avg
        )
        if bull:
            return Signal.BUY_CANDIDATE

        bear = (
            ctx.ema_fast < ctx.ema_slow
            and ctx.rsi < self.config.rsi_bull_threshold
            and ctx.close < ctx.vwap
            and ctx.volume > ctx.volume_avg
        )
        if bear:
            return Signal.SELL_CANDIDATE

        return Signal.NEUTRAL

    def exit_signal(self, ctx: StrategyContext) -> bool:
        # Momentum flip: fast EMA drops below slow EMA.
        return math.isfinite(ctx.ema_fast) and math.isfinite(ctx.ema_slow) and ctx.ema_fast < ctx.ema_slow

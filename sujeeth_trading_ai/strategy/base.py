"""Strategy engine foundations (Section 7).

A strategy NEVER says "buy now". It says "this setup deserves further
analysis" — Signal.BUY_CANDIDATE. The signal engine (backtest/paper
loop) then consults the risk engine, which has final say.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"  # reserved for later versions


class Signal(str, Enum):
    """What a strategy may emit. Note the vocabulary: candidates, not orders."""

    BUY_CANDIDATE = "BUY_CANDIDATE"
    SELL_CANDIDATE = "SELL_CANDIDATE"
    EXIT = "EXIT"
    NEUTRAL = "NEUTRAL"


@dataclass(slots=True)
class StrategyContext:
    """Everything a strategy is allowed to know about bar ``i``.

    Built causally by the engine — a strategy physically cannot see
    bars > i, which is how look-ahead bias is prevented.
    """

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    ema_fast: float
    ema_slow: float
    rsi: float
    vwap: float
    atr: float
    volume_avg: float
    macd_histogram: float


class Strategy:
    """Base class. Subclasses implement entry_signal() and exit_signal()."""

    name: str = "base"

    def entry_signal(self, ctx: StrategyContext) -> Signal:
        raise NotImplementedError

    def exit_signal(self, ctx: StrategyContext) -> bool:
        """Return True when an open position should be closed at this bar's close."""
        raise NotImplementedError

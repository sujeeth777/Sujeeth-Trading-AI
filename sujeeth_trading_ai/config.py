"""
Central configuration for Sujeeth Trading AI.

All money is in INR (₹). All percentages are fractions of 1 (0.005 = 0.5%).

The values here follow the design document:
  - Max risk per trade:      0.5%  of capital   (Section 11)
  - Max daily loss:          1.5%  of capital   (Section 14)
  - Minimum reward : risk:   2 : 1              (Section 13)
  - Stops derived from ATR (volatility), never fixed rupee amounts (Section 5)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path


@dataclass(slots=True)
class StrategyConfig:
    """Parameters of the rule-based candidate strategy (Section 7)."""

    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_bull_threshold: float = 50.0   # RSI must be above this for longs
    volume_avg_window: int = 20        # "volume > average volume"
    vwap_window: int = 20              # rolling VWAP window for daily bars
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5   # stop  = entry - 1.5 * ATR
    atr_target_multiplier: float = 3.0 # target = entry + 3.0 * ATR (2:1)


@dataclass(slots=True)
class RiskConfig:
    """Risk engine limits. The risk engine always has veto power (Section 10)."""

    initial_capital: float = 10_000.0
    max_risk_per_trade_pct: float = 0.005   # 0.5% of current equity per trade
    max_daily_loss_pct: float = 0.015       # 1.5% halt for the session
    min_reward_risk: float = 2.0            # reject trades below 2:1 R:R
    min_stop_distance: float = 0.05         # reject if stop closer than 5 paise


@dataclass(slots=True)
class RegimeConfig:
    """Market regime detection thresholds (Section 6)."""

    atr_pct_high_vol: float = 0.030     # ATR/close above 3% => HIGH_VOLATILITY
    slope_lookback: int = 10            # bars used to measure EMA slope
    slope_trend_threshold: float = 0.02 # |EMA move| over lookback => TRENDING


@dataclass(slots=True)
class BacktestConfig:
    """Backtest engine behaviour."""

    warmup_bars: int = 60               # bars skipped so indicators stabilise
    allow_shorts: bool = False          # v0.1 is long-only


@dataclass(slots=True)
class AppConfig:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "AppConfig":
        """Load configuration from a JSON file, filling any missing keys with defaults."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls()
        for section_name, section_value in raw.items():
            if not hasattr(cfg, section_name):
                raise ValueError(f"Unknown config section: {section_name!r}")
            section = getattr(cfg, section_name)
            for key, value in section_value.items():
                if not hasattr(section, key):
                    raise ValueError(f"Unknown config key: {section_name}.{key}")
                setattr(section, key, value)
        return cfg

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _dataclass_repr(obj) -> str:  # pragma: no cover - debug helper
    return json.dumps(asdict(obj), indent=2) if is_dataclass(obj) else repr(obj)

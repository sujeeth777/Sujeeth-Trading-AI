"""Sujeeth Trading AI — an AI-assisted algorithmic trading research laboratory.

Core principle: the AI proposes, the strategy/risk/execution layers decide.

Version 0.1 (current): historical data -> indicators -> market regime ->
rule-based strategy -> risk engine -> backtest -> report.
"""

__version__ = "0.1.0"

from sujeeth_trading_ai.config import AppConfig, RiskConfig, StrategyConfig  # noqa: F401

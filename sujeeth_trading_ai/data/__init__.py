"""Layer 1 + 2 — Market Data Engine and Data Processing.

Raw market data goes in; a validated, cleaned, time-sorted OHLCV
DataFrame comes out. Nothing downstream is allowed to touch unclean data.
"""

from sujeeth_trading_ai.data.loader import load_csv, validate_ohlcv
from sujeeth_trading_ai.data.synthetic import generate_synthetic_ohlcv

__all__ = ["load_csv", "validate_ohlcv", "generate_synthetic_ohlcv"]

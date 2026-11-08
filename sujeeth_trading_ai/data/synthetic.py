"""Layer 1 — synthetic market data generator.

Produces deterministic, regime-shifted OHLCV series so the whole
pipeline runs offline with no broker or data vendor. The generator walks
through segments (trend up / sideways / high volatility / trend down) so
the regime detector and backtest engine see realistic variety.

Synthetic data is for plumbing tests only — never for strategy conclusions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SegmentName = str  # one of: "trend_up", "trend_down", "sideways", "high_vol"

_DEFAULT_SEGMENTS: list[tuple[SegmentName, int]] = [
    ("trend_up", 120),
    ("sideways", 100),
    ("high_vol", 80),
    ("trend_up", 90),
    ("trend_down", 90),
    ("sideways", 120),
]


def generate_synthetic_ohlcv(
    bars: int = 500,
    seed: int = 42,
    start_price: float = 1_000.0,
    start: str = "2025-01-01",
    segments: list[tuple[SegmentName, int]] | None = None,
) -> pd.DataFrame:
    """Generate a daily-bar OHLCV DataFrame with regime shifts.

    Parameters
    ----------
    bars:
        Approximate number of bars; segment lengths are scaled to match.
    seed:
        Seed for the numpy generator — same seed, same data, always.
    start_price:
        Price of the first close.
    start:
        First business date of the series.

    Returns
    -------
    pd.DataFrame with DatetimeIndex (business days) and
    open/high/low/close/volume columns, guaranteed OHLC-valid.
    """
    rng = np.random.default_rng(seed)
    segments = segments or _DEFAULT_SEGMENTS

    total_weight = sum(n for _, n in segments)
    scaled = [(name, max(10, round(n * bars / total_weight))) for name, n in segments]

    per_bar_drift = {
        "trend_up": 0.0012,
        "trend_down": -0.0012,
        "sideways": 0.0,
        "high_vol": 0.0,
    }
    per_bar_vol = {
        "trend_up": 0.010,
        "trend_down": 0.012,
        "sideways": 0.006,
        "high_vol": 0.028,
    }

    closes: list[float] = [start_price]
    labels: list[SegmentName] = ["warmup"]
    price = start_price
    for name, n in scaled:
        for _ in range(n):
            shock = rng.normal(0.0, per_bar_vol[name])
            price = max(1.0, price * (1.0 + per_bar_drift[name] + shock))
            closes.append(price)
            labels.append(name)

    index = pd.bdate_range(start=start, periods=len(closes))
    close = pd.Series(closes, index=index, name="close")

    # Build OHLC around each close with regime-scaled intraday ranges.
    rows = []
    base_volume = 1_000_000.0
    prev_close = close.iloc[0]
    for ts, c, label in zip(
        close.index[1:], close.to_numpy()[1:], labels[1:], strict=True
    ):
        rng_bar = {
            "trend_up": 0.008,
            "trend_down": 0.010,
            "sideways": 0.005,
            "high_vol": 0.022,
        }[label]
        open_ = prev_close * (1 + rng.normal(0, rng_bar / 3))
        gap_range = abs(c - open_) + c * rng_bar * abs(rng.normal(0.6, 0.2))
        high = max(open_, c) + gap_range * abs(rng.normal(0.3, 0.1))
        low = min(open_, c) - gap_range * abs(rng.normal(0.3, 0.1))
        volume = base_volume * (1 + abs(rng.normal(0, 0.35))) * (
            1.8 if label == "high_vol" else 1.0
        )
        rows.append((open_, high, low, c, volume))
        prev_close = c

    # `rows` intentionally has one fewer entry than `closes` (the first close
    # only seeds prev_close), so align it with close.index[1:].
    df = pd.DataFrame(
        rows,
        index=close.index[1 : 1 + len(rows)],
        columns=["open", "high", "low", "close", "volume"],
    ).astype("float64")
    return df

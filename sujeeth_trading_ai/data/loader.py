"""Layer 2 — Data Processing.

Pipeline (Section 4 of the design document):

    Raw CSV rows
        -> timestamp normalisation (UTC-naive DatetimeIndex)
        -> duplicate removal
        -> OHLCV validation (bad rows dropped, never "repaired")
        -> time-sorted DataFrame

The rest of the system consumes only the DataFrame this module produces.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# Aliases accepted in input CSVs, mapped to canonical lower-case names.
_COLUMN_ALIASES = {
    "timestamp": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "qty": "volume",
}


def load_csv(path, symbol: str | None = None) -> pd.DataFrame:
    """Load an OHLCV CSV file into the canonical DataFrame format.

    The result has a DatetimeIndex named ``timestamp`` and the columns
    ``open, high, low, close, volume`` (float64). Rows that fail OHLCV
    validation are dropped; a summary of what was removed is printed.

    Parameters
    ----------
    path:
        Path to the CSV file.
    symbol:
        Optional symbol label stored in ``df.attrs["symbol"]``.
    """
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    renamed = raw.rename(columns=_COLUMN_ALIASES)

    missing = [c for c in REQUIRED_COLUMNS if c not in renamed.columns]
    if "timestamp" not in renamed.columns:
        missing.append("timestamp")
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = renamed[["timestamp", *REQUIRED_COLUMNS]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("timestamp")
    df = df.sort_index()

    # Timestamp normalisation -> duplicates -> validation, in that order.
    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    dropped_dupes = before - len(df)

    df = validate_ohlcv(df, verbose=False)
    if df.empty:
        raise ValueError("No valid OHLCV rows remain after validation")

    if symbol:
        df.attrs["symbol"] = symbol
    print(
        f"[data] loaded {path}: {len(df)} rows "
        f"({dropped_dupes} duplicate, {before - dropped_dupes - len(df)} invalid rows dropped)"
    )
    return df


def validate_ohlcv(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Return only rows that satisfy OHLCV integrity rules.

    A row is valid when prices/volume are positive and:
        high >= max(open, close)   and   low <= min(open, close)

    Invalid rows are dropped, not repaired — silently inventing prices
    would poison every layer above this one.
    """
    missing_cols = [c for c in ("open", "high", "low", "close", "volume") if c not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing OHLCV columns: {missing_cols}")

    numeric = df.select_dtypes(include="number")
    if len(numeric.columns) < 5:  # pragma: no cover - defensive
        raise ValueError("OHLCV columns must be numeric")

    positive = (
        (df[["open", "high", "low", "close", "volume"]] > 0).all(axis=1)
    )
    consistent = (
        (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
    )
    valid = positive & consistent
    if verbose and (~valid).any():
        print(f"[data] dropped {(~valid).sum()} invalid OHLCV rows")
    return df.loc[valid].astype("float64")

"""CLI runner — the v0.1 research laboratory entry point.

Usage:
    python -m sujeeth_trading_ai --synthetic --bars 500 --seed 42
    python -m sujeeth_trading_ai --csv data/sample_reliance_daily.csv --symbol RELIANCE
    python -m sujeeth_trading_ai --csv my_data.csv --config my_config.json

Pipeline: data -> indicators -> regime -> strategy -> risk -> backtest -> report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sujeeth_trading_ai import __version__
from sujeeth_trading_ai.analytics.report import format_report, save_outputs
from sujeeth_trading_ai.backtest.engine import BacktestEngine
from sujeeth_trading_ai.backtest.metrics import compute_metrics
from sujeeth_trading_ai.config import AppConfig
from sujeeth_trading_ai.data import generate_synthetic_ohlcv, load_csv
from sujeeth_trading_ai.regime.detector import detect_regime, regime_summary
from sujeeth_trading_ai.strategy import EmaRsiVwapStrategy


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sujeeth-trading-ai",
        description="Sujeeth Trading AI v0.1 — trading research laboratory "
        "(backtesting only; no real money, no broker connection).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="path to an OHLCV CSV file")
    source.add_argument("--synthetic", action="store_true",
                        help="generate deterministic synthetic data instead")
    p.add_argument("--symbol", default=None, help="symbol label (default: inferred)")
    p.add_argument("--bars", type=int, default=500, help="synthetic bar count")
    p.add_argument("--seed", type=int, default=42, help="synthetic data seed")
    p.add_argument("--start", default="2024-01-01", help="synthetic start date")
    p.add_argument("--config", default=None, help="optional JSON config overrides")
    p.add_argument("--output", default="output", help="directory for the report files")
    p.add_argument("--quiet", action="store_true", help="suppress the printed report")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    cfg = AppConfig.from_json(args.config) if args.config else AppConfig()
    symbol = args.symbol or ("SYNTH" if args.synthetic else Path(args.csv).stem.upper())

    # Layer 1+2 — data
    if args.synthetic:
        df = generate_synthetic_ohlcv(
            bars=args.bars, seed=args.seed, start=args.start,
        )
        print(f"[data] generated {len(df)} synthetic daily bars (seed={args.seed})")
    else:
        df = load_csv(args.csv, symbol=symbol)

    # Layer 4 — regime labels for per-regime analytics
    regimes = detect_regime(df, cfg.regime)
    print(f"[regime] bar shares: {regime_summary(regimes)}")

    # Layers 5-8 — strategy, signal loop, risk engine, backtest
    strategy = EmaRsiVwapStrategy(cfg.strategy)
    engine = BacktestEngine(strategy, cfg.risk, cfg.backtest, symbol=symbol)
    result = engine.run(df, regimes=regimes)

    # Layer 19/21 — metrics + report
    metrics = compute_metrics(result)
    paths = save_outputs(result, symbol, args.output, metrics)
    if not args.quiet:
        print()
        print(format_report(result, symbol, metrics))
    print("\n[output] wrote:")
    for kind, path in paths.items():
        print(f"  {kind:>7}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

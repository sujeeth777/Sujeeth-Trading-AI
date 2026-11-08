"""Tests for the analytics/report layer and the CLI wiring."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sujeeth_trading_ai import __main__ as cli
from sujeeth_trading_ai.analytics.report import format_report, save_outputs
from sujeeth_trading_ai.backtest import BacktestEngine, compute_metrics
from sujeeth_trading_ai.config import AppConfig
from sujeeth_trading_ai.data import generate_synthetic_ohlcv
from sujeeth_trading_ai.regime import detect_regime
from sujeeth_trading_ai.strategy import EmaRsiVwapStrategy


def _result(symbol="T"):
    df = generate_synthetic_ohlcv(bars=400, seed=42)
    cfg = AppConfig()
    engine = BacktestEngine(EmaRsiVwapStrategy(cfg.strategy), cfg.risk, cfg.backtest, symbol=symbol)
    return engine.run(df, regimes=detect_regime(df, cfg.regime)), df


class TestReport:
    def test_report_contains_key_sections(self, capsys):
        result, _ = _result()
        text = format_report(result, "T")
        for needle in (
            "BACKTEST REPORT",
            "Initial Capital",
            "Win Rate",
            "Profit Factor",
            "Max Drawdown",
            "PERFORMANCE BY MARKET REGIME",
            "TRADE LOG",
        ):
            assert needle in text

    def test_save_outputs_writes_four_files(self, tmp_path):
        result, _ = _result("OUT")
        paths = save_outputs(result, "OUT", tmp_path)
        assert set(paths) == {"report", "trades", "metrics", "equity"}
        for p in paths.values():
            assert p.exists() and p.stat().st_size > 0
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        assert metrics["symbol"] == "OUT"
        assert metrics["total_trades"] == len(result.trades)


class TestCli:
    def test_synthetic_end_to_end(self, tmp_path, capsys):
        rc = cli.main([
            "--synthetic", "--bars", "300", "--seed", "42",
            "--output", str(tmp_path / "out"),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "BACKTEST REPORT" in out
        assert (tmp_path / "out" / "SYNTH_report.txt").exists()

    def test_csv_end_to_end(self, tmp_path, capsys):
        df = generate_synthetic_ohlcv(bars=300, seed=7)
        csv = tmp_path / "mydata.csv"
        out_df = df.copy()
        out_df.insert(0, "timestamp", out_df.index.strftime("%Y-%m-%d"))
        out_df.to_csv(csv, index=False)

        rc = cli.main(["--csv", str(csv), "--symbol", "MYDATA",
                       "--output", str(tmp_path / "out"), "--quiet"])
        assert rc == 0
        assert (tmp_path / "out" / "MYDATA_trades.csv").exists()

    def test_config_override_is_applied(self, tmp_path):
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps(
            {"risk": {"initial_capital": 50_000.0, "max_risk_per_trade_pct": 0.01}}
        ), encoding="utf-8")
        cfg = cli.AppConfig.from_json(cfg_file)
        assert cfg.risk.initial_capital == 50_000.0
        assert cfg.risk.max_risk_per_trade_pct == 0.01
        # Untouched defaults survive:
        assert cfg.risk.max_daily_loss_pct == AppConfig().risk.max_daily_loss_pct

    def test_unknown_config_key_raises(self, tmp_path):
        cfg_file = tmp_path / "bad.json"
        cfg_file.write_text(json.dumps({"risk": {"nope": 1}}), encoding="utf-8")
        with pytest.raises(ValueError, match="nope"):
            cli.AppConfig.from_json(cfg_file)

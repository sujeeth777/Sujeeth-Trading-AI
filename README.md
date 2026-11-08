# Sujeeth Trading AI

An AI-assisted algorithmic trading research laboratory for Indian markets
(NSE/BSE), designed to eventually connect to a Zerodha account — but starting
completely in backtesting mode, with virtual money and zero broker access.

> **Core principle:** the AI proposes, it never controls money.
> The strategy, risk engine and execution safeguards decide what is actually
> allowed. The risk engine has veto power — always.

## What this is (v0.1)

A working **trading research laboratory**: give it historical market data and
it returns strategy performance, trade-by-trade history, an equity curve and
a risk-audited backtest report.

```
MARKET DATA ──► DATA PROCESSING ──► TECHNICAL ANALYSIS ──► REGIME DETECTION
                                                                     │
BACKTEST REPORT ◄── ANALYTICS ◄── BACKTEST ENGINE ◄── RISK ENGINE ◄──┘
                                                        ▲
                                                 STRATEGY ENGINE
                                            (BUY_CANDIDATE, never "BUY")
```

## Quick start

```bash
pip install -r requirements.txt

# Run a backtest on deterministic synthetic data (no downloads needed)
python -m sujeeth_trading_ai --synthetic --bars 500 --seed 42

# Run on your own OHLCV CSV (columns: timestamp,open,high,low,close,volume)
python -m sujeeth_trading_ai --csv data/sample_reliance_daily.csv --symbol RELIANCE

# Custom configuration
python -m sujeeth_trading_ai --synthetic --config config.example.json
```

Every run writes `output/<SYMBOL>_report.txt`, `_trades.csv`, `_metrics.json`
and `_equity_curve.csv`.

## The layers (design doc → code)

| Layer | Module | What it does |
|---|---|---|
| 1–2. Data engine | `sujeeth_trading_ai/data/` | CSV loading, timestamp normalisation, duplicate removal, OHLCV validation. Bad rows are dropped, never repaired. |
| 3. Technical analysis | `sujeeth_trading_ai/indicators/` | EMA, RSI, MACD, VWAP, ATR — all causal (value at bar *i* uses bars 0..*i* only). |
| 4. Regime detection | `sujeeth_trading_ai/regime/` | Classifies every bar: `TRENDING`, `SIDEWAYS`, `HIGH_VOLATILITY`. |
| 5. Strategy engine | `sujeeth_trading_ai/strategy/` | Explicit rules → `BUY_CANDIDATE`. A candidate, not an order. |
| 8. Risk engine | `sujeeth_trading_ai/risk/` | Position sizing (0.5%/trade), reward:risk ≥ 2 gate, 1.5% daily loss halt, kill switch. **Veto power over everything.** |
| 18. Backtester | `sujeeth_trading_ai/backtest/` | Strict time order, next-bar-open fills, stop-first on ambiguous bars, no look-ahead. |
| 19/21. Analytics | `sujeeth_trading_ai/analytics/` | Win rate, profit factor, max drawdown, per-regime breakdown, reports. |

### The v0.1 strategy (deliberately transparent)

```
BUY CANDIDATE IF:
    EMA9 > EMA21            (short-term momentum up)
AND RSI > 50                (momentum confirms)
AND close > VWAP            (above the volume benchmark)
AND volume > 20-bar average (participation confirms)

EXIT WHEN:
    EMA9 crosses below EMA21
Stops/targets: stop = entry − 1.5×ATR, target = entry + 3×ATR (2:1 by design)
```

Rules first, ML later: a transparent baseline lets us honestly ask
*"does adding AI actually improve performance?"* after 500 recorded trades.

### Risk engine worked example (unit-tested)

```
Capital ₹10,000 · max risk/trade 0.5% = ₹50
Entry ₹100, stop ₹95 → risk/share ₹5
Quantity = ⌊50 / 5⌋ = 10 shares        ← tests/test_risk.py asserts exactly this
```

Lose ₹150 (1.5%) in a session → trading halted until the next session.
No "let's make the money back". Ever.

## Honest backtest guarantees

- **No look-ahead bias** — decisions use bar *i* information; fills happen at
  bar *i+1*'s open. Enforced by test.
- **Worst-case intrabar assumption** — if a bar touches both stop and target,
  the stop is taken.
- **Equity conservation** — `final equity ≡ initial capital + Σ trade P&L`,
  enforced by test.
- **Determinism** — same data + seed ⇒ identical trades, enforced by test.

Run the suite: `python -m pytest tests/` (66 tests).

## Configuration

Defaults live in `sujeeth_trading_ai/config.py` and can be overridden with a
JSON file (`--config`):

```json
{
  "risk": {
    "initial_capital": 10000.0,
    "max_risk_per_trade_pct": 0.005,
    "max_daily_loss_pct": 0.015,
    "min_reward_risk": 2.0
  },
  "strategy": { "ema_fast": 9, "ema_slow": 21, "rsi_bull_threshold": 50.0 }
}
```

See `config.example.json` for all keys.

## Roadmap

| Version | Scope | Status |
|---|---|---|
| 0.1 | Historical data → indicators → strategy → backtest → report | ✅ **this repo** |
| 0.2 | Live market data → strategy → paper trading | planned |
| 0.3 | + risk dashboard, trade database | planned |
| 0.4 | + AI analyst, regime-aware gating | planned |
| 0.5 | Multi-agent (technical/sentiment/risk), walk-forward validation | planned |
| 1.0 | Full stack; only after extensive testing: Zerodha Kite Connect → NSE | planned |

**Zerodha integration comes last, not first.** No code in this repository can
place a real order.

## Sample data

`data/sample_reliance_daily.csv` is **synthetic** placeholder data in the
expected CSV format (seeded, reproducible) so the pipeline runs end-to-end
out of the box. Swap in real historical OHLCV exports whenever ready.

## Disclaimer

Educational software for trading research. Nothing here is financial advice,
and no component of this system executes real trades.

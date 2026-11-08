"""Tests for Layer 8 — the risk engine.

Section 12's worked example is a unit test:
    Capital ₹10,000, max risk/trade 0.5% = ₹50,
    entry ₹100, stop ₹95 -> risk/share ₹5 -> quantity = 10 shares.
"""

from __future__ import annotations

import pytest

from sujeeth_trading_ai.config import RiskConfig
from sujeeth_trading_ai.risk import PositionProposal, RiskEngine


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine(RiskConfig())


def approve(engine, entry=100.0, stop=95.0, target=110.0, equity=10_000.0,
            cash=10_000.0, session_start=10_000.0, current=10_000.0):
    return engine.evaluate(
        PositionProposal("LONG", entry, stop, target),
        equity=equity, cash=cash,
        session_start_equity=session_start, current_equity=current,
    )


class TestPositionSizing:
    def test_document_example_ten_shares(self, engine):
        d = approve(engine)
        assert d.approved
        assert d.quantity == 10

    def test_quantity_scales_with_equity(self, engine):
        d = approve(engine, equity=100_000.0, cash=100_000.0)
        assert d.quantity == 100

    def test_risk_share_too_big_for_budget(self, engine):
        # risk/share ₹200 vs budget ₹50 -> fractional 0.25 -> reject
        d = approve(engine, entry=1000.0, stop=800.0, target=1500.0)
        assert not d.approved
        assert "risk budget" in d.reasons[0]

    def test_quantity_capped_by_cash(self, engine):
        d = approve(engine, cash=600.0)  # 600/100 = 6 shares affordable < 10
        assert d.approved
        assert d.quantity == 6
        assert any("capped by available cash" in r for r in d.reasons)

    def test_no_cash_rejects(self, engine):
        d = approve(engine, cash=0.0)
        assert not d.approved
        assert "insufficient cash" in d.reasons[0]


class TestRewardRiskGate:
    def test_exactly_2_to_1_is_allowed(self, engine):
        assert approve(engine).approved

    def test_below_2_to_1_rejected(self, engine):
        d = approve(engine, target=109.0)  # 9/5 = 1.8
        assert not d.approved
        assert "reward:risk" in d.reasons[0]

    def test_target_below_entry_rejected(self, engine):
        assert not approve(engine, target=99.0).approved


class TestStopSanity:
    def test_stop_too_close_rejected(self, engine):
        d = approve(engine, stop=99.99)
        assert not d.approved
        assert "stop too close" in d.reasons[0]

    def test_non_finite_levels_rejected(self, engine):
        d = approve(engine, entry=float("nan"))
        assert not d.approved

    def test_short_side_disabled_in_v01(self, engine):
        d = engine.evaluate(
            PositionProposal("SHORT", 100.0, 105.0, 90.0),
            equity=10_000.0, cash=10_000.0,
            session_start_equity=10_000.0, current_equity=10_000.0,
        )
        assert not d.approved
        assert "not enabled" in d.reasons[0]


class TestDailyLossLimit:
    def test_under_limit_trades(self, engine):
        d = approve(engine, session_start=10_000.0, current=9_900.0)  # -1.0%
        assert d.approved

    def test_at_limit_rejects(self, engine):
        d = approve(engine, session_start=10_000.0, current=9_850.0)  # -1.5%
        assert not d.approved
        assert "daily loss limit" in d.reasons[0]

    def test_limit_amount_is_150_rupees(self, engine):
        assert engine.daily_loss_limit_rupees(10_000.0) == pytest.approx(150.0)

    def test_zero_start_equity_halts(self, engine):
        assert engine.is_halted_for_day(0.0, 0.0)


class TestKillSwitch:
    def test_engaged_blocks_everything(self, engine):
        engine.engage_kill_switch()
        d = approve(engine)
        assert not d.approved
        assert d.reasons == ["kill switch engaged"]

    def test_release_restores_trading(self, engine):
        engine.engage_kill_switch()
        engine.release_kill_switch()
        assert approve(engine).approved

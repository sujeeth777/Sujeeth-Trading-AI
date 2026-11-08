"""Layer 8 — Risk Management Engine (Sections 11-15).

This engine is more important than the AI. It has veto power over every
candidate signal (Section 10): the risk engine wins. Always.

Responsibilities
----------------
1. Position sizing (Section 12):
       qty = floor(max_risk_rupees / risk_per_share)
   with risk_per_share = |entry - stop| and max_risk = 0.5% of equity.

2. Reward:risk gate (Section 13):
   (target - entry) / (entry - stop) must be >= 2.0 for longs.

3. Daily loss limit (Section 14):
   Lose 1.5% of session-start equity -> no new trades until the next
   session. No "let's make the money back".

4. Kill switch (Section 15):
   One flag, honoured everywhere, zero exceptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sujeeth_trading_ai.config import RiskConfig


@dataclass(slots=True)
class PositionProposal:
    """What the signal engine asks permission for."""

    side: str                 # "LONG" (SHORT reserved for later versions)
    entry_price: float
    stop_price: float
    target_price: float


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    quantity: int = 0
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - debug convenience
        state = "APPROVE" if self.approved else "REJECT"
        return f"{state} qty={self.quantity} [{', '.join(self.reasons) or 'ok'}]"


@dataclass(slots=True)
class RiskEngine:
    """Stateful guard. The backtest/paper loop must consult it per trade."""

    config: RiskConfig
    kill_switch_engaged: bool = False

    # --- kill switch (Section 15) -----------------------------------------
    def engage_kill_switch(self) -> None:
        """Emergency stop: no new trades, anywhere, until manually released."""
        self.kill_switch_engaged = True

    def release_kill_switch(self) -> None:
        self.kill_switch_engaged = False

    # --- daily loss limit (Section 14) ------------------------------------
    def daily_loss_limit_rupees(self, session_start_equity: float) -> float:
        return self.config.max_daily_loss_pct * session_start_equity

    def is_halted_for_day(self, session_start_equity: float, current_equity: float) -> bool:
        """True once the day's drawdown has reached the loss limit."""
        if session_start_equity <= 0:
            return True  # capital gone -> halt
        loss = session_start_equity - current_equity
        return loss >= self.daily_loss_limit_rupees(session_start_equity)

    # --- the veto gate (Section 10) ----------------------------------------
    def evaluate(
        self,
        proposal: PositionProposal,
        equity: float,
        cash: float,
        session_start_equity: float,
        current_equity: float,
    ) -> RiskDecision:
        """Approve or veto a candidate, and size it if approved."""
        reasons: list[str] = []

        if self.kill_switch_engaged:
            return RiskDecision(False, 0, ["kill switch engaged"])

        if self.is_halted_for_day(session_start_equity, current_equity):
            return RiskDecision(False, 0, ["daily loss limit reached"])

        if proposal.side != "LONG":
            return RiskDecision(False, 0, [f"side {proposal.side} not enabled in v0.1"])

        entry, stop, target = proposal.entry_price, proposal.stop_price, proposal.target_price
        if not all(math.isfinite(p) and p > 0 for p in (entry, stop, target)):
            return RiskDecision(False, 0, ["non-finite or non-positive price levels"])

        risk_per_share = entry - stop
        if risk_per_share < self.config.min_stop_distance:
            return RiskDecision(False, 0, [f"stop too close (< ₹{self.config.min_stop_distance:.2f})"])

        reward_per_share = target - entry
        if reward_per_share <= 0:
            return RiskDecision(False, 0, ["target not above entry"])

        reward_risk = reward_per_share / risk_per_share
        # Epsilon guards exact-ratio brackets: entry-2k*ATR over k*ATR is
        # mathematically 2.0 but can evaluate to 1.9999999999999998 in floats.
        if reward_risk < self.config.min_reward_risk - 1e-9:
            reasons.append(f"reward:risk {reward_risk:.2f} < {self.config.min_reward_risk:.2f}")
            return RiskDecision(False, 0, reasons)

        # --- position sizing (Section 12) ---
        max_risk_rupees = equity * self.config.max_risk_per_trade_pct
        quantity = int(math.floor(max_risk_rupees / risk_per_share))
        if quantity < 1:
            return RiskDecision(False, 0, [
                f"risk budget ₹{max_risk_rupees:.2f} < risk/share ₹{risk_per_share:.2f}"
            ])

        # Never spend more than available cash.
        affordable = int(math.floor(cash / entry))
        if affordable < 1:
            return RiskDecision(False, 0, [f"insufficient cash (₹{cash:.2f})"])
        if affordable < quantity:
            quantity = affordable
            reasons.append("quantity capped by available cash")

        reasons.append(
            f"risk ₹{risk_per_share * quantity:.2f} <= ₹{max_risk_rupees:.2f} budget; "
            f"R:R {reward_risk:.2f}"
        )
        return RiskDecision(True, quantity, reasons)

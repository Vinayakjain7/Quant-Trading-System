"""Position sizing & portfolio risk controls.

Sizing uses the more conservative of two rules:
  1. Risk-based: never lose more than `risk_per_trade` of capital if the
     ATR-based stop is hit  ->  qty = (capital * risk_per_trade) / stop_distance
  2. Allocation cap: never put more than `max_allocation` of capital in one name.

Plus hard limits: max concurrent positions and a portfolio drawdown halt.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SizingResult:
    quantity: int
    stop_price: float
    risk_amount: float
    notional: float
    reason: str


def position_size(
    capital: float,
    price: float,
    atr: float,
    risk_cfg: dict,
) -> SizingResult:
    """Return a SizingResult. quantity == 0 means 'do not take this trade'."""
    if price <= 0:
        return SizingResult(0, 0.0, 0.0, 0.0, "invalid price")

    stop_mult = risk_cfg["atr_stop_mult"]
    risk_per_trade = risk_cfg["risk_per_trade"]
    max_allocation = risk_cfg["max_allocation"]

    # Fall back to a 5% stop if ATR is unavailable.
    stop_distance = stop_mult * atr if atr and atr > 0 else 0.05 * price
    stop_price = max(price - stop_distance, 0.0)

    risk_amount = capital * risk_per_trade
    qty_risk = int(risk_amount // stop_distance) if stop_distance > 0 else 0

    max_notional = capital * max_allocation
    qty_alloc = int(max_notional // price)

    qty = max(min(qty_risk, qty_alloc), 0)
    reason = "risk-limited" if qty_risk <= qty_alloc else "allocation-capped"

    return SizingResult(
        quantity=qty,
        stop_price=round(stop_price, 2),
        risk_amount=round(qty * stop_distance, 2),
        notional=round(qty * price, 2),
        reason=reason if qty > 0 else "size rounds to zero",
    )


def can_open_new(open_positions: int, risk_cfg: dict) -> bool:
    return open_positions < int(risk_cfg["max_positions"])


def drawdown_halt(realized_pnl: float, capital: float, risk_cfg: dict) -> bool:
    """True if cumulative drawdown breaches the limit -> stop new trades."""
    if capital <= 0:
        return True
    return (realized_pnl / capital) < float(risk_cfg["max_drawdown"])

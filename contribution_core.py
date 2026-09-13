"""Pure, state-free rules for releasing an annual Roth contribution budget."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math


POLICY_REVISION = "roth-50-10x5-sma50-dd10-dd20-v1"
INITIAL_FRACTION = 0.50
TRANCHE_FRACTION = 0.10
SCHEDULE_MONTHS = (3, 5, 7, 9, 11)
LOOKBACK_SESSIONS = 63
DRAWDOWN_FIRST = -0.10
DRAWDOWN_ALL = -0.20


@dataclass(frozen=True)
class ReleaseDecision:
    calendar_fraction: float
    target_fraction: float
    bull_pullback: bool
    drawdown: float
    use_pullback: bool
    use_drawdown_10: bool
    use_drawdown_20: bool
    reasons: tuple[str, ...]


def calendar_fraction(as_of: date) -> float:
    """Cumulative fraction due by date, with full deployment by November."""
    if not isinstance(as_of, date):
        raise ValueError("as_of must be a date")
    elapsed = sum(as_of.month >= month for month in SCHEDULE_MONTHS)
    return min(1.0, INITIAL_FRACTION + TRANCHE_FRACTION * elapsed)


def evaluate_release(
    *,
    as_of: date,
    qqq_close: float,
    qqq_sma_50: float,
    qqq_sma_200: float,
    qqq_high_63: float,
    pullback_used: bool,
    drawdown_10_used: bool,
    drawdown_20_used: bool,
) -> ReleaseDecision:
    """Return the cumulative budget fraction that should have been released.

    Calendar milestones are floors. A bullish-regime pullback and a 10% drawdown
    each pull one future tranche forward once. A 20% drawdown releases everything.
    """
    numbers = (qqq_close, qqq_sma_50, qqq_sma_200, qqq_high_63)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in numbers):
        raise ValueError("QQQ contribution inputs must be positive and finite")
    flags = (pullback_used, drawdown_10_used, drawdown_20_used)
    if any(not isinstance(value, bool) for value in flags):
        raise ValueError("Contribution trigger flags must be boolean")

    drawdown = qqq_close / qqq_high_63 - 1.0
    bull_pullback = qqq_sma_200 < qqq_close < qqq_sma_50
    use_pullback = bull_pullback and not pullback_used
    use_drawdown_10 = drawdown <= DRAWDOWN_FIRST + 1e-12 and not drawdown_10_used
    use_drawdown_20 = drawdown <= DRAWDOWN_ALL + 1e-12 and not drawdown_20_used

    calendar = calendar_fraction(as_of)
    used_single_tranches = (
        int(pullback_used or use_pullback)
        + int(drawdown_10_used or use_drawdown_10)
    )
    target = min(1.0, calendar + TRANCHE_FRACTION * used_single_tranches)
    reasons: list[str] = []
    if use_pullback:
        reasons.append("QQQ_PULLBACK_ABOVE_SMA200")
    if use_drawdown_10:
        reasons.append("QQQ_DRAWDOWN_10")
    if drawdown_20_used or use_drawdown_20:
        target = 1.0
    if use_drawdown_20:
        reasons.append("QQQ_DRAWDOWN_20_DEPLOY_REMAINDER")
    if not reasons:
        reasons.append("CALENDAR_MILESTONE")

    return ReleaseDecision(
        calendar_fraction=calendar,
        target_fraction=target,
        bull_pullback=bull_pullback,
        drawdown=drawdown,
        use_pullback=use_pullback,
        use_drawdown_10=use_drawdown_10,
        use_drawdown_20=use_drawdown_20,
        reasons=tuple(reasons),
    )

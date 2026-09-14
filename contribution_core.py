"""Pure rules for a single annual contribution release."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math


POLICY_REVISION = "static-annual-contribution-v1"


@dataclass(frozen=True)
class ReleaseDecision:
    target_fraction: float
    reasons: tuple[str, ...]


def evaluate_release(*, as_of: date, already_released: float, budget: float) -> ReleaseDecision:
    """Release the remaining configured budget once per calendar year.

    No market prices, indicators, drawdowns, or regime signals are consulted.
    """
    if not isinstance(as_of, date):
        raise ValueError("as_of must be a date")
    values = (already_released, budget)
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("Contribution values must be finite and nonnegative")
    if budget <= 0 or already_released > budget + 0.005:
        raise ValueError("Contribution budget is invalid")
    fraction = min(1.0, already_released / budget) if budget else 0.0
    return ReleaseDecision(fraction, ("ANNUAL_CONTRIBUTION",))

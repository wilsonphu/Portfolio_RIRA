"""Pure rules for the static annual Roth IRA allocation."""

from __future__ import annotations

import math


TQQQ = "TQQQ"
DBMF = "DBMF"
UGL = "UGL"
ZROZ = "ZROZ"
BTAL = "BTAL"
CASH = "CASH"

DECISION_SEMANTIC_REVISION = "static-annual-tqqq35-dbmf25-ugl20-zroz15-btal5-v1"
MODEL_HISTORY_START = "2019-01-01"

STATIC_WEIGHTS = {
    TQQQ: 0.40,
    DBMF: 0.32,
    UGL: 0.15,
    ZROZ: 0.05,
    BTAL: 0.08,
}

ADVERTISED_DAILY_MULTIPLIERS = {
    TQQQ: 3.0,
    DBMF: 1.0,
    UGL: 2.0,
    ZROZ: 1.0,
    BTAL: 1.0,
    CASH: 0.0,
}


def target_weights() -> dict[str, float]:
    """Return a fresh copy of the immutable 35/25/20/15/5 target."""
    return dict(STATIC_WEIGHTS)


def advertised_daily_exposure(weights: dict[str, float] | None = None) -> float:
    """Return the advertised gross daily exposure of the supplied allocation."""
    selected = target_weights() if weights is None else dict(weights)
    unknown = set(selected) - set(ADVERTISED_DAILY_MULTIPLIERS)
    if unknown:
        raise ValueError(f"Unknown exposure components: {sorted(unknown)}")
    exposure = sum(
        float(weight) * ADVERTISED_DAILY_MULTIPLIERS[ticker]
        for ticker, weight in selected.items()
    )
    if not math.isfinite(exposure):
        raise ValueError("Advertised exposure is invalid")
    return float(exposure)


def validate_target() -> None:
    if set(STATIC_WEIGHTS) != {TQQQ, DBMF, UGL, ZROZ, BTAL}:
        raise RuntimeError("Static target universe is invalid")
    if not math.isclose(sum(STATIC_WEIGHTS.values()), 1.0, abs_tol=1e-12):
        raise RuntimeError("Static target must sum to one")
    if any(
        not isinstance(weight, (int, float))
        or isinstance(weight, bool)
        or not math.isfinite(weight)
        or weight < 0
        for weight in STATIC_WEIGHTS.values()
    ):
        raise RuntimeError("Static target contains invalid weights")
    if not math.isclose(advertised_daily_exposure(), 1.90, abs_tol=1e-12):
        raise RuntimeError("Static target exposure is invalid")

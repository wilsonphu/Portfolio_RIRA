"""Pure signal and allocation rules for the Roth equity router.

This module has no filesystem, network, email, environment, or production
state side effects. Production and tests import the same state transition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


QQQ = "QQQ"
TQQQ = "TQQQ"
UPRO = "UPRO"
DBMF = "DBMF"
ZROZ = "ZROZ"
UGL = "UGL"
CASH = "CASH"

DECISION_SEMANTIC_REVISION = "tqqq-upro-sma200-router-v1"
MODEL_HISTORY_START = "2010-03-11"
SMA_WINDOW = 200
BULLISH_ENTRY_CLOSES = 2
EQUITY_WEIGHT = 0.40
DIVERSIFIER_WEIGHT = 0.20
MAX_ADVERTISED_DAILY_EXPOSURE = 2.00


@dataclass(frozen=True)
class EquityRouterState:
    """State required for distinct-close TQQQ entry confirmation."""

    tqqq_active: bool = False
    bullish_streak: int = 0
    last_processed_signal_date: str = ""
    switch_date: str = ""


@dataclass(frozen=True)
class EquityRouterTransition:
    state: EquityRouterState
    reason: str
    structural_change: bool


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def target_weights(tqqq_active: bool) -> dict[str, float]:
    """Return the exact 40/20/20/20 strategic allocation."""
    active = _strict_bool(tqqq_active, "tqqq_active")
    return {
        TQQQ if active else UPRO: EQUITY_WEIGHT,
        DBMF: DIVERSIFIER_WEIGHT,
        ZROZ: DIVERSIFIER_WEIGHT,
        UGL: DIVERSIFIER_WEIGHT,
    }


def advertised_daily_exposure(tqqq_active: bool) -> float:
    """Both router states carry 40% in a 3x daily equity product."""
    _strict_bool(tqqq_active, "tqqq_active")
    return MAX_ADVERTISED_DAILY_EXPOSURE


def advance_equity_router(
    state: EquityRouterState,
    *,
    signal_date: pd.Timestamp,
    trend_positive: bool,
    entry_closes: int = BULLISH_ENTRY_CLOSES,
) -> EquityRouterTransition:
    """Process one completed close without same-date double counting.

    A bearish close selects UPRO immediately. TQQQ requires two distinct
    bullish closes, including the current close. This state machine changes
    index exposure, not the portfolio's nominal 3x equity multiplier.
    """
    if not isinstance(state, EquityRouterState):
        raise ValueError("state must be an EquityRouterState")
    if not isinstance(entry_closes, int) or isinstance(entry_closes, bool) or entry_closes < 1:
        raise ValueError("entry_closes must be a positive integer")
    trend = _strict_bool(trend_positive, "trend_positive")
    session = pd.Timestamp(signal_date).normalize()
    session_text = session.date().isoformat()
    if state.last_processed_signal_date:
        previous = pd.Timestamp(state.last_processed_signal_date).normalize()
        if previous > session:
            raise ValueError("Router state is ahead of the signal date")
        if previous == session:
            return EquityRouterTransition(state, "SAME_DATE", False)

    if not trend:
        changed = state.tqqq_active
        next_state = replace(
            state,
            tqqq_active=False,
            bullish_streak=0,
            last_processed_signal_date=session_text,
            switch_date=session_text if changed else state.switch_date,
        )
        return EquityRouterTransition(
            next_state,
            "TREND_SWITCH_TO_UPRO" if changed else "UPRO_HOLD",
            changed,
        )

    streak = state.bullish_streak + 1
    if state.tqqq_active:
        next_state = replace(
            state,
            bullish_streak=streak,
            last_processed_signal_date=session_text,
        )
        return EquityRouterTransition(next_state, "TQQQ_HOLD", False)

    if streak >= entry_closes:
        next_state = replace(
            state,
            tqqq_active=True,
            bullish_streak=streak,
            last_processed_signal_date=session_text,
            switch_date=session_text,
        )
        return EquityRouterTransition(next_state, "TREND_SWITCH_TO_TQQQ", True)

    next_state = replace(
        state,
        bullish_streak=streak,
        last_processed_signal_date=session_text,
    )
    return EquityRouterTransition(next_state, "TQQQ_ENTRY_PENDING", False)

"""Pure signal and allocation rules for the Roth crisis hedge.

This module has no filesystem, network, email, environment, or production
state side effects. Production and tests import the same state transition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


QQQ = "QQQ"
SPY = "SPY"
TQQQ = "TQQQ"
BTAL = "BTAL"
DBMF = "DBMF"
ZROZ = "ZROZ"
UGL = "UGL"
CASH = "CASH"

DECISION_SEMANTIC_REVISION = "tqqq40-btal10-extreme-bear-v1"
MODEL_HISTORY_START = "2010-03-11"
SMA_WINDOW = 200
SHORT_SMA_WINDOW = 50
EXTREME_BEAR_RATIO = 0.92
RECOVERY_RATIO = 0.98
CONFIRMATION_CLOSES = 2
TQQQ_WEIGHT = 0.40
HEDGED_TQQQ_WEIGHT = 0.30
BTAL_WEIGHT = 0.10
DIVERSIFIER_WEIGHT = 0.20
MAX_ADVERTISED_DAILY_EXPOSURE = 2.00
HEDGED_ADVERTISED_DAILY_EXPOSURE = 1.80


@dataclass(frozen=True)
class CrisisHedgeState:
    """State required for distinct-close BTAL entry and exit confirmation."""

    btal_active: bool = False
    entry_streak: int = 0
    exit_streak: int = 0
    last_processed_signal_date: str = ""
    switch_date: str = ""


@dataclass(frozen=True)
class CrisisHedgeTransition:
    state: CrisisHedgeState
    reason: str
    structural_change: bool


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def target_weights(btal_active: bool) -> dict[str, float]:
    """Return the exact normal or extreme-bear strategic allocation."""
    hedged = _strict_bool(btal_active, "btal_active")
    weights = {
        TQQQ: HEDGED_TQQQ_WEIGHT if hedged else TQQQ_WEIGHT,
        DBMF: DIVERSIFIER_WEIGHT,
        ZROZ: DIVERSIFIER_WEIGHT,
        UGL: DIVERSIFIER_WEIGHT,
    }
    if hedged:
        weights[BTAL] = BTAL_WEIGHT
    return weights


def advertised_daily_exposure(btal_active: bool) -> float:
    """Return advertised gross daily exposure for the selected hedge state."""
    hedged = _strict_bool(btal_active, "btal_active")
    return HEDGED_ADVERTISED_DAILY_EXPOSURE if hedged else MAX_ADVERTISED_DAILY_EXPOSURE


def crisis_conditions(
    *,
    qqq_close: float,
    qqq_sma_200: float,
    spy_close: float,
    spy_sma_200: float,
    spy_sma_50: float,
) -> tuple[bool, bool]:
    """Return the extreme-entry and recovery-exit conditions."""
    values = (qqq_close, qqq_sma_200, spy_close, spy_sma_200, spy_sma_50)
    if not all(
        not isinstance(value, bool)
        and isinstance(value, (int, float, np.number))
        and np.isfinite(value)
        and value > 0
        for value in values
    ):
        raise ValueError("Crisis-hedge inputs must be positive and finite")
    extreme = qqq_close <= EXTREME_BEAR_RATIO * qqq_sma_200 and spy_close < spy_sma_200
    recovery = qqq_close >= RECOVERY_RATIO * qqq_sma_200 and spy_close > spy_sma_50
    return bool(extreme), bool(recovery)


def advance_crisis_hedge(
    state: CrisisHedgeState,
    *,
    signal_date: pd.Timestamp,
    extreme_bearish: bool,
    recovery_confirmed: bool,
    confirmation_closes: int = CONFIRMATION_CLOSES,
) -> CrisisHedgeTransition:
    """Process one completed close without same-date double counting.

    BTAL enters only after two distinct extreme-bear closes and exits only
    after two distinct recovery closes. The conditions use separate asymmetric
    thresholds, creating an explicit deadband.
    """
    if not isinstance(state, CrisisHedgeState):
        raise ValueError("state must be a CrisisHedgeState")
    if (
        not isinstance(confirmation_closes, int)
        or isinstance(confirmation_closes, bool)
        or confirmation_closes < 1
    ):
        raise ValueError("confirmation_closes must be a positive integer")
    extreme = _strict_bool(extreme_bearish, "extreme_bearish")
    recovery = _strict_bool(recovery_confirmed, "recovery_confirmed")
    session = pd.Timestamp(signal_date).normalize()
    session_text = session.date().isoformat()
    if state.last_processed_signal_date:
        previous = pd.Timestamp(state.last_processed_signal_date).normalize()
        if previous > session:
            raise ValueError("Crisis-hedge state is ahead of the signal date")
        if previous == session:
            return CrisisHedgeTransition(state, "SAME_DATE", False)

    if state.btal_active:
        exit_streak = state.exit_streak + 1 if recovery else 0
        if exit_streak >= confirmation_closes:
            next_state = replace(
                state,
                btal_active=False,
                entry_streak=0,
                exit_streak=0,
                last_processed_signal_date=session_text,
                switch_date=session_text,
            )
            return CrisisHedgeTransition(next_state, "EXTREME_HEDGE_EXIT", True)
        next_state = replace(
            state,
            entry_streak=0,
            exit_streak=exit_streak,
            last_processed_signal_date=session_text,
        )
        reason = "EXTREME_HEDGE_EXIT_PENDING" if recovery else "EXTREME_HEDGE_HOLD"
        return CrisisHedgeTransition(next_state, reason, False)

    entry_streak = state.entry_streak + 1 if extreme else 0
    if entry_streak >= confirmation_closes:
        next_state = replace(
            state,
            btal_active=True,
            entry_streak=0,
            exit_streak=0,
            last_processed_signal_date=session_text,
            switch_date=session_text,
        )
        return CrisisHedgeTransition(next_state, "EXTREME_HEDGE_ENTRY", True)
    next_state = replace(
        state,
        entry_streak=entry_streak,
        exit_streak=0,
        last_processed_signal_date=session_text,
    )
    reason = "EXTREME_HEDGE_ENTRY_PENDING" if extreme else "TQQQ_HOLD"
    return CrisisHedgeTransition(next_state, reason, False)

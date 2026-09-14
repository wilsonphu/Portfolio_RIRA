#!/usr/bin/env python3
"""Stateful Roth allocator with a capped extreme-bear BTAL hedge.

The normal allocation holds 40% TQQQ. After two completed broad-market closes
meeting the extreme-bear threshold, 10% moves from TQQQ to BTAL. The hedge
exits after two completed recovery closes. DBMF, ZROZ, and UGL remain 20% each.
Confirmed broker shares and cash are always the source of current weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import smtplib
import sys
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd

import alpha_core as core
import contribution_core as contribution
import performance_core as performance


# Configuration and universe
MARKET_INDEX = core.QQQ
CONFIRMATION_INDEX = core.SPY
GROWTH_EQUITY = core.TQQQ
CRISIS_HEDGE = core.BTAL
# UPRO remains valuation/liquidation-only so a version-18 state can migrate
# without silently dropping a real broker holding.
LEGACY_UPRO = "UPRO"
DBMF = core.DBMF
ZROZ = core.ZROZ
LEVERAGED_GOLD = core.UGL
TREASURY_RESERVE = "SGOV"
CASH_ASSET = core.CASH

STRATEGIC_TICKERS = (
    GROWTH_EQUITY,
    CRISIS_HEDGE,
    DBMF,
    ZROZ,
    LEVERAGED_GOLD,
    TREASURY_RESERVE,
)
TACTICAL_TICKERS = (GROWTH_EQUITY, CRISIS_HEDGE)
VALUATION_TICKERS = (*STRATEGIC_TICKERS, LEGACY_UPRO)
ALL_TICKERS = tuple(
    dict.fromkeys((MARKET_INDEX, CONFIRMATION_INDEX, *VALUATION_TICKERS))
)
TRADED_TICKERS = frozenset(VALUATION_TICKERS)
PORTFOLIO_COMPONENTS = TRADED_TICKERS | {CASH_ASSET}

ADVERTISED_DAILY_MULTIPLIERS = {
    GROWTH_EQUITY: 3.0,
    CRISIS_HEDGE: 1.0,
    LEGACY_UPRO: 3.0,
    DBMF: 1.0,
    ZROZ: 1.0,
    LEVERAGED_GOLD: 2.0,
    TREASURY_RESERVE: 0.0,
    CASH_ASSET: 0.0,
}

SMA_WINDOW = core.SMA_WINDOW
SHORT_TREND_WINDOW = core.SHORT_SMA_WINDOW
MOMENTUM_WINDOW = 252
REQUIRED_SIGNAL_ROWS = MOMENTUM_WINDOW + core.CONFIRMATION_CLOSES
MODEL_START_DATE = core.MODEL_HISTORY_START
NOTIFICATION_WEIGHT_TOLERANCE = 0.005
TRANSACTION_COST_SCENARIOS_BPS = (5, 10, 25)
MARKET_CLOSE_BUFFER_MINUTES = 15
NEW_YORK = ZoneInfo("America/New_York")

LIFECYCLE_SPRINT = "SPRINT"
LIFECYCLE_GLIDE_225 = "GLIDE_225"
LIFECYCLE_TWO_X = "TWO_X"
LIFECYCLE_PHI = "PHI"
LIFECYCLE_ONE_THREE = "ONE_THREE"
LIFECYCLE_ONE_X = "ONE_X"
LIFECYCLE_RETIREMENT = "RETIREMENT"
LIFECYCLE_STAGES = (
    LIFECYCLE_SPRINT,
    LIFECYCLE_GLIDE_225,
    LIFECYCLE_TWO_X,
    LIFECYCLE_PHI,
    LIFECYCLE_ONE_THREE,
    LIFECYCLE_ONE_X,
    LIFECYCLE_RETIREMENT,
)
LIFECYCLE_EXPOSURE_CEILINGS = {
    LIFECYCLE_SPRINT: None,
    LIFECYCLE_GLIDE_225: 2.25,
    LIFECYCLE_TWO_X: 2.0,
    LIFECYCLE_PHI: (1.0 + 5.0**0.5) / 2.0,
    LIFECYCLE_ONE_THREE: 1.30,
    LIFECYCLE_ONE_X: 1.0,
    LIFECYCLE_RETIREMENT: 0.75,
}
LIFECYCLE_VALUE_THRESHOLDS_2026 = (
    (250_000.0, LIFECYCLE_GLIDE_225),
    (500_000.0, LIFECYCLE_TWO_X),
    (1_000_000.0, LIFECYCLE_PHI),
    (2_000_000.0, LIFECYCLE_ONE_THREE),
    (5_000_000.0, LIFECYCLE_ONE_X),
)
LIFECYCLE_AGE_THRESHOLDS = (
    (45.0, LIFECYCLE_GLIDE_225),
    (50.0, LIFECYCLE_TWO_X),
    (55.0, LIFECYCLE_PHI),
    (59.5, LIFECYCLE_ONE_THREE),
    (65.0, LIFECYCLE_ONE_X),
    (70.0, LIFECYCLE_RETIREMENT),
)
LIFECYCLE_INFLATION_RATE = 0.025
LIFECYCLE_ANCHOR_DATE = date(2026, 8, 14)
LIFECYCLE_ANCHOR_AGE = 23.0

STRATEGY_REVISION = "tqqq40-btal10-extreme-bear-8-2-annual-v4"
STATE_VERSION = 19
DECISION_AUDIT_SCHEMA_VERSION = 11
APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / "roth_ira_state.json"
LOG_FILE = APP_DIR / "roth_ira.log"
DECISION_AUDIT_FILE = APP_DIR / "roth_ira_decision.json"


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strategy_manifest() -> dict[str, object]:
    return {
        "revision": STRATEGY_REVISION,
        "quantitative_core_revision": core.DECISION_SEMANTIC_REVISION,
        "signal": {
            "primary_index": MARKET_INDEX,
            "confirmation_index": CONFIRMATION_INDEX,
            "sma_sessions": SMA_WINDOW,
            "short_sma_sessions": SHORT_TREND_WINDOW,
            "extreme_entry": {
                "qqq_ratio_to_sma200_lte": core.EXTREME_BEAR_RATIO,
                "spy_below_sma200": True,
                "distinct_closes": core.CONFIRMATION_CLOSES,
            },
            "recovery_exit": {
                "qqq_ratio_to_sma200_gte": core.RECOVERY_RATIO,
                "spy_above_sma50": True,
                "distinct_closes": core.CONFIRMATION_CLOSES,
            },
            "same_date": "idempotent",
            "supporting_health_checks": {
                "qqq_sma_sessions": SHORT_TREND_WINDOW,
                "qqq_momentum_sessions": MOMENTUM_WINDOW,
                "effect": "dashboard_only",
            },
        },
        "allocation": {
            "normal_tqqq_weight": core.TQQQ_WEIGHT,
            "extreme_bear_tqqq_weight": core.HEDGED_TQQQ_WEIGHT,
            "extreme_bear_btal_weight": core.BTAL_WEIGHT,
            "fixed_sleeves": {
                DBMF: core.DIVERSIFIER_WEIGHT,
                ZROZ: core.DIVERSIFIER_WEIGHT,
                LEVERAGED_GOLD: core.DIVERSIFIER_WEIGHT,
            },
            "maximum_advertised_daily_exposure": core.MAX_ADVERTISED_DAILY_EXPOSURE,
        },
        "lifecycle": {
            "stages": list(LIFECYCLE_STAGES),
            "exposure_ceilings": LIFECYCLE_EXPOSURE_CEILINGS,
            "value_thresholds_2026": list(LIFECYCLE_VALUE_THRESHOLDS_2026),
            "age_thresholds": list(LIFECYCLE_AGE_THRESHOLDS),
            "inflation_rate": LIFECYCLE_INFLATION_RATE,
            "anchor_date": LIFECYCLE_ANCHOR_DATE.isoformat(),
            "anchor_age": LIFECYCLE_ANCHOR_AGE,
            "mapping": "proportional_risky_target_to_SGOV",
            "ratchet": "one_way",
        },
        "execution": {
            "signal": "completed_close",
            "fill": "next_session",
            "missed_sessions": "replay_all",
            "ordinary_rebalance": "annual_only",
            "crisis_switch": "move_10pp_between_TQQQ_and_BTAL_without_rebalancing_other_sleeves",
            "holdings_source": "confirmed_shares_and_cash",
            "annual_rebalance": "first_completed_XNYS_signal_each_calendar_year",
        },
    }


def calculate_strategy_fingerprint(manifest: dict[str, object] | None = None) -> str:
    return canonical_sha256(strategy_manifest() if manifest is None else manifest)


STRATEGY_FINGERPRINT = calculate_strategy_fingerprint()
EXPECTED_STRATEGY_FINGERPRINT = "373c3655762642fd99f4dff471840c2133cbc1caa8bf83a4c7c6428e69cecf36"


@dataclass(frozen=True)
class StrategyDecision:
    target_weights: dict[str, float]
    hedge_state: core.CrisisHedgeState
    transition_reason: str
    structural_change: bool
    trend_positive: bool
    qqq_close: float
    qqq_sma_200: float
    qqq_sma_50: float
    qqq_momentum_252: float
    spy_close: float
    spy_sma_200: float
    spy_sma_50: float
    extreme_bearish: bool
    recovery_confirmed: bool
    processed_signal_dates: tuple[str, ...] = ()
    transition_path: tuple[str, ...] = ()
    lifecycle_stage: str = LIFECYCLE_SPRINT
    lifecycle_reason: str = ""
    estimated_investor_age: float = LIFECYCLE_ANCHOR_AGE
    lifecycle_value_stage: str = LIFECYCLE_SPRINT
    lifecycle_age_stage: str = LIFECYCLE_SPRINT
    lifecycle_stage_advanced: bool = False


@dataclass(frozen=True)
class LifecycleSelection:
    stage: str
    reason: str
    estimated_age: float
    value_stage: str
    age_stage: str
    advanced: bool


@dataclass(frozen=True)
class RebalancePlan:
    execution_weights: dict[str, float]
    rebalance_due: bool
    full_transition: bool
    reason: str
    one_way_turnover: float
    individual_orders: int
    annual_rebalance_due: bool = False
    annual_rebalance_year: int = 0


@dataclass(frozen=True)
class NotificationDecision:
    kind: str
    reason: str
    previous_recommendation_date: str = ""
    supersedes_recommendation_date: str = ""

    @property
    def should_send(self) -> bool:
        return self.kind != "NONE"


@dataclass(frozen=True)
class ExecutionDiagnostics:
    current_daily_exposure: float
    strategic_daily_exposure: float
    destination_daily_exposure: float
    gross_security_trade_fraction: float
    estimated_costs: dict[int, float]


@dataclass(frozen=True)
class ContributionPlan:
    enabled: bool = False
    year: int = 0
    budget: float = 0.0
    released_amount: float = 0.0
    due_amount: float = 0.0
    projected_released_amount: float = 0.0
    calendar_fraction: float = 0.0
    target_fraction: float = 0.0
    qqq_drawdown_63: float = 0.0
    bull_pullback: bool = False
    reasons: tuple[str, ...] = ()
    use_pullback: bool = False
    use_drawdown_10: bool = False
    use_drawdown_20: bool = False
    retry: bool = False
    allocation_dollars: dict[str, float] = field(default_factory=dict)
    estimated_units: dict[str, float] = field(default_factory=dict)

    @property
    def notification_due(self) -> bool:
        return self.enabled and self.due_amount >= 0.01


@dataclass
class PortfolioState:
    state_version: int = STATE_VERSION
    shares: dict[str, float] = field(default_factory=dict)
    cash_balance: float = 0.0
    target_weights: dict[str, float] = field(default_factory=dict)
    portfolio_value: float = 0.0

    strategy_initialized: bool = False
    btal_active: bool = False
    btal_entry_streak: int = 0
    btal_exit_streak: int = 0
    btal_switch_date: str = ""
    last_processed_signal_date: str = ""
    lifecycle_stage: str = LIFECYCLE_SPRINT
    lifecycle_stage_date: str = ""

    executed_btal_active: bool = False
    executed_lifecycle_stage: str = LIFECYCLE_SPRINT
    executed_strategy_fingerprint: str = ""
    last_completed_annual_rebalance_year: int = 0

    pending_recommendation_date: str = ""
    pending_recommendation_weights: dict[str, float] = field(default_factory=dict)
    pending_recommendation_btal_active: bool = False
    pending_recommendation_lifecycle_stage: str = ""
    pending_recommendation_notified: bool = False
    pending_recommendation_supersedes_date: str = ""
    pending_recommendation_fingerprint: str = ""
    pending_recommendation_annual_year: int = 0

    contribution_plan_year: int = 0
    contribution_policy_revision: str = ""
    contribution_budget: float = 0.0
    contribution_released_amount: float = 0.0
    contribution_pullback_used: bool = False
    contribution_drawdown_10_used: bool = False
    contribution_drawdown_20_used: bool = False
    pending_contribution_date: str = ""
    pending_contribution_amount: float = 0.0
    pending_contribution_reason: str = ""
    pending_contribution_pullback: bool = False
    pending_contribution_drawdown_10: bool = False
    pending_contribution_drawdown_20: bool = False
    last_contribution_notice_date: str = ""

    last_processed_data_fingerprint: str = ""
    last_delivered_decision_hash: str = ""
    last_delivered_signal_date: str = ""
    last_delivered_notification_kind: str = ""
    last_updated: str = ""


@dataclass(frozen=True)
class StrategyRun:
    price_data: pd.DataFrame
    decision: StrategyDecision
    state: PortfolioState
    planning_state: PortfolioState
    portfolio_value: float
    current_weights: dict[str, float]
    execution_table: pd.DataFrame
    signal_date: pd.Timestamp
    market_data_fingerprint: str
    rebalance_plan: RebalancePlan
    execution_diagnostics: ExecutionDiagnostics
    contribution_plan: ContributionPlan = field(default_factory=ContributionPlan)


_configured_amount = os.environ.get("ROTH_IRA_AMOUNT", "").strip()
try:
    ROTH_IRA_AMOUNT = float(_configured_amount) if _configured_amount else None
except ValueError as exc:
    raise RuntimeError("ROTH_IRA_AMOUNT must be numeric") from exc

logger = logging.getLogger("roth_ira")
logger.setLevel(logging.INFO)
logger.propagate = False


def configure_logging(*, persist_log: bool) -> None:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if persist_log:
        handlers.insert(0, logging.FileHandler(LOG_FILE))
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def _is_number(value: object, *, positive: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _parse_date(value: str, name: str) -> date | None:
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a string")
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must use YYYY-MM-DD") from exc


def _with_cash(weights: dict[str, float]) -> dict[str, float]:
    result = dict(weights)
    result.setdefault(CASH_ASSET, 0.0)
    return result


def _weights_close(left: dict[str, float], right: dict[str, float], tolerance: float = 1e-9) -> bool:
    return all(
        abs(left.get(ticker, 0.0) - right.get(ticker, 0.0)) <= tolerance
        for ticker in set(left) | set(right)
    )


def advertised_daily_exposure(weights: dict[str, float]) -> float:
    unknown = set(weights) - set(ADVERTISED_DAILY_MULTIPLIERS)
    if unknown:
        raise ValueError(f"Unknown exposure components: {sorted(unknown)}")
    exposure = sum(float(weight) * ADVERTISED_DAILY_MULTIPLIERS[ticker] for ticker, weight in weights.items())
    if not np.isfinite(exposure):
        raise ValueError("Advertised exposure is invalid")
    return float(exposure)


def target_weights(btal_active: bool, lifecycle_stage: str = LIFECYCLE_SPRINT) -> dict[str, float]:
    if lifecycle_stage not in LIFECYCLE_STAGES:
        raise ValueError(f"Unknown lifecycle stage: {lifecycle_stage}")
    aggressive = core.target_weights(btal_active)
    ceiling = LIFECYCLE_EXPOSURE_CEILINGS[lifecycle_stage]
    exposure = advertised_daily_exposure(aggressive)
    if ceiling is None or exposure <= ceiling + 1e-12:
        return aggressive
    fraction = float(ceiling) / exposure
    result = {ticker: weight * fraction for ticker, weight in aggressive.items()}
    result[TREASURY_RESERVE] = 1.0 - sum(result.values())
    if not np.isclose(sum(result.values()), 1.0, atol=1e-12):
        raise RuntimeError("Target weights do not sum to one")
    return result


def validate_configuration() -> None:
    if STRATEGY_FINGERPRINT != calculate_strategy_fingerprint():
        raise RuntimeError("Strategy fingerprint is internally inconsistent")
    if STRATEGY_FINGERPRINT != EXPECTED_STRATEGY_FINGERPRINT:
        raise RuntimeError("Decision boundaries changed without fingerprint review")
    for stage in LIFECYCLE_STAGES:
        ceiling = LIFECYCLE_EXPOSURE_CEILINGS[stage]
        for hedged in (False, True):
            weights = target_weights(hedged, stage)
            if not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
                raise RuntimeError("Allocation invariant failed")
            exposure = advertised_daily_exposure(weights)
            expected = core.advertised_daily_exposure(hedged)
            if ceiling is not None:
                expected = min(expected, ceiling)
            if not np.isclose(exposure, expected, atol=1e-12):
                raise RuntimeError("Lifecycle exposure invariant failed")
    value_stages = [stage for _, stage in LIFECYCLE_VALUE_THRESHOLDS_2026]
    age_stages = [stage for _, stage in LIFECYCLE_AGE_THRESHOLDS]
    if value_stages != sorted(value_stages, key=_lifecycle_rank):
        raise RuntimeError("Lifecycle value stages are not monotonic")
    if age_stages != sorted(age_stages, key=_lifecycle_rank):
        raise RuntimeError("Lifecycle age stages are not monotonic")


def _validate_weights(weights: object, name: str, *, require_total: bool) -> None:
    if not isinstance(weights, dict):
        raise RuntimeError(f"{name} must be a mapping")
    if set(weights) - PORTFOLIO_COMPONENTS:
        raise RuntimeError(f"{name} contains unsupported holdings")
    if any(not _is_number(value) for value in weights.values()):
        raise RuntimeError(f"{name} contains invalid weights")
    if require_total and not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
        raise RuntimeError(f"{name} must sum to one")


def validate_state(state: PortfolioState) -> None:
    if state.state_version != STATE_VERSION:
        raise RuntimeError("Portfolio state version is invalid")
    if not isinstance(state.shares, dict) or set(state.shares) - TRADED_TICKERS:
        raise RuntimeError("shares contains unsupported holdings")
    if any(not _is_number(value) for value in state.shares.values()):
        raise RuntimeError("shares contains invalid quantities")
    if not _is_number(state.cash_balance) or not _is_number(state.portfolio_value):
        raise RuntimeError("Portfolio balances are invalid")
    _validate_weights(state.target_weights, "target_weights", require_total=bool(state.target_weights))
    for name in (
        "strategy_initialized",
        "btal_active",
        "executed_btal_active",
        "pending_recommendation_btal_active",
        "pending_recommendation_notified",
        "contribution_pullback_used",
        "contribution_drawdown_10_used",
        "contribution_drawdown_20_used",
        "pending_contribution_pullback",
        "pending_contribution_drawdown_10",
        "pending_contribution_drawdown_20",
    ):
        if not isinstance(getattr(state, name), bool):
            raise RuntimeError(f"{name} must be boolean")
    if (
        not isinstance(state.btal_entry_streak, int)
        or isinstance(state.btal_entry_streak, bool)
        or state.btal_entry_streak < 0
    ):
        raise RuntimeError("btal_entry_streak is invalid")
    if (
        not isinstance(state.btal_exit_streak, int)
        or isinstance(state.btal_exit_streak, bool)
        or state.btal_exit_streak < 0
    ):
        raise RuntimeError("btal_exit_streak is invalid")
    for name in (
        "last_completed_annual_rebalance_year",
        "pending_recommendation_annual_year",
    ):
        value = getattr(state, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(f"{name} is invalid")
    if (
        state.lifecycle_stage not in LIFECYCLE_STAGES
        or state.executed_lifecycle_stage not in LIFECYCLE_STAGES
    ):
        raise RuntimeError("Lifecycle state is invalid")
    _validate_weights(
        state.pending_recommendation_weights,
        "pending_recommendation_weights",
        require_total=bool(state.pending_recommendation_date),
    )
    if state.pending_recommendation_date:
        pending_date = _parse_date(
            state.pending_recommendation_date, "pending_recommendation_date"
        )
        superseded_date = _parse_date(
            state.pending_recommendation_supersedes_date,
            "pending_recommendation_supersedes_date",
        )
        if superseded_date and pending_date and superseded_date >= pending_date:
            raise RuntimeError(
                "pending_recommendation_supersedes_date must precede the pending date"
            )
        if not _is_sha256(state.pending_recommendation_fingerprint):
            raise RuntimeError("Pending recommendation fingerprint is invalid")
        if state.pending_recommendation_lifecycle_stage not in LIFECYCLE_STAGES:
            raise RuntimeError("Pending lifecycle stage is invalid")
        pending_has_btal = state.pending_recommendation_weights.get(CRISIS_HEDGE, 0.0) > 0
        if state.pending_recommendation_btal_active != pending_has_btal:
            raise RuntimeError("Pending crisis-hedge state is inconsistent")
    elif any((state.pending_recommendation_weights, state.pending_recommendation_notified,
              state.pending_recommendation_supersedes_date, state.pending_recommendation_fingerprint,
              state.pending_recommendation_lifecycle_stage,
              state.pending_recommendation_annual_year,
              state.pending_recommendation_btal_active)):
        raise RuntimeError("Pending recommendation state is inconsistent")
    if not isinstance(state.contribution_plan_year, int) or isinstance(
        state.contribution_plan_year, bool
    ) or state.contribution_plan_year < 0:
        raise RuntimeError("contribution_plan_year is invalid")
    for name in (
        "contribution_budget",
        "contribution_released_amount",
        "pending_contribution_amount",
    ):
        if not _is_number(getattr(state, name)):
            raise RuntimeError(f"{name} is invalid")
    contribution_fields = (
        state.contribution_policy_revision,
        state.contribution_budget,
        state.contribution_released_amount,
        state.pending_contribution_amount,
        state.contribution_pullback_used,
        state.contribution_drawdown_10_used,
        state.contribution_drawdown_20_used,
        state.pending_contribution_date,
        state.pending_contribution_reason,
        state.pending_contribution_pullback,
        state.pending_contribution_drawdown_10,
        state.pending_contribution_drawdown_20,
        state.last_contribution_notice_date,
    )
    if state.contribution_plan_year == 0:
        if any(contribution_fields):
            raise RuntimeError("Disabled contribution plan contains state")
    else:
        if state.contribution_policy_revision != contribution.POLICY_REVISION:
            raise RuntimeError("Contribution policy revision is unsupported")
        if state.contribution_budget <= 0:
            raise RuntimeError("Enabled contribution plan requires a positive budget")
        if state.contribution_released_amount > state.contribution_budget + 0.005:
            raise RuntimeError("Released contribution exceeds its budget")
        if state.pending_contribution_date:
            _parse_date(state.pending_contribution_date, "pending_contribution_date")
            if state.pending_contribution_amount < 0.01 or not state.pending_contribution_reason:
                raise RuntimeError("Pending contribution state is incomplete")
            if (
                state.contribution_released_amount + state.pending_contribution_amount
                > state.contribution_budget + 0.005
            ):
                raise RuntimeError("Pending contribution exceeds its budget")
        elif any(
            (
                state.pending_contribution_amount,
                state.pending_contribution_reason,
                state.pending_contribution_pullback,
                state.pending_contribution_drawdown_10,
                state.pending_contribution_drawdown_20,
            )
        ):
            raise RuntimeError("Pending contribution state is inconsistent")
    for name in (
        "btal_switch_date",
        "last_processed_signal_date",
        "lifecycle_stage_date",
        "last_delivered_signal_date",
        "last_contribution_notice_date",
    ):
        _parse_date(getattr(state, name), name)
    if state.last_updated:
        if not isinstance(state.last_updated, str):
            raise RuntimeError("last_updated must be a string")
        try:
            datetime.fromisoformat(state.last_updated)
        except ValueError as exc:
            raise RuntimeError("last_updated must be an ISO-8601 timestamp") from exc
    for name in ("last_processed_data_fingerprint", "last_delivered_decision_hash"):
        value = getattr(state, name)
        if value and not _is_sha256(value):
            raise RuntimeError(f"{name} is invalid")


def _migrate_mapping(value: object, name: str) -> dict[str, float]:
    """Validate migrated holdings without silently discarding account data."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for ticker, quantity in value.items():
        symbol = str(ticker).upper()
        if symbol not in PORTFOLIO_COMPONENTS:
            raise RuntimeError(f"{name} contains unsupported holding {symbol}")
        if not _is_number(quantity):
            raise RuntimeError(f"{name} contains an invalid quantity for {symbol}")
        result[symbol] = float(quantity)
    return result


def _migrate_state(payload: dict[str, object]) -> PortfolioState:
    version = payload.get("state_version")
    if not isinstance(version, int) or isinstance(version, bool) or not 2 <= version < STATE_VERSION:
        raise RuntimeError(f"Unsupported state version: {version!r}")
    shares = _migrate_mapping(payload.get("shares"), "shares")
    target = _migrate_mapping(payload.get("target_weights"), "target_weights")
    pending = _migrate_mapping(
        payload.get("pending_recommendation_weights"),
        "pending_recommendation_weights",
    )
    new_names = {item.name for item in fields(PortfolioState)}
    migrated = {
        name: payload[name]
        for name in new_names
        if name in payload
        and name
        not in {
            "state_version",
            "shares",
            "target_weights",
            "pending_recommendation_weights",
            "strategy_initialized",
            "btal_active",
            "btal_entry_streak",
            "btal_exit_streak",
            "btal_switch_date",
            "executed_btal_active",
            "pending_recommendation_btal_active",
            "last_processed_signal_date",
            "last_processed_data_fingerprint",
        }
    }
    migrated.update(
        state_version=STATE_VERSION,
        shares=shares,
        target_weights=target,
        pending_recommendation_weights=pending,
        # A strategy revision must evaluate two fresh completed sessions. It
        # must not inherit the old TQQQ/UPRO router's signal state.
        strategy_initialized=False,
        btal_active=False,
        btal_entry_streak=0,
        btal_exit_streak=0,
        btal_switch_date="",
        last_processed_signal_date="",
        last_processed_data_fingerprint="",
        executed_btal_active=shares.get(CRISIS_HEDGE, 0.0) > 0,
        pending_recommendation_btal_active=pending.get(CRISIS_HEDGE, 0.0) > 0,
    )
    if "last_completed_annual_rebalance_year" not in payload:
        previous_signal = str(payload.get("last_processed_signal_date", ""))
        try:
            migrated["last_completed_annual_rebalance_year"] = (
                date.fromisoformat(previous_signal).year if previous_signal else 0
            )
        except ValueError:
            migrated["last_completed_annual_rebalance_year"] = 0
    state = PortfolioState(**migrated)
    if not pending:
        _clear_pending(state)
    if state.target_weights and not np.isclose(sum(state.target_weights.values()), 1.0, atol=1e-9):
        state.target_weights = {}
    return state


def load_state() -> PortfolioState:
    if not STATE_FILE.exists():
        return PortfolioState()
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Portfolio state is unreadable") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Portfolio state must be an object")
    version = payload.get("state_version")
    if version == STATE_VERSION:
        allowed = {item.name for item in fields(PortfolioState)}
        unknown = set(payload) - allowed
        if unknown:
            raise RuntimeError(f"Portfolio state contains unknown fields: {sorted(unknown)}")
        state = PortfolioState(**payload)
    else:
        state = _migrate_state(payload)
        logger.info("Migrated state version %s to %s", version, STATE_VERSION)
    validate_state(state)
    return state


def save_state(state: PortfolioState) -> None:
    validate_state(state)
    state.last_updated = datetime.now(NEW_YORK).isoformat()
    temporary = STATE_FILE.with_suffix(f"{STATE_FILE.suffix}.tmp")
    temporary.write_text(json.dumps(asdict(state), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, STATE_FILE)


def _nyse_calendar():
    return xcals.get_calendar("XNYS")


def expected_completed_session(now_new_york: datetime | None = None) -> pd.Timestamp:
    current = now_new_york or datetime.now(NEW_YORK)
    stamp = pd.Timestamp(current)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(NEW_YORK)
    else:
        stamp = stamp.tz_convert(NEW_YORK)
    calendar = _nyse_calendar()
    today = pd.Timestamp(stamp.date())
    sessions = calendar.sessions_in_range(today - pd.Timedelta(days=14), today + pd.Timedelta(days=1))
    now_utc = stamp.tz_convert("UTC")
    if calendar.is_session(today):
        opened = calendar.session_open(today)
        safe_close = calendar.session_close(today) + pd.Timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES)
        if opened <= now_utc < safe_close:
            raise RuntimeError("The latest daily bar is not final")
    complete = [
        session
        for session in sessions
        if calendar.session_close(session)
        + pd.Timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES)
        <= now_utc
    ]
    if not complete:
        raise RuntimeError("No completed XNYS session is available")
    return pd.Timestamp(complete[-1]).tz_localize(None).normalize()


def required_nyse_sessions(ending_session: pd.Timestamp, count: int) -> pd.DatetimeIndex:
    if count <= 0:
        raise ValueError("count must be positive")
    ending = pd.Timestamp(ending_session).normalize()
    sessions = pd.DatetimeIndex(
        _nyse_calendar().sessions_in_range(
            ending - pd.Timedelta(days=max(30, count * 3)), ending
        )
    )
    if sessions.tz is not None:
        sessions = sessions.tz_convert(None)
    if len(sessions) < count:
        raise RuntimeError("XNYS calendar returned insufficient sessions")
    return sessions.normalize()[-count:]


def _extract_yfinance_prices(data: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    requested = list(tickers)
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise RuntimeError("yfinance returned no market data")
    try:
        if isinstance(data.columns, pd.MultiIndex):
            prices = (
                data["Close"].copy()
                if "Close" in data.columns.get_level_values(0)
                else data.xs("Close", axis=1, level=1).copy()
            )
        else:
            close = data["Close"]
            prices = close.to_frame(requested[0]) if isinstance(close, pd.Series) else pd.DataFrame(close)
    except KeyError as exc:
        raise RuntimeError("yfinance response is missing Close") from exc
    missing = set(requested) - set(map(str, prices.columns))
    if missing:
        raise RuntimeError(f"Incomplete market-data universe: {sorted(missing)}")
    prices = prices.reindex(columns=requested)
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise RuntimeError("Market data requires a DatetimeIndex")
    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert(None)
    prices.index = prices.index.normalize()
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise RuntimeError("Market dates are duplicated or unsorted")
    return prices


def _require_positive(values: pd.DataFrame | pd.Series, description: str) -> None:
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not (numeric > 0).all():
        raise RuntimeError(f"{description} contains missing or invalid values")


def validate_session_continuity(index: pd.DatetimeIndex, expected: pd.Timestamp, count: int) -> None:
    received = pd.DatetimeIndex(index[-count:]).normalize()
    required = required_nyse_sessions(expected, count)
    if len(required.difference(received)) or len(received.difference(required)):
        raise RuntimeError("Market-data session continuity failed")


def download_market_data(
    tickers: Iterable[str] = ALL_TICKERS,
    *,
    now_new_york: datetime | None = None,
) -> pd.DataFrame:
    import yfinance as yf

    expected = expected_completed_session(now_new_york)
    requested = list(tickers)
    raw = yf.download(
        requested,
        start=MODEL_START_DATE,
        end=(expected.date() + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    prices = _extract_yfinance_prices(raw, requested)
    received = pd.Timestamp(prices.index[-1]).normalize()
    if received != expected:
        raise RuntimeError(f"Market data is stale: expected {expected.date()}, received {received.date()}")
    if len(prices) < REQUIRED_SIGNAL_ROWS:
        raise RuntimeError("Insufficient market history")
    for ticker in (MARKET_INDEX, CONFIRMATION_INDEX):
        history = prices[ticker].dropna()
        validate_session_continuity(history.index, expected, REQUIRED_SIGNAL_ROWS)
        _require_positive(
            history.iloc[-REQUIRED_SIGNAL_ROWS:], f"{ticker} signal history"
        )
    _require_positive(prices.loc[received, requested], "Latest valuation prices")
    return prices


def market_data_fingerprint(prices: pd.DataFrame) -> str:
    if len(prices) < REQUIRED_SIGNAL_ROWS:
        raise ValueError("Insufficient data for fingerprint")
    qqq = pd.to_numeric(prices[MARKET_INDEX], errors="coerce")
    spy = pd.to_numeric(prices[CONFIRMATION_INDEX], errors="coerce")
    close = float(qqq.iloc[-1])
    sma_200 = float(qqq.iloc[-SMA_WINDOW:].mean())
    sma_50 = float(qqq.iloc[-SHORT_TREND_WINDOW:].mean())
    momentum_252 = float(close / qqq.iloc[-(MOMENTUM_WINDOW + 1)] - 1.0)
    spy_close = float(spy.iloc[-1])
    spy_sma_200 = float(spy.iloc[-SMA_WINDOW:].mean())
    spy_sma_50 = float(spy.iloc[-SHORT_TREND_WINDOW:].mean())
    values = (close, sma_200, sma_50, momentum_252, spy_close, spy_sma_200, spy_sma_50)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Signal inputs are invalid for fingerprinting")
    extreme, recovery = core.crisis_conditions(
        qqq_close=close,
        qqq_sma_200=sma_200,
        spy_close=spy_close,
        spy_sma_200=spy_sma_200,
        spy_sma_50=spy_sma_50,
    )
    return canonical_sha256(
        {
            "semantic_version": "extreme-bear-normalized-inputs-v3",
            "session": pd.Timestamp(prices.index[-1]).date().isoformat(),
            "qqq_close": round(close, 4),
            "qqq_sma_200": round(sma_200, 4),
            "qqq_sma_50": round(sma_50, 4),
            "qqq_momentum_252": round(momentum_252, 6),
            "spy_close": round(spy_close, 4),
            "spy_sma_200": round(spy_sma_200, 4),
            "spy_sma_50": round(spy_sma_50, 4),
            "extreme_bearish": extreme,
            "recovery_confirmed": recovery,
        }
    )


def _hedge_state(state: PortfolioState) -> core.CrisisHedgeState:
    return core.CrisisHedgeState(
        btal_active=state.btal_active,
        entry_streak=state.btal_entry_streak,
        exit_streak=state.btal_exit_streak,
        last_processed_signal_date=state.last_processed_signal_date,
        switch_date=state.btal_switch_date,
    )


def _apply_hedge(state: PortfolioState, hedge: core.CrisisHedgeState) -> None:
    state.strategy_initialized = True
    state.btal_active = hedge.btal_active
    state.btal_entry_streak = hedge.entry_streak
    state.btal_exit_streak = hedge.exit_streak
    state.last_processed_signal_date = hedge.last_processed_signal_date
    state.btal_switch_date = hedge.switch_date


def _latest_decision(prices: pd.DataFrame, state: PortfolioState) -> StrategyDecision:
    session = pd.Timestamp(prices.index[-1]).normalize()
    qqq = pd.to_numeric(prices[MARKET_INDEX], errors="coerce")
    spy = pd.to_numeric(prices[CONFIRMATION_INDEX], errors="coerce")
    close = float(qqq.iloc[-1])
    average = float(qqq.iloc[-SMA_WINDOW:].mean())
    short_average = float(qqq.iloc[-SHORT_TREND_WINDOW:].mean())
    momentum = float(close / qqq.iloc[-(MOMENTUM_WINDOW + 1)] - 1.0)
    spy_close = float(spy.iloc[-1])
    spy_average = float(spy.iloc[-SMA_WINDOW:].mean())
    spy_short_average = float(spy.iloc[-SHORT_TREND_WINDOW:].mean())
    if (
        not all(
            np.isfinite(value)
            for value in (
                close,
                average,
                short_average,
                momentum,
                spy_close,
                spy_average,
                spy_short_average,
            )
        )
        or close <= 0
        or average <= 0
        or short_average <= 0
        or spy_close <= 0
        or spy_average <= 0
        or spy_short_average <= 0
    ):
        raise RuntimeError("Crisis-hedge signal inputs are invalid")
    trend = close > average
    extreme, recovery = core.crisis_conditions(
        qqq_close=close,
        qqq_sma_200=average,
        spy_close=spy_close,
        spy_sma_200=spy_average,
        spy_sma_50=spy_short_average,
    )
    transition = core.advance_crisis_hedge(
        _hedge_state(state),
        signal_date=session,
        extreme_bearish=extreme,
        recovery_confirmed=recovery,
    )
    return StrategyDecision(
        target_weights=target_weights(transition.state.btal_active),
        hedge_state=transition.state,
        transition_reason=transition.reason,
        structural_change=transition.structural_change,
        trend_positive=trend,
        qqq_close=close,
        qqq_sma_200=average,
        qqq_sma_50=short_average,
        qqq_momentum_252=momentum,
        spy_close=spy_close,
        spy_sma_200=spy_average,
        spy_sma_50=spy_short_average,
        extreme_bearish=extreme,
        recovery_confirmed=recovery,
    )


def calculate_strategy_decision(prices: pd.DataFrame, state: PortfolioState) -> StrategyDecision:
    if prices.empty:
        raise RuntimeError("Strategy requires market data")
    latest = pd.Timestamp(prices.index[-1]).normalize()
    if state.strategy_initialized and state.last_processed_signal_date:
        prior = pd.Timestamp(state.last_processed_signal_date).normalize()
        if prior > latest:
            raise RuntimeError("Portfolio state is ahead of market data")
        unseen = pd.DatetimeIndex(prices.index[prices.index > prior])
        sessions = unseen if len(unseen) else pd.DatetimeIndex([latest])
    else:
        sessions = pd.DatetimeIndex(prices.index[-core.CONFIRMATION_CLOSES:])
    working = copy.deepcopy(state)
    decisions = []
    reasons = []
    dates = []
    changed = False
    for session in sessions:
        history = prices.loc[:session]
        if len(history) < MOMENTUM_WINDOW + 1:
            raise RuntimeError("Insufficient history for crisis-hedge replay")
        decision = _latest_decision(history, working)
        _apply_hedge(working, decision.hedge_state)
        decisions.append(decision)
        reasons.append(decision.transition_reason)
        dates.append(pd.Timestamp(session).date().isoformat())
        changed = changed or decision.structural_change
    return replace(
        decisions[-1],
        structural_change=changed,
        processed_signal_dates=tuple(dates),
        transition_path=tuple(reasons),
    )


def _lifecycle_rank(stage: str) -> int:
    try:
        return LIFECYCLE_STAGES.index(stage)
    except ValueError as exc:
        raise ValueError(f"Unknown lifecycle stage: {stage}") from exc


def estimated_investor_age(as_of: date) -> float:
    configured = os.environ.get("INVESTOR_BIRTH_DATE", "").strip()
    if configured:
        try:
            born = date.fromisoformat(configured)
        except ValueError as exc:
            raise RuntimeError("INVESTOR_BIRTH_DATE must use YYYY-MM-DD") from exc
        if born >= as_of:
            raise RuntimeError("INVESTOR_BIRTH_DATE must precede signal date")
        birthday = (born.month, born.day)
        return float(as_of.year - born.year - ((as_of.month, as_of.day) < birthday))
    return LIFECYCLE_ANCHOR_AGE + (as_of - LIFECYCLE_ANCHOR_DATE).days / 365.2425


def inflation_adjusted_value_threshold(base: float, as_of: date) -> float:
    years = (as_of - LIFECYCLE_ANCHOR_DATE).days / 365.2425
    return float(base * (1.0 + LIFECYCLE_INFLATION_RATE) ** years)


def _stage_for_value(value: float, as_of: date) -> str:
    stage = LIFECYCLE_SPRINT
    for threshold, candidate in LIFECYCLE_VALUE_THRESHOLDS_2026:
        if value >= inflation_adjusted_value_threshold(threshold, as_of):
            stage = candidate
    return stage


def _stage_for_age(age: float) -> str:
    stage = LIFECYCLE_SPRINT
    for threshold, candidate in LIFECYCLE_AGE_THRESHOLDS:
        if age >= threshold:
            stage = candidate
    return stage


def select_lifecycle_stage(prior: str, value: float, as_of: date) -> LifecycleSelection:
    if prior not in LIFECYCLE_STAGES or not _is_number(value, positive=True):
        raise ValueError("Invalid lifecycle inputs")
    age = estimated_investor_age(as_of)
    value_stage = _stage_for_value(value, as_of)
    age_stage = _stage_for_age(age)
    selected = max((prior, value_stage, age_stage), key=_lifecycle_rank)
    advanced = _lifecycle_rank(selected) > _lifecycle_rank(prior)
    if selected == prior and not advanced:
        reason = "RATCHET_HOLD"
    elif _lifecycle_rank(age_stage) >= _lifecycle_rank(value_stage):
        reason = "AGE_CEILING"
    else:
        reason = "VALUE_MILESTONE"
    return LifecycleSelection(selected, reason, age, value_stage, age_stage, advanced)


def apply_lifecycle_policy(
    decision: StrategyDecision,
    state: PortfolioState,
    portfolio_value: float,
    signal_date: pd.Timestamp,
) -> StrategyDecision:
    selected = select_lifecycle_stage(
        state.lifecycle_stage, portfolio_value, pd.Timestamp(signal_date).date()
    )
    return replace(
        decision,
        target_weights=target_weights(decision.hedge_state.btal_active, selected.stage),
        lifecycle_stage=selected.stage,
        lifecycle_reason=selected.reason,
        estimated_investor_age=selected.estimated_age,
        lifecycle_value_stage=selected.value_stage,
        lifecycle_age_stage=selected.age_stage,
        lifecycle_stage_advanced=selected.advanced,
    )


def _apply_lifecycle(state: PortfolioState, decision: StrategyDecision, signal_date: pd.Timestamp) -> None:
    if _lifecycle_rank(decision.lifecycle_stage) < _lifecycle_rank(state.lifecycle_stage):
        raise RuntimeError("Lifecycle ratchet cannot move backward")
    if decision.lifecycle_stage != state.lifecycle_stage:
        state.lifecycle_stage = decision.lifecycle_stage
        state.lifecycle_stage_date = pd.Timestamp(signal_date).date().isoformat()


def validate_same_date_data_fingerprint(
    state: PortfolioState, signal_date: pd.Timestamp, fingerprint: str
) -> None:
    if not _is_sha256(fingerprint):
        raise ValueError("Market-data fingerprint is invalid")
    if (
        state.last_processed_signal_date == pd.Timestamp(signal_date).date().isoformat()
        and state.last_processed_data_fingerprint
        and state.last_processed_data_fingerprint != fingerprint
    ):
        raise RuntimeError("Market data changed for an already processed signal date")


def validate_holdings_against_prices(state: PortfolioState, prices: pd.DataFrame) -> None:
    missing = set(state.shares) - set(prices.columns)
    if missing:
        raise RuntimeError(f"Holdings have no current prices: {sorted(missing)}")
    for ticker, quantity in state.shares.items():
        if quantity > 0:
            price = float(prices[ticker].iloc[-1])
            if not np.isfinite(price) or price <= 0:
                raise RuntimeError(f"Invalid price for confirmed holding {ticker}")


def existing_portfolio_value(state: PortfolioState, prices: pd.DataFrame) -> float:
    validate_holdings_against_prices(state, prices)
    value = float(state.cash_balance)
    for ticker, quantity in state.shares.items():
        value += float(quantity) * float(prices[ticker].iloc[-1])
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("Portfolio value must be positive")
    return value


def existing_weights(state: PortfolioState, prices: pd.DataFrame) -> dict[str, float]:
    value = existing_portfolio_value(state, prices)
    result = {
        ticker: float(quantity) * float(prices[ticker].iloc[-1]) / value
        for ticker, quantity in state.shares.items()
        if quantity > 1e-14
    }
    if state.cash_balance > 1e-10:
        result[CASH_ASSET] = state.cash_balance / value
    if not np.isclose(sum(result.values()), 1.0, atol=1e-9):
        raise RuntimeError("Current weights do not sum to one")
    return result


def resolve_portfolio_value(
    roth_amount: float | None, state: PortfolioState, prices: pd.DataFrame
) -> tuple[float, PortfolioState]:
    planning = copy.deepcopy(state)
    if state.shares or state.cash_balance > 0:
        if roth_amount is not None:
            raise RuntimeError("--roth-amount is valid only for a new all-cash state")
        return existing_portfolio_value(planning, prices), planning
    amount = roth_amount if roth_amount is not None else ROTH_IRA_AMOUNT
    if not _is_number(amount, positive=True):
        raise RuntimeError("No portfolio value is available; initialize explicitly")
    planning.cash_balance = float(amount)
    return float(amount), planning


def _allocate_contribution(
    amount: float,
    state: PortfolioState,
    prices: pd.DataFrame,
    target: dict[str, float],
    portfolio_value: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Allocate new cash across target underweights without assuming any fills."""
    if amount < 0.01:
        return {}, {}
    post_contribution_value = portfolio_value + amount
    deficits = {}
    for ticker, weight in target.items():
        current_value = state.shares.get(ticker, 0.0) * float(prices[ticker].iloc[-1])
        deficits[ticker] = max(0.0, weight * post_contribution_value - current_value)
    total_deficit = sum(deficits.values())
    proportions = (
        {ticker: deficit / total_deficit for ticker, deficit in deficits.items()}
        if total_deficit > 1e-12
        else dict(target)
    )
    dollars = {ticker: round(amount * weight, 2) for ticker, weight in proportions.items()}
    residual = round(amount - sum(dollars.values()), 2)
    if dollars and abs(residual) >= 0.01:
        largest = max(dollars, key=dollars.get)
        dollars[largest] = round(dollars[largest] + residual, 2)
    dollars = {ticker: value for ticker, value in dollars.items() if value >= 0.01}
    units = {
        ticker: value / float(prices[ticker].iloc[-1]) for ticker, value in dollars.items()
    }
    return dollars, units


def build_contribution_plan(
    state: PortfolioState,
    decision: StrategyDecision,
    prices: pd.DataFrame,
    portfolio_value: float,
    signal_date: pd.Timestamp,
) -> ContributionPlan:
    plan_year = state.contribution_plan_year
    year = pd.Timestamp(signal_date).year
    if plan_year == 0 or state.contribution_budget <= 0 or plan_year != year:
        return ContributionPlan(
            enabled=plan_year > 0,
            year=plan_year,
            budget=state.contribution_budget,
            released_amount=state.contribution_released_amount,
        )

    qqq = prices[MARKET_INDEX]
    high_63 = float(qqq.iloc[-contribution.LOOKBACK_SESSIONS :].max())
    rule = contribution.evaluate_release(
        as_of=pd.Timestamp(signal_date).date(),
        qqq_close=decision.qqq_close,
        qqq_sma_50=decision.qqq_sma_50,
        qqq_sma_200=decision.qqq_sma_200,
        qqq_high_63=high_63,
        pullback_used=state.contribution_pullback_used,
        drawdown_10_used=state.contribution_drawdown_10_used,
        drawdown_20_used=state.contribution_drawdown_20_used,
    )
    retry = bool(state.pending_contribution_date)
    if retry:
        due = state.pending_contribution_amount
        reasons = tuple(state.pending_contribution_reason.split("+"))
        use_pullback = state.pending_contribution_pullback
        use_drawdown_10 = state.pending_contribution_drawdown_10
        use_drawdown_20 = state.pending_contribution_drawdown_20
    else:
        target_amount = round(state.contribution_budget * rule.target_fraction, 2)
        due = max(0.0, round(target_amount - state.contribution_released_amount, 2))
        reasons = rule.reasons
        use_pullback = rule.use_pullback
        use_drawdown_10 = rule.use_drawdown_10
        use_drawdown_20 = rule.use_drawdown_20
    projected = min(
        state.contribution_budget,
        round(state.contribution_released_amount + due, 2),
    )
    dollars, units = _allocate_contribution(
        due, state, prices, decision.target_weights, portfolio_value
    )
    return ContributionPlan(
        enabled=True,
        year=plan_year,
        budget=state.contribution_budget,
        released_amount=state.contribution_released_amount,
        due_amount=due,
        projected_released_amount=projected,
        calendar_fraction=rule.calendar_fraction,
        target_fraction=rule.target_fraction,
        qqq_drawdown_63=rule.drawdown,
        bull_pullback=rule.bull_pullback,
        reasons=reasons,
        use_pullback=use_pullback,
        use_drawdown_10=use_drawdown_10,
        use_drawdown_20=use_drawdown_20,
        retry=retry,
        allocation_dollars=dollars,
        estimated_units=units,
    )


def _one_way_turnover(
    existing: dict[str, float], execution: dict[str, float], components: Iterable[str]
) -> float:
    """Half the sum of absolute weight changes, i.e. one-way turnover."""
    return 0.5 * sum(
        abs(execution.get(item, 0.0) - existing.get(item, 0.0)) for item in components
    )


def _individual_orders(
    existing: dict[str, float], execution: dict[str, float], components: Iterable[str]
) -> int:
    """Count of non-cash securities whose weight changes materially."""
    return sum(
        item != CASH_ASSET
        and abs(execution.get(item, 0.0) - existing.get(item, 0.0)) > 1e-9
        for item in components
    )


def build_rebalance_plan(
    existing: dict[str, float],
    decision: StrategyDecision,
    state: PortfolioState,
) -> RebalancePlan:
    target = _with_cash(decision.target_weights)
    signal_year = date.fromisoformat(
        decision.hedge_state.last_processed_signal_date
    ).year
    annual_due = signal_year > state.last_completed_annual_rebalance_year
    strategy_changed = state.executed_strategy_fingerprint != STRATEGY_FINGERPRINT
    lifecycle_changed = state.executed_lifecycle_stage != decision.lifecycle_stage
    tactical_weight = sum(existing.get(ticker, 0.0) for ticker in TACTICAL_TICKERS)
    tactical_missing = tactical_weight <= 1e-12
    legacy_position = existing.get(LEGACY_UPRO, 0.0) > 1e-12
    hedge_misaligned = (
        decision.hedge_state.btal_active
        and existing.get(CRISIS_HEDGE, 0.0) <= 1e-12
    ) or (
        not decision.hedge_state.btal_active
        and existing.get(CRISIS_HEDGE, 0.0) > 1e-12
    )
    full = strategy_changed or lifecycle_changed or annual_due or tactical_missing or legacy_position
    due = full or hedge_misaligned or decision.structural_change
    if strategy_changed:
        reason = "STRATEGY_REVISION_TRANSITION"
    elif lifecycle_changed:
        reason = "LIFECYCLE_STAGE_ADVANCE"
    elif legacy_position:
        reason = "LEGACY_UPRO_LIQUIDATION"
    elif tactical_missing:
        reason = "MISSING_TACTICAL_POSITION"
    elif decision.structural_change:
        reason = decision.transition_reason
    elif hedge_misaligned:
        reason = "CRISIS_HEDGE_REALIGNMENT"
    elif annual_due:
        reason = "ANNUAL_REBALANCE"
    else:
        reason = "HOLD"
    if full or not due:
        execution = target
    else:
        execution = dict(existing)
        tactical_weight = sum(execution.pop(ticker, 0.0) for ticker in TACTICAL_TICKERS)
        target_tactical = sum(target.get(ticker, 0.0) for ticker in TACTICAL_TICKERS)
        if tactical_weight > 1e-12 and target_tactical > 1e-12:
            for ticker in TACTICAL_TICKERS:
                ratio = target.get(ticker, 0.0) / target_tactical
                if ratio > 0:
                    execution[ticker] = tactical_weight * ratio
    components = set(existing) | set(execution)
    one_way = _one_way_turnover(existing, execution, components) if due else 0.0
    orders = _individual_orders(existing, execution, components) if due else 0
    if due and orders == 0 and one_way <= 1e-12:
        due, full, one_way = False, False, 0.0
        reason = "ANNUAL_REVIEW" if annual_due else "CONFIRMED_TARGET_STATE"
    return RebalancePlan(
        execution,
        due,
        full,
        reason,
        float(one_way),
        int(orders),
        annual_due,
        signal_year if annual_due else 0,
    )


def calculate_execution_table(
    prices: pd.DataFrame,
    target: dict[str, float],
    portfolio_value: float,
    state: PortfolioState,
    *,
    actionable: bool,
) -> pd.DataFrame:
    rows = []
    for ticker in sorted(set(state.shares) | set(target) | {CASH_ASSET}):
        price = 1.0 if ticker == CASH_ASSET else float(prices[ticker].iloc[-1])
        current = state.cash_balance if ticker == CASH_ASSET else state.shares.get(ticker, 0.0)
        weight = float(target.get(ticker, 0.0))
        estimated = portfolio_value * weight / price
        delta = estimated - current
        action = "HOLD"
        if actionable and abs(delta * price) > 0.005:
            action = "CASH AFTER TRADES" if ticker == CASH_ASSET else ("BUY" if delta > 0 else "SELL")
        rows.append({
            "Ticker": ticker,
            "Price": price,
            "CurrentUnits": current,
            "TargetPct": weight,
            "EstimatedUnits": estimated,
            "DeltaUnits": delta if actionable else 0.0,
            "DeltaValue": delta * price if actionable else 0.0,
            "Action": action,
        })
    return pd.DataFrame(rows)


def calculate_execution_diagnostics(
    table: pd.DataFrame,
    portfolio_value: float,
    current: dict[str, float],
    strategic: dict[str, float],
    destination: dict[str, float],
) -> ExecutionDiagnostics:
    security = table["Ticker"] != CASH_ASSET
    gross = float(np.abs(table.loc[security, "DeltaValue"]).sum())
    return ExecutionDiagnostics(
        advertised_daily_exposure(current),
        advertised_daily_exposure(_with_cash(strategic)),
        advertised_daily_exposure(_with_cash(destination)),
        gross / portfolio_value,
        {bps: gross * bps / 10_000 for bps in TRANSACTION_COST_SCENARIOS_BPS},
    )


def preserve_pending_delivery_plan(
    plan: RebalancePlan, state: PortfolioState, current: dict[str, float]
) -> RebalancePlan:
    if not (
        plan.rebalance_due
        and state.pending_recommendation_date
        and not state.pending_recommendation_notified
        and state.pending_recommendation_fingerprint == STRATEGY_FINGERPRINT
        and _weights_close(
            state.pending_recommendation_weights,
            plan.execution_weights,
            NOTIFICATION_WEIGHT_TOLERANCE,
        )
    ):
        return plan
    retry = dict(state.pending_recommendation_weights)
    components = set(retry) | set(current)
    return replace(
        plan,
        execution_weights=retry,
        reason="PENDING_DELIVERY_RETRY",
        one_way_turnover=_one_way_turnover(current, retry, components),
        individual_orders=_individual_orders(current, retry, components),
    )


def run_strategy(roth_amount: float | None) -> StrategyRun:
    prices = download_market_data(ALL_TICKERS)
    signal_date = pd.Timestamp(prices.index[-1]).normalize()
    fingerprint = market_data_fingerprint(prices)
    state = load_state()
    validate_same_date_data_fingerprint(state, signal_date, fingerprint)
    value, planning = resolve_portfolio_value(roth_amount, state, prices)
    decision = calculate_strategy_decision(prices, state)
    decision = apply_lifecycle_policy(decision, state, value, signal_date)
    _apply_hedge(planning, decision.hedge_state)
    _apply_lifecycle(planning, decision, signal_date)
    current = existing_weights(planning, prices)
    plan = preserve_pending_delivery_plan(
        build_rebalance_plan(current, decision, planning), state, current
    )
    table_target = (
        plan.execution_weights if plan.rebalance_due else _with_cash(decision.target_weights)
    )
    table = calculate_execution_table(
        prices, table_target, value, planning, actionable=plan.rebalance_due
    )
    diagnostics = calculate_execution_diagnostics(
        table, value, current, decision.target_weights, table_target
    )
    contribution_plan = build_contribution_plan(
        state, decision, prices, value, signal_date
    )
    return StrategyRun(
        price_data=prices,
        decision=decision,
        state=state,
        planning_state=planning,
        portfolio_value=value,
        current_weights=current,
        execution_table=table,
        signal_date=signal_date,
        market_data_fingerprint=fingerprint,
        rebalance_plan=plan,
        execution_diagnostics=diagnostics,
        contribution_plan=contribution_plan,
    )


def _pending_matches(run: StrategyRun) -> bool:
    state = run.state
    return (
        bool(state.pending_recommendation_date)
        and state.pending_recommendation_fingerprint == STRATEGY_FINGERPRINT
        and state.pending_recommendation_btal_active == run.decision.hedge_state.btal_active
        and state.pending_recommendation_lifecycle_stage == run.decision.lifecycle_stage
        and state.pending_recommendation_annual_year
        == run.rebalance_plan.annual_rebalance_year
        and _weights_close(
            state.pending_recommendation_weights,
            run.rebalance_plan.execution_weights,
            NOTIFICATION_WEIGHT_TOLERANCE,
        )
    )


def decide_notification(run: StrategyRun) -> NotificationDecision:
    state = run.state
    actionable = run.rebalance_plan.rebalance_due and run.rebalance_plan.individual_orders > 0
    if actionable:
        if not state.pending_recommendation_date:
            return NotificationDecision("ACTION", "NEW_RECOMMENDATION")
        if _pending_matches(run):
            if state.pending_recommendation_notified:
                if run.contribution_plan.notification_due:
                    kind = (
                        "CONTRIBUTION_RETRY"
                        if run.contribution_plan.retry
                        else "CONTRIBUTION"
                    )
                    return NotificationDecision(kind, "+".join(run.contribution_plan.reasons))
                return NotificationDecision(
                    "NONE",
                    "IDENTICAL_PENDING_RECOMMENDATION",
                    state.pending_recommendation_date,
                )
            kind = "UPDATE_RETRY" if state.pending_recommendation_supersedes_date else "RETRY"
            return NotificationDecision(
                kind,
                "UNDELIVERED_PENDING_RECOMMENDATION",
                state.pending_recommendation_date,
                state.pending_recommendation_supersedes_date,
            )
        return NotificationDecision(
            "UPDATE",
            "MATERIAL_RECOMMENDATION_UPDATE",
            state.pending_recommendation_supersedes_date or state.pending_recommendation_date,
        )
    if state.pending_recommendation_date:
        return NotificationDecision(
            "CANCELLATION",
            "PENDING_ACTION_NO_LONGER_REQUIRED",
            state.pending_recommendation_supersedes_date or state.pending_recommendation_date,
        )
    if run.rebalance_plan.annual_rebalance_due:
        return NotificationDecision("ANNUAL_REVIEW", "ANNUAL_ALLOCATION_CONFIRMED")
    if run.contribution_plan.notification_due:
        kind = "CONTRIBUTION_RETRY" if run.contribution_plan.retry else "CONTRIBUTION"
        return NotificationDecision(kind, "+".join(run.contribution_plan.reasons))
    return NotificationDecision("NONE", "HOLD")


def request_notification_resend(run: StrategyRun) -> None:
    """Explicitly re-arm one pending portfolio email without changing its target."""
    if not run.state.pending_recommendation_date:
        raise RuntimeError("There is no pending portfolio recommendation to resend")
    run.state.pending_recommendation_notified = False


def _clear_pending(state: PortfolioState) -> None:
    state.pending_recommendation_date = ""
    state.pending_recommendation_weights = {}
    state.pending_recommendation_btal_active = False
    state.pending_recommendation_lifecycle_stage = ""
    state.pending_recommendation_notified = False
    state.pending_recommendation_supersedes_date = ""
    state.pending_recommendation_fingerprint = ""
    state.pending_recommendation_annual_year = 0


def _clear_pending_contribution(state: PortfolioState) -> None:
    state.pending_contribution_date = ""
    state.pending_contribution_amount = 0.0
    state.pending_contribution_reason = ""
    state.pending_contribution_pullback = False
    state.pending_contribution_drawdown_10 = False
    state.pending_contribution_drawdown_20 = False


def persist_signal_run(run: StrategyRun) -> None:
    state = run.state
    if not state.shares and state.cash_balance == 0:
        state.cash_balance = run.planning_state.cash_balance
    _apply_hedge(state, run.decision.hedge_state)
    _apply_lifecycle(state, run.decision, run.signal_date)
    state.portfolio_value = round(run.portfolio_value, 2)
    state.last_processed_data_fingerprint = run.market_data_fingerprint
    if run.rebalance_plan.reason == "CONFIRMED_TARGET_STATE":
        state.target_weights = dict(run.rebalance_plan.execution_weights)
        state.executed_btal_active = run.decision.hedge_state.btal_active
        state.executed_lifecycle_stage = run.decision.lifecycle_stage
        state.executed_strategy_fingerprint = STRATEGY_FINGERPRINT
    save_state(state)


def prepare_notification_delivery(run: StrategyRun, notification: NotificationDecision) -> None:
    if not notification.should_send:
        return
    state = run.state
    if notification.kind in {"ACTION", "UPDATE"}:
        state.pending_recommendation_date = run.signal_date.date().isoformat()
        state.pending_recommendation_weights = dict(run.rebalance_plan.execution_weights)
        state.pending_recommendation_btal_active = run.decision.hedge_state.btal_active
        state.pending_recommendation_lifecycle_stage = run.decision.lifecycle_stage
        state.pending_recommendation_notified = False
        state.pending_recommendation_supersedes_date = (
            notification.previous_recommendation_date
            if notification.kind == "UPDATE"
            else ""
        )
        state.pending_recommendation_fingerprint = STRATEGY_FINGERPRINT
        state.pending_recommendation_annual_year = run.rebalance_plan.annual_rebalance_year
    elif notification.kind in {"RETRY", "UPDATE_RETRY"}:
        if not state.pending_recommendation_date:
            raise RuntimeError("Cannot retry a missing recommendation")
        state.pending_recommendation_notified = False
    elif notification.kind not in {
        "CANCELLATION",
        "ANNUAL_REVIEW",
        "CONTRIBUTION",
        "CONTRIBUTION_RETRY",
    }:
        raise ValueError("Unsupported notification kind")


def prepare_contribution_delivery(run: StrategyRun) -> None:
    plan = run.contribution_plan
    if not plan.notification_due:
        return
    state = run.state
    if plan.retry:
        if not state.pending_contribution_date:
            raise RuntimeError("Cannot retry a missing contribution notice")
        return
    state.pending_contribution_date = run.signal_date.date().isoformat()
    state.pending_contribution_amount = plan.due_amount
    state.pending_contribution_reason = "+".join(plan.reasons)
    state.pending_contribution_pullback = plan.use_pullback
    state.pending_contribution_drawdown_10 = plan.use_drawdown_10
    state.pending_contribution_drawdown_20 = plan.use_drawdown_20


BENCHMARK_BOOKS = {
    "unlevered_qqq": {MARKET_INDEX: 1.00},
    "qqq60_zroz40": {MARKET_INDEX: 0.60, ZROZ: 0.40},
    "permanent_growth": {
        GROWTH_EQUITY: 0.40,
        DBMF: 0.20,
        ZROZ: 0.20,
        LEVERAGED_GOLD: 0.20,
    },
    "permanent_10pct_btal": {
        GROWTH_EQUITY: 0.30,
        CRISIS_HEDGE: 0.10,
        DBMF: 0.20,
        ZROZ: 0.20,
        LEVERAGED_GOLD: 0.20,
    },
}
LIVE_BOOK = {
    GROWTH_EQUITY: 0.40,
    DBMF: 0.20,
    ZROZ: 0.20,
    LEVERAGED_GOLD: 0.20,
}


def build_performance_diagnostics(run: StrategyRun) -> dict[str, object]:
    """Build measurement-only telemetry that can never block a decision."""
    try:
        report = performance.build_performance_report(
            run.price_data,
            reference_books=BENCHMARK_BOOKS,
            live_book=LIVE_BOOK,
            index_ticker=MARKET_INDEX,
            confirmation_ticker=CONFIRMATION_INDEX,
            growth=GROWTH_EQUITY,
            hedge=CRISIS_HEDGE,
            cash_proxy=TREASURY_RESERVE,
            sma_window=SMA_WINDOW,
            short_sma_window=SHORT_TREND_WINDOW,
            entry_ratio=core.EXTREME_BEAR_RATIO,
            exit_ratio=core.RECOVERY_RATIO,
            multipliers=ADVERTISED_DAILY_MULTIPLIERS,
            current_weights=run.current_weights,
        )
    except Exception as error:  # noqa: BLE001 - diagnostics cannot halt trading
        logger.warning("Performance diagnostics unavailable: %s", error)
        return {"available": False, "reason": type(error).__name__}

    return {
        "available": True,
        "note": (
            "Paper track from price history assuming a clean annually rebalanced "
            "book. Not the dollar-weighted return of the account."
        ),
        "sample_sessions": report.sample_sessions,
        "paper_track": report.paper_track,
        "realized_volatility": report.realized_volatility,
        "sleeve_risk_contribution": report.sleeve_risk_contribution,
        "estimated_short_rate": report.estimated_short_rate,
        "financing_drag": report.financing_drag,
    }


def build_decision_audit(
    run: StrategyRun, notification: NotificationDecision, delivery_status: str
) -> dict[str, object]:
    payload = {
        "schema_version": DECISION_AUDIT_SCHEMA_VERSION,
        "strategy": {"revision": STRATEGY_REVISION, "fingerprint": STRATEGY_FINGERPRINT},
        "signal": {
            "date": run.signal_date.date().isoformat(),
            "data_fingerprint": run.market_data_fingerprint,
            "qqq_close": run.decision.qqq_close,
            "qqq_sma_200": run.decision.qqq_sma_200,
            "trend_positive": run.decision.trend_positive,
            "btal_active": run.decision.hedge_state.btal_active,
            "btal_entry_streak": run.decision.hedge_state.entry_streak,
            "btal_exit_streak": run.decision.hedge_state.exit_streak,
            "transition": run.decision.transition_reason,
            "transition_path": list(run.decision.transition_path),
            "processed_dates": list(run.decision.processed_signal_dates),
            "qqq_sma_50": run.decision.qqq_sma_50,
            "qqq_momentum_252": run.decision.qqq_momentum_252,
            "spy_close": run.decision.spy_close,
            "spy_sma_200": run.decision.spy_sma_200,
            "spy_sma_50": run.decision.spy_sma_50,
            "extreme_bearish": run.decision.extreme_bearish,
            "recovery_confirmed": run.decision.recovery_confirmed,
        },
        "portfolio": {
            "value": run.portfolio_value,
            "current_weights": run.current_weights,
            "strategic_weights": run.decision.target_weights,
            "execution_weights": run.rebalance_plan.execution_weights,
            "rebalance": asdict(run.rebalance_plan),
            "lifecycle_stage": run.decision.lifecycle_stage,
        },
        "lifecycle": {
            "stage": run.decision.lifecycle_stage,
            "reason": run.decision.lifecycle_reason,
            "value_stage": run.decision.lifecycle_value_stage,
            "age_stage": run.decision.lifecycle_age_stage,
            "advanced": run.decision.lifecycle_stage_advanced,
            "estimated_investor_age": run.decision.estimated_investor_age,
        },
        "exposure": {
            "current": run.execution_diagnostics.current_daily_exposure,
            "strategic": run.execution_diagnostics.strategic_daily_exposure,
            "destination": run.execution_diagnostics.destination_daily_exposure,
            "gross_security_trade_fraction": (
                run.execution_diagnostics.gross_security_trade_fraction
            ),
            "estimated_costs_by_bps": {
                str(bps): amount
                for bps, amount in sorted(run.execution_diagnostics.estimated_costs.items())
            },
        },
        "contribution": asdict(run.contribution_plan),
        "contribution_policy": {"revision": contribution.POLICY_REVISION},
        "notification": asdict(notification),
        "delivery_status": delivery_status,
    }
    payload["decision_hash"] = canonical_sha256(payload)
    # Diagnostics are outside the decision hash so changing measurements cannot
    # make an unchanged trading decision appear different.
    payload["diagnostics"] = build_performance_diagnostics(run)
    return payload


def write_decision_audit(
    run: StrategyRun,
    notification: NotificationDecision,
    delivery_status: str,
) -> dict[str, object]:
    payload = build_decision_audit(run, notification, delivery_status)
    temporary = DECISION_AUDIT_FILE.with_suffix(f"{DECISION_AUDIT_FILE.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, DECISION_AUDIT_FILE)
    return payload


def persist_notification_delivery(
    run: StrategyRun,
    notification: NotificationDecision,
    *,
    delivered_decision_hash: str | None = None,
) -> None:
    if not notification.should_send:
        raise ValueError("A NONE notification cannot be delivered")
    state = run.state
    delivered_decision_hash = delivered_decision_hash or build_decision_audit(
        run, notification, "DELIVERED"
    )["decision_hash"]
    if not _is_sha256(delivered_decision_hash):
        raise ValueError("Delivered decision hash is invalid")
    if notification.kind in {"ACTION", "UPDATE", "RETRY", "UPDATE_RETRY"}:
        if not state.pending_recommendation_date:
            raise RuntimeError("Delivered action lacks pending state")
        state.pending_recommendation_notified = True
        state.pending_recommendation_supersedes_date = ""
    elif notification.kind == "CANCELLATION":
        _clear_pending(state)
    elif notification.kind == "ANNUAL_REVIEW":
        state.last_completed_annual_rebalance_year = run.rebalance_plan.annual_rebalance_year
    elif notification.kind not in {"CONTRIBUTION", "CONTRIBUTION_RETRY"}:
        raise ValueError("Unsupported notification kind")
    if run.contribution_plan.notification_due:
        if not state.pending_contribution_date:
            raise RuntimeError("Delivered contribution notice lacks pending state")
        state.contribution_released_amount = min(
            state.contribution_budget,
            round(
                state.contribution_released_amount + state.pending_contribution_amount,
                2,
            ),
        )
        state.contribution_pullback_used |= state.pending_contribution_pullback
        state.contribution_drawdown_10_used |= state.pending_contribution_drawdown_10
        state.contribution_drawdown_20_used |= state.pending_contribution_drawdown_20
        state.last_contribution_notice_date = state.pending_contribution_date
        _clear_pending_contribution(state)
    state.last_delivered_decision_hash = str(delivered_decision_hash)
    state.last_delivered_signal_date = run.signal_date.date().isoformat()
    state.last_delivered_notification_kind = notification.kind
    save_state(state)


def validate_execution_confirmation(state: PortfolioState, signal_date: str) -> bool:
    if not state.pending_recommendation_date:
        raise RuntimeError("There is no pending recommendation to confirm")
    if signal_date == state.pending_recommendation_date:
        return state.pending_recommendation_fingerprint == STRATEGY_FINGERPRINT
    supplied = _parse_date(signal_date, "executed_signal_date")
    pending = _parse_date(state.pending_recommendation_date, "pending_recommendation_date")
    if supplied and pending and supplied < pending:
        return False
    raise RuntimeError(f"Expected signal {state.pending_recommendation_date}, got {signal_date}")


def confirm_execution(shares: dict[str, float], cash: float, signal_date: str) -> PortfolioState:
    state = load_state()
    matches = validate_execution_confirmation(state, signal_date)
    state.shares = dict(shares)
    state.cash_balance = float(cash)
    if matches:
        state.target_weights = dict(state.pending_recommendation_weights)
        state.executed_btal_active = state.pending_recommendation_btal_active
        state.executed_lifecycle_stage = state.pending_recommendation_lifecycle_stage
        state.executed_strategy_fingerprint = state.pending_recommendation_fingerprint
        if state.pending_recommendation_annual_year:
            state.last_completed_annual_rebalance_year = state.pending_recommendation_annual_year
        _clear_pending(state)
    else:
        state.target_weights = {}
        state.executed_btal_active = shares.get(CRISIS_HEDGE, 0.0) > 0
        state.executed_lifecycle_stage = state.lifecycle_stage
        state.executed_strategy_fingerprint = ""
    save_state(state)
    return state


def sync_holdings(shares: dict[str, float], cash: float) -> PortfolioState:
    state = load_state()
    if state.pending_recommendation_date:
        raise RuntimeError("Cannot sync while a recommendation is pending")
    state.shares = dict(shares)
    state.cash_balance = float(cash)
    state.executed_btal_active = shares.get(CRISIS_HEDGE, 0.0) > 0
    save_state(state)
    return state


def configure_contribution_plan(year: int, budget: float) -> PortfolioState:
    if not STATE_FILE.exists():
        raise RuntimeError("Initialize or synchronize portfolio holdings first")
    if not isinstance(year, int) or isinstance(year, bool) or year < 2000:
        raise ValueError("Contribution year is invalid")
    current_year = datetime.now(NEW_YORK).year
    if year != current_year:
        raise ValueError(
            f"Contribution year must be the current New York year ({current_year})"
        )
    if not _is_number(budget, positive=True):
        raise ValueError("Contribution budget must be positive and finite")
    state = load_state()
    if state.pending_contribution_date:
        raise RuntimeError("Cannot replace a contribution plan while delivery is pending")
    state.contribution_plan_year = year
    state.contribution_policy_revision = contribution.POLICY_REVISION
    state.contribution_budget = round(float(budget), 2)
    state.contribution_released_amount = 0.0
    state.contribution_pullback_used = False
    state.contribution_drawdown_10_used = False
    state.contribution_drawdown_20_used = False
    state.last_contribution_notice_date = ""
    _clear_pending_contribution(state)
    save_state(state)
    return state


def disable_contribution_plan() -> PortfolioState:
    if not STATE_FILE.exists():
        raise RuntimeError("No portfolio state exists")
    state = load_state()
    if state.pending_contribution_date:
        raise RuntimeError("Cannot disable a contribution plan while delivery is pending")
    state.contribution_plan_year = 0
    state.contribution_policy_revision = ""
    state.contribution_budget = 0.0
    state.contribution_released_amount = 0.0
    state.contribution_pullback_used = False
    state.contribution_drawdown_10_used = False
    state.contribution_drawdown_20_used = False
    state.last_contribution_notice_date = ""
    _clear_pending_contribution(state)
    save_state(state)
    return state


def log_decision(run: StrategyRun) -> None:
    logger.info(
        "signal=%s hedge=%s extreme=%s recovery=%s transition=%s lifecycle=%s "
        "rebalance=%s reason=%s turnover=%.4f strategy=%s data=%s",
        run.signal_date.date(),
        "ACTIVE" if run.decision.hedge_state.btal_active else "OFF",
        run.decision.extreme_bearish,
        run.decision.recovery_confirmed,
        run.decision.transition_reason,
        run.decision.lifecycle_stage,
        run.rebalance_plan.rebalance_due,
        run.rebalance_plan.reason,
        run.rebalance_plan.one_way_turnover,
        STRATEGY_FINGERPRINT[:12],
        run.market_data_fingerprint[:12],
    )


DASHBOARD_WIDTH = 74
_LIFECYCLE_CEILING_LABEL = {
    stage: ("base target" if ceiling is None else f"{ceiling:.2f}x ceiling")
    for stage, ceiling in LIFECYCLE_EXPOSURE_CEILINGS.items()
}
_REASON_LABELS = {
    "STRATEGY_REVISION_TRANSITION": "portfolio initialization",
    "LIFECYCLE_STAGE_ADVANCE": "lifecycle risk reduction",
    "MISSING_TACTICAL_POSITION": "TQQQ position missing",
    "CRISIS_HEDGE_REALIGNMENT": "crisis hedge needs to be realigned",
    "LEGACY_UPRO_LIQUIDATION": "legacy UPRO position must be replaced",
    "EXTREME_HEDGE_ENTRY": "extreme-bear hedge activated",
    "EXTREME_HEDGE_EXIT": "market recovery confirmed",
    "ANNUAL_REBALANCE": "annual rebalance",
    "ANNUAL_REVIEW": "annual review complete",
    "CONFIRMED_TARGET_STATE": "confirmed holdings match",
    "HOLD": "no change",
    "CALENDAR_MILESTONE": "scheduled contribution",
    "QQQ_PULLBACK_ABOVE_SMA200": "bull-market pullback",
    "QQQ_DRAWDOWN_10": "10% QQQ drawdown",
    "QQQ_DRAWDOWN_20_DEPLOY_REMAINDER": "20% QQQ drawdown",
}


def _visible_execution_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Rows worth showing: any non-cash ticker that is held or targeted."""
    return table[
        (table["Ticker"] != CASH_ASSET)
        & (
            (table["Action"] != "HOLD")
            | (table["TargetPct"] > 1e-12)
            | (table["CurrentUnits"] > 1e-12)
        )
    ]


def _dashboard_rule(character: str = "-") -> str:
    return character * DASHBOARD_WIDTH


def _dashboard_field(label: str, value: str) -> str:
    return f"{label:<14}{value}"


def _reason_label(reason: str) -> str:
    return _REASON_LABELS.get(reason, reason.replace("_", " ").lower())


def _contribution_reason(plan: ContributionPlan) -> str:
    return " + ".join(_reason_label(reason) for reason in plan.reasons)


def build_dashboard(run: StrategyRun) -> str:
    decision = run.decision
    plan = run.rebalance_plan
    trend_gap = decision.qqq_close / decision.qqq_sma_200 - 1.0
    spy_long_gap = decision.spy_close / decision.spy_sma_200 - 1.0
    spy_short_gap = decision.spy_close / decision.spy_sma_50 - 1.0
    contribution_plan = run.contribution_plan
    contribution_due = contribution_plan.notification_due
    status = "ACTION REQUIRED" if plan.rebalance_due or contribution_due else "NO ACTION"
    status_reason = (
        _reason_label(plan.reason)
        if plan.rebalance_due or not contribution_due
        else _contribution_reason(contribution_plan)
    )

    lines = [
        _dashboard_rule("="),
        "ROTH IRA PORTFOLIO UPDATE".center(DASHBOARD_WIDTH).rstrip(),
        _dashboard_rule("="),
        _dashboard_field("Signal close", str(run.signal_date.date())),
        _dashboard_field("Value", f"${run.portfolio_value:,.2f}"),
        _dashboard_field("Status", status),
        _dashboard_field("Reason", status_reason),
        "",
        "WHAT TO DO",
        _dashboard_rule(),
    ]
    if contribution_due:
        lines.append(
            _dashboard_field("Deposit", f"${contribution_plan.due_amount:,.2f} now")
        )
    lines.append(
        _dashboard_field(
            "Portfolio",
            "Place the orders below next session" if plan.rebalance_due else "No trades",
        )
    )

    lines.extend([
        "",
        "TARGET",
        _dashboard_rule(),
        _dashboard_field(
            "Crisis hedge",
            "ACTIVE: TQQQ 30% + BTAL 10%"
            if decision.hedge_state.btal_active
            else "OFF: TQQQ 40%",
        ),
        _dashboard_field(
            "Allocation",
            "  ".join(
                f"{ticker} {weight:.0%}"
                for ticker, weight in decision.target_weights.items()
                if weight > 1e-12
            ),
        ),
        "",
        "SIGNAL",
        _dashboard_rule(),
        _dashboard_field(
            "QQQ / SMA200",
            f"{trend_gap:+.1%}  |  hedge entry at -8.0%",
        ),
        _dashboard_field(
            "SPY confirms",
            f"{spy_long_gap:+.1%} vs SMA200  |  {spy_short_gap:+.1%} vs SMA50",
        ),
        _dashboard_field(
            "Signal state",
            f"entry {decision.hedge_state.entry_streak}/2  |  "
            f"exit {decision.hedge_state.exit_streak}/2",
        ),
        _dashboard_field(
            "Lifecycle",
            f"{decision.lifecycle_stage} "
            f"({_LIFECYCLE_CEILING_LABEL.get(decision.lifecycle_stage, 'unknown')})",
        ),
    ])
    if len(decision.processed_signal_dates) > 1:
        lines.append(
            _dashboard_field(
                "Replay", f"{len(decision.processed_signal_dates)} closes processed in order"
            )
        )

    if contribution_plan.enabled:
        lines.extend([
            "",
            "CONTRIBUTIONS",
            _dashboard_rule(),
            _dashboard_field(
                str(contribution_plan.year),
                f"${contribution_plan.released_amount:,.2f} of "
                f"${contribution_plan.budget:,.2f} already notified",
            ),
        ])
        if contribution_plan.year != run.signal_date.year:
            lines.append(_dashboard_field("Status", "Plan is not active for this year"))
        else:
            lines.extend([
                _dashboard_field(
                    "QQQ drawdown", f"{contribution_plan.qqq_drawdown_63:.1%} from 63-session high"
                ),
                _dashboard_field(
                    "Setup",
                    "pullback above SMA200" if contribution_plan.bull_pullback else "no pullback trigger",
                ),
            ])
        if contribution_plan.notification_due:
            lines.extend([
                _dashboard_field(
                    "Why now",
                    f"${contribution_plan.due_amount:,.2f}  "
                    f"({_contribution_reason(contribution_plan)})",
                ),
                f"{'Ticker':<10}{'Buy dollars':>16}{'Est. units':>18}",
                _dashboard_rule(),
            ])
            for ticker, dollars in contribution_plan.allocation_dollars.items():
                lines.append(
                    f"{ticker:<10}${dollars:>15,.2f}"
                    f"{contribution_plan.estimated_units[ticker]:>18,.4f}"
                )

    if plan.rebalance_due:
        trades = _visible_execution_rows(run.execution_table)
        trades = trades[trades["Action"] != "HOLD"]
        lines.extend([
            "",
            f"PORTFOLIO ORDERS  ({run.signal_date.date()} close estimates)",
            _dashboard_rule(),
            f"{'Action':<8}{'Ticker':<8}{'Target':>8} {'Trade $':>13}{'Est. shares':>15}",
            _dashboard_rule(),
        ])
        for _, row in trades.iterrows():
            lines.append(
                f"{row['Action']:<8}{row['Ticker']:<8}{row['TargetPct']:>8.0%} "
                f"${abs(row['DeltaValue']):>12,.2f}{abs(row['DeltaUnits']):>15,.4f}"
            )
        lines.extend([
            _dashboard_rule(),
            "Recalculate shares at next-session prices, execute, then confirm holdings.",
        ])

    lines.append(_dashboard_rule("="))
    return "\n".join(lines)


def notification_subject(run: StrategyRun, notification: NotificationDecision) -> str:
    label = {
        "ACTION": "Action Required",
        "UPDATE": "Action Updated",
        "RETRY": "Action Required (Retry)",
        "UPDATE_RETRY": "Action Updated (Retry)",
        "CANCELLATION": "Action Cancelled",
        "ANNUAL_REVIEW": "Annual Review",
        "CONTRIBUTION": "Contribution Due",
        "CONTRIBUTION_RETRY": "Contribution Due (Retry)",
    }.get(notification.kind)
    if label is None:
        raise ValueError("A NONE notification has no subject")
    if notification.kind in {"RETRY", "UPDATE_RETRY"}:
        subject_date = notification.previous_recommendation_date
    elif notification.kind == "CONTRIBUTION_RETRY":
        subject_date = run.state.pending_contribution_date
    else:
        subject_date = run.signal_date.date().isoformat()
    return f"ROTH IRA {label} - {subject_date}"


def notification_text_body(
    run: StrategyRun, notification: NotificationDecision, dashboard: str
) -> str:
    if notification.kind == "CANCELLATION":
        prefix = (
            f"The action for signal {notification.previous_recommendation_date} "
            "is no longer required. Do not execute it.\n\n"
        )
    elif notification.kind == "UPDATE":
        prefix = (
            "This recommendation replaces the action for signal "
            f"{notification.previous_recommendation_date}.\n\n"
        )
    elif notification.kind == "RETRY":
        prefix = (
            "Delivery of the action for signal "
            f"{notification.previous_recommendation_date} is being retried.\n\n"
        )
    elif notification.kind == "UPDATE_RETRY":
        prefix = (
            "Delivery of the updated action for signal "
            f"{notification.previous_recommendation_date} is being retried; "
            "it replaces the previously delivered action for signal "
            f"{notification.supersedes_recommendation_date}.\n\n"
        )
    elif notification.kind == "ACTION":
        prefix = "A portfolio update is required.\n\n"
    elif notification.kind == "ANNUAL_REVIEW":
        prefix = "Your annual allocation review is complete; no trades are required.\n\n"
    elif notification.kind == "CONTRIBUTION":
        prefix = "A scheduled or accelerated Roth IRA contribution is due.\n\n"
    elif notification.kind == "CONTRIBUTION_RETRY":
        prefix = "Delivery of your Roth IRA contribution notice is being retried.\n\n"
    else:
        raise ValueError("A NONE notification has no body")
    return prefix + dashboard


NOTIFICATION_STATUS_LABELS = {
    "ACTION": "ACTION REQUIRED",
    "UPDATE": "UPDATED ACTION REQUIRED",
    "RETRY": "ACTION DELIVERY RETRY",
    "UPDATE_RETRY": "UPDATED ACTION DELIVERY RETRY",
    "CANCELLATION": "PREVIOUS ACTION CANCELLED",
    "ANNUAL_REVIEW": "ANNUAL REVIEW COMPLETE",
    "CONTRIBUTION": "ROTH CONTRIBUTION DUE",
    "CONTRIBUTION_RETRY": "ROTH CONTRIBUTION DELIVERY RETRY",
}
NOTIFICATION_ACCENTS = {
    "ACTION": "#b45309",
    "UPDATE": "#b45309",
    "RETRY": "#b45309",
    "UPDATE_RETRY": "#b45309",
    "CANCELLATION": "#6b7280",
    "ANNUAL_REVIEW": "#15803d",
    "CONTRIBUTION": "#1d4ed8",
    "CONTRIBUTION_RETRY": "#1d4ed8",
}
_ACTION_COLORS = {"BUY": "#15803d", "SELL": "#b91c1c"}

_EMAIL_BASE = "font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif"
_CELL = "padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:14px"
_HEAD_CELL = (
    "padding:9px 12px;border-bottom:2px solid #d1d5db;font-size:12px;"
    "letter-spacing:.06em;text-transform:uppercase;color:#6b7280;font-weight:600"
)


def _email_metric(label: str, value: str) -> str:
    return (
        '<td style="padding:10px 14px;border:1px solid #e5e7eb;'
        'background:#f9fafb;vertical-align:top">'
        f'<div style="font-size:12px;letter-spacing:.04em;text-transform:uppercase;'
        f'color:#6b7280">{label}</div>'
        f'<div style="font-size:16px;color:#111827;padding-top:4px">{value}</div></td>'
    )


def build_email_html(run: StrategyRun, notification: NotificationDecision) -> str:
    decision = run.decision
    plan = run.rebalance_plan
    visible = _visible_execution_rows(run.execution_table)
    hedge_status = "TQQQ 30% + BTAL 10%" if decision.hedge_state.btal_active else "TQQQ 40%"
    status = NOTIFICATION_STATUS_LABELS[notification.kind]
    accent = NOTIFICATION_ACCENTS[notification.kind]

    target_rows = "".join(
        "<tr>"
        f'<td style="{_CELL};font-weight:600;color:#111827">{ticker}</td>'
        f'<td style="{_CELL};text-align:right;font-weight:600;color:#111827">'
        f'{weight:.0%}</td>'
        "</tr>"
        for ticker, weight in decision.target_weights.items()
        if weight > 1e-12
    )
    order_rows = "".join(
        "<tr>"
        f'<td style="{_CELL};font-weight:700;'
        f'color:{_ACTION_COLORS.get(row["Action"], "#111827")}">{row["Action"]}</td>'
        f'<td style="{_CELL};font-weight:600;color:#111827">{row["Ticker"]}</td>'
        f'<td style="{_CELL};text-align:right;color:#374151">{row["TargetPct"]:.0%}</td>'
        f'<td style="{_CELL};text-align:right;color:#374151">'
        f'${abs(row["DeltaValue"]):,.2f}</td>'
        f'<td style="{_CELL};text-align:right;color:#374151">'
        f'{abs(row["DeltaUnits"]):,.4f}</td>'
        "</tr>"
        for _, row in visible[visible["Action"] != "HOLD"].iterrows()
    )
    metrics = (
        "<tr>"
        + _email_metric("Signal close", str(run.signal_date.date()))
        + _email_metric("Account value", f"${run.portfolio_value:,.2f}")
        + "</tr><tr>"
        + _email_metric("Tactical sleeve", hedge_status)
        + _email_metric("Lifecycle", decision.lifecycle_stage)
        + "</tr>"
    )

    if notification.kind == "CANCELLATION":
        action_text = "Do not execute the previous recommendation. It has been cancelled."
    elif plan.rebalance_due and run.contribution_plan.notification_due:
        action_text = (
            f"1. Deposit ${run.contribution_plan.due_amount:,.2f}. "
            "2. Place both the portfolio orders and contribution buys below "
            "using next-session prices."
        )
    elif plan.rebalance_due:
        action_text = "Place the portfolio orders below using next-session prices."
    elif run.contribution_plan.notification_due:
        action_text = (
            f"Deposit ${run.contribution_plan.due_amount:,.2f} and place the "
            "contribution buys below using next-session prices."
        )
    else:
        action_text = "No portfolio trades are required."

    orders_html = ""
    if plan.rebalance_due:
        orders_html = f"""
<tr><td style="padding:4px 24px 18px">
  <div style="font-size:17px;font-weight:700;color:#111827;padding-bottom:9px">
    Portfolio orders</div>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
   style="border-collapse:collapse;border:1px solid #e5e7eb">
    <tr style="background:#f9fafb">
      <th style="{_HEAD_CELL};text-align:left">Action</th>
      <th style="{_HEAD_CELL};text-align:left">Ticker</th>
      <th style="{_HEAD_CELL};text-align:right">Target</th>
      <th style="{_HEAD_CELL};text-align:right">Trade value</th>
      <th style="{_HEAD_CELL};text-align:right">Est. shares</th>
    </tr>{order_rows}
  </table>
  <div style="font-size:14px;color:#374151;padding-top:9px;line-height:1.5">
    Share quantities use the signal close. Recalculate them at the price you can execute,
    then confirm the completed holdings and remaining cash.
  </div>
</td></tr>"""

    contribution_html = ""
    if run.contribution_plan.notification_due:
        contribution_rows = "".join(
            "<tr>"
            f'<td style="{_CELL};font-weight:600">{ticker}</td>'
            f'<td style="{_CELL};text-align:right">${dollars:,.2f}</td>'
            f'<td style="{_CELL};text-align:right">'
            f'{run.contribution_plan.estimated_units[ticker]:,.4f}</td></tr>'
            for ticker, dollars in run.contribution_plan.allocation_dollars.items()
        )
        contribution_html = f"""
<tr><td style="padding:4px 24px 18px">
  <div style="font-size:17px;font-weight:700;color:#1d4ed8;padding-bottom:8px">
    Contribution: ${run.contribution_plan.due_amount:,.2f}</div>
  <div style="font-size:14px;color:#374151;padding-bottom:9px">
    {_contribution_reason(run.contribution_plan)} &middot; QQQ drawdown
    {run.contribution_plan.qqq_drawdown_63:.1%} from its 63-session high</div>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
   style="border-collapse:collapse;border:1px solid #e5e7eb">
    <tr style="background:#f9fafb">
      <th style="{_HEAD_CELL};text-align:left">Ticker</th>
      <th style="{_HEAD_CELL};text-align:right">Buy dollars</th>
      <th style="{_HEAD_CELL};text-align:right">Est. units</th>
    </tr>{contribution_rows}
  </table>
</td></tr>"""
    return f"""<!doctype html>
<html><body style="margin:0;padding:24px 12px;background:#f3f4f6;{_EMAIL_BASE}">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"
 style="max-width:680px;margin:0 auto;background:#ffffff;border:1px solid #e5e7eb">
<tr><td style="padding:20px 24px;border-bottom:3px solid {accent}">
  <div style="font-size:21px;font-weight:600;color:{accent};padding-top:6px">{status}</div>
  <div style="font-size:15px;color:#374151;padding-top:5px">
    {_reason_label(plan.reason if plan.rebalance_due else notification.reason)}</div>
</td></tr>
<tr><td style="padding:18px 24px 8px">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
   style="border-collapse:collapse">{metrics}</table>
</td></tr>
<tr><td style="padding:8px 24px 18px">
  <div style="padding:14px 16px;background:#fff7ed;border-left:4px solid {accent};
   font-size:16px;color:#111827;line-height:1.55;font-weight:600">{action_text}</div>
</td></tr>
<tr><td style="padding:0 24px 18px">
  <div style="font-size:17px;font-weight:700;color:#111827;padding-bottom:9px">
    Target allocation</div>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
   style="border-collapse:collapse;border:1px solid #e5e7eb">
    <tr style="background:#f9fafb">
      <th style="{_HEAD_CELL};text-align:left">Holding</th>
      <th style="{_HEAD_CELL};text-align:right">Weight</th>
    </tr>{target_rows}
  </table>
</td></tr>
<tr><td style="padding:0 24px 18px">
  <div style="font-size:17px;font-weight:700;color:#111827;padding-bottom:7px">
    Why this signal</div>
  <div style="font-size:15px;color:#374151;line-height:1.55">
    QQQ is <strong>{decision.qqq_close / decision.qqq_sma_200 - 1:+.1%}</strong>
    versus its 200-day average (BTAL entry threshold: -8.0%). SPY is
    <strong>{decision.spy_close / decision.spy_sma_200 - 1:+.1%}</strong>
    versus its 200-day average and
    <strong>{decision.spy_close / decision.spy_sma_50 - 1:+.1%}</strong>
    versus its 50-day average.
  </div>
</td></tr>
{contribution_html}
{orders_html}
</table></body></html>"""


def send_email(subject: str, text_body: str, html_body: str) -> None:
    address = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("RECEIVER_EMAIL")
    if not all((address, password, recipient)):
        raise RuntimeError(
            "GMAIL_ADDRESS, GMAIL_APP_PASSWORD, and RECEIVER_EMAIL are required "
            "only when a portfolio notification is due"
        )
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = address
    message["To"] = recipient
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(address, password)
            server.send_message(message)
    except Exception:
        logger.exception("Email delivery failed")
        raise
    logger.info("Email sent")


def parse_executed_shares(entries: list[str] | None) -> tuple[dict[str, float] | None, float]:
    if entries is None:
        return None, 0.0
    holdings: dict[str, float] = {}
    cash: float | None = None
    for entry in entries:
        name, separator, value_text = entry.partition("=")
        name = name.strip().upper()
        if separator != "=":
            raise ValueError(f"Invalid executed holding: {entry!r}")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value: {entry!r}") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Value must be nonnegative and finite: {entry!r}")
        if name == CASH_ASSET:
            if cash is not None:
                raise ValueError("CASH was supplied more than once")
            cash = value
        elif name not in TRADED_TICKERS or name in holdings:
            raise ValueError(f"Invalid or duplicate holding: {entry!r}")
        else:
            holdings[name] = value
    if cash is None:
        raise ValueError("Complete holdings must include CASH, even when CASH=0")
    return holdings, cash


def parse_signal_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError("--executed-signal-date must be a valid date") from exc


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROTH IRA TQQQ/BTAL crisis-hedge engine")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--confirm-execution", action="store_true")
    mode.add_argument("--sync-holdings", action="store_true")
    mode.add_argument("--configure-contributions", action="store_true")
    mode.add_argument("--disable-contributions", action="store_true")
    mode.add_argument("--resend-notification", action="store_true")
    parser.add_argument(
        "--test", action="store_true",
        help="Generate a report without state, email, audit, or log persistence",
    )
    parser.add_argument("--roth-amount", type=float, default=None)
    parser.add_argument("--executed-shares", nargs="+", metavar="TICKER=SHARES")
    parser.add_argument("--executed-signal-date", metavar="YYYY-MM-DD")
    parser.add_argument("--contribution-budget", type=float)
    parser.add_argument("--contribution-year", type=int)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    configure_logging(persist_log=not args.test)
    validate_configuration()
    if args.roth_amount is not None and (
        not np.isfinite(args.roth_amount) or args.roth_amount <= 0
    ):
        parser.error("--roth-amount must be positive and finite")
    if args.contribution_budget is not None and (
        not np.isfinite(args.contribution_budget) or args.contribution_budget <= 0
    ):
        parser.error("--contribution-budget must be positive and finite")
    try:
        executed_shares, executed_cash = parse_executed_shares(args.executed_shares)
        executed_signal_date = parse_signal_date(args.executed_signal_date)
    except ValueError as exc:
        parser.error(str(exc))

    if args.configure_contributions:
        if (
            args.test
            or args.roth_amount is not None
            or executed_shares is not None
            or executed_signal_date is not None
            or args.contribution_budget is None
        ):
            parser.error(
                "--configure-contributions requires only --contribution-budget "
                "and optional --contribution-year"
            )
        contribution_year = args.contribution_year or datetime.now(NEW_YORK).year
        try:
            state = configure_contribution_plan(
                contribution_year, args.contribution_budget
            )
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        print(
            f"Contribution plan configured for {state.contribution_plan_year}: "
            f"${state.contribution_budget:,.2f} remaining budget."
        )
        return
    if args.disable_contributions:
        if any(
            (
                args.test,
                args.roth_amount is not None,
                executed_shares is not None,
                executed_signal_date is not None,
                args.contribution_budget is not None,
                args.contribution_year is not None,
            )
        ):
            parser.error("--disable-contributions cannot be combined with other inputs")
        disable_contribution_plan()
        print("Contribution plan disabled.")
        return
    if args.contribution_budget is not None or args.contribution_year is not None:
        parser.error("Contribution inputs require --configure-contributions")

    if args.resend_notification and (
        args.test
        or args.roth_amount is not None
        or executed_shares is not None
        or executed_signal_date is not None
    ):
        parser.error("--resend-notification cannot be combined with other inputs")

    if args.confirm_execution:
        if args.test or args.roth_amount is not None:
            parser.error("--confirm-execution cannot be combined with --test or --roth-amount")
        if executed_shares is None or executed_signal_date is None:
            parser.error("--confirm-execution requires --executed-shares and --executed-signal-date")
        confirm_execution(executed_shares, executed_cash, executed_signal_date)
        print(f"Confirmed execution for signal {executed_signal_date}.")
        return
    if args.sync_holdings:
        if args.test or args.roth_amount is not None or executed_signal_date:
            parser.error("--sync-holdings cannot be combined with other modes or initialization")
        if executed_shares is None:
            parser.error("--sync-holdings requires --executed-shares")
        sync_holdings(executed_shares, executed_cash)
        print("Broker holdings synchronized.")
        return
    if executed_shares is not None or executed_signal_date is not None:
        parser.error("Execution fields require --confirm-execution or --sync-holdings")

    run = run_strategy(args.roth_amount)
    if args.resend_notification:
        try:
            request_notification_resend(run)
        except RuntimeError as exc:
            parser.error(str(exc))
    dashboard = build_dashboard(run)
    print(dashboard)
    notification = decide_notification(run)
    if args.test:
        return

    log_decision(run)
    logger.info("notification kind=%s reason=%s", notification.kind, notification.reason)
    prepare_notification_delivery(run, notification)
    prepare_contribution_delivery(run)
    persist_signal_run(run)
    audit = write_decision_audit(
        run, notification, "STAGED" if notification.should_send else "NOT_REQUIRED"
    )
    if notification.should_send:
        send_email(
            notification_subject(run, notification),
            notification_text_body(run, notification, dashboard),
            build_email_html(run, notification),
        )
        persist_notification_delivery(
            run, notification, delivered_decision_hash=str(audit["decision_hash"])
        )
        write_decision_audit(run, notification, "DELIVERED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("ROTH IRA engine failed")
        sys.exit(1)

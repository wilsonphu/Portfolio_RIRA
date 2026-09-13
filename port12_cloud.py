#!/usr/bin/env python3
"""Stateful Roth allocator with a 40% TQQQ/UPRO equity router.

Completed QQQ closes drive one decision: TQQQ after two closes above SMA200,
UPRO immediately after a failed close. DBMF, ZROZ, and UGL remain 20% each.
Confirmed broker shares and cash are always the source of current weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import shutil
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


# Configuration and universe
MARKET_INDEX = core.QQQ
GROWTH_EQUITY = core.TQQQ
DEFENSIVE_EQUITY = core.UPRO
DBMF = core.DBMF
ZROZ = core.ZROZ
LEVERAGED_GOLD = core.UGL
TREASURY_RESERVE = "SGOV"
CASH_ASSET = core.CASH

LEGACY_HOLDINGS = frozenset(
    {"SOXL", "QLD", "GLD", "TECL", "SPMO", "USD", "QQQM", "GLDM", "SMH"}
)
STRATEGIC_TICKERS = (
    GROWTH_EQUITY,
    DEFENSIVE_EQUITY,
    DBMF,
    ZROZ,
    LEVERAGED_GOLD,
    TREASURY_RESERVE,
)
EQUITY_TICKERS = (GROWTH_EQUITY, DEFENSIVE_EQUITY)
VALUATION_TICKERS = tuple(sorted(set(STRATEGIC_TICKERS) | LEGACY_HOLDINGS))
ALL_TICKERS = tuple(dict.fromkeys((MARKET_INDEX, *VALUATION_TICKERS)))
TRADED_TICKERS = frozenset(VALUATION_TICKERS)
PORTFOLIO_COMPONENTS = TRADED_TICKERS | {CASH_ASSET}

ADVERTISED_DAILY_MULTIPLIERS = {
    GROWTH_EQUITY: 3.0,
    DEFENSIVE_EQUITY: 3.0,
    DBMF: 1.0,
    ZROZ: 1.0,
    LEVERAGED_GOLD: 2.0,
    TREASURY_RESERVE: 0.0,
    "SOXL": 3.0,
    "QLD": 2.0,
    "GLD": 1.0,
    "TECL": 3.0,
    "SPMO": 1.0,
    "USD": 2.0,
    "QQQM": 1.0,
    "GLDM": 1.0,
    "SMH": 1.0,
    CASH_ASSET: 0.0,
}

SMA_WINDOW = core.SMA_WINDOW
REQUIRED_SIGNAL_ROWS = SMA_WINDOW + 1
MODEL_START_DATE = core.MODEL_HISTORY_START
REBALANCE_BAND = 0.05
REBALANCE_DESTINATION = 0.025
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

STRATEGY_REVISION = "tqqq-upro40-dbmf20-zroz20-ugl20-sma200-v1"
STATE_VERSION = 15
DECISION_AUDIT_SCHEMA_VERSION = 5
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
            "index": MARKET_INDEX,
            "trend": "close_strictly_above_sma",
            "sma_sessions": SMA_WINDOW,
            "tqqq_entry_closes": core.BULLISH_ENTRY_CLOSES,
            "upro_entry": "immediate_on_failed_trend",
            "same_date": "idempotent",
        },
        "allocation": {
            "equity_weight": core.EQUITY_WEIGHT,
            "equity_choices": [GROWTH_EQUITY, DEFENSIVE_EQUITY],
            "fixed_sleeves": {
                DBMF: core.DIVERSIFIER_WEIGHT,
                ZROZ: core.DIVERSIFIER_WEIGHT,
                LEVERAGED_GOLD: core.DIVERSIFIER_WEIGHT,
            },
            "maximum_advertised_daily_exposure": core.MAX_ADVERTISED_DAILY_EXPOSURE,
            "soxl": "not_a_strategic_holding",
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
            "individual_and_equity_drift_trigger": REBALANCE_BAND,
            "inner_destination": REBALANCE_DESTINATION,
            "structural_switch": "exact_target",
            "holdings_source": "confirmed_shares_and_cash",
        },
    }


def calculate_strategy_fingerprint(manifest: dict[str, object] | None = None) -> str:
    return canonical_sha256(strategy_manifest() if manifest is None else manifest)


STRATEGY_FINGERPRINT = calculate_strategy_fingerprint()
EXPECTED_STRATEGY_FINGERPRINT = "c5b105419ff3212209a8117a376a6ac96026d595e508687bb18e8b0755b477d9"


@dataclass(frozen=True)
class StrategyDecision:
    target_weights: dict[str, float]
    router_state: core.EquityRouterState
    transition_reason: str
    structural_change: bool
    trend_positive: bool
    qqq_close: float
    qqq_sma_200: float
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
    individual_drift_triggered: bool = False
    aggregate_equity_drift_triggered: bool = False


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


@dataclass
class PortfolioState:
    state_version: int = STATE_VERSION
    shares: dict[str, float] = field(default_factory=dict)
    cash_balance: float = 0.0
    target_weights: dict[str, float] = field(default_factory=dict)
    portfolio_value: float = 0.0

    strategy_initialized: bool = False
    tqqq_active: bool = False
    tqqq_bullish_streak: int = 0
    tqqq_switch_date: str = ""
    last_processed_signal_date: str = ""
    lifecycle_stage: str = LIFECYCLE_SPRINT
    lifecycle_stage_date: str = ""

    executed_tqqq_active: bool = False
    executed_lifecycle_stage: str = LIFECYCLE_SPRINT
    executed_strategy_fingerprint: str = ""

    pending_recommendation_date: str = ""
    pending_recommendation_weights: dict[str, float] = field(default_factory=dict)
    pending_recommendation_tqqq_active: bool = False
    pending_recommendation_lifecycle_stage: str = ""
    pending_recommendation_notified: bool = False
    pending_recommendation_supersedes_date: str = ""
    pending_recommendation_fingerprint: str = ""

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


def target_weights(tqqq_active: bool, lifecycle_stage: str = LIFECYCLE_SPRINT) -> dict[str, float]:
    if lifecycle_stage not in LIFECYCLE_STAGES:
        raise ValueError(f"Unknown lifecycle stage: {lifecycle_stage}")
    aggressive = core.target_weights(tqqq_active)
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
    for active in (False, True):
        weights = target_weights(active)
        if not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
            raise RuntimeError("Allocation invariant failed")
        if not np.isclose(advertised_daily_exposure(weights), core.MAX_ADVERTISED_DAILY_EXPOSURE):
            raise RuntimeError("Exposure invariant failed")


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
    for name in ("strategy_initialized", "tqqq_active", "executed_tqqq_active", "pending_recommendation_tqqq_active", "pending_recommendation_notified"):
        if not isinstance(getattr(state, name), bool):
            raise RuntimeError(f"{name} must be boolean")
    if not isinstance(state.tqqq_bullish_streak, int) or isinstance(state.tqqq_bullish_streak, bool) or state.tqqq_bullish_streak < 0:
        raise RuntimeError("tqqq_bullish_streak is invalid")
    if state.lifecycle_stage not in LIFECYCLE_STAGES or state.executed_lifecycle_stage not in LIFECYCLE_STAGES:
        raise RuntimeError("Lifecycle state is invalid")
    _validate_weights(
        state.pending_recommendation_weights,
        "pending_recommendation_weights",
        require_total=bool(state.pending_recommendation_date),
    )
    if state.pending_recommendation_date:
        _parse_date(state.pending_recommendation_date, "pending_recommendation_date")
        if not _is_sha256(state.pending_recommendation_fingerprint):
            raise RuntimeError("Pending recommendation fingerprint is invalid")
        if state.pending_recommendation_lifecycle_stage not in LIFECYCLE_STAGES:
            raise RuntimeError("Pending lifecycle stage is invalid")
    elif any((state.pending_recommendation_weights, state.pending_recommendation_notified,
              state.pending_recommendation_supersedes_date, state.pending_recommendation_fingerprint,
              state.pending_recommendation_lifecycle_stage)):
        raise RuntimeError("Pending recommendation state is inconsistent")
    for name in ("tqqq_switch_date", "last_processed_signal_date", "lifecycle_stage_date", "last_delivered_signal_date"):
        _parse_date(getattr(state, name), name)
    for name in ("last_processed_data_fingerprint", "last_delivered_decision_hash"):
        value = getattr(state, name)
        if value and not _is_sha256(value):
            raise RuntimeError(f"{name} is invalid")


def _safe_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for ticker, quantity in value.items():
        name = str(ticker).upper()
        if name in PORTFOLIO_COMPONENTS and _is_number(quantity):
            result[name] = float(quantity)
    return result


def _migrate_state(payload: dict[str, object]) -> PortfolioState:
    version = payload.get("state_version")
    if not isinstance(version, int) or isinstance(version, bool) or not 2 <= version < STATE_VERSION:
        raise RuntimeError(f"Unsupported state version: {version!r}")
    shares = _safe_mapping(payload.get("shares"))
    pending = _safe_mapping(payload.get("pending_recommendation_weights"))
    prior_stage = payload.get("lifecycle_stage", LIFECYCLE_SPRINT)
    lifecycle = prior_stage if prior_stage in LIFECYCLE_STAGES else LIFECYCLE_SPRINT
    executed_stage = payload.get("executed_lifecycle_stage", lifecycle)
    if executed_stage not in LIFECYCLE_STAGES:
        executed_stage = lifecycle
    pending_stage = payload.get("pending_recommendation_lifecycle_stage", "")
    if pending and pending_stage not in LIFECYCLE_STAGES:
        pending_stage = lifecycle
    pending_date = str(payload.get("pending_recommendation_date", "")) if pending else ""
    pending_fp = str(payload.get("pending_recommendation_fingerprint", "")) if pending_date else ""
    if pending_date and not _is_sha256(pending_fp):
        pending_fp = "0" * 64
    state = PortfolioState(
        shares=shares,
        cash_balance=float(payload.get("cash_balance", 0.0)),
        target_weights=_safe_mapping(payload.get("target_weights")),
        portfolio_value=float(payload.get("portfolio_value", 0.0)),
        strategy_initialized=False,
        tqqq_active=False,
        tqqq_bullish_streak=0,
        lifecycle_stage=lifecycle,
        lifecycle_stage_date=str(payload.get("lifecycle_stage_date", "")),
        executed_tqqq_active=shares.get(GROWTH_EQUITY, 0.0) > 0 and shares.get(DEFENSIVE_EQUITY, 0.0) <= 0,
        executed_lifecycle_stage=executed_stage,
        executed_strategy_fingerprint=str(payload.get("executed_strategy_fingerprint", "")),
        pending_recommendation_date=pending_date,
        pending_recommendation_weights=pending,
        pending_recommendation_tqqq_active=pending.get(GROWTH_EQUITY, 0.0) > pending.get(DEFENSIVE_EQUITY, 0.0),
        pending_recommendation_lifecycle_stage=pending_stage if pending_date else "",
        pending_recommendation_notified=bool(payload.get("pending_recommendation_notified", False)) if pending_date else False,
        pending_recommendation_supersedes_date=str(payload.get("pending_recommendation_supersedes_date", "")) if pending_date else "",
        pending_recommendation_fingerprint=pending_fp,
        last_delivered_decision_hash=str(payload.get("last_delivered_decision_hash", "")),
        last_delivered_signal_date=str(payload.get("last_delivered_signal_date", "")),
        last_delivered_notification_kind=str(payload.get("last_delivered_notification_kind", "")),
        last_updated=str(payload.get("last_updated", "")),
    )
    if state.target_weights and not np.isclose(sum(state.target_weights.values()), 1.0, atol=1e-9):
        state.target_weights = {}
    return state


def load_state(*, backup_legacy: bool = True) -> PortfolioState:
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
        if backup_legacy:
            stamp = datetime.now(NEW_YORK).strftime("%Y%m%dT%H%M%S")
            shutil.copy2(STATE_FILE, STATE_FILE.with_name(f"{STATE_FILE.stem}.v{version}.{stamp}.backup.json"))
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
    complete = [session for session in sessions if calendar.session_close(session) + pd.Timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES) <= now_utc]
    if not complete:
        raise RuntimeError("No completed XNYS session is available")
    return pd.Timestamp(complete[-1]).tz_localize(None).normalize()


def required_nyse_sessions(ending_session: pd.Timestamp, count: int) -> pd.DatetimeIndex:
    if count <= 0:
        raise ValueError("count must be positive")
    ending = pd.Timestamp(ending_session).normalize()
    sessions = pd.DatetimeIndex(_nyse_calendar().sessions_in_range(ending - pd.Timedelta(days=max(30, count * 3)), ending))
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
            prices = data["Close"].copy() if "Close" in data.columns.get_level_values(0) else data.xs("Close", axis=1, level=1).copy()
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


def download_market_data(tickers: Iterable[str] = ALL_TICKERS, *, now_new_york: datetime | None = None) -> pd.DataFrame:
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
    validate_session_continuity(prices[MARKET_INDEX].dropna().index, expected, REQUIRED_SIGNAL_ROWS)
    _require_positive(prices[MARKET_INDEX].dropna().iloc[-REQUIRED_SIGNAL_ROWS:], "QQQ signal history")
    _require_positive(prices.loc[received, requested], "Latest valuation prices")
    return prices


def market_data_fingerprint(prices: pd.DataFrame) -> str:
    if len(prices) < REQUIRED_SIGNAL_ROWS:
        raise ValueError("Insufficient data for fingerprint")
    return canonical_sha256({
        "sessions": [pd.Timestamp(value).date().isoformat() for value in prices.index[-REQUIRED_SIGNAL_ROWS:]],
        "qqq": [float(value) for value in prices[MARKET_INDEX].iloc[-REQUIRED_SIGNAL_ROWS:]],
        "latest": {ticker: float(prices[ticker].iloc[-1]) for ticker in VALUATION_TICKERS},
    })


def _router_state(state: PortfolioState) -> core.EquityRouterState:
    return core.EquityRouterState(
        tqqq_active=state.tqqq_active,
        bullish_streak=state.tqqq_bullish_streak,
        last_processed_signal_date=state.last_processed_signal_date,
        switch_date=state.tqqq_switch_date,
    )


def _apply_router(state: PortfolioState, router: core.EquityRouterState) -> None:
    state.strategy_initialized = True
    state.tqqq_active = router.tqqq_active
    state.tqqq_bullish_streak = router.bullish_streak
    state.last_processed_signal_date = router.last_processed_signal_date
    state.tqqq_switch_date = router.switch_date


def _latest_decision(prices: pd.DataFrame, state: PortfolioState) -> StrategyDecision:
    session = pd.Timestamp(prices.index[-1]).normalize()
    qqq = pd.to_numeric(prices[MARKET_INDEX], errors="coerce")
    close = float(qqq.iloc[-1])
    average = float(qqq.iloc[-SMA_WINDOW:].mean())
    if not np.isfinite(close) or not np.isfinite(average) or close <= 0 or average <= 0:
        raise RuntimeError("QQQ trend inputs are invalid")
    trend = close > average
    transition = core.advance_equity_router(
        _router_state(state), signal_date=session, trend_positive=trend
    )
    return StrategyDecision(
        target_weights=target_weights(transition.state.tqqq_active),
        router_state=transition.state,
        transition_reason=transition.reason,
        structural_change=transition.structural_change,
        trend_positive=trend,
        qqq_close=close,
        qqq_sma_200=average,
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
        sessions = pd.DatetimeIndex(prices.index[-core.BULLISH_ENTRY_CLOSES:])
    working = copy.deepcopy(state)
    decisions = []
    reasons = []
    dates = []
    changed = False
    for session in sessions:
        history = prices.loc[:session]
        if len(history) < SMA_WINDOW:
            raise RuntimeError("Insufficient history for router replay")
        decision = _latest_decision(history, working)
        _apply_router(working, decision.router_state)
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
        target_weights=target_weights(decision.router_state.tqqq_active, selected.stage),
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


def drift_triggers(
    existing: dict[str, float], target: dict[str, float], *, band: float = REBALANCE_BAND
) -> tuple[bool, bool]:
    if not 0 < band < 1:
        raise ValueError("band must be between zero and one")
    if not existing:
        return True, True
    desired = _with_cash(target)
    individual = any(
        abs(desired.get(ticker, 0.0) - existing.get(ticker, 0.0)) >= band - 1e-12
        for ticker in set(existing) | set(desired)
    )
    current_equity = sum(existing.get(ticker, 0.0) for ticker in EQUITY_TICKERS)
    target_equity = sum(desired.get(ticker, 0.0) for ticker in EQUITY_TICKERS)
    aggregate = abs(current_equity - target_equity) >= band - 1e-12
    return individual, aggregate


def inner_band_rebalance_weights(
    existing: dict[str, float],
    target: dict[str, float],
    destination: float = REBALANCE_DESTINATION,
    *,
    trigger_band: float = REBALANCE_BAND,
) -> dict[str, float]:
    if not existing:
        return dict(target)
    if not 0 < destination < trigger_band < 1:
        raise ValueError("Destination must be inside the trigger band")
    desired_mapping = _with_cash(target)
    tickers = sorted(set(existing) | set(desired_mapping))
    current = np.array([existing.get(ticker, 0.0) for ticker in tickers])
    desired = np.array([desired_mapping.get(ticker, 0.0) for ticker in tickers])
    if not np.isclose(current.sum(), 1.0, atol=1e-9) or not np.isclose(desired.sum(), 1.0, atol=1e-9):
        raise RuntimeError("Weight maps must sum to one")
    lower = np.maximum(0.0, desired - destination)
    upper = np.minimum(1.0, desired + destination)
    low_shift = float(np.min(current - upper)) - 1.0
    high_shift = float(np.max(current - lower)) + 1.0
    for _ in range(100):
        shift = (low_shift + high_shift) / 2.0
        candidate = np.clip(current - shift, lower, upper)
        if candidate.sum() > 1.0:
            low_shift = shift
        else:
            high_shift = shift
    projected = np.clip(current - high_shift, lower, upper)
    remainder = 1.0 - projected.sum()
    if abs(remainder) > 1e-10:
        slack = upper - projected if remainder > 0 else projected - lower
        for index in np.argsort(-slack):
            adjustment = min(abs(remainder), slack[index])
            projected[index] += adjustment if remainder > 0 else -adjustment
            remainder += -adjustment if remainder > 0 else adjustment
            if abs(remainder) <= 1e-12:
                break
    if not np.isclose(projected.sum(), 1.0, atol=1e-9):
        raise RuntimeError("Inner-band projection failed")
    return {ticker: float(weight) for ticker, weight in zip(tickers, projected) if weight > 1e-12}


def build_rebalance_plan(
    existing: dict[str, float],
    decision: StrategyDecision,
    state: PortfolioState,
    *,
    rebalance_band: float = REBALANCE_BAND,
    rebalance_destination: float = REBALANCE_DESTINATION,
) -> RebalancePlan:
    target = _with_cash(decision.target_weights)
    strategy_changed = state.executed_strategy_fingerprint != STRATEGY_FINGERPRINT
    lifecycle_changed = state.executed_lifecycle_stage != decision.lifecycle_stage
    router_changed = state.executed_tqqq_active != decision.router_state.tqqq_active
    legacy_exit = any(existing.get(ticker, 0.0) > 1e-12 for ticker in LEGACY_HOLDINGS)
    individual, aggregate = drift_triggers(existing, target, band=rebalance_band)
    full = strategy_changed or lifecycle_changed or router_changed or legacy_exit
    due = full or individual or aggregate
    if strategy_changed:
        reason = "STRATEGY_REVISION_TRANSITION"
    elif legacy_exit:
        reason = "LEGACY_POSITION_EXIT"
    elif lifecycle_changed:
        reason = "LIFECYCLE_STAGE_ADVANCE"
    elif router_changed:
        reason = decision.transition_reason
    elif individual:
        reason = "INDIVIDUAL_DRIFT_BAND"
    elif aggregate:
        reason = "AGGREGATE_EQUITY_DRIFT_BAND"
    else:
        reason = "HOLD"
    execution = target if full or not due else inner_band_rebalance_weights(
        existing, target, rebalance_destination, trigger_band=rebalance_band
    )
    components = set(existing) | set(execution)
    one_way = 0.5 * sum(abs(execution.get(item, 0.0) - existing.get(item, 0.0)) for item in components) if due else 0.0
    orders = sum(
        item != CASH_ASSET and abs(execution.get(item, 0.0) - existing.get(item, 0.0)) > 1e-9
        for item in components
    ) if due else 0
    if due and orders == 0 and one_way <= 1e-12:
        due, full, reason, one_way = False, False, "CONFIRMED_TARGET_STATE", 0.0
    return RebalancePlan(execution, due, full, reason, float(one_way), int(orders), individual, aggregate)


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
        rows.append({"Ticker": ticker, "Price": price, "CurrentUnits": current, "TargetPct": weight,
                     "EstimatedUnits": estimated, "DeltaUnits": delta if actionable else 0.0,
                     "DeltaValue": delta * price if actionable else 0.0, "Action": action})
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
    one_way = 0.5 * sum(abs(retry.get(item, 0.0) - current.get(item, 0.0)) for item in components)
    orders = sum(item != CASH_ASSET and abs(retry.get(item, 0.0) - current.get(item, 0.0)) > 1e-9 for item in components)
    return replace(plan, execution_weights=retry, reason="PENDING_DELIVERY_RETRY", one_way_turnover=one_way, individual_orders=orders)


def run_strategy(roth_amount: float | None, *, backup_legacy_state: bool = True) -> StrategyRun:
    prices = download_market_data(ALL_TICKERS)
    signal_date = pd.Timestamp(prices.index[-1]).normalize()
    fingerprint = market_data_fingerprint(prices)
    state = load_state(backup_legacy=backup_legacy_state)
    validate_same_date_data_fingerprint(state, signal_date, fingerprint)
    value, planning = resolve_portfolio_value(roth_amount, state, prices)
    decision = calculate_strategy_decision(prices, state)
    decision = apply_lifecycle_policy(decision, state, value, signal_date)
    _apply_router(planning, decision.router_state)
    _apply_lifecycle(planning, decision, signal_date)
    current = existing_weights(planning, prices)
    plan = preserve_pending_delivery_plan(
        build_rebalance_plan(current, decision, planning), state, current
    )
    table_target = plan.execution_weights if plan.rebalance_due else _with_cash(decision.target_weights)
    table = calculate_execution_table(prices, table_target, value, planning, actionable=plan.rebalance_due)
    diagnostics = calculate_execution_diagnostics(table, value, current, decision.target_weights, table_target)
    return StrategyRun(prices, decision, state, planning, value, current, table, signal_date, fingerprint, plan, diagnostics)


def _pending_matches(run: StrategyRun) -> bool:
    state = run.state
    return (
        bool(state.pending_recommendation_date)
        and state.pending_recommendation_fingerprint == STRATEGY_FINGERPRINT
        and state.pending_recommendation_tqqq_active == run.decision.router_state.tqqq_active
        and state.pending_recommendation_lifecycle_stage == run.decision.lifecycle_stage
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
                return NotificationDecision("NONE", "IDENTICAL_PENDING_RECOMMENDATION", state.pending_recommendation_date)
            kind = "UPDATE_RETRY" if state.pending_recommendation_supersedes_date else "RETRY"
            return NotificationDecision(kind, "UNDELIVERED_PENDING_RECOMMENDATION", state.pending_recommendation_date, state.pending_recommendation_supersedes_date)
        return NotificationDecision("UPDATE", "MATERIAL_RECOMMENDATION_UPDATE", state.pending_recommendation_supersedes_date or state.pending_recommendation_date)
    if state.pending_recommendation_date:
        return NotificationDecision("CANCELLATION", "PENDING_ACTION_NO_LONGER_REQUIRED", state.pending_recommendation_supersedes_date or state.pending_recommendation_date)
    return NotificationDecision("NONE", "HOLD")


def _clear_pending(state: PortfolioState) -> None:
    state.pending_recommendation_date = ""
    state.pending_recommendation_weights = {}
    state.pending_recommendation_tqqq_active = False
    state.pending_recommendation_lifecycle_stage = ""
    state.pending_recommendation_notified = False
    state.pending_recommendation_supersedes_date = ""
    state.pending_recommendation_fingerprint = ""


def persist_signal_run(run: StrategyRun) -> None:
    state = run.state
    if not state.shares and state.cash_balance == 0:
        state.cash_balance = run.planning_state.cash_balance
    _apply_router(state, run.decision.router_state)
    _apply_lifecycle(state, run.decision, run.signal_date)
    state.portfolio_value = round(run.portfolio_value, 2)
    state.last_processed_data_fingerprint = run.market_data_fingerprint
    if run.rebalance_plan.reason == "CONFIRMED_TARGET_STATE":
        state.target_weights = dict(run.rebalance_plan.execution_weights)
        state.executed_tqqq_active = run.decision.router_state.tqqq_active
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
        state.pending_recommendation_tqqq_active = run.decision.router_state.tqqq_active
        state.pending_recommendation_lifecycle_stage = run.decision.lifecycle_stage
        state.pending_recommendation_notified = False
        state.pending_recommendation_supersedes_date = notification.previous_recommendation_date if notification.kind == "UPDATE" else ""
        state.pending_recommendation_fingerprint = STRATEGY_FINGERPRINT
    elif notification.kind in {"RETRY", "UPDATE_RETRY"}:
        if not state.pending_recommendation_date:
            raise RuntimeError("Cannot retry a missing recommendation")
        state.pending_recommendation_notified = False
    elif notification.kind != "CANCELLATION":
        raise ValueError("Unsupported notification kind")


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
            "tqqq_active": run.decision.router_state.tqqq_active,
            "bullish_streak": run.decision.router_state.bullish_streak,
            "transition": run.decision.transition_reason,
            "processed_dates": list(run.decision.processed_signal_dates),
        },
        "portfolio": {
            "value": run.portfolio_value,
            "current_weights": run.current_weights,
            "strategic_weights": run.decision.target_weights,
            "execution_weights": run.rebalance_plan.execution_weights,
            "rebalance": asdict(run.rebalance_plan),
            "lifecycle_stage": run.decision.lifecycle_stage,
        },
        "notification": asdict(notification),
        "delivery_status": delivery_status,
    }
    payload["decision_hash"] = canonical_sha256(payload)
    return payload


def write_decision_audit(run: StrategyRun, notification: NotificationDecision, delivery_status: str) -> dict[str, object]:
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
    delivered_decision_hash = delivered_decision_hash or build_decision_audit(run, notification, "DELIVERED")["decision_hash"]
    if not _is_sha256(delivered_decision_hash):
        raise ValueError("Delivered decision hash is invalid")
    if notification.kind in {"ACTION", "UPDATE", "RETRY", "UPDATE_RETRY"}:
        if not state.pending_recommendation_date:
            raise RuntimeError("Delivered action lacks pending state")
        state.pending_recommendation_notified = True
        state.pending_recommendation_supersedes_date = ""
    elif notification.kind == "CANCELLATION":
        _clear_pending(state)
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
        state.executed_tqqq_active = state.pending_recommendation_tqqq_active
        state.executed_lifecycle_stage = state.pending_recommendation_lifecycle_stage
        state.executed_strategy_fingerprint = state.pending_recommendation_fingerprint
    else:
        state.target_weights = {}
        state.executed_tqqq_active = shares.get(GROWTH_EQUITY, 0.0) > shares.get(DEFENSIVE_EQUITY, 0.0)
        state.executed_lifecycle_stage = state.lifecycle_stage
        state.executed_strategy_fingerprint = ""
    _clear_pending(state)
    save_state(state)
    return state


def sync_holdings(shares: dict[str, float], cash: float) -> PortfolioState:
    state = load_state()
    if state.pending_recommendation_date:
        raise RuntimeError("Cannot sync while a recommendation is pending")
    state.shares = dict(shares)
    state.cash_balance = float(cash)
    save_state(state)
    return state


def log_decision(run: StrategyRun) -> None:
    logger.info(
        "signal=%s router=%s trend=%s transition=%s lifecycle=%s "
        "rebalance=%s reason=%s turnover=%.4f strategy=%s data=%s",
        run.signal_date.date(),
        GROWTH_EQUITY if run.decision.router_state.tqqq_active else DEFENSIVE_EQUITY,
        run.decision.trend_positive,
        run.decision.transition_reason,
        run.decision.lifecycle_stage,
        run.rebalance_plan.rebalance_due,
        run.rebalance_plan.reason,
        run.rebalance_plan.one_way_turnover,
        STRATEGY_FINGERPRINT[:12],
        run.market_data_fingerprint[:12],
    )


def build_dashboard(run: StrategyRun) -> str:
    decision = run.decision
    plan = run.rebalance_plan
    diagnostics = run.execution_diagnostics
    active_fund = GROWTH_EQUITY if decision.router_state.tqqq_active else DEFENSIVE_EQUITY
    lines = [
        "=" * 108,
        "ROTH IRA - TQQQ / UPRO EQUITY ROUTER",
        "=" * 108,
        f"Signal close: {run.signal_date.date()} | Portfolio value: ${run.portfolio_value:,.2f}",
        (
            f"QQQ trend: {'POSITIVE' if decision.trend_positive else 'NEGATIVE'} "
            f"({decision.qqq_close / decision.qqq_sma_200 - 1:+.2%} vs SMA200) | "
            f"40% equity fund: {active_fund} | bullish streak: "
            f"{decision.router_state.bullish_streak}/{core.BULLISH_ENTRY_CLOSES}"
        ),
        (
            f"Lifecycle: {decision.lifecycle_stage} ({decision.lifecycle_reason}) | "
            f"estimated age {decision.estimated_investor_age:.1f} | "
            f"value/age gates {decision.lifecycle_value_stage}/{decision.lifecycle_age_stage}"
        ),
        (
            f"Transition: {decision.transition_reason} | "
            f"Rebalance: {'YES' if plan.rebalance_due else 'NO'} | "
            f"Reason: {plan.reason} | One-way turnover: {plan.one_way_turnover:.1%}"
        ),
        (
            f"Advertised daily exposure: current {diagnostics.current_daily_exposure:.2f}x, "
            f"strategic {diagnostics.strategic_daily_exposure:.2f}x, "
            f"destination {diagnostics.destination_daily_exposure:.2f}x"
        ),
        "Cost sensitivity: "
        + " / ".join(
            f"{bps}bp ${diagnostics.estimated_costs[bps]:,.2f}"
            for bps in TRANSACTION_COST_SCENARIOS_BPS
        ),
        f"Strategy {STRATEGY_REVISION} ({STRATEGY_FINGERPRINT[:12]}) | Data {run.market_data_fingerprint[:12]}",
        "-" * 108,
        (
            f"{'Ticker':<8}{'Price':>12}{'Current':>14}{'Target':>11}"
            f"{'Est. units':>15}{'Delta':>14}{'Est. trade':>16}{'Action':>13}"
        ),
        "-" * 108,
    ]
    if len(decision.processed_signal_dates) > 1:
        lines.insert(
            5,
            f"Catch-up replay: {len(decision.processed_signal_dates)} completed sessions | "
            f"path {' -> '.join(decision.transition_path[-10:])}",
        )
    for _, row in run.execution_table.iterrows():
        lines.append(
            f"{row['Ticker']:<8}${row['Price']:>11,.2f}{row['CurrentUnits']:>14,.4f}"
            f"{row['TargetPct']:>10.1%}{row['EstimatedUnits']:>15,.4f}"
            f"{row['DeltaUnits']:>14,.4f}${row['DeltaValue']:>15,.2f}{row['Action']:>13}"
        )
    lines.extend(
        [
            "-" * 108,
            "Signal-close quantities are estimates. Recalculate at executable next-session prices, then confirm holdings.",
            (
                "TQQQ and UPRO both target 3x daily returns. This router changes Nasdaq/S&P exposure; "
                "it does not reduce the leverage multiplier."
            ),
            f"Drift trigger/destination: {REBALANCE_BAND:.0%}/{REBALANCE_DESTINATION:.1%}; no margin or options.",
            "=" * 108,
        ]
    )
    return "\n".join(lines)


def notification_subject(run: StrategyRun, notification: NotificationDecision) -> str:
    label = {
        "ACTION": "Action Required",
        "UPDATE": "Action Updated",
        "RETRY": "Action Required (Retry)",
        "UPDATE_RETRY": "Action Updated (Retry)",
        "CANCELLATION": "Action Cancelled",
    }.get(notification.kind)
    if label is None:
        raise ValueError("A NONE notification has no subject")
    subject_date = (
        notification.previous_recommendation_date
        if notification.kind in {"RETRY", "UPDATE_RETRY"}
        else run.signal_date.date().isoformat()
    )
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
    else:
        raise ValueError("A NONE notification has no body")
    return prefix + dashboard


def build_email_html(run: StrategyRun, notification: NotificationDecision) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{row['Ticker']}</td><td>${row['Price']:,.2f}</td>"
        f"<td>{row['CurrentUnits']:,.4f}</td><td>{row['TargetPct']:.1%}</td>"
        f"<td>{row['EstimatedUnits']:,.4f}</td><td>{row['DeltaUnits']:,.4f}</td>"
        f"<td>{row['Action']}</td></tr>"
        for _, row in run.execution_table.iterrows()
    )
    status = {
        "ACTION": "ACTION REQUIRED",
        "UPDATE": "UPDATED ACTION REQUIRED",
        "RETRY": "ACTION DELIVERY RETRY",
        "UPDATE_RETRY": "UPDATED ACTION DELIVERY RETRY",
        "CANCELLATION": "PREVIOUS ACTION CANCELLED",
    }[notification.kind]
    active = GROWTH_EQUITY if run.decision.router_state.tqqq_active else DEFENSIVE_EQUITY
    return f"""<!doctype html><html><body style="font-family:Arial,sans-serif">
<h2>ROTH IRA TQQQ / UPRO Router</h2>
<h3>{status}: {run.rebalance_plan.reason}</h3>
<p>Signal close {run.signal_date.date()} &middot; Portfolio ${run.portfolio_value:,.2f}
&middot; 40% equity fund {active} &middot; Lifecycle {run.decision.lifecycle_stage}</p>
<table style="border-collapse:collapse" cellpadding="7">
<tr><th>Ticker</th><th>Price</th><th>Current</th><th>Target</th>
<th>Est. units</th><th>Delta</th><th>Action</th></tr>{rows}</table>
<p>Signal-close estimates only. Recalculate at executable prices next session,
then confirm complete post-trade holdings and CASH.</p>
<p>TQQQ and UPRO both target 3x daily returns; this switch changes the index exposure.</p>
</body></html>"""


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
    parser = argparse.ArgumentParser(description="ROTH IRA TQQQ/UPRO allocation engine")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--confirm-execution", action="store_true")
    mode.add_argument("--sync-holdings", action="store_true")
    parser.add_argument(
        "--test", action="store_true",
        help="Generate a report without state, email, audit, or log persistence",
    )
    parser.add_argument("--roth-amount", type=float, default=None)
    parser.add_argument("--executed-shares", nargs="+", metavar="TICKER=SHARES")
    parser.add_argument("--executed-signal-date", metavar="YYYY-MM-DD")
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
    try:
        executed_shares, executed_cash = parse_executed_shares(args.executed_shares)
        executed_signal_date = parse_signal_date(args.executed_signal_date)
    except ValueError as exc:
        parser.error(str(exc))

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

    run = run_strategy(args.roth_amount, backup_legacy_state=not args.test)
    dashboard = build_dashboard(run)
    print(dashboard)
    notification = decide_notification(run)
    if args.test:
        return

    log_decision(run)
    logger.info("notification kind=%s reason=%s", notification.kind, notification.reason)
    prepare_notification_delivery(run, notification)
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

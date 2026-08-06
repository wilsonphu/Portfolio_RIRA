#!/usr/bin/env python3
"""
ROTH IRA - Barbell Momentum Allocation Engine
==============================================
Production daily allocation engine for a Roth IRA.

Strategy
--------
Regime filter (QQQ, 2-of-3 consensus):
  - 200-session SMA
  - 50-session Donchian midband
  - 50-session VWMA

Bull volatility tiers use max(10-session, 30-session) realized QQQ volatility:
  - LOW (<15%):       45% Leader / 15% Follower / 25% SMH / 15% QLD
  - MODERATE (15-22%):25% Leader / 10% Follower / 45% SMH / 20% QLD
  - HIGH (>22%):      85% SMH / 15% GLD

Risk-state behavior:
  - Immediate movement to a more defensive volatility tier
  - Two consecutive completed closes before re-risking one or more tiers
  - SOXL/TECL leader review every 21 completed trading sessions
  - 5 percentage-point drift trigger
  - Drift-only trades stop at the 2.5 percentage-point inner boundary
  - Regime, volatility-tier, and leader changes transition to the exact target

Bear allocation:
  - 80% SPMO / 20% GLD

Operational workflow
--------------------
1. Generate a signal after the latest completed market close:

     python port12_cloud.py

2. Trade manually during the next trading session. Recalculate actual orders from
   current executable prices; report quantities are signal-close estimates.

3. Confirm the final post-trade holdings and remaining cash in a separate command:

     python port12_cloud.py --confirm-execution \
       --executed-signal-date 2026-08-04 \
       --executed-shares SOXL=1.2 SMH=3.4 CASH=12.50

4. To record a contribution, withdrawal, dividend, or broker correction when no
   recommendation is pending:

     python port12_cloud.py --sync-holdings \
       --executed-shares SOXL=1.2 SMH=3.4 CASH=1012.50

Environment variables
---------------------
ROTH_IRA_AMOUNT      First-run cash balance only
GMAIL_ADDRESS        Gmail sender, required only when a notification is due
GMAIL_APP_PASSWORD   Gmail app password, required only when a notification is due
RECEIVER_EMAIL       Report recipient, required only when a notification is due

Important
---------
- No forward-fill, backfill, interpolation, or synthetic price/volume data.
- Signals use only completed daily bars.
- HOLD runs are persisted silently. New, materially changed, and cancelled
  recommendations are emailed once; identical pending actions are not resent.
- Confirmed share counts and cash, never a prior model target, remain the source
  of truth for portfolio valuation and rebalance decisions.
"""

from __future__ import annotations

import argparse
import copy
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

import numpy as np
import pandas as pd
import exchange_calendars as xcals

# =============================================================================
# 1. UNIVERSE
# =============================================================================
MARKET_INDEX = "QQQ"  # signal only

LEVERAGED_SEMICONDUCTOR = "SOXL"
LEVERAGED_TECH = "TECL"
SEMICONDUCTOR_ETF = "SMH"
LEVERAGED_INDEX = "QLD"
DEFENSIVE_EQUITY = "SPMO"
HEDGE_ASSET = "GLD"
CASH_ASSET = "CASH"

LEADER_CANDIDATES = (LEVERAGED_SEMICONDUCTOR, LEVERAGED_TECH)
LEVERAGED_SECTOR_ETFS = frozenset(LEADER_CANDIDATES)

ALL_TICKERS = (
    MARKET_INDEX,
    LEVERAGED_SEMICONDUCTOR,
    LEVERAGED_TECH,
    SEMICONDUCTOR_ETF,
    LEVERAGED_INDEX,
    DEFENSIVE_EQUITY,
    HEDGE_ASSET,
)
TRADED_TICKERS = frozenset(ALL_TICKERS) - {MARKET_INDEX}
PORTFOLIO_COMPONENTS = TRADED_TICKERS | {CASH_ASSET}

# =============================================================================
# 2. STRATEGY CONSTANTS
# =============================================================================
SMA_WINDOW = 200
DONCHIAN_WINDOW = 50
VWMA_WINDOW = 50
VOLATILITY_FAST_WINDOW = 10
VOLATILITY_SLOW_WINDOW = 30
MOMENTUM_WINDOW = 15
HISTORY_DAYS = 750

LOW_VOL_THRESHOLD = 0.15
MODERATE_VOL_THRESHOLD = 0.22
VOLATILITY_RERISK_PERSISTENCE = 2

REBALANCE_BAND = 0.05
REBALANCE_DESTINATION = 0.025
NOTIFICATION_WEIGHT_TOLERANCE = 0.005
MAX_LEVERAGED_POSITION = 0.45
LEADER_SWITCH_THRESHOLD = 0.05
SECTOR_REBALANCE_DAYS = 21

LOW_VOL_ALLOCATION = {
    "Leader": 0.45,
    "Follower": 0.15,
    SEMICONDUCTOR_ETF: 0.25,
    LEVERAGED_INDEX: 0.15,
}
MODERATE_VOL_ALLOCATION = {
    "Leader": 0.25,
    "Follower": 0.10,
    SEMICONDUCTOR_ETF: 0.45,
    LEVERAGED_INDEX: 0.20,
}
HIGH_VOL_ALLOCATION = {SEMICONDUCTOR_ETF: 0.85, HEDGE_ASSET: 0.15}
BEAR_ALLOCATION = {DEFENSIVE_EQUITY: 0.80, HEDGE_ASSET: 0.20}

# =============================================================================
# 3. APPLICATION CONFIGURATION
# =============================================================================
NEW_YORK = ZoneInfo("America/New_York")
MARKET_CLOSE_BUFFER_MINUTES = 15
STATE_VERSION = 5

APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / "roth_ira_state.json"
LOG_FILE = APP_DIR / "roth_ira.log"

_configured_roth_amount = os.environ.get("ROTH_IRA_AMOUNT", "").strip()
try:
    ROTH_IRA_AMOUNT = (
        float(_configured_roth_amount)
        if _configured_roth_amount
        else None
    )
except ValueError as exc:
    raise RuntimeError("ROTH_IRA_AMOUNT must be numeric when provided") from exc

if ROTH_IRA_AMOUNT is not None and (
    not np.isfinite(ROTH_IRA_AMOUNT) or ROTH_IRA_AMOUNT <= 0
):
    raise RuntimeError("ROTH_IRA_AMOUNT must be positive and finite")

logger = logging.getLogger("roth_ira")
logger.setLevel(logging.INFO)
logger.propagate = False


def configure_logging(*, persist_log: bool) -> None:
    """Configure CLI logging without creating files during imports or --test runs."""
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if persist_log:
        handlers.insert(0, logging.FileHandler(LOG_FILE))

    for existing_handler in logger.handlers:
        existing_handler.close()
    logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

# =============================================================================
# 4. DATA CLASSES
# =============================================================================
@dataclass(frozen=True)
class StrategyResult:
    target_weights: dict[str, float]
    regime: str
    leader: str
    volatility_tier: str
    annualized_volatility: float
    raw_volatility_tier: str


@dataclass(frozen=True)
class VolatilityTierDecision:
    tier: str
    raw_tier: str
    pending_tier: str = ""
    pending_days: int = 0
    transition: str = "NONE"
    processed_sessions: int = 0


@dataclass(frozen=True)
class RebalancePlan:
    execution_weights: dict[str, float]
    rebalance_due: bool
    full_transition: bool
    reason: str
    one_way_turnover: float
    individual_orders: int


@dataclass(frozen=True)
class NotificationDecision:
    kind: str
    reason: str
    previous_recommendation_date: str = ""
    supersedes_recommendation_date: str = ""

    @property
    def should_send(self) -> bool:
        return self.kind != "NONE"


@dataclass
class PortfolioState:
    state_version: int = STATE_VERSION
    shares: dict[str, float] = field(default_factory=dict)
    cash_balance: float = 0.0
    target_weights: dict[str, float] = field(default_factory=dict)
    portfolio_value: float = 0.0

    # Latest signal state.
    leader: str | None = None
    volatility_tier: str = "N/A"
    pending_volatility_tier: str = ""
    pending_volatility_days: int = 0
    last_processed_signal_date: str = ""
    regime: str = "UNKNOWN"

    # Last confirmed execution state.
    executed_leader: str | None = None
    executed_regime: str = "UNKNOWN"
    executed_volatility_tier: str = "N/A"

    # Recommendation awaiting manual execution confirmation.
    pending_recommendation_date: str = ""
    pending_recommendation_leader: str | None = None
    pending_recommendation_regime: str = "UNKNOWN"
    pending_recommendation_tier: str = "N/A"
    pending_recommendation_weights: dict[str, float] = field(default_factory=dict)
    pending_recommendation_notified: bool = False
    pending_recommendation_supersedes_date: str = ""

    last_sector_rebalance: str = ""
    last_updated: str = ""


@dataclass(frozen=True)
class StrategyRun:
    price_data: pd.DataFrame
    latest_indicators: pd.Series
    result: StrategyResult
    state: PortfolioState
    planning_state: PortfolioState
    portfolio_value: float
    current_weights: dict[str, float]
    execution_table: pd.DataFrame
    signal_date: pd.Timestamp
    sector_review_due: bool
    rebalance_plan: RebalancePlan
    tier_decision: VolatilityTierDecision

# =============================================================================
# 5. GENERIC VALIDATION HELPERS
# =============================================================================
def _is_valid_number(value: object, *, allow_zero: bool = True) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not np.isfinite(value):
        return False
    return value >= 0 if allow_zero else value > 0


def _parse_iso_date(value: str, field_name: str) -> date | None:
    if not isinstance(value, str):
        raise RuntimeError(f"{field_name} must be a string")
    if not value:
        return None
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} is not a valid date") from exc


def _weights_close(
    left: dict[str, float],
    right: dict[str, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    tickers = set(left) | set(right)
    return all(
        abs(left.get(ticker, 0.0) - right.get(ticker, 0.0)) <= tolerance
        for ticker in tickers
    )


def _with_cash_target(weights: dict[str, float]) -> dict[str, float]:
    result = dict(weights)
    result.setdefault(CASH_ASSET, 0.0)
    return result


def validate_configuration() -> None:
    if len(ALL_TICKERS) != len(set(ALL_TICKERS)):
        raise RuntimeError("Ticker universe contains duplicates")
    if MARKET_INDEX in TRADED_TICKERS:
        raise RuntimeError("The signal ticker cannot be traded")
    if not (0 < VOLATILITY_FAST_WINDOW < VOLATILITY_SLOW_WINDOW):
        raise RuntimeError("Volatility windows must be positive and ordered")
    if not (0 < LOW_VOL_THRESHOLD < MODERATE_VOL_THRESHOLD):
        raise RuntimeError("Volatility thresholds must be positive and ordered")
    if VOLATILITY_RERISK_PERSISTENCE < 2:
        raise RuntimeError("Re-risk persistence must require at least two closes")
    if not (0 < REBALANCE_DESTINATION < REBALANCE_BAND):
        raise RuntimeError("The rebalance destination must be inside the trigger band")
    if not (0 < NOTIFICATION_WEIGHT_TOLERANCE < REBALANCE_BAND):
        raise RuntimeError("The notification tolerance must be inside the drift band")
    if not (0 < MAX_LEVERAGED_POSITION <= 1):
        raise RuntimeError("The leveraged-position cap is invalid")

    allowed_slots = TRADED_TICKERS | {"Leader", "Follower"}
    templates = {
        "LOW_VOL_ALLOCATION": LOW_VOL_ALLOCATION,
        "MODERATE_VOL_ALLOCATION": MODERATE_VOL_ALLOCATION,
        "HIGH_VOL_ALLOCATION": HIGH_VOL_ALLOCATION,
        "BEAR_ALLOCATION": BEAR_ALLOCATION,
    }
    for name, template in templates.items():
        if set(template) - allowed_slots:
            raise RuntimeError(f"{name} contains an invalid ticker or slot")
        if not np.isclose(sum(template.values()), 1.0, atol=1e-12):
            raise RuntimeError(f"{name} must sum to 1.0")
        if any(not _is_valid_number(weight) for weight in template.values()):
            raise RuntimeError(f"{name} contains an invalid weight")

# =============================================================================
# 6. STATE PERSISTENCE
# =============================================================================
def _validate_weight_mapping(
    mapping: object,
    field_name: str,
    *,
    require_sum_one: bool,
) -> dict[str, float]:
    if not isinstance(mapping, dict):
        raise RuntimeError(f"{field_name} must be a mapping")
    invalid_tickers = set(mapping) - PORTFOLIO_COMPONENTS
    if invalid_tickers:
        raise RuntimeError(
            f"{field_name} contains invalid components: {sorted(invalid_tickers)}"
        )
    for ticker, weight in mapping.items():
        if not _is_valid_number(weight):
            raise RuntimeError(f"{field_name} contains an invalid weight for {ticker}")
    if mapping and require_sum_one and not np.isclose(
        sum(mapping.values()), 1.0, atol=1e-9
    ):
        raise RuntimeError(f"{field_name} must sum to 1.0")
    return mapping


def validate_state(state: PortfolioState) -> None:
    today = datetime.now(NEW_YORK).date()

    if state.state_version != STATE_VERSION:
        raise RuntimeError("Portfolio state has the wrong version after migration")

    if not isinstance(state.shares, dict):
        raise RuntimeError("shares must be a mapping")
    invalid_share_tickers = set(state.shares) - TRADED_TICKERS
    if invalid_share_tickers:
        raise RuntimeError(
            f"State contains invalid holdings: {sorted(invalid_share_tickers)}"
        )
    for ticker, shares in state.shares.items():
        if not _is_valid_number(shares):
            raise RuntimeError(f"Invalid share count for {ticker}")

    if not _is_valid_number(state.cash_balance):
        raise RuntimeError("Portfolio state contains an invalid cash balance")
    if not _is_valid_number(state.portfolio_value):
        raise RuntimeError("Portfolio state contains an invalid portfolio value")

    _validate_weight_mapping(
        state.target_weights,
        "target_weights",
        require_sum_one=True,
    )
    _validate_weight_mapping(
        state.pending_recommendation_weights,
        "pending_recommendation_weights",
        require_sum_one=True,
    )

    if state.leader is not None and state.leader not in LEADER_CANDIDATES:
        raise RuntimeError("Portfolio state contains an invalid signal leader")
    if state.executed_leader is not None and state.executed_leader not in LEADER_CANDIDATES:
        raise RuntimeError("Portfolio state contains an invalid executed leader")
    if (
        state.pending_recommendation_leader is not None
        and state.pending_recommendation_leader not in LEADER_CANDIDATES
    ):
        raise RuntimeError("Portfolio state contains an invalid pending leader")

    if state.regime not in {"UNKNOWN", "BULL", "BEAR"}:
        raise RuntimeError("Portfolio state contains an invalid signal regime")
    if state.executed_regime not in {"UNKNOWN", "BULL", "BEAR"}:
        raise RuntimeError("Portfolio state contains an invalid executed regime")
    if state.pending_recommendation_regime not in {"UNKNOWN", "BULL", "BEAR"}:
        raise RuntimeError("Portfolio state contains an invalid pending regime")

    valid_tiers = {"N/A", "LOW", "MODERATE", "HIGH"}
    if state.volatility_tier not in valid_tiers:
        raise RuntimeError("Portfolio state contains an invalid signal tier")
    if state.executed_volatility_tier not in valid_tiers:
        raise RuntimeError("Portfolio state contains an invalid executed tier")
    if state.pending_recommendation_tier not in valid_tiers:
        raise RuntimeError("Portfolio state contains an invalid pending tier")
    if state.pending_volatility_tier not in {"", "LOW", "MODERATE", "HIGH"}:
        raise RuntimeError("Portfolio state contains an invalid pending volatility tier")
    if (
        isinstance(state.pending_volatility_days, bool)
        or not isinstance(state.pending_volatility_days, int)
        or state.pending_volatility_days < 0
        or state.pending_volatility_days >= VOLATILITY_RERISK_PERSISTENCE
    ):
        raise RuntimeError("Portfolio state contains an invalid pending-tier count")
    if state.pending_volatility_days == 0 and state.pending_volatility_tier:
        raise RuntimeError("A pending tier requires a positive pending-day count")
    if state.pending_volatility_days > 0 and not state.pending_volatility_tier:
        raise RuntimeError("Pending volatility days require a pending tier")

    if state.regime == "BULL" and state.volatility_tier not in {
        "LOW",
        "MODERATE",
        "HIGH",
    }:
        raise RuntimeError("Bull signal state requires a bull volatility tier")
    if state.regime in {"BEAR", "UNKNOWN"} and state.volatility_tier != "N/A":
        raise RuntimeError("Bear or unknown signal state must use tier N/A")

    if state.executed_regime == "BULL":
        if state.executed_volatility_tier not in {"LOW", "MODERATE", "HIGH"}:
            raise RuntimeError("Bull execution state requires a bull tier")
        if state.executed_leader is None:
            raise RuntimeError("Bull execution state requires an executed leader")
    elif state.executed_regime in {"BEAR", "UNKNOWN"}:
        if state.executed_volatility_tier != "N/A":
            raise RuntimeError("Bear or unknown execution state must use tier N/A")

    pending_date = _parse_iso_date(
        state.pending_recommendation_date,
        "pending_recommendation_date",
    )
    supersedes_date = _parse_iso_date(
        state.pending_recommendation_supersedes_date,
        "pending_recommendation_supersedes_date",
    )
    if not isinstance(state.pending_recommendation_notified, bool):
        raise RuntimeError("pending_recommendation_notified must be a boolean")
    if pending_date:
        if pending_date > today:
            raise RuntimeError("Pending recommendation date cannot be in the future")
        if supersedes_date and supersedes_date > today:
            raise RuntimeError(
                "Pending superseded recommendation date cannot be in the future"
            )
        if supersedes_date and state.pending_recommendation_notified:
            raise RuntimeError(
                "A delivered recommendation cannot retain superseded outbox state"
            )
        if not state.pending_recommendation_weights:
            raise RuntimeError("Pending recommendation is missing weights")
        if state.pending_recommendation_leader is None:
            raise RuntimeError("Pending recommendation is missing a leader")
        if state.pending_recommendation_regime == "UNKNOWN":
            raise RuntimeError("Pending recommendation is missing a regime")
        if (
            state.pending_recommendation_regime == "BULL"
            and state.pending_recommendation_tier not in {"LOW", "MODERATE", "HIGH"}
        ):
            raise RuntimeError("Bull pending recommendation requires a bull tier")
        if (
            state.pending_recommendation_regime == "BEAR"
            and state.pending_recommendation_tier != "N/A"
        ):
            raise RuntimeError("Bear pending recommendation must use tier N/A")
    else:
        if (
            state.pending_recommendation_leader is not None
            or state.pending_recommendation_regime != "UNKNOWN"
            or state.pending_recommendation_tier != "N/A"
            or state.pending_recommendation_weights
            or state.pending_recommendation_notified
            or state.pending_recommendation_supersedes_date
        ):
            raise RuntimeError("Portfolio state contains an inconsistent pending recommendation")

    for field_name in ("last_processed_signal_date", "last_sector_rebalance"):
        parsed = _parse_iso_date(getattr(state, field_name), field_name)
        if parsed and parsed > today:
            raise RuntimeError(f"{field_name} cannot be in the future")

    if not isinstance(state.last_updated, str):
        raise RuntimeError("last_updated must be a string")
    if state.last_updated:
        try:
            datetime.fromisoformat(state.last_updated)
        except ValueError as exc:
            raise RuntimeError("last_updated is not a valid timestamp") from exc


def _backup_legacy_state(version: object) -> Path:
    label = "legacy" if version is None else f"v{version}"
    timestamp = datetime.now(NEW_YORK).strftime("%Y%m%dT%H%M%S")
    backup = STATE_FILE.with_name(
        f"{STATE_FILE.stem}.{label}.{timestamp}.backup{STATE_FILE.suffix}"
    )
    shutil.copy2(STATE_FILE, backup)
    return backup


def _migrate_state_payload(
    payload: dict[str, object],
    version: object,
    *,
    backup_legacy: bool,
) -> dict[str, object]:
    known_legacy_keys = {
        "shares",
        "target_weights",
        "portfolio_value",
        "leader",
        "volatility_tier",
        "regime",
        "last_sector_rebalance",
        "last_updated",
    }
    if version is None and not known_legacy_keys.issubset(payload):
        raise RuntimeError("Unversioned state does not match the known legacy schema")

    backup = _backup_legacy_state(version) if backup_legacy else None
    defaults = asdict(PortfolioState())
    valid_fields = {item.name for item in fields(PortfolioState)}
    for key, value in payload.items():
        if key in valid_fields:
            defaults[key] = value

    defaults["state_version"] = STATE_VERSION

    # Versions before 3 did not safely distinguish signal state from confirmed
    # execution state. Preserve holdings, but force a conservative next transition.
    numeric_version = version if isinstance(version, int) else 1
    if numeric_version < STATE_VERSION:
        # Earlier schemas persisted the action before attempting SMTP and could
        # not prove delivery. A one-time retry is safer than suppressing it.
        defaults["pending_recommendation_notified"] = False
    if numeric_version < 3:
        defaults.update(
            {
                "executed_leader": None,
                "executed_regime": "UNKNOWN",
                "executed_volatility_tier": "N/A",
                "pending_recommendation_date": "",
                "pending_recommendation_leader": None,
                "pending_recommendation_regime": "UNKNOWN",
                "pending_recommendation_tier": "N/A",
                "pending_recommendation_weights": {},
                "pending_volatility_tier": "",
                "pending_volatility_days": 0,
                "last_processed_signal_date": "",
            }
        )

    logger.warning(
        "Migrated state version %r to version %s; backup=%s",
        version,
        STATE_VERSION,
        backup or "disabled",
    )
    return defaults


def load_state(*, backup_legacy: bool = True) -> PortfolioState:
    if not STATE_FILE.exists():
        return PortfolioState()

    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {STATE_FILE}; refusing to infer holdings") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Portfolio state must be a JSON object")

    version = payload.get("state_version")
    if version != STATE_VERSION:
        if version not in {None, 1, 2, 3, 4}:
            raise RuntimeError(f"Unsupported state version: {version!r}")
        payload = _migrate_state_payload(
            payload,
            version,
            backup_legacy=backup_legacy,
        )

    expected_fields = {item.name for item in fields(PortfolioState)}
    missing_fields = expected_fields - set(payload)
    unexpected_fields = set(payload) - expected_fields
    if missing_fields or unexpected_fields:
        raise RuntimeError(
            "Portfolio state schema mismatch: "
            f"missing={sorted(missing_fields)}, "
            f"unexpected={sorted(unexpected_fields)}"
        )

    try:
        state = PortfolioState(**payload)
    except TypeError as exc:
        raise RuntimeError("Portfolio state contains unsupported fields") from exc

    validate_state(state)
    return state


def save_state(state: PortfolioState) -> None:
    validate_state(state)
    state.last_updated = datetime.now(NEW_YORK).isoformat()
    payload = json.dumps(asdict(state), indent=4, allow_nan=False)
    temporary = STATE_FILE.with_suffix(f"{STATE_FILE.suffix}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, STATE_FILE)
    logger.info(
        "State saved: signal=%s/%s leader=%s executed=%s/%s holdings=%s "
        "pending_signal=%s notified=%s",
        state.regime,
        state.volatility_tier,
        state.leader,
        state.executed_regime,
        state.executed_volatility_tier,
        len(state.shares),
        state.pending_recommendation_date or "NONE",
        state.pending_recommendation_notified,
    )

# =============================================================================
# 7. MARKET DATA
# =============================================================================
def _nyse_calendar() -> object:
    return xcals.get_calendar("XNYS")


def expected_completed_session(now_new_york: datetime | None = None) -> pd.Timestamp:
    now_new_york = now_new_york or datetime.now(NEW_YORK)
    today = now_new_york.date()
    calendar = _nyse_calendar()
    sessions = calendar.sessions_in_range(
        pd.Timestamp(today - timedelta(days=14)),
        pd.Timestamp(today + timedelta(days=1)),
    )
    if len(sessions) == 0:
        raise RuntimeError("NYSE calendar returned no sessions")

    now_utc = pd.Timestamp(now_new_york).tz_convert("UTC")
    today_session = pd.Timestamp(today)
    if calendar.is_session(today_session):
        market_open = calendar.session_open(today_session)
        safe_close = calendar.session_close(today_session) + pd.Timedelta(
            minutes=MARKET_CLOSE_BUFFER_MINUTES
        )
        if market_open <= now_utc < safe_close:
            raise RuntimeError(
                "The latest daily bar is not final; run before the session opens "
                f"or after {MARKET_CLOSE_BUFFER_MINUTES} minutes past the close"
            )

    completed = [
        session
        for session in sessions
        if calendar.session_close(session)
        + pd.Timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES)
        <= now_utc
    ]
    if not completed:
        raise RuntimeError("No completed NYSE session is available")
    return pd.Timestamp(completed[-1]).tz_localize(None).normalize()


def _extract_yfinance_frames(
    data: pd.DataFrame,
    tickers: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty:
        raise RuntimeError("yfinance returned no market data")
    try:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"].copy()
            volumes = data["Volume"].copy()
        else:
            prices = pd.DataFrame(data["Close"])
            volumes = pd.DataFrame(data["Volume"])
    except KeyError as exc:
        raise RuntimeError("yfinance response is missing Close or Volume") from exc

    requested = list(tickers)
    missing_prices = set(requested) - set(prices.columns)
    missing_volumes = set(requested) - set(volumes.columns)
    if missing_prices or missing_volumes:
        raise RuntimeError(
            "Incomplete market-data universe: "
            f"prices={sorted(missing_prices)}, volumes={sorted(missing_volumes)}"
        )

    prices = prices.reindex(columns=requested)
    volumes = volumes.reindex(columns=requested)
    if not prices.index.equals(volumes.index):
        raise RuntimeError("Price and volume indexes differ")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise RuntimeError("Market data does not use a DatetimeIndex")

    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert(None)
        volumes.index = volumes.index.tz_convert(None)
    prices.index = prices.index.normalize()
    volumes.index = volumes.index.normalize()
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise RuntimeError("Market-data dates are duplicated or unsorted")
    return prices, volumes


def _require_finite_positive(series: pd.Series, description: str) -> None:
    try:
        values = series.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{description} contains nonnumeric values") from exc
    if not np.isfinite(values).all() or not (values > 0).all():
        raise RuntimeError(f"{description} contains missing or invalid values")


def download_market_data(
    tickers: Iterable[str],
    days: int = HISTORY_DAYS,
    *,
    now_new_york: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_session = expected_completed_session(now_new_york)
    start_date = (expected_session.date() - timedelta(days=days)).isoformat()
    # yfinance end is exclusive.
    end_date = (expected_session.date() + timedelta(days=1)).isoformat()
    requested = list(tickers)

    logger.info(
        "Downloading %s from %s through completed session %s",
        requested,
        start_date,
        expected_session.date(),
    )
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for signal generation; install it with "
            "`pip install yfinance`"
        ) from exc

    data = yf.download(
        requested,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    prices, volumes = _extract_yfinance_frames(data, requested)

    if prices.empty:
        raise RuntimeError("No market-data rows were returned")
    received_session = pd.Timestamp(prices.index[-1]).normalize()
    if received_session != expected_session:
        raise RuntimeError(
            "Market data is stale or misdated: "
            f"expected {expected_session.date()}, received {received_session.date()}"
        )

    required_price_rows = max(
        SMA_WINDOW,
        DONCHIAN_WINDOW,
        VWMA_WINDOW,
        VOLATILITY_SLOW_WINDOW + 1,
    )
    if len(prices) < required_price_rows:
        raise RuntimeError(
            f"Insufficient market history: {len(prices)} rows; "
            f"need at least {required_price_rows}"
        )

    # No filling or row deletion. Each signal series is validated for its exact
    # required window, and unrelated historical NaNs do not alter session counts.
    _require_finite_positive(
        prices[MARKET_INDEX].iloc[-required_price_rows:],
        f"{MARKET_INDEX} prices in the indicator window",
    )
    _require_finite_positive(
        volumes[MARKET_INDEX].iloc[-VWMA_WINDOW:],
        f"{MARKET_INDEX} volumes in the VWMA window",
    )
    for ticker in LEADER_CANDIDATES:
        _require_finite_positive(
            prices[ticker].iloc[-(MOMENTUM_WINDOW + 1):],
            f"{ticker} prices in the momentum window",
        )
    _require_finite_positive(
        prices.loc[received_session, list(ALL_TICKERS)],
        "latest prices for the full universe",
    )

    return prices, volumes

# =============================================================================
# 8. INDICATORS AND STRATEGY
# =============================================================================
def calculate_indicators(
    price_data: pd.DataFrame,
    volume_data: pd.DataFrame,
) -> pd.DataFrame:
    indicators = pd.DataFrame(index=price_data.index)
    index_close = price_data[MARKET_INDEX]

    indicators["sma_200"] = index_close.rolling(
        SMA_WINDOW,
        min_periods=SMA_WINDOW,
    ).mean()
    channel_high = index_close.rolling(
        DONCHIAN_WINDOW,
        min_periods=DONCHIAN_WINDOW,
    ).max()
    channel_low = index_close.rolling(
        DONCHIAN_WINDOW,
        min_periods=DONCHIAN_WINDOW,
    ).min()
    indicators["donchian_mid"] = (channel_high + channel_low) / 2.0

    qqq_volume = volume_data[MARKET_INDEX]
    price_times_volume = index_close * qqq_volume
    volume_sum = qqq_volume.rolling(
        VWMA_WINDOW,
        min_periods=VWMA_WINDOW,
    ).sum()
    indicators["vwma_50"] = (
        price_times_volume.rolling(
            VWMA_WINDOW,
            min_periods=VWMA_WINDOW,
        ).sum()
        / volume_sum
    )

    indicators["sma_signal"] = (index_close >= indicators["sma_200"]).astype(int)
    indicators["donchian_signal"] = (
        index_close >= indicators["donchian_mid"]
    ).astype(int)
    indicators["vwma_signal"] = (index_close >= indicators["vwma_50"]).astype(int)
    indicators["bullish_consensus"] = (
        indicators[["sma_signal", "donchian_signal", "vwma_signal"]].sum(axis=1)
        >= 2
    ).astype(int)

    returns = index_close.pct_change(fill_method=None)
    indicators["volatility_10"] = returns.rolling(
        VOLATILITY_FAST_WINDOW,
        min_periods=VOLATILITY_FAST_WINDOW,
    ).std() * np.sqrt(252)
    indicators["volatility_30"] = returns.rolling(
        VOLATILITY_SLOW_WINDOW,
        min_periods=VOLATILITY_SLOW_WINDOW,
    ).std() * np.sqrt(252)
    indicators["annualized_volatility"] = pd.concat(
        [indicators["volatility_10"], indicators["volatility_30"]],
        axis=1,
    ).max(axis=1, skipna=False)

    indicators["soxl_momentum"] = price_data[
        LEVERAGED_SEMICONDUCTOR
    ].pct_change(MOMENTUM_WINDOW, fill_method=None)
    indicators["tecl_momentum"] = price_data[
        LEVERAGED_TECH
    ].pct_change(MOMENTUM_WINDOW, fill_method=None)
    return indicators


def validate_latest_indicators(latest: pd.Series) -> None:
    required = (
        "sma_200",
        "donchian_mid",
        "vwma_50",
        "volatility_10",
        "volatility_30",
        "annualized_volatility",
        "soxl_momentum",
        "tecl_momentum",
    )
    invalid: list[str] = []
    for name in required:
        try:
            value = float(latest[name])
            valid = np.isfinite(value)
            if name in {
                "volatility_10",
                "volatility_30",
                "annualized_volatility",
            }:
                valid = valid and value > 0
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            invalid.append(name)
    if invalid:
        raise RuntimeError(f"Latest indicators are incomplete: {invalid}")


def classify_volatility(annualized_volatility: float) -> str:
    if not np.isfinite(annualized_volatility) or annualized_volatility <= 0:
        raise ValueError("Volatility must be positive and finite")
    if annualized_volatility < LOW_VOL_THRESHOLD:
        return "LOW"
    if annualized_volatility <= MODERATE_VOL_THRESHOLD:
        return "MODERATE"
    return "HIGH"


def classify_volatility_with_persistence(
    annualized_volatility: float,
    existing_tier: str,
    pending_tier: str,
    pending_days: int,
) -> VolatilityTierDecision:
    raw_tier = classify_volatility(annualized_volatility)
    defensiveness = {"LOW": 0, "MODERATE": 1, "HIGH": 2}

    if existing_tier not in defensiveness:
        return VolatilityTierDecision(
            tier=raw_tier,
            raw_tier=raw_tier,
            transition="INITIAL",
        )
    if raw_tier == existing_tier:
        return VolatilityTierDecision(
            tier=existing_tier,
            raw_tier=raw_tier,
            transition="UNCHANGED",
        )
    if defensiveness[raw_tier] > defensiveness[existing_tier]:
        return VolatilityTierDecision(
            tier=raw_tier,
            raw_tier=raw_tier,
            transition="DE_RISK",
        )

    next_days = pending_days + 1 if pending_tier == raw_tier else 1
    if next_days >= VOLATILITY_RERISK_PERSISTENCE:
        return VolatilityTierDecision(
            tier=raw_tier,
            raw_tier=raw_tier,
            transition="RE_RISK",
        )
    return VolatilityTierDecision(
        tier=existing_tier,
        raw_tier=raw_tier,
        pending_tier=raw_tier,
        pending_days=next_days,
        transition="DELAY_RE_RISK",
    )


def replay_volatility_state(
    indicators: pd.DataFrame,
    state: PortfolioState,
) -> VolatilityTierDecision:
    latest_date = pd.Timestamp(indicators.index[-1]).normalize()
    current_tier = state.volatility_tier
    pending_tier = state.pending_volatility_tier
    pending_days = state.pending_volatility_days

    if state.last_processed_signal_date:
        last_processed = pd.Timestamp(state.last_processed_signal_date).normalize()
        if last_processed > latest_date:
            raise RuntimeError("State was processed after the latest market-data session")
        if last_processed not in indicators.index:
            raise RuntimeError(
                "The last processed signal date is outside or missing from the "
                "downloaded history; increase HISTORY_DAYS or migrate state explicitly"
            )
        unprocessed = indicators.loc[indicators.index > last_processed]
    else:
        # On first use, initialize from the latest close only; do not fabricate an
        # execution history by replaying years of signals.
        unprocessed = indicators.iloc[[-1]]

    if unprocessed.empty:
        latest = indicators.iloc[-1]
        if int(latest["bullish_consensus"]) != 1:
            return VolatilityTierDecision(
                tier="N/A",
                raw_tier="N/A",
                transition="SAME_DATE",
                processed_sessions=0,
            )
        raw = classify_volatility(float(latest["annualized_volatility"]))
        return VolatilityTierDecision(
            tier=current_tier,
            raw_tier=raw,
            pending_tier=pending_tier,
            pending_days=pending_days,
            transition="SAME_DATE",
            processed_sessions=0,
        )

    final_decision: VolatilityTierDecision | None = None
    processed_sessions = 0
    for _, row in unprocessed.iterrows():
        processed_sessions += 1
        if int(row["bullish_consensus"]) != 1:
            current_tier = "N/A"
            pending_tier = ""
            pending_days = 0
            final_decision = VolatilityTierDecision(
                tier="N/A",
                raw_tier="N/A",
                transition="BEAR",
            )
            continue

        volatility = float(row["annualized_volatility"])
        if not np.isfinite(volatility):
            raise RuntimeError(
                "An unprocessed completed session has unavailable volatility"
            )
        decision = classify_volatility_with_persistence(
            volatility,
            current_tier,
            pending_tier,
            pending_days,
        )
        current_tier = decision.tier
        pending_tier = decision.pending_tier
        pending_days = decision.pending_days
        final_decision = decision

    assert final_decision is not None
    return replace(final_decision, processed_sessions=processed_sessions)


def select_leader(
    soxl_momentum: float,
    tecl_momentum: float,
    existing_leader: str | None,
) -> str:
    if not np.isfinite(soxl_momentum) or not np.isfinite(tecl_momentum):
        raise ValueError("Leader momentum is unavailable")
    if existing_leader == LEVERAGED_SEMICONDUCTOR:
        if tecl_momentum - soxl_momentum > LEADER_SWITCH_THRESHOLD:
            return LEVERAGED_TECH
        return LEVERAGED_SEMICONDUCTOR
    if existing_leader == LEVERAGED_TECH:
        if soxl_momentum - tecl_momentum > LEADER_SWITCH_THRESHOLD:
            return LEVERAGED_SEMICONDUCTOR
        return LEVERAGED_TECH
    return (
        LEVERAGED_SEMICONDUCTOR
        if soxl_momentum >= tecl_momentum
        else LEVERAGED_TECH
    )


def apply_allocation_template(
    template: dict[str, float],
    leader: str,
) -> dict[str, float]:
    follower = (
        LEVERAGED_TECH
        if leader == LEVERAGED_SEMICONDUCTOR
        else LEVERAGED_SEMICONDUCTOR
    )
    target: dict[str, float] = {}
    for slot, original_weight in template.items():
        ticker = leader if slot == "Leader" else follower if slot == "Follower" else slot
        weight = original_weight
        if ticker in LEVERAGED_SECTOR_ETFS and weight > MAX_LEVERAGED_POSITION:
            excess = weight - MAX_LEVERAGED_POSITION
            weight = MAX_LEVERAGED_POSITION
            target[SEMICONDUCTOR_ETF] = target.get(SEMICONDUCTOR_ETF, 0.0) + excess
        target[ticker] = target.get(ticker, 0.0) + weight

    if not np.isclose(sum(target.values()), 1.0, atol=1e-12):
        raise RuntimeError("Resolved target does not sum to 1.0")
    if any(
        target.get(ticker, 0.0) > MAX_LEVERAGED_POSITION + 1e-12
        for ticker in LEVERAGED_SECTOR_ETFS
    ):
        raise RuntimeError("Resolved target exceeds the leveraged-position cap")
    return target


def determine_target_allocation(
    latest: pd.Series,
    existing_leader: str | None,
    allow_leader_review: bool,
    tier_decision: VolatilityTierDecision,
) -> StrategyResult:
    annualized_volatility = float(latest["annualized_volatility"])
    leader = existing_leader or LEVERAGED_SEMICONDUCTOR
    if allow_leader_review:
        leader = select_leader(
            float(latest["soxl_momentum"]),
            float(latest["tecl_momentum"]),
            existing_leader,
        )

    if int(latest["bullish_consensus"]) != 1:
        return StrategyResult(
            target_weights=dict(BEAR_ALLOCATION),
            regime="BEAR",
            leader=leader,
            volatility_tier="N/A",
            annualized_volatility=annualized_volatility,
            raw_volatility_tier="N/A",
        )

    if tier_decision.tier not in {"LOW", "MODERATE", "HIGH"}:
        raise RuntimeError("Bull allocation requires a valid volatility tier")
    template = {
        "LOW": LOW_VOL_ALLOCATION,
        "MODERATE": MODERATE_VOL_ALLOCATION,
        "HIGH": HIGH_VOL_ALLOCATION,
    }[tier_decision.tier]
    target = apply_allocation_template(template, leader)
    return StrategyResult(
        target_weights=target,
        regime="BULL",
        leader=leader,
        volatility_tier=tier_decision.tier,
        annualized_volatility=annualized_volatility,
        raw_volatility_tier=tier_decision.raw_tier,
    )

# =============================================================================
# 9. PORTFOLIO AND REBALANCING
# =============================================================================
def validate_holdings_against_prices(
    state: PortfolioState,
    price_data: pd.DataFrame,
) -> None:
    missing = set(state.shares) - set(price_data.columns)
    if missing:
        raise RuntimeError(f"Holdings have no current price data: {sorted(missing)}")
    if MARKET_INDEX in state.shares:
        raise RuntimeError(f"{MARKET_INDEX} is signal-only and cannot be held")


def existing_portfolio_value(
    state: PortfolioState,
    price_data: pd.DataFrame,
) -> float:
    validate_holdings_against_prices(state, price_data)
    total = float(state.cash_balance)
    for ticker, shares in state.shares.items():
        price = float(price_data[ticker].iloc[-1])
        if not np.isfinite(price) or price <= 0:
            raise RuntimeError(f"Invalid latest price for held ticker {ticker}")
        total += shares * price
    if not np.isfinite(total) or total < 0:
        raise RuntimeError("Marked-to-market portfolio value is invalid")
    return total


def existing_weights(
    state: PortfolioState,
    price_data: pd.DataFrame,
) -> dict[str, float]:
    total = existing_portfolio_value(state, price_data)
    if total <= 0:
        return {}
    weights = {
        ticker: shares * float(price_data[ticker].iloc[-1]) / total
        for ticker, shares in state.shares.items()
        if shares > 0
    }
    if state.cash_balance > 0:
        weights[CASH_ASSET] = state.cash_balance / total
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
        raise RuntimeError("Current portfolio weights do not sum to 1.0")
    return weights


def should_rebalance(
    existing: dict[str, float],
    target: dict[str, float],
    band: float = REBALANCE_BAND,
) -> bool:
    if not existing:
        return True
    tickers = set(existing) | set(target)
    return any(
        abs(target.get(ticker, 0.0) - existing.get(ticker, 0.0))
        >= band - 1e-12
        for ticker in tickers
    )


def inner_band_rebalance_weights(
    existing: dict[str, float],
    target: dict[str, float],
    destination: float = REBALANCE_DESTINATION,
) -> dict[str, float]:
    if not existing:
        return dict(target)
    if not (0 < destination < REBALANCE_BAND):
        raise ValueError("Destination must be positive and inside the trigger band")

    tickers = sorted(set(existing) | set(target))
    current = np.array([existing.get(ticker, 0.0) for ticker in tickers], dtype=float)
    desired = np.array([target.get(ticker, 0.0) for ticker in tickers], dtype=float)
    if not np.isclose(current.sum(), 1.0, atol=1e-9):
        raise RuntimeError("Existing portfolio weights do not sum to 1.0")
    if not np.isclose(desired.sum(), 1.0, atol=1e-12):
        raise RuntimeError("Target portfolio weights do not sum to 1.0")

    lower = np.maximum(0.0, desired - destination)
    upper = np.minimum(1.0, desired + destination)
    for index, ticker in enumerate(tickers):
        if ticker in LEVERAGED_SECTOR_ETFS:
            upper[index] = min(upper[index], MAX_LEVERAGED_POSITION)
    if lower.sum() > 1.0 + 1e-12 or upper.sum() < 1.0 - 1e-12:
        raise RuntimeError("Inner-band bounds cannot form a fully invested portfolio")

    lower_shift = float(np.min(current - upper)) - 1.0
    upper_shift = float(np.max(current - lower)) + 1.0
    for _ in range(100):
        shift = (lower_shift + upper_shift) / 2.0
        candidate = np.clip(current - shift, lower, upper)
        if candidate.sum() > 1.0:
            lower_shift = shift
        else:
            upper_shift = shift
    weights = np.clip(current - upper_shift, lower, upper)

    remainder = 1.0 - float(weights.sum())
    if abs(remainder) > 1e-10:
        slack = upper - weights if remainder > 0 else weights - lower
        for index in np.argsort(-slack):
            adjustment = min(abs(remainder), float(slack[index]))
            weights[index] += adjustment if remainder > 0 else -adjustment
            remainder += -adjustment if remainder > 0 else adjustment
            if abs(remainder) <= 1e-12:
                break

    if not np.isclose(weights.sum(), 1.0, atol=1e-9):
        raise RuntimeError("Inner-band projection did not preserve total weight")
    return {
        ticker: float(weight)
        for ticker, weight in zip(tickers, weights)
        if weight > 1e-12
    }


def build_rebalance_plan(
    existing: dict[str, float],
    result: StrategyResult,
    state: PortfolioState,
) -> RebalancePlan:
    target = _with_cash_target(result.target_weights)
    initial_allocation = state.executed_regime == "UNKNOWN" or not existing
    regime_changed = (
        not initial_allocation and state.executed_regime != result.regime
    )
    tier_changed = (
        not initial_allocation
        and result.regime == "BULL"
        and (
            state.executed_regime != "BULL"
            or state.executed_volatility_tier != result.volatility_tier
        )
    )
    leader_changed = (
        not initial_allocation
        and result.regime == "BULL"
        and state.executed_leader != result.leader
    )
    full_transition = initial_allocation or regime_changed or tier_changed or leader_changed
    drift_exceeded = should_rebalance(existing, target)
    rebalance_due = full_transition or drift_exceeded

    if initial_allocation:
        reason = "INITIAL_ALLOCATION"
    elif regime_changed:
        reason = "REGIME_TRANSITION"
    elif tier_changed:
        reason = "VOLATILITY_TIER_TRANSITION"
    elif leader_changed:
        reason = "LEADER_TRANSITION"
    elif drift_exceeded:
        reason = "DRIFT_BAND"
    else:
        reason = "HOLD"

    execution = (
        target
        if full_transition or not rebalance_due
        else inner_band_rebalance_weights(existing, target)
    )

    if rebalance_due:
        components = set(existing) | set(execution)
        one_way_turnover = 0.5 * sum(
            abs(execution.get(component, 0.0) - existing.get(component, 0.0))
            for component in components
        )
        individual_orders = sum(
            component != CASH_ASSET
            and abs(execution.get(component, 0.0) - existing.get(component, 0.0)) > 1e-9
            for component in components
        )
    else:
        one_way_turnover = 0.0
        individual_orders = 0

    if rebalance_due and individual_orders == 0 and one_way_turnover <= 1e-12:
        rebalance_due = False
        full_transition = False
        reason = "CONFIRMED_TARGET_STATE"
        one_way_turnover = 0.0

    return RebalancePlan(
        execution_weights=execution,
        rebalance_due=rebalance_due,
        full_transition=full_transition,
        reason=reason,
        one_way_turnover=one_way_turnover,
        individual_orders=individual_orders,
    )


def sector_review_due(
    last_sector_review: str,
    signal_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
) -> bool:
    if not last_sector_review:
        return True
    previous = pd.Timestamp(last_sector_review).normalize()
    completed = trading_dates[
        (trading_dates > previous) & (trading_dates <= signal_date)
    ]
    return len(completed) >= SECTOR_REBALANCE_DAYS


def resolve_portfolio_value(
    roth_amount: float | None,
    state: PortfolioState,
    price_data: pd.DataFrame,
) -> tuple[float, PortfolioState]:
    planning_state = copy.deepcopy(state)
    has_recorded_portfolio = bool(state.shares) or state.cash_balance > 0

    if has_recorded_portfolio:
        if roth_amount is not None:
            raise RuntimeError(
                "--roth-amount is first-run only. Record contributions with "
                "--sync-holdings and an updated CASH value."
            )
        value = existing_portfolio_value(planning_state, price_data)
    else:
        starting_cash = roth_amount if roth_amount is not None else ROTH_IRA_AMOUNT
        if starting_cash is None:
            raise RuntimeError(
                "No portfolio value is available; supply --roth-amount or "
                "ROTH_IRA_AMOUNT on the first run"
            )
        if not np.isfinite(starting_cash) or starting_cash <= 0:
            raise RuntimeError("First-run Roth IRA amount must be positive and finite")
        planning_state.cash_balance = float(starting_cash)
        value = float(starting_cash)

    if value <= 0:
        raise RuntimeError("Portfolio value must be positive")
    return value, planning_state


def calculate_execution_table(
    price_data: pd.DataFrame,
    destination_weights: dict[str, float],
    portfolio_value: float,
    current_state: PortfolioState,
    *,
    actionable: bool,
) -> pd.DataFrame:
    components = set(destination_weights) | set(current_state.shares)
    if current_state.cash_balance > 0 or CASH_ASSET in destination_weights:
        components.add(CASH_ASSET)

    rows: list[dict[str, object]] = []
    for component in components:
        target_weight = destination_weights.get(component, 0.0)
        target_value = target_weight * portfolio_value
        if component == CASH_ASSET:
            price = 1.0
            current_units = current_state.cash_balance
            estimated_units = target_value
            delta_units = estimated_units - current_units
            action = "CASH AFTER TRADES" if actionable else "HOLD"
        else:
            price = float(price_data[component].iloc[-1])
            if not np.isfinite(price) or price <= 0:
                raise RuntimeError(f"Invalid latest price for {component}")
            current_units = current_state.shares.get(component, 0.0)
            estimated_units = target_value / price
            delta_units = estimated_units - current_units
            if not actionable or abs(delta_units) <= 0.00005:
                action = "HOLD"
            elif delta_units > 0:
                action = "BUY"
            else:
                action = "SELL"

        rows.append(
            {
                "Ticker": component,
                "Price": price,
                "CurrentUnits": current_units,
                "TargetPct": target_weight,
                "TargetValue": target_value,
                "EstimatedUnits": estimated_units,
                "DeltaUnits": delta_units if actionable else 0.0,
                "DeltaValue": delta_units * price if actionable else 0.0,
                "Action": action,
            }
        )

    order = {ticker: index for index, ticker in enumerate((*ALL_TICKERS, CASH_ASSET))}
    rows.sort(key=lambda row: order.get(str(row["Ticker"]), 999))
    return pd.DataFrame(rows)

# =============================================================================
# 10. ORCHESTRATION AND STATE TRANSITIONS
# =============================================================================
def run_strategy(
    roth_amount: float | None,
    *,
    backup_legacy_state: bool = True,
) -> StrategyRun:
    price_data, volume_data = download_market_data(ALL_TICKERS)
    indicators = calculate_indicators(price_data, volume_data)
    latest = indicators.iloc[-1]
    validate_latest_indicators(latest)

    state = load_state(backup_legacy=backup_legacy_state)
    signal_date = pd.Timestamp(price_data.index[-1]).normalize()

    tier_decision = replay_volatility_state(indicators, state)
    sector_due = sector_review_due(
        state.last_sector_rebalance,
        signal_date,
        price_data.index,
    )
    result = determine_target_allocation(
        latest,
        state.leader,
        sector_due,
        tier_decision,
    )

    portfolio_value, planning_state = resolve_portfolio_value(
        roth_amount,
        state,
        price_data,
    )
    current_weights = existing_weights(planning_state, price_data)
    plan = build_rebalance_plan(current_weights, result, planning_state)

    # Preserve the exact staged action during an SMTP retry when the current
    # recommendation has not changed materially.
    if (
        plan.rebalance_due
        and state.pending_recommendation_date
        and not state.pending_recommendation_notified
        and _weights_close(
            state.pending_recommendation_weights,
            plan.execution_weights,
            tolerance=NOTIFICATION_WEIGHT_TOLERANCE,
        )
    ):
        retry_weights = dict(state.pending_recommendation_weights)
        retry_components = set(current_weights) | set(retry_weights)
        plan = replace(
            plan,
            execution_weights=retry_weights,
            reason="PENDING_DELIVERY_RETRY",
            one_way_turnover=0.5
            * sum(
                abs(
                    retry_weights.get(component, 0.0)
                    - current_weights.get(component, 0.0)
                )
                for component in retry_components
            ),
            individual_orders=sum(
                component != CASH_ASSET
                and abs(
                    retry_weights.get(component, 0.0)
                    - current_weights.get(component, 0.0)
                )
                > 1e-9
                for component in retry_components
            ),
        )

    table_weights = (
        plan.execution_weights
        if plan.rebalance_due
        else _with_cash_target(result.target_weights)
    )
    execution_table = calculate_execution_table(
        price_data,
        table_weights,
        portfolio_value,
        planning_state,
        actionable=plan.rebalance_due,
    )

    return StrategyRun(
        price_data=price_data,
        latest_indicators=latest,
        result=result,
        state=state,
        planning_state=planning_state,
        portfolio_value=portfolio_value,
        current_weights=current_weights,
        execution_table=execution_table,
        signal_date=signal_date,
        sector_review_due=sector_due,
        rebalance_plan=plan,
        tier_decision=tier_decision,
    )


def _pending_recommendation_matches(strategy_run: StrategyRun) -> bool:
    state = strategy_run.state
    return (
        bool(state.pending_recommendation_date)
        and _weights_close(
            state.pending_recommendation_weights,
            strategy_run.rebalance_plan.execution_weights,
            tolerance=NOTIFICATION_WEIGHT_TOLERANCE,
        )
    )


def decide_notification(strategy_run: StrategyRun) -> NotificationDecision:
    """Return the one notification, if any, warranted by confirmed holdings."""
    state = strategy_run.state
    pending_date = state.pending_recommendation_date
    supersedes_date = state.pending_recommendation_supersedes_date
    pending = bool(pending_date)
    actionable = (
        strategy_run.rebalance_plan.rebalance_due
        and strategy_run.rebalance_plan.individual_orders > 0
    )

    if actionable:
        if not pending:
            return NotificationDecision("ACTION", "NEW_RECOMMENDATION")
        if _pending_recommendation_matches(strategy_run):
            if state.pending_recommendation_notified:
                return NotificationDecision(
                    "NONE",
                    "IDENTICAL_PENDING_RECOMMENDATION",
                    pending_date,
                )
            return NotificationDecision(
                (
                    "UPDATE_RETRY"
                    if supersedes_date
                    else "RETRY"
                ),
                (
                    "UNDELIVERED_RECOMMENDATION_UPDATE"
                    if supersedes_date
                    else "UNDELIVERED_PENDING_RECOMMENDATION"
                ),
                pending_date,
                supersedes_date,
            )
        return NotificationDecision(
            "UPDATE",
            "MATERIAL_RECOMMENDATION_UPDATE",
            supersedes_date or pending_date,
        )

    if pending:
        return NotificationDecision(
            "CANCELLATION",
            "PENDING_ACTION_NO_LONGER_REQUIRED",
            supersedes_date or pending_date,
        )
    return NotificationDecision("NONE", "HOLD")


def persist_signal_run(strategy_run: StrategyRun) -> None:
    """Persist holdings valuation and signal progress, but not email delivery."""
    state = strategy_run.state
    result = strategy_run.result
    signal_date = strategy_run.signal_date.date().isoformat()

    # On the first signal run, persist the starting amount as actual cash.
    if not state.shares and state.cash_balance == 0:
        state.cash_balance = strategy_run.planning_state.cash_balance

    state.portfolio_value = round(strategy_run.portfolio_value, 2)
    state.regime = result.regime
    state.volatility_tier = result.volatility_tier
    state.leader = result.leader
    state.pending_volatility_tier = strategy_run.tier_decision.pending_tier
    state.pending_volatility_days = strategy_run.tier_decision.pending_days
    state.last_processed_signal_date = signal_date

    if strategy_run.sector_review_due:
        state.last_sector_rebalance = signal_date

    if strategy_run.rebalance_plan.reason == "CONFIRMED_TARGET_STATE":
        state.target_weights = dict(strategy_run.rebalance_plan.execution_weights)
        state.executed_leader = result.leader
        state.executed_regime = result.regime
        state.executed_volatility_tier = result.volatility_tier

    save_state(state)


def prepare_notification_delivery(
    strategy_run: StrategyRun,
    notification: NotificationDecision,
) -> None:
    """Stage an exact, confirmable recommendation before attempting SMTP."""
    if not notification.should_send:
        return

    state = strategy_run.state
    if notification.kind in {"ACTION", "UPDATE"}:
        result = strategy_run.result
        state.pending_recommendation_date = (
            strategy_run.signal_date.date().isoformat()
        )
        state.pending_recommendation_leader = result.leader
        state.pending_recommendation_regime = result.regime
        state.pending_recommendation_tier = result.volatility_tier
        state.pending_recommendation_weights = dict(
            strategy_run.rebalance_plan.execution_weights
        )
        state.pending_recommendation_notified = False
        state.pending_recommendation_supersedes_date = (
            notification.previous_recommendation_date
            if notification.kind == "UPDATE"
            else ""
        )
    elif notification.kind in {"RETRY", "UPDATE_RETRY"}:
        if not state.pending_recommendation_date:
            raise RuntimeError("Cannot retry a missing pending recommendation")
        state.pending_recommendation_notified = False
    elif notification.kind != "CANCELLATION":
        raise ValueError(f"Unsupported notification kind: {notification.kind}")


def persist_notification_delivery(
    strategy_run: StrategyRun,
    notification: NotificationDecision,
) -> None:
    """Record a notification only after SMTP delivery succeeds."""
    if not notification.should_send:
        raise ValueError("A NONE notification cannot be persisted as delivered")

    state = strategy_run.state
    if notification.kind in {"ACTION", "UPDATE", "RETRY", "UPDATE_RETRY"}:
        if not state.pending_recommendation_date:
            raise RuntimeError("Delivered recommendation is missing its outbox state")
        state.pending_recommendation_notified = True
        state.pending_recommendation_supersedes_date = ""
    elif notification.kind == "CANCELLATION":
        state.pending_recommendation_date = ""
        state.pending_recommendation_leader = None
        state.pending_recommendation_regime = "UNKNOWN"
        state.pending_recommendation_tier = "N/A"
        state.pending_recommendation_weights = {}
        state.pending_recommendation_notified = False
        state.pending_recommendation_supersedes_date = ""
    else:
        raise ValueError(f"Unsupported notification kind: {notification.kind}")

    save_state(state)


def validate_execution_confirmation(
    state: PortfolioState,
    executed_signal_date: str,
) -> bool:
    if not state.pending_recommendation_date:
        raise RuntimeError("There is no pending recommendation to confirm")
    if state.pending_recommendation_date == executed_signal_date:
        return True

    executed_date = _parse_iso_date(executed_signal_date, "executed_signal_date")
    pending_date = _parse_iso_date(
        state.pending_recommendation_date,
        "pending_recommendation_date",
    )
    if executed_date and pending_date and executed_date < pending_date:
        logger.warning(
            "Reconciling fills for superseded signal %s; current pending signal %s "
            "will be cleared and recalculated from confirmed holdings",
            executed_signal_date,
            state.pending_recommendation_date,
        )
        return False

    raise RuntimeError(
        "Confirmed fills do not match the pending recommendation: "
        f"expected {state.pending_recommendation_date}, got {executed_signal_date}"
    )


def confirm_execution(
    executed_shares: dict[str, float],
    executed_cash: float,
    executed_signal_date: str,
) -> PortfolioState:
    state = load_state()
    matches_current_recommendation = validate_execution_confirmation(
        state,
        executed_signal_date,
    )

    state.shares = dict(executed_shares)
    state.cash_balance = float(executed_cash)
    if matches_current_recommendation:
        state.target_weights = dict(state.pending_recommendation_weights)
        state.executed_leader = state.pending_recommendation_leader
        state.executed_regime = state.pending_recommendation_regime
        state.executed_volatility_tier = state.pending_recommendation_tier
    else:
        state.target_weights = {}
        state.executed_leader = None
        state.executed_regime = "UNKNOWN"
        state.executed_volatility_tier = "N/A"

    state.pending_recommendation_date = ""
    state.pending_recommendation_leader = None
    state.pending_recommendation_regime = "UNKNOWN"
    state.pending_recommendation_tier = "N/A"
    state.pending_recommendation_weights = {}
    state.pending_recommendation_notified = False
    state.pending_recommendation_supersedes_date = ""

    save_state(state)
    logger.info(
        "Execution confirmed for signal %s: holdings=%s",
        executed_signal_date,
        len(executed_shares),
    )
    return state


def sync_holdings(
    executed_shares: dict[str, float],
    executed_cash: float,
) -> PortfolioState:
    state = load_state()
    if state.pending_recommendation_date:
        raise RuntimeError(
            "Cannot sync holdings while a recommendation is pending; confirm or "
            "resolve that recommendation first"
        )
    state.shares = dict(executed_shares)
    state.cash_balance = float(executed_cash)
    save_state(state)
    logger.info("Holdings synchronized: holdings=%s", len(executed_shares))
    return state


def log_decision(strategy_run: StrategyRun) -> None:
    plan = strategy_run.rebalance_plan
    target = _with_cash_target(strategy_run.result.target_weights)
    components = set(strategy_run.current_weights) | set(target)
    drifts = {
        component: target.get(component, 0.0)
        - strategy_run.current_weights.get(component, 0.0)
        for component in sorted(components)
    }
    logger.info(
        "decision signal_date=%s regime=%s tier=%s raw_tier=%s leader=%s "
        "volatility=%.6f volatility_10=%.6f volatility_30=%.6f "
        "soxl_momentum=%.6f tecl_momentum=%.6f sector_due=%s "
        "tier_transition=%s replayed_sessions=%s rebalance_due=%s "
        "full_transition=%s reason=%s one_way_turnover=%.6f orders=%s "
        "current=%s strategic_target=%s execution_target=%s drift=%s",
        strategy_run.signal_date.date(),
        strategy_run.result.regime,
        strategy_run.result.volatility_tier,
        strategy_run.result.raw_volatility_tier,
        strategy_run.result.leader,
        strategy_run.result.annualized_volatility,
        strategy_run.latest_indicators["volatility_10"],
        strategy_run.latest_indicators["volatility_30"],
        strategy_run.latest_indicators["soxl_momentum"],
        strategy_run.latest_indicators["tecl_momentum"],
        strategy_run.sector_review_due,
        strategy_run.tier_decision.transition,
        strategy_run.tier_decision.processed_sessions,
        plan.rebalance_due,
        plan.full_transition,
        plan.reason,
        plan.one_way_turnover,
        plan.individual_orders,
        strategy_run.current_weights,
        target,
        plan.execution_weights,
        drifts,
    )

# =============================================================================
# 11. REPORTING
# =============================================================================
def build_dashboard(strategy_run: StrategyRun) -> str:
    result = strategy_run.result
    latest = strategy_run.latest_indicators
    plan = strategy_run.rebalance_plan
    border = "=" * 132
    divider = "-" * 132
    follower = (
        LEVERAGED_TECH
        if result.leader == LEVERAGED_SEMICONDUCTOR
        else LEVERAGED_SEMICONDUCTOR
    )
    allocation_title = (
        "EXECUTION DESTINATION - ESTIMATES AT SIGNAL CLOSE"
        if plan.rebalance_due
        else "STRATEGIC TARGET - NO TRADE"
    )

    lines = [
        border,
        "  ROTH IRA - BARBELL MOMENTUM ENGINE",
        border,
        f"  Signal Date: {strategy_run.signal_date.date()}   "
        f"Portfolio Value: ${strategy_run.portfolio_value:,.2f}",
        f"  Regime: {result.regime}   "
        f"Consensus: SMA:{int(latest['sma_signal'])} "
        f"Donchian:{int(latest['donchian_signal'])} "
        f"VWMA:{int(latest['vwma_signal'])}",
        f"  QQQ Vol: {result.annualized_volatility:.1%} "
        f"(10d {latest['volatility_10']:.1%}, 30d {latest['volatility_30']:.1%})   "
        f"Tier: {result.volatility_tier} (raw {result.raw_volatility_tier})",
        f"  Leader: {result.leader}   Follower: {follower}   "
        f"Sector Review Due: {'YES' if strategy_run.sector_review_due else 'NO'}",
        f"  Rebalance: {'YES' if plan.rebalance_due else 'NO'}   "
        f"Reason: {plan.reason}   One-way Turnover: {plan.one_way_turnover:.1%}   "
        f"Estimated Orders: {plan.individual_orders}",
        divider,
        f"  {allocation_title}",
        divider,
        (
            f"  {'Ticker':<7}{'Price':>11}{'Current':>13}{'Target %':>11}"
            f"{'Target $':>14}{'Est. Units':>14}{'Delta':>13}{'Est. Trade $':>15}"
            f"{'Action':>16}"
        ),
        divider,
    ]

    for _, row in strategy_run.execution_table.iterrows():
        lines.append(
            f"  {row['Ticker']:<7}"
            f"${row['Price']:>10,.2f}"
            f"{row['CurrentUnits']:>13,.4f}"
            f"{row['TargetPct'] * 100:>10.1f}%"
            f"${row['TargetValue']:>13,.2f}"
            f"{row['EstimatedUnits']:>14,.4f}"
            f"{row['DeltaUnits']:>13,.4f}"
            f"${row['DeltaValue']:>14,.2f}"
            f"{row['Action']:>16}"
        )

    lines.extend(
        [
            divider,
            "  Quantities are estimates using the signal close. Recalculate actual "
            "orders from next-session executable prices.",
            f"  Drift trigger/destination: {REBALANCE_BAND:.0%} / "
            f"{REBALANCE_DESTINATION:.1%}   "
            f"Leader review: {SECTOR_REBALANCE_DAYS} sessions   "
            f"Re-risk confirmation: {VOLATILITY_RERISK_PERSISTENCE} closes",
            border,
        ]
    )
    return "\n".join(lines)


def build_email_html(
    strategy_run: StrategyRun,
    notification: NotificationDecision,
) -> str:
    result = strategy_run.result
    plan = strategy_run.rebalance_plan
    status = {
        "ACTION": "ACTION REQUIRED",
        "UPDATE": "UPDATED ACTION REQUIRED",
        "RETRY": "ACTION REQUIRED - DELIVERY RETRY",
        "UPDATE_RETRY": "UPDATED ACTION REQUIRED - DELIVERY RETRY",
        "CANCELLATION": "PREVIOUS ACTION CANCELLED",
    }.get(notification.kind, "HOLD")
    color = "#b42318" if plan.rebalance_due else "#667085"
    rows: list[str] = []
    for _, row in strategy_run.execution_table.iterrows():
        rows.append(
            "<tr>"
            f"<td>{row['Ticker']}</td>"
            f"<td>${row['Price']:,.2f}</td>"
            f"<td>{row['CurrentUnits']:,.4f}</td>"
            f"<td>{row['TargetPct']:.1%}</td>"
            f"<td>{row['EstimatedUnits']:,.4f}</td>"
            f"<td>{row['DeltaUnits']:,.4f}</td>"
            f"<td>${row['DeltaValue']:,.2f}</td>"
            f"<td>{row['Action']}</td>"
            "</tr>"
        )
    return f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background:#f7f8fa; color:#1d2939; }}
.card {{ max-width:900px; margin:20px auto; background:white; border-radius:10px; overflow:hidden; box-shadow:0 2px 10px rgba(0,0,0,.08); }}
.header {{ background:#101828; color:white; padding:22px; text-align:center; }}
.status {{ padding:14px; text-align:center; font-weight:700; color:{color}; background:#f2f4f7; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th, td {{ padding:8px; border-bottom:1px solid #eaecf0; text-align:right; }}
th:first-child, td:first-child {{ text-align:left; }}
.note {{ padding:16px; color:#667085; font-size:12px; }}
</style></head>
<body><div class="card">
<div class="header"><h2>ROTH IRA - BARBELL MOMENTUM ENGINE</h2>
<div>Signal date: {strategy_run.signal_date.date()}</div></div>
<div class="status">{status}: {plan.reason}</div>
<div class="note">Regime: {result.regime} | Tier: {result.volatility_tier} | Leader: {result.leader} |
Portfolio: ${strategy_run.portfolio_value:,.2f} | One-way turnover: {plan.one_way_turnover:.1%}</div>
<table><thead><tr><th>Ticker</th><th>Price</th><th>Current</th><th>Target</th><th>Est. Units</th><th>Delta</th><th>Est. Trade</th><th>Action</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="note">Quantities use the signal close and are estimates. Recalculate actual orders from next-session executable prices.</div>
</div></body></html>
"""


def notification_subject(
    strategy_run: StrategyRun,
    notification: NotificationDecision,
) -> str:
    label = {
        "ACTION": "Action Required",
        "UPDATE": "Action Updated",
        "RETRY": "Action Required (Retry)",
        "UPDATE_RETRY": "Action Updated (Retry)",
        "CANCELLATION": "Action Cancelled",
    }.get(notification.kind)
    if label is None:
        raise ValueError("A NONE notification has no email subject")
    subject_date = (
        notification.previous_recommendation_date
        if notification.kind in {"RETRY", "UPDATE_RETRY"}
        else strategy_run.signal_date.date().isoformat()
    )
    return f"ROTH IRA {label} - {subject_date}"


def notification_text_body(
    strategy_run: StrategyRun,
    notification: NotificationDecision,
    dashboard: str,
) -> str:
    if notification.kind == "CANCELLATION":
        prefix = (
            "The portfolio action for signal "
            f"{notification.previous_recommendation_date} is no longer required. "
            "Do not execute the prior recommendation.\n\n"
        )
    elif notification.kind == "UPDATE":
        prefix = (
            "This recommendation replaces the portfolio action for signal "
            f"{notification.previous_recommendation_date}.\n\n"
        )
    elif notification.kind == "RETRY":
        prefix = (
            "Delivery of the portfolio action for signal "
            f"{notification.previous_recommendation_date} is being retried.\n\n"
        )
    elif notification.kind == "UPDATE_RETRY":
        prefix = (
            "Delivery of the updated portfolio action for signal "
            f"{notification.previous_recommendation_date} is being retried. "
            "It replaces the previously delivered action for signal "
            f"{notification.supersedes_recommendation_date}.\n\n"
        )
    elif notification.kind == "ACTION":
        prefix = "A portfolio update is required.\n\n"
    else:
        raise ValueError("A NONE notification has no email body")
    return prefix + dashboard


def send_email(subject: str, text_body: str, html_body: str) -> None:
    address = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("RECEIVER_EMAIL")
    if not all((address, password, recipient)):
        raise RuntimeError(
            "GMAIL_ADDRESS, GMAIL_APP_PASSWORD, and RECEIVER_EMAIL are required "
            "when a portfolio notification is due"
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

# =============================================================================
# 12. COMMAND-LINE PARSING
# =============================================================================
def parse_executed_shares(
    entries: list[str] | None,
) -> tuple[dict[str, float] | None, float]:
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
            continue
        if name not in TRADED_TICKERS or name in holdings:
            raise ValueError(f"Invalid or duplicate holding: {entry!r}")
        holdings[name] = value

    if cash is None:
        cash = 0.0
    return holdings, cash


def parse_signal_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError("--executed-signal-date must be a valid date") from exc


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROTH IRA - Barbell Momentum Allocation Engine"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--confirm-execution",
        action="store_true",
        help="Confirm a pending recommendation and exit",
    )
    mode.add_argument(
        "--sync-holdings",
        action="store_true",
        help="Synchronize actual holdings/cash when no recommendation is pending",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate the report without saving state or sending email",
    )
    parser.add_argument(
        "--roth-amount",
        type=float,
        default=None,
        help="First-run cash amount only",
    )
    parser.add_argument(
        "--executed-shares",
        nargs="+",
        metavar="TICKER=SHARES",
        help="Actual holdings plus optional CASH amount",
    )
    parser.add_argument(
        "--executed-signal-date",
        metavar="YYYY-MM-DD",
        help="Signal date of the pending recommendation being confirmed",
    )
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
            parser.error(
                "--confirm-execution requires --executed-shares and "
                "--executed-signal-date"
            )
        confirm_execution(executed_shares, executed_cash, executed_signal_date)
        print(
            f"Confirmed execution for signal {executed_signal_date}: "
            f"{len(executed_shares)} holdings, cash ${executed_cash:,.2f}"
        )
        return

    if args.sync_holdings:
        if args.test or args.roth_amount is not None or executed_signal_date is not None:
            parser.error(
                "--sync-holdings cannot be combined with --test, --roth-amount, "
                "or --executed-signal-date"
            )
        if executed_shares is None:
            parser.error("--sync-holdings requires --executed-shares")
        sync_holdings(executed_shares, executed_cash)
        print(
            f"Synchronized {len(executed_shares)} holdings and cash "
            f"${executed_cash:,.2f}"
        )
        return

    if executed_shares is not None or executed_signal_date is not None:
        parser.error(
            "Execution fields require --confirm-execution or --sync-holdings"
        )

    strategy_run = run_strategy(
        args.roth_amount,
        backup_legacy_state=not args.test,
    )
    log_decision(strategy_run)
    dashboard = build_dashboard(strategy_run)
    print(dashboard)
    notification = decide_notification(strategy_run)
    logger.info(
        "notification kind=%s reason=%s",
        notification.kind,
        notification.reason,
    )

    if not args.test:
        # Atomically checkpoint confirmed holdings, signal progression, and the
        # exact outbox action before SMTP. Delivery is marked separately so an
        # ambiguous failure remains both confirmable and retryable.
        prepare_notification_delivery(strategy_run, notification)
        persist_signal_run(strategy_run)
        if notification.should_send:
            send_email(
                notification_subject(strategy_run, notification),
                notification_text_body(strategy_run, notification, dashboard),
                build_email_html(strategy_run, notification),
            )
            persist_notification_delivery(strategy_run, notification)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("ROTH IRA engine failed")
        sys.exit(1)

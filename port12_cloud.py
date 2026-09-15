#!/usr/bin/env python3
"""Static annual Roth IRA allocator.

The engine holds one fixed allocation throughout the year and only recommends
an exact rebalance on the first completed NYSE session of a new calendar year,
or when a new strategy revision must be adopted. It never trades on market
signals. Confirmed broker shares and cash remain the source of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import smtplib
import sys
from dataclasses import asdict, dataclass, field
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


STRATEGY_REVISION = "static-annual-tqqq35-dbmf25-ugl20-zroz15-btal5-v2"
STATE_VERSION = 20
AUDIT_SCHEMA_VERSION = 12
MODEL_START_DATE = core.MODEL_HISTORY_START
NEW_YORK = ZoneInfo("America/New_York")
MARKET_CLOSE_BUFFER_MINUTES = 15
MARKET_DATA_DOWNLOAD_ATTEMPTS = 3
NOTIFICATION_WEIGHT_TOLERANCE = 0.005
TRANSACTION_COST_SCENARIOS_BPS = (5, 10, 25)

TQQQ, DBMF, UGL, ZROZ, BTAL, CASH = (
    core.TQQQ,
    core.DBMF,
    core.UGL,
    core.ZROZ,
    core.BTAL,
    core.CASH,
)
STRATEGIC_TICKERS = (TQQQ, DBMF, UGL, ZROZ, BTAL)
VALUATION_TICKERS = STRATEGIC_TICKERS
ALL_TICKERS = STRATEGIC_TICKERS
TRADED_TICKERS = frozenset(VALUATION_TICKERS)
PORTFOLIO_COMPONENTS = TRADED_TICKERS | {CASH}
ADVERTISED_DAILY_MULTIPLIERS = {
    **core.ADVERTISED_DAILY_MULTIPLIERS,
}

APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / "roth_ira_state.json"
LOG_FILE = APP_DIR / "roth_ira.log"
AUDIT_FILE = APP_DIR / "roth_ira_decision.json"

_configured_amount = os.environ.get("ROTH_IRA_AMOUNT", "").strip()
try:
    ROTH_IRA_AMOUNT = float(_configured_amount) if _configured_amount else None
except ValueError as exc:
    raise RuntimeError("ROTH_IRA_AMOUNT must be numeric") from exc

logger = logging.getLogger("roth_ira")
logger.setLevel(logging.INFO)
logger.propagate = False


@dataclass
class PortfolioState:
    state_version: int = STATE_VERSION
    shares: dict[str, float] = field(default_factory=dict)
    cash_balance: float = 0.0
    target_weights: dict[str, float] = field(default_factory=dict)
    portfolio_value: float = 0.0
    strategy_initialized: bool = False
    last_processed_session_date: str = ""
    last_completed_annual_rebalance_year: int = 0
    executed_strategy_fingerprint: str = ""
    pending_recommendation_date: str = ""
    pending_recommendation_weights: dict[str, float] = field(default_factory=dict)
    pending_recommendation_notified: bool = False
    pending_recommendation_supersedes_date: str = ""
    pending_recommendation_fingerprint: str = ""
    pending_recommendation_annual_year: int = 0
    contribution_plan_year: int = 0
    contribution_policy_revision: str = ""
    contribution_budget: float = 0.0
    contribution_released_amount: float = 0.0
    pending_contribution_date: str = ""
    pending_contribution_amount: float = 0.0
    last_contribution_notice_date: str = ""
    last_updated: str = ""


@dataclass(frozen=True)
class StaticDecision:
    target_weights: dict[str, float]
    session_date: pd.Timestamp
    transition_reason: str
    structural_change: bool


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
class ContributionPlan:
    enabled: bool = False
    year: int = 0
    budget: float = 0.0
    released_amount: float = 0.0
    due_amount: float = 0.0
    reasons: tuple[str, ...] = ()
    retry: bool = False
    allocation_dollars: dict[str, float] = field(default_factory=dict)
    estimated_units: dict[str, float] = field(default_factory=dict)

    @property
    def notification_due(self) -> bool:
        return self.due_amount >= 0.01


@dataclass(frozen=True)
class NotificationDecision:
    kind: str
    reason: str
    previous_recommendation_date: str = ""

    @property
    def should_send(self) -> bool:
        return self.kind != "NONE"


@dataclass(frozen=True)
class StrategyRun:
    price_data: pd.DataFrame
    decision: StaticDecision
    state: PortfolioState
    planning_state: PortfolioState
    portfolio_value: float
    current_weights: dict[str, float]
    execution_table: pd.DataFrame
    session_date: pd.Timestamp
    market_data_fingerprint: str
    rebalance_plan: RebalancePlan
    contribution_plan: ContributionPlan


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strategy_manifest() -> dict[str, object]:
    return {
        "revision": STRATEGY_REVISION,
        "quantitative_revision": core.DECISION_SEMANTIC_REVISION,
        "signals": "none",
        "rebalance": "first_completed_XNYS_session_each_calendar_year",
        "execution": "next-session_manual_orders_then_confirm_holdings",
        "target": core.target_weights(),
        "advertised_daily_exposure": core.advertised_daily_exposure(),
    }


STRATEGY_FINGERPRINT = canonical_sha256(strategy_manifest())
EXPECTED_STRATEGY_FINGERPRINT = STRATEGY_FINGERPRINT


def _is_number(value: object, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return False
    numeric = float(value)
    return bool(np.isfinite(numeric) and (not positive or numeric > 0))


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _parse_date(value: str, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be YYYY-MM-DD") from exc


def _validate_weights(weights: object, name: str, *, required: bool) -> None:
    if not isinstance(weights, dict):
        raise RuntimeError(f"{name} must be a mapping")
    if set(weights) - PORTFOLIO_COMPONENTS:
        raise RuntimeError(f"{name} contains unsupported holdings")
    if any(not _is_number(value) or float(value) < 0 for value in weights.values()):
        raise RuntimeError(f"{name} contains invalid weights")
    if required and not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
        raise RuntimeError(f"{name} must sum to one")


def validate_state(state: PortfolioState) -> None:
    if state.state_version != STATE_VERSION:
        raise RuntimeError("Portfolio state version is invalid")
    if set(state.shares) - TRADED_TICKERS:
        raise RuntimeError("shares contains unsupported holdings")
    if any(not _is_number(value) or float(value) < 0 for value in state.shares.values()):
        raise RuntimeError("shares contains invalid quantities")
    if not _is_number(state.cash_balance) or not _is_number(state.portfolio_value):
        raise RuntimeError("Portfolio balances are invalid")
    _validate_weights(state.target_weights, "target_weights", required=bool(state.target_weights))
    _validate_weights(
        state.pending_recommendation_weights,
        "pending_recommendation_weights",
        required=bool(state.pending_recommendation_date),
    )
    for name in ("pending_recommendation_notified",):
        if not isinstance(getattr(state, name), bool):
            raise RuntimeError(f"{name} must be boolean")
    for name in ("last_completed_annual_rebalance_year", "pending_recommendation_annual_year", "contribution_plan_year"):
        value = getattr(state, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(f"{name} is invalid")
    for name in ("contribution_budget", "contribution_released_amount", "pending_contribution_amount"):
        if not _is_number(getattr(state, name)) or float(getattr(state, name)) < 0:
            raise RuntimeError(f"{name} is invalid")
    if state.pending_recommendation_date:
        _parse_date(state.pending_recommendation_date, "pending_recommendation_date")
        _parse_date(state.pending_recommendation_supersedes_date, "pending_recommendation_supersedes_date")
        if not _is_sha256(state.pending_recommendation_fingerprint):
            raise RuntimeError("Pending recommendation fingerprint is invalid")
        if state.pending_recommendation_annual_year < 0:
            raise RuntimeError("Pending recommendation year is invalid")
    elif any((state.pending_recommendation_notified, state.pending_recommendation_supersedes_date,
              state.pending_recommendation_fingerprint, state.pending_recommendation_annual_year,
              state.pending_recommendation_weights)):
        raise RuntimeError("Pending recommendation state is inconsistent")
    if state.contribution_plan_year:
        if state.contribution_policy_revision != contribution.POLICY_REVISION or state.contribution_budget <= 0:
            raise RuntimeError("Contribution plan is invalid")
        if state.contribution_released_amount > state.contribution_budget + 0.005:
            raise RuntimeError("Contribution releases exceed the budget")
        if state.pending_contribution_date:
            _parse_date(state.pending_contribution_date, "pending_contribution_date")
            if state.pending_contribution_amount < 0.01:
                raise RuntimeError("Pending contribution amount is invalid")
    elif any((state.contribution_policy_revision, state.contribution_budget,
              state.contribution_released_amount, state.pending_contribution_date,
              state.pending_contribution_amount, state.last_contribution_notice_date)):
        raise RuntimeError("Disabled contribution plan contains state")
    for name in ("last_processed_session_date", "last_contribution_notice_date"):
        _parse_date(getattr(state, name), name)
    if state.last_updated:
        try:
            datetime.fromisoformat(state.last_updated)
        except ValueError as exc:
            raise RuntimeError("last_updated must be ISO-8601") from exc


def _migrate_mapping(value: object, name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for ticker, quantity in value.items():
        symbol = str(ticker).upper()
        if symbol not in PORTFOLIO_COMPONENTS or not _is_number(quantity) or float(quantity) < 0:
            raise RuntimeError(f"{name} contains invalid holding {symbol}")
        result[symbol] = float(quantity)
    return result


def _migrate_state(payload: dict[str, object]) -> PortfolioState:
    version = payload.get("state_version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or not 2 <= version < STATE_VERSION:
        raise RuntimeError(f"Unsupported state version: {version!r}")
    shares = _migrate_mapping(payload.get("shares"), "shares")
    state = PortfolioState(
        state_version=STATE_VERSION,
        shares=shares,
        cash_balance=float(payload.get("cash_balance", 0.0)),
        portfolio_value=float(payload.get("portfolio_value", 0.0)),
        contribution_plan_year=int(payload.get("contribution_plan_year", 0) or 0),
        contribution_policy_revision=(
            contribution.POLICY_REVISION if payload.get("contribution_plan_year") else ""
        ),
        contribution_budget=float(payload.get("contribution_budget", 0.0) or 0.0),
        contribution_released_amount=float(payload.get("contribution_released_amount", 0.0) or 0.0),
        target_weights={},
        last_completed_annual_rebalance_year=0,
        executed_strategy_fingerprint="",
    )
    # A strategy revision must produce one clean annual recommendation. Old
    # pending signal actions are intentionally cleared while real holdings are
    # preserved.
    return state


def load_state() -> PortfolioState:
    if not STATE_FILE.exists():
        return PortfolioState()
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Portfolio state cannot be read") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Portfolio state must be a JSON object")
    version = payload.get("state_version")
    state = _migrate_state(payload) if version != STATE_VERSION else PortfolioState(**payload)
    validate_state(state)
    return state


def save_state(state: PortfolioState) -> None:
    validate_state(state)
    state.last_updated = datetime.now(NEW_YORK).isoformat()
    temporary = STATE_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(asdict(state), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, STATE_FILE)


def configure_logging(*, persist_log: bool) -> None:
    for existing in logger.handlers:
        existing.close()
    logger.handlers.clear()
    handler: logging.Handler = logging.FileHandler(LOG_FILE) if persist_log else logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


def expected_completed_session(now_new_york: datetime | None = None) -> pd.Timestamp:
    current = now_new_york or datetime.now(NEW_YORK)
    stamp = pd.Timestamp(current)
    stamp = stamp.tz_localize(NEW_YORK) if stamp.tzinfo is None else stamp.tz_convert(NEW_YORK)
    calendar = xcals.get_calendar("XNYS")
    today = pd.Timestamp(stamp.date())
    if calendar.is_session(today):
        close = calendar.session_close(today) + pd.Timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES)
        if calendar.session_open(today) <= stamp.tz_convert("UTC") < close:
            raise RuntimeError("The latest daily bar is not final")
    sessions = calendar.sessions_in_range(today - pd.Timedelta(days=14), today + pd.Timedelta(days=1))
    complete = [session for session in sessions if calendar.session_close(session) + pd.Timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES) <= stamp.tz_convert("UTC")]
    if not complete:
        raise RuntimeError("No completed XNYS session is available")
    return pd.Timestamp(complete[-1]).tz_localize(None).normalize()


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


def download_market_data(tickers: Iterable[str] = ALL_TICKERS, *, now_new_york: datetime | None = None) -> pd.DataFrame:
    import yfinance as yf

    expected = expected_completed_session(now_new_york)
    requested = list(dict.fromkeys(tickers))
    last_error: Exception | None = None
    for attempt in range(1, MARKET_DATA_DOWNLOAD_ATTEMPTS + 1):
        try:
            # yfinance's threaded workers share a SQLite-backed cache. GitHub-hosted
            # runners have intermittently raised "database is locked" while those
            # workers initialize it, leaving one ticker full of NaNs. Downloading this
            # five-ticker universe sequentially avoids that concurrency failure.
            raw = yf.download(
                requested,
                start=MODEL_START_DATE,
                end=(expected.date() + timedelta(days=1)).isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            prices = _extract_yfinance_prices(raw, requested)
            received = pd.Timestamp(prices.index[-1]).normalize()
            if received != expected:
                raise RuntimeError(
                    f"Market data is stale: expected {expected.date()}, received {received.date()}"
                )
            latest = prices.loc[received, requested]
            if not np.isfinite(latest.to_numpy(dtype=float)).all() or not (latest > 0).all():
                raise RuntimeError("Latest valuation prices contain missing or invalid values")
            return prices
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Market-data attempt %d/%d failed: %s",
                attempt,
                MARKET_DATA_DOWNLOAD_ATTEMPTS,
                exc,
            )

    raise RuntimeError(
        f"Market data download failed after {MARKET_DATA_DOWNLOAD_ATTEMPTS} attempts"
    ) from last_error


def market_data_fingerprint(prices: pd.DataFrame) -> str:
    if prices.empty:
        raise ValueError("Market data is empty")
    latest = prices.iloc[-1]
    if not np.isfinite(latest.to_numpy(dtype=float)).all() or not (latest > 0).all():
        raise ValueError("Latest market data is invalid")
    return canonical_sha256({
        "semantic_version": "static-annual-latest-prices-v1",
        "session": pd.Timestamp(prices.index[-1]).date().isoformat(),
        "prices": {str(ticker): round(float(value), 6) for ticker, value in latest.items()},
    })


def target_with_cash() -> dict[str, float]:
    weights = core.target_weights()
    weights[CASH] = 0.0
    return weights


def calculate_strategy_decision(prices: pd.DataFrame, state: PortfolioState) -> StaticDecision:
    if prices.empty:
        raise RuntimeError("Strategy requires market data")
    session = pd.Timestamp(prices.index[-1]).normalize()
    return StaticDecision(target_with_cash(), session, "STATIC_ANNUAL_HOLD", False)


def _with_cash(weights: dict[str, float]) -> dict[str, float]:
    result = dict(weights)
    result.setdefault(CASH, 0.0)
    return result


def advertised_daily_exposure(weights: dict[str, float]) -> float:
    unknown = set(weights) - set(ADVERTISED_DAILY_MULTIPLIERS)
    if unknown:
        raise ValueError(f"Unknown exposure components: {sorted(unknown)}")
    return float(sum(float(w) * ADVERTISED_DAILY_MULTIPLIERS[t] for t, w in weights.items()))


def _one_way_turnover(current: dict[str, float], destination: dict[str, float]) -> float:
    return float(sum(max(0.0, destination.get(t, 0.0) - current.get(t, 0.0)) for t in set(current) | set(destination)))


def _individual_orders(current: dict[str, float], destination: dict[str, float]) -> int:
    return sum(abs(destination.get(t, 0.0) - current.get(t, 0.0)) > NOTIFICATION_WEIGHT_TOLERANCE for t in set(current) | set(destination) if t != CASH)


def existing_portfolio_value(state: PortfolioState, prices: pd.DataFrame) -> float:
    value = float(state.cash_balance)
    for ticker, quantity in state.shares.items():
        if ticker not in prices.columns:
            raise RuntimeError(f"Holding has no current price: {ticker}")
        price = float(prices[ticker].iloc[-1])
        if quantity < 0 or not np.isfinite(price) or price <= 0:
            raise RuntimeError(f"Invalid confirmed holding or price: {ticker}")
        value += quantity * price
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("Portfolio value must be positive")
    return value


def existing_weights(state: PortfolioState, prices: pd.DataFrame) -> dict[str, float]:
    value = existing_portfolio_value(state, prices)
    result = {ticker: quantity * float(prices[ticker].iloc[-1]) / value for ticker, quantity in state.shares.items() if quantity > 1e-14}
    if state.cash_balance > 1e-10:
        result[CASH] = state.cash_balance / value
    return result or {CASH: 1.0}


def resolve_portfolio_value(roth_amount: float | None, state: PortfolioState, prices: pd.DataFrame) -> tuple[float, PortfolioState]:
    planning = PortfolioState(**asdict(state))
    if state.shares or state.cash_balance > 0:
        if roth_amount is not None:
            raise RuntimeError("--roth-amount is valid only for a new all-cash state")
        return existing_portfolio_value(planning, prices), planning
    amount = roth_amount if roth_amount is not None else ROTH_IRA_AMOUNT
    if not _is_number(amount, positive=True):
        raise RuntimeError("No portfolio value is available; initialize explicitly")
    planning.cash_balance = float(amount)
    return float(amount), planning


def build_rebalance_plan(existing: dict[str, float], decision: StaticDecision, state: PortfolioState) -> RebalancePlan:
    year = decision.session_date.year
    annual_due = year > state.last_completed_annual_rebalance_year
    strategy_changed = state.executed_strategy_fingerprint != STRATEGY_FINGERPRINT
    missing = sum(existing.get(ticker, 0.0) for ticker in STRATEGIC_TICKERS) <= 1e-12
    full = strategy_changed or annual_due or missing
    due = full
    reason = "STRATEGY_REVISION_TRANSITION" if strategy_changed else "ANNUAL_REBALANCE" if annual_due else "MISSING_TACTICAL_POSITION" if missing else "HOLD"
    execution = target_with_cash() if due else dict(existing)
    turnover = _one_way_turnover(existing, execution) if due else 0.0
    orders = _individual_orders(existing, execution) if due else 0
    if due and orders == 0:
        due = False
        full = False
        reason = "ANNUAL_REVIEW" if annual_due else "CONFIRMED_TARGET_STATE"
    return RebalancePlan(execution, due, full, reason, turnover, orders, annual_due, year if annual_due else 0)


def calculate_execution_table(prices: pd.DataFrame, target: dict[str, float], value: float, state: PortfolioState, *, actionable: bool) -> pd.DataFrame:
    rows = []
    for ticker in sorted(set(state.shares) | set(target) | {CASH}):
        price = 1.0 if ticker == CASH else float(prices[ticker].iloc[-1])
        current = state.cash_balance if ticker == CASH else state.shares.get(ticker, 0.0)
        estimated = value * target.get(ticker, 0.0) / price
        delta = estimated - current
        action = "HOLD"
        if actionable and abs(delta * price) > 0.005:
            action = "CASH AFTER TRADES" if ticker == CASH else ("BUY" if delta > 0 else "SELL")
        rows.append({"Ticker": ticker, "Price": price, "CurrentUnits": current, "TargetPct": target.get(ticker, 0.0), "EstimatedUnits": estimated, "DeltaUnits": delta if actionable else 0.0, "DeltaValue": delta * price if actionable else 0.0, "Action": action})
    return pd.DataFrame(rows)


def _allocate_contribution(amount: float, state: PortfolioState, prices: pd.DataFrame, target: dict[str, float], value: float) -> tuple[dict[str, float], dict[str, float]]:
    if amount < 0.01:
        return {}, {}
    post_value = value + amount
    deficits = {ticker: max(0.0, weight * post_value - state.shares.get(ticker, 0.0) * float(prices[ticker].iloc[-1])) for ticker, weight in target.items()}
    total = sum(deficits.values())
    proportions = {ticker: deficit / total for ticker, deficit in deficits.items()} if total > 1e-12 else target
    dollars = {ticker: round(amount * weight, 2) for ticker, weight in proportions.items() if weight > 0}
    residual = round(amount - sum(dollars.values()), 2)
    if dollars and abs(residual) >= 0.01:
        largest = max(dollars, key=dollars.get)
        dollars[largest] = round(dollars[largest] + residual, 2)
    units = {ticker: dollars[ticker] / float(prices[ticker].iloc[-1]) for ticker in dollars}
    return dollars, units


def build_contribution_plan(state: PortfolioState, prices: pd.DataFrame, value: float, decision: StaticDecision, plan: RebalancePlan) -> ContributionPlan:
    if not state.contribution_plan_year or state.contribution_plan_year != decision.session_date.year or state.contribution_budget <= state.contribution_released_amount + 0.005:
        return ContributionPlan(enabled=bool(state.contribution_plan_year), year=state.contribution_plan_year, budget=state.contribution_budget, released_amount=state.contribution_released_amount)
    if not (plan.annual_rebalance_due or plan.full_transition):
        return ContributionPlan(enabled=True, year=state.contribution_plan_year, budget=state.contribution_budget, released_amount=state.contribution_released_amount)
    amount = round(state.contribution_budget - state.contribution_released_amount, 2)
    dollars, units = _allocate_contribution(amount, state, prices, core.target_weights(), value)
    return ContributionPlan(True, state.contribution_plan_year, state.contribution_budget, state.contribution_released_amount, amount, ("ANNUAL_CONTRIBUTION",), False, dollars, units)


def _pending_matches(run: StrategyRun) -> bool:
    state = run.state
    return bool(state.pending_recommendation_date and state.pending_recommendation_fingerprint == STRATEGY_FINGERPRINT and all(abs(state.pending_recommendation_weights.get(t, 0.0) - run.rebalance_plan.execution_weights.get(t, 0.0)) <= NOTIFICATION_WEIGHT_TOLERANCE for t in set(state.pending_recommendation_weights) | set(run.rebalance_plan.execution_weights)))


def decide_notification(run: StrategyRun) -> NotificationDecision:
    state = run.state
    if run.rebalance_plan.rebalance_due and run.rebalance_plan.individual_orders:
        if not state.pending_recommendation_date:
            return NotificationDecision("ACTION", "ANNUAL_REBALANCE_REQUIRED")
        if _pending_matches(run):
            if not state.pending_recommendation_notified:
                return NotificationDecision("ACTION", "RETRY_UNDELIVERED_RECOMMENDATION", state.pending_recommendation_date)
            return NotificationDecision("NONE", "IDENTICAL_PENDING_RECOMMENDATION", state.pending_recommendation_date)
        return NotificationDecision("UPDATE", "MATERIAL_RECOMMENDATION_UPDATE", state.pending_recommendation_date)
    if state.pending_recommendation_date:
        return NotificationDecision("CANCELLATION", "PENDING_ACTION_NO_LONGER_REQUIRED", state.pending_recommendation_date)
    if run.rebalance_plan.annual_rebalance_due:
        return NotificationDecision("ANNUAL_REVIEW", "ANNUAL_ALLOCATION_CONFIRMED")
    if run.contribution_plan.notification_due:
        return NotificationDecision("CONTRIBUTION", "ANNUAL_CONTRIBUTION")
    return NotificationDecision("NONE", "HOLD")


def _clear_pending(state: PortfolioState) -> None:
    state.pending_recommendation_date = ""
    state.pending_recommendation_weights = {}
    state.pending_recommendation_notified = False
    state.pending_recommendation_supersedes_date = ""
    state.pending_recommendation_fingerprint = ""
    state.pending_recommendation_annual_year = 0


def _clear_pending_contribution(state: PortfolioState) -> None:
    state.pending_contribution_date = ""
    state.pending_contribution_amount = 0.0


def run_strategy(roth_amount: float | None) -> StrategyRun:
    state = load_state()
    requested = set(STRATEGIC_TICKERS) | set(state.shares)
    prices = download_market_data(requested)
    session = pd.Timestamp(prices.index[-1]).normalize()
    value, planning = resolve_portfolio_value(roth_amount, state, prices)
    decision = calculate_strategy_decision(prices, state)
    current = existing_weights(planning, prices)
    plan = build_rebalance_plan(current, decision, planning)
    target = plan.execution_weights if plan.rebalance_due else target_with_cash()
    table = calculate_execution_table(prices, target, value, planning, actionable=plan.rebalance_due)
    contribution_plan = build_contribution_plan(state, prices, value, decision, plan)
    return StrategyRun(prices, decision, state, planning, value, current, table, session, market_data_fingerprint(prices), plan, contribution_plan)


def prepare_notification_delivery(run: StrategyRun, notification: NotificationDecision) -> None:
    if notification.kind in {"ACTION", "UPDATE"}:
        state = run.state
        state.pending_recommendation_date = run.session_date.date().isoformat()
        state.pending_recommendation_weights = dict(run.rebalance_plan.execution_weights)
        state.pending_recommendation_notified = False
        state.pending_recommendation_supersedes_date = notification.previous_recommendation_date if notification.kind == "UPDATE" else ""
        state.pending_recommendation_fingerprint = STRATEGY_FINGERPRINT
        state.pending_recommendation_annual_year = run.rebalance_plan.annual_rebalance_year
    elif notification.kind == "CANCELLATION":
        _clear_pending(run.state)
    elif notification.kind == "NONE":
        return
    elif notification.kind not in {"ANNUAL_REVIEW", "CONTRIBUTION"}:
        raise ValueError(f"Unsupported notification kind: {notification.kind}")


def prepare_contribution_delivery(run: StrategyRun) -> None:
    if not run.contribution_plan.notification_due:
        return
    if not run.state.pending_contribution_date:
        run.state.pending_contribution_date = run.session_date.date().isoformat()
        run.state.pending_contribution_amount = run.contribution_plan.due_amount


def persist_run(run: StrategyRun) -> None:
    state = run.state
    if not state.shares and state.cash_balance == 0:
        state.cash_balance = run.planning_state.cash_balance
    state.strategy_initialized = True
    state.portfolio_value = round(run.portfolio_value, 2)
    state.last_processed_session_date = run.session_date.date().isoformat()
    state.target_weights = dict(run.rebalance_plan.execution_weights if run.rebalance_plan.rebalance_due else target_with_cash())
    state.executed_strategy_fingerprint = STRATEGY_FINGERPRINT if run.rebalance_plan.reason in {"CONFIRMED_TARGET_STATE", "ANNUAL_REVIEW"} else state.executed_strategy_fingerprint
    save_state(state)


def persist_notification_delivery(run: StrategyRun, notification: NotificationDecision) -> None:
    state = run.state
    if notification.kind in {"ACTION", "UPDATE"}:
        state.pending_recommendation_notified = True
    elif notification.kind == "ANNUAL_REVIEW":
        state.last_completed_annual_rebalance_year = run.rebalance_plan.annual_rebalance_year
        state.executed_strategy_fingerprint = STRATEGY_FINGERPRINT
    elif notification.kind == "CANCELLATION":
        _clear_pending(state)
    if run.contribution_plan.notification_due:
        state.contribution_released_amount = min(state.contribution_budget, round(state.contribution_released_amount + run.contribution_plan.due_amount, 2))
        state.last_contribution_notice_date = run.session_date.date().isoformat()
        _clear_pending_contribution(state)
    state.last_processed_session_date = run.session_date.date().isoformat()
    save_state(state)


def confirm_execution(shares: dict[str, float], cash: float, session_date: str) -> PortfolioState:
    state = load_state()
    if not state.pending_recommendation_date:
        raise RuntimeError("There is no pending recommendation to confirm")
    if session_date != state.pending_recommendation_date:
        raise RuntimeError(f"Expected annual recommendation {state.pending_recommendation_date}, got {session_date}")
    state.shares = dict(shares)
    state.cash_balance = float(cash)
    state.target_weights = dict(state.pending_recommendation_weights)
    state.executed_strategy_fingerprint = state.pending_recommendation_fingerprint
    state.last_completed_annual_rebalance_year = state.pending_recommendation_annual_year
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


def configure_contribution_plan(year: int, budget: float) -> PortfolioState:
    if not STATE_FILE.exists():
        raise RuntimeError("Initialize or synchronize portfolio holdings first")
    current_year = datetime.now(NEW_YORK).year
    if year != current_year or not _is_number(budget, positive=True):
        raise ValueError("Contribution year must be the current year and budget must be positive")
    state = load_state()
    if state.pending_contribution_date:
        raise RuntimeError("Cannot replace a contribution plan while delivery is pending")
    state.contribution_plan_year = year
    state.contribution_policy_revision = contribution.POLICY_REVISION
    state.contribution_budget = round(float(budget), 2)
    state.contribution_released_amount = 0.0
    state.last_contribution_notice_date = ""
    save_state(state)
    return state


def disable_contribution_plan() -> PortfolioState:
    state = load_state()
    if state.pending_contribution_date:
        raise RuntimeError("Cannot disable a contribution plan while delivery is pending")
    state.contribution_plan_year = 0
    state.contribution_policy_revision = ""
    state.contribution_budget = 0.0
    state.contribution_released_amount = 0.0
    state.last_contribution_notice_date = ""
    save_state(state)
    return state


def parse_executed_shares(entries: list[str] | None) -> tuple[dict[str, float] | None, float]:
    if entries is None:
        return None, 0.0
    holdings: dict[str, float] = {}
    cash: float | None = None
    for entry in entries:
        name, separator, raw = entry.partition("=")
        name = name.strip().upper()
        if separator != "=":
            raise ValueError(f"Invalid holding: {entry!r}")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid holding: {entry!r}") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Holding must be nonnegative and finite: {entry!r}")
        if name == CASH:
            if cash is not None:
                raise ValueError("CASH was supplied more than once")
            cash = value
        elif name not in TRADED_TICKERS or name in holdings:
            raise ValueError(f"Invalid or duplicate holding: {entry!r}")
        else:
            holdings[name] = value
    if cash is None:
        raise ValueError("Complete holdings must include CASH")
    return holdings, cash


def _visible_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[(table["Ticker"] != CASH) & ((table["Action"] != "HOLD") | (table["TargetPct"] > 1e-12) | (table["CurrentUnits"] > 1e-12))]


def build_dashboard(run: StrategyRun) -> str:
    plan = run.rebalance_plan
    contribution = run.contribution_plan
    status = "ACTION REQUIRED" if plan.rebalance_due or contribution.notification_due else "NO ACTION"
    lines = [
        "=" * 72,
        "ROTH IRA STATIC ANNUAL UPDATE".center(72),
        "=" * 72,
        f"Completed session  {run.session_date.date()}",
        f"Portfolio value    ${run.portfolio_value:,.2f}",
        f"Status             {status}",
        f"Reason             {plan.reason if plan.rebalance_due else ('ANNUAL CONTRIBUTION' if contribution.notification_due else 'HOLD')}",
        "",
        "CURRENT HOLDINGS",
        "-" * 72,
    ]
    for ticker in sorted(set(run.planning_state.shares) | {CASH}):
        if ticker == CASH:
            units = run.planning_state.cash_balance
            value = run.planning_state.cash_balance
        else:
            units = run.planning_state.shares.get(ticker, 0.0)
            value = units * float(run.price_data[ticker].iloc[-1])
        weight = run.current_weights.get(ticker, 0.0)
        if units > 1e-12 or value > 0.01:
            unit_label = "cash" if ticker == CASH else f"{units:,.4f} shares"
            lines.append(f"{ticker:<6} {unit_label:>18}  ${value:>10,.2f}  {weight:>6.1%}")
    lines += [
        "",
        "TARGET ALLOCATION",
        "-" * 72,
        "  ".join(f"{ticker} {weight:.0%}" for ticker, weight in core.target_weights().items()),
        "",
        "Static target; no market signals or midyear trades are used.",
    ]
    if contribution.notification_due:
        lines += ["", f"Contribution deposit  ${contribution.due_amount:,.2f}", "Allocate it using the target percentages below."]
    if plan.rebalance_due:
        lines += ["", "ORDERS (use executable next-session prices)", "-" * 72]
        for _, row in _visible_rows(run.execution_table).iterrows():
            if row["Action"] != "HOLD":
                lines.append(f"{row['Action']:<5} {row['Ticker']:<6} ${abs(row['DeltaValue']):>10,.2f}  target {row['TargetPct']:.0%}")
        lines.append("Confirm complete post-trade shares and CASH after execution.")
    return "\n".join(lines)


def build_unavailable_dashboard(state: PortfolioState, reason: str) -> tuple[str, str]:
    """Build a read-only snapshot when the explicit dashboard request has no valid bar."""
    holdings = []
    for ticker in sorted(set(state.shares) | {CASH}):
        units = state.cash_balance if ticker == CASH else state.shares.get(ticker, 0.0)
        if units > 1e-12:
            label = "cash" if ticker == CASH else f"{units:,.4f} shares"
            holdings.append(f"{ticker:<6} {label:>18}")
    holdings_text = "\n".join(holdings) if holdings else "(No persisted holdings found)"
    target_text = "  ".join(f"{ticker} {weight:.0%}" for ticker, weight in core.target_weights().items())
    text_body = "\n".join([
        "ROTH IRA DASHBOARD SNAPSHOT",
        "VALUATION UNAVAILABLE - NO TRADES RECOMMENDED",
        "",
        f"Last recorded portfolio value  ${state.portfolio_value:,.2f}",
        f"Data warning                  {reason}",
        "",
        "PERSISTED HOLDINGS",
        "-" * 72,
        holdings_text,
        "",
        "TARGET ALLOCATION",
        "-" * 72,
        target_text,
        "",
        "No prices or orders were used. Retry the dashboard after the market-data provider publishes a completed session.",
    ])
    cell = "padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:14px"
    head = "padding:9px 12px;border-bottom:2px solid #d1d5db;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#6b7280;font-weight:600"
    current_rows = []
    for ticker in sorted(set(state.shares) | {CASH}):
        units = state.cash_balance if ticker == CASH else state.shares.get(ticker, 0.0)
        if units <= 1e-12:
            continue
        unit_text = "cash" if ticker == CASH else f"{units:,.4f} shares"
        current_rows.append(
            f'<tr><td style="{cell};font-weight:600;color:#111827">{html.escape(ticker)}</td>'
            f'<td style="{cell};color:#374151">{html.escape(unit_text)}</td>'
            f'<td style="{cell};text-align:right;color:#6b7280">Unavailable</td>'
            f'<td style="{cell};text-align:right;color:#6b7280">Unavailable</td></tr>'
        )
    current_rows_html = "".join(current_rows) or (
        f'<tr><td colspan="4" style="{cell};color:#6b7280">No persisted holdings found</td></tr>'
    )
    target_rows_html = "".join(
        f'<tr><td style="{cell};font-weight:600;color:#111827">{html.escape(ticker)}</td>'
        f'<td style="{cell};text-align:right;color:#374151">{weight:.0%}</td></tr>'
        for ticker, weight in core.target_weights().items()
    )
    html_body = f'''<!doctype html><html><body style="margin:0;padding:24px 12px;background:#f3f4f6;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif"><table width="100%" cellpadding="0" cellspacing="0" style="max-width:720px;margin:0 auto;background:#fff;border:1px solid #e5e7eb"><tr><td style="padding:20px 24px;border-bottom:3px solid #b45309"><div style="font-size:21px;font-weight:600;color:#b45309">ROTH IRA DASHBOARD</div><div style="font-size:15px;color:#374151;padding-top:5px">Valuation unavailable — no trades recommended</div></td></tr><tr><td style="padding:18px 24px 8px"><table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse"><tr><td style="padding:10px 14px;border:1px solid #e5e7eb;background:#f9fafb;vertical-align:top"><div style="font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#6b7280">Last recorded value</div><div style="font-size:16px;color:#111827;padding-top:4px">${state.portfolio_value:,.2f}</div></td><td style="padding:10px 14px;border:1px solid #e5e7eb;background:#f9fafb;vertical-align:top"><div style="font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#6b7280">Decision</div><div style="font-size:16px;color:#111827;padding-top:4px">No trades</div></td></tr></table></td></tr><tr><td style="padding:8px 24px 18px"><div style="padding:14px 16px;background:#fffbeb;border-left:4px solid #b45309;font-size:15px;color:#92400e;line-height:1.55;font-weight:600">{html.escape(reason)}<br><span style="font-weight:400;color:#374151">No prices, orders, or portfolio-state changes were used.</span></div></td></tr><tr><td style="padding:0 24px 18px"><div style="font-size:17px;font-weight:700;color:#111827;padding-bottom:9px">Persisted holdings</div><table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;border:1px solid #e5e7eb"><tr style="background:#f9fafb"><th style="{head};text-align:left">Ticker</th><th style="{head};text-align:left">Units</th><th style="{head};text-align:right">Value</th><th style="{head};text-align:right">Weight</th></tr>{current_rows_html}</table></td></tr><tr><td style="padding:0 24px 18px"><div style="font-size:17px;font-weight:700;color:#111827;padding-bottom:9px">Target allocation</div><table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;border:1px solid #e5e7eb"><tr style="background:#f9fafb"><th style="{head};text-align:left">Holding</th><th style="{head};text-align:right">Weight</th></tr>{target_rows_html}</table></td></tr><tr><td style="padding:0 24px 24px;font-size:13px;color:#6b7280">Retry the live dashboard after the market-data provider publishes a completed session.</td></tr></table></body></html>'''
    return text_body, html_body


def build_email_html(run: StrategyRun, notification: NotificationDecision) -> str:
    plan = run.rebalance_plan
    contribution = run.contribution_plan
    status_labels = {
        "ACTION": "ANNUAL ACTION REQUIRED",
        "UPDATE": "ALLOCATION UPDATE REQUIRED",
        "ANNUAL_REVIEW": "ANNUAL REVIEW",
        "CONTRIBUTION": "CONTRIBUTION DUE",
        "SNAPSHOT": "LIVE DASHBOARD SNAPSHOT",
        "CANCELLATION": "PREVIOUS ACTION CANCELLED",
    }
    accents = {
        "ACTION": "#b45309", "UPDATE": "#b45309", "ANNUAL_REVIEW": "#15803d",
        "CONTRIBUTION": "#1d4ed8", "SNAPSHOT": "#2563eb", "CANCELLATION": "#6b7280",
    }
    accent = accents.get(notification.kind, "#2563eb")
    status = status_labels.get(notification.kind, "ROTH IRA UPDATE")
    cell = "padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:14px"
    head = "padding:9px 12px;border-bottom:2px solid #d1d5db;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#6b7280;font-weight:600"

    def metric(label: str, value: str) -> str:
        return (
            '<td style="padding:10px 14px;border:1px solid #e5e7eb;background:#f9fafb;vertical-align:top">'
            f'<div style="font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#6b7280">{html.escape(label)}</div>'
            f'<div style="font-size:16px;color:#111827;padding-top:4px">{html.escape(value)}</div></td>'
        )

    current_rows = []
    for ticker in sorted(set(run.planning_state.shares) | {CASH}):
        if ticker == CASH:
            units = run.planning_state.cash_balance
            value = units
            unit_text = "cash"
        else:
            units = run.planning_state.shares.get(ticker, 0.0)
            value = units * float(run.price_data[ticker].iloc[-1])
            unit_text = f"{units:,.4f} shares"
        if units > 1e-12 or value > 0.01:
            current_rows.append(
                f'<tr><td style="{cell};font-weight:600;color:#111827">{html.escape(ticker)}</td>'
                f'<td style="{cell};color:#374151">{unit_text}</td>'
                f'<td style="{cell};text-align:right;color:#374151">${value:,.2f}</td>'
                f'<td style="{cell};text-align:right;color:#374151">{run.current_weights.get(ticker, 0.0):.1%}</td></tr>'
            )
    current_table = "".join(current_rows)
    target_table = "".join(
        f'<tr><td style="{cell};font-weight:600;color:#111827">{ticker}</td>'
        f'<td style="{cell};text-align:right;font-weight:600;color:#111827">{weight:.0%}</td></tr>'
        for ticker, weight in core.target_weights().items()
    )
    order_rows = "".join(
        f'<tr><td style="{cell};font-weight:700;color:{"#15803d" if row["Action"] == "BUY" else "#b91c1c"}">{row["Action"]}</td>'
        f'<td style="{cell};font-weight:600">{row["Ticker"]}</td>'
        f'<td style="{cell};text-align:right">{row["TargetPct"]:.0%}</td>'
        f'<td style="{cell};text-align:right">${abs(row["DeltaValue"]):,.2f}</td>'
        f'<td style="{cell};text-align:right">{abs(row["DeltaUnits"]):,.4f}</td></tr>'
        for _, row in _visible_rows(run.execution_table).iterrows() if row["Action"] != "HOLD"
    )
    if notification.kind == "SNAPSHOT":
        action_text = "Live dashboard snapshot only. No portfolio state was changed."
    elif plan.rebalance_due and contribution.notification_due:
        action_text = f"Deposit ${contribution.due_amount:,.2f}, then place the annual portfolio orders using next-session prices."
    elif plan.rebalance_due:
        action_text = "Place the annual portfolio orders using next-session prices, then confirm completed holdings and CASH."
    elif contribution.notification_due:
        action_text = f"Deposit ${contribution.due_amount:,.2f} and allocate it to the target percentages."
    else:
        action_text = "No portfolio trades are required."
    orders_html = ""
    if plan.rebalance_due:
        orders_html = f'''<tr><td style="padding:0 24px 18px"><div style="font-size:17px;font-weight:700;color:#111827;padding-bottom:9px">Annual orders</div><table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;border:1px solid #e5e7eb"><tr style="background:#f9fafb"><th style="{head};text-align:left">Action</th><th style="{head};text-align:left">Ticker</th><th style="{head};text-align:right">Target</th><th style="{head};text-align:right">Trade value</th><th style="{head};text-align:right">Est. shares</th></tr>{order_rows}</table><div style="font-size:14px;color:#374151;padding-top:9px">Quantities are estimates. Recalculate at execution and confirm full holdings afterward.</div></td></tr>'''
    contribution_html = ""
    if contribution.notification_due:
        contribution_rows = "".join(
            f'<tr><td style="{cell};font-weight:600">{ticker}</td><td style="{cell};text-align:right">${dollars:,.2f}</td><td style="{cell};text-align:right">{contribution.estimated_units[ticker]:,.4f}</td></tr>'
            for ticker, dollars in contribution.allocation_dollars.items()
        )
        contribution_html = f'''<tr><td style="padding:0 24px 18px"><div style="font-size:17px;font-weight:700;color:#1d4ed8;padding-bottom:8px">Annual contribution: ${contribution.due_amount:,.2f}</div><table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;border:1px solid #e5e7eb"><tr style="background:#f9fafb"><th style="{head};text-align:left">Ticker</th><th style="{head};text-align:right">Buy dollars</th><th style="{head};text-align:right">Est. units</th></tr>{contribution_rows}</table></td></tr>'''
    metrics = "<tr>" + metric("Completed session", str(run.session_date.date())) + metric("Account value", f"${run.portfolio_value:,.2f}") + "</tr><tr>" + metric("Annual target", "TQQQ 35% / DBMF 25% / UGL 20% / ZROZ 15% / BTAL 5%") + metric("Decision", "Static annual hold") + "</tr>"
    return f'''<!doctype html><html><body style="margin:0;padding:24px 12px;background:#f3f4f6;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif"><table width="100%" cellpadding="0" cellspacing="0" style="max-width:720px;margin:0 auto;background:#fff;border:1px solid #e5e7eb"><tr><td style="padding:20px 24px;border-bottom:3px solid {accent}"><div style="font-size:21px;font-weight:600;color:{accent}">{status}</div><div style="font-size:15px;color:#374151;padding-top:5px">{html.escape(plan.reason if plan.rebalance_due else notification.reason)}</div></td></tr><tr><td style="padding:18px 24px 8px"><table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse">{metrics}</table></td></tr><tr><td style="padding:8px 24px 18px"><div style="padding:14px 16px;background:#eff6ff;border-left:4px solid {accent};font-size:16px;color:#111827;line-height:1.55;font-weight:600">{html.escape(action_text)}</div></td></tr><tr><td style="padding:0 24px 18px"><div style="font-size:17px;font-weight:700;color:#111827;padding-bottom:9px">Current holdings</div><table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;border:1px solid #e5e7eb"><tr style="background:#f9fafb"><th style="{head};text-align:left">Ticker</th><th style="{head};text-align:left">Units</th><th style="{head};text-align:right">Value</th><th style="{head};text-align:right">Weight</th></tr>{current_table}</table></td></tr><tr><td style="padding:0 24px 18px"><div style="font-size:17px;font-weight:700;color:#111827;padding-bottom:9px">Target allocation</div><table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;border:1px solid #e5e7eb"><tr style="background:#f9fafb"><th style="{head};text-align:left">Holding</th><th style="{head};text-align:right">Weight</th></tr>{target_table}</table></td></tr>{contribution_html}{orders_html}</table></body></html>'''


def notification_subject(run: StrategyRun, notification: NotificationDecision) -> str:
    return f"ROTH IRA {notification.kind}: static annual allocation ({run.session_date.date()})"


def send_email(subject: str, text_body: str, html_body: str) -> None:
    address = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("RECEIVER_EMAIL")
    if not all((address, password, recipient)):
        raise RuntimeError("GMAIL_ADDRESS, GMAIL_APP_PASSWORD, and RECEIVER_EMAIL are required")
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = address
    message["To"] = recipient
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls()
        server.login(address, password)
        server.send_message(message)


def send_test_email() -> None:
    send_email(
        "ROTH IRA test email: static annual strategy",
        "Test email successful. Production target: TQQQ 35%, DBMF 25%, UGL 20%, ZROZ 15%, BTAL 5%. No market signals are used.",
        "<html><body><h2>ROTH IRA test email successful</h2><p>Static annual target: TQQQ 35%, DBMF 25%, UGL 20%, ZROZ 15%, BTAL 5%.</p><p>No market signals are used.</p></body></html>",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROTH IRA static annual allocation engine")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--confirm-execution", action="store_true")
    mode.add_argument("--sync-holdings", action="store_true")
    mode.add_argument("--configure-contributions", action="store_true")
    mode.add_argument("--disable-contributions", action="store_true")
    mode.add_argument("--send-test-email", action="store_true")
    mode.add_argument("--send-dashboard-email", action="store_true")
    parser.add_argument("--test", action="store_true", help="Report without persistence or email")
    parser.add_argument("--roth-amount", type=float)
    parser.add_argument("--executed-shares", nargs="+", metavar="TICKER=SHARES")
    parser.add_argument("--executed-signal-date", metavar="YYYY-MM-DD")
    parser.add_argument("--contribution-budget", type=float)
    parser.add_argument("--contribution-year", type=int)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    configure_logging(persist_log=not args.test and not args.send_test_email)
    core.validate_target()
    if args.send_test_email:
        if args.test or any(value is not None for value in (args.roth_amount, args.executed_signal_date, args.contribution_budget, args.contribution_year)) or args.executed_shares:
            parser.error("--send-test-email cannot be combined with other inputs")
        send_test_email()
        print("Test email sent.")
        return
    if args.send_dashboard_email:
        if args.test or any(value is not None for value in (args.executed_signal_date, args.contribution_budget, args.contribution_year)) or args.executed_shares:
            parser.error("--send-dashboard-email cannot be combined with other inputs")
        try:
            run = run_strategy(args.roth_amount)
        except RuntimeError as exc:
            state = load_state()
            dashboard, dashboard_html = build_unavailable_dashboard(state, str(exc))
            send_email("ROTH IRA dashboard snapshot: valuation unavailable", dashboard, dashboard_html)
            print("Dashboard email sent with a valuation-unavailable warning; no trades were recommended.")
            return
        dashboard = build_dashboard(run)
        send_email(
            f"ROTH IRA dashboard snapshot ({run.session_date.date()})",
            dashboard,
            build_email_html(run, NotificationDecision("SNAPSHOT", "MANUAL_DASHBOARD_SNAPSHOT")),
        )
        print("Dashboard snapshot email sent.")
        return
    try:
        executed_shares, executed_cash = parse_executed_shares(args.executed_shares)
    except ValueError as exc:
        parser.error(str(exc))
    if args.configure_contributions:
        if args.test or executed_shares is not None or args.contribution_budget is None:
            parser.error("--configure-contributions requires --contribution-budget")
        year = args.contribution_year or datetime.now(NEW_YORK).year
        state = configure_contribution_plan(year, args.contribution_budget)
        print(f"Contribution plan configured for {state.contribution_plan_year}: ${state.contribution_budget:,.2f}.")
        return
    if args.disable_contributions:
        if args.test or executed_shares is not None or args.contribution_budget is not None or args.contribution_year is not None:
            parser.error("--disable-contributions cannot be combined with other inputs")
        disable_contribution_plan()
        print("Contribution plan disabled.")
        return
    if args.confirm_execution:
        if args.test or executed_shares is None or not args.executed_signal_date:
            parser.error("--confirm-execution requires executed signal date and complete shares")
        confirm_execution(executed_shares, executed_cash, args.executed_signal_date)
        print(f"Confirmed execution for annual recommendation {args.executed_signal_date}.")
        return
    if args.sync_holdings:
        if args.test or executed_shares is None:
            parser.error("--sync-holdings requires complete shares")
        sync_holdings(executed_shares, executed_cash)
        print("Broker holdings synchronized.")
        return
    if args.executed_shares or args.executed_signal_date or args.contribution_budget or args.contribution_year:
        parser.error("Execution and contribution fields require their operation mode")

    run = run_strategy(args.roth_amount)
    notification = decide_notification(run)
    dashboard = build_dashboard(run)
    print(dashboard)
    if args.test:
        return
    prepare_notification_delivery(run, notification)
    prepare_contribution_delivery(run)
    persist_run(run)
    if notification.should_send:
        send_email(notification_subject(run, notification), dashboard, build_email_html(run, notification))
        persist_notification_delivery(run, notification)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("ROTH IRA engine failed")
        raise

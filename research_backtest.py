#!/usr/bin/env python3
"""Auditable, production-parity research harness for the ROTH IRA engine.

This module is deliberately isolated from the live CLI.  It never loads or
writes production state, never sends email, and never fabricates missing ETF
history.  Signals are formed after close ``t`` and queued trades execute at the
adjusted open of the next validated XNYS session.

The harness is exploratory.  The historical sample has already been inspected,
so its fold/bootstrap diagnostics are pseudo-out-of-sample rather than proof of
future performance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import port12_cloud as engine


RESEARCH_SCHEMA_VERSION = 1
DEFAULT_START = "2014-01-01"
DEFAULT_COSTS_BPS = (0.0, 5.0, 10.0, 25.0)
PRIMARY_COST_BPS = 10.0
MASTER_SEED = 20_260_805
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
MIN_INFERENTIAL_BOOTSTRAP_SAMPLES = 9_999
STARTING_CASH = 10_000.0
SPY = "SPY"
RESEARCH_TICKERS = (*engine.ALL_TICKERS, SPY)
FDR_Q = 0.10


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    category: str = "candidate"
    kind: str = "original"
    rebalance_band: float = engine.REBALANCE_BAND
    rebalance_destination: float = engine.REBALANCE_DESTINATION
    bullish_reentry_closes: int = 1
    volatility_hysteresis: bool = False
    momentum_window: int = engine.MOMENTUM_WINDOW
    equal_leaders: bool = False


VARIANTS: dict[str, Variant] = {
    "original": Variant(
        name="original",
        description="Locked production Original ROTH IRA Barbell",
        category="baseline",
    ),
    "c1_wide_buffer": Variant(
        name="c1_wide_buffer",
        description="7.5pp drift trigger and 3.75pp destination",
        rebalance_band=0.075,
        rebalance_destination=0.0375,
    ),
    "c2_bull_reentry_2": Variant(
        name="c2_bull_reentry_2",
        description="Immediate bearish exit; two bullish closes to re-enter",
        bullish_reentry_closes=2,
    ),
    "c3_vol_hysteresis": Variant(
        name="c3_vol_hysteresis",
        description="Immediate de-risk; 20%/13% two-close volatility re-risk",
        volatility_hysteresis=True,
    ),
    "c4_momentum_63": Variant(
        name="c4_momentum_63",
        description="63-session SOXL/TECL leader momentum",
        momentum_window=63,
    ),
    "c5_equal_leaders": Variant(
        name="c5_equal_leaders",
        description="Equal SOXL/TECL weights in low and moderate tiers",
        equal_leaders=True,
    ),
    "benchmark_spmo_smh": Variant(
        name="benchmark_spmo_smh",
        description="Buffered 50% SPMO / 50% SMH",
        category="benchmark",
        kind="static_buffered",
    ),
    "benchmark_qld": Variant(
        name="benchmark_qld",
        description="100% QLD buy and hold",
        category="benchmark",
        kind="buy_hold_qld",
    ),
    "benchmark_spy": Variant(
        name="benchmark_spy",
        description="100% SPY buy and hold",
        category="benchmark",
        kind="buy_hold_spy",
    ),
}

DEFAULT_VARIANT_NAMES = tuple(VARIANTS)
DECLARED_CANDIDATE_FAMILY = tuple(
    sorted(
        name
        for name, variant in VARIANTS.items()
        if variant.category == "candidate"
    )
)
DECLARED_FAMILY_FINGERPRINT = engine.canonical_sha256(
    {
        name: asdict(VARIANTS[name])
        for name in DECLARED_CANDIDATE_FAMILY
    }
)


@dataclass(frozen=True)
class MarketData:
    opens: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame
    scoring_dates: pd.DatetimeIndex
    common_start: pd.Timestamp
    final_session: pd.Timestamp
    fingerprint: str
    requested_start: str
    requested_end_exclusive: str
    source: str


@dataclass(frozen=True)
class PendingOrder:
    signal_date: pd.Timestamp
    execution_weights: dict[str, float]
    result: engine.StrategyResult
    reason: str
    median_dollar_volume20: dict[str, float]


@dataclass
class RuntimeState:
    portfolio: engine.PortfolioState = field(
        default_factory=lambda: engine.PortfolioState(cash_balance=STARTING_CASH)
    )
    pending_order: PendingOrder | None = None
    bullish_streak: int = 0
    has_executed_trade: bool = False


@dataclass(frozen=True)
class FillResult:
    shares: dict[str, float]
    cash: float
    pretrade_nav: float
    posttrade_nav: float
    gross_notional: float
    gross_fraction: float
    one_way_fraction: float
    cost: float
    cost_fraction: float
    security_orders: int
    trade_notional: dict[str, float]


@dataclass(frozen=True)
class BacktestResult:
    variant: Variant
    cost_bps: float
    ledger: pd.DataFrame
    metrics: dict[str, object]
    unfilled_final_order: bool


def _finite_positive_frame(
    frame: pd.DataFrame,
    description: str,
    *,
    allow_preinception_missing: bool,
) -> None:
    values = frame.to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values <= 0)
    if invalid.any() and not allow_preinception_missing:
        row, column = np.argwhere(invalid)[0]
        raise RuntimeError(
            f"{description} contains missing or invalid data at "
            f"{frame.index[row].date().isoformat()} / {frame.columns[column]}"
        )


def _normalize_frame(frame: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        raise RuntimeError("Market data does not use a DatetimeIndex")
    if result.index.tz is not None:
        result.index = result.index.tz_convert(None)
    result.index = result.index.normalize()
    result.index.name = None
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise RuntimeError("Market-data dates are duplicated or unsorted")
    result.columns = [str(item) for item in result.columns]
    result.columns.name = None
    missing = set(tickers) - set(result.columns)
    if missing:
        raise RuntimeError(f"Market data is missing tickers: {sorted(missing)}")
    return result.reindex(columns=list(tickers)).apply(pd.to_numeric, errors="coerce")


def _extract_field(
    raw: pd.DataFrame,
    field_name: str,
    tickers: Sequence[str],
) -> pd.DataFrame:
    if raw.empty:
        raise RuntimeError("yfinance returned no market data")
    if isinstance(raw.columns, pd.MultiIndex):
        first = set(str(item) for item in raw.columns.get_level_values(0))
        second = set(str(item) for item in raw.columns.get_level_values(1))
        if field_name in first:
            frame = raw[field_name]
        elif field_name in second:
            frame = raw.xs(field_name, axis=1, level=1)
        else:
            raise RuntimeError(f"yfinance response is missing {field_name}")
    else:
        if len(tickers) != 1 or field_name not in raw.columns:
            raise RuntimeError(f"yfinance response is missing {field_name}")
        frame = pd.DataFrame({tickers[0]: raw[field_name]}, index=raw.index)
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=tickers[0])
    return _normalize_frame(frame, tickers)


def _calendar_sessions(
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
) -> pd.DatetimeIndex:
    sessions = pd.DatetimeIndex(
        engine._nyse_calendar().sessions_in_range(  # noqa: SLF001
            pd.Timestamp(start).normalize(),
            pd.Timestamp(end_inclusive).normalize(),
        )
    )
    if sessions.tz is not None:
        sessions = sessions.tz_convert(None)
    return sessions.normalize()


def _last_session_before(exclusive_end: str) -> pd.Timestamp:
    end = pd.Timestamp(exclusive_end).normalize()
    sessions = _calendar_sessions(end - pd.Timedelta(days=14), end - pd.Timedelta(days=1))
    if len(sessions) == 0:
        raise RuntimeError("XNYS calendar returned no session before --end")
    return pd.Timestamp(sessions[-1]).normalize()


def _full_data_fingerprint(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
) -> str:
    digest = hashlib.sha256()
    for name, frame in (("Open", opens), ("Close", closes), ("Volume", volumes)):
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
        canonical = frame.copy()
        canonical.index.name = "Session"
        canonical.columns.name = "Ticker"
        rendered = canonical.to_csv(
            index=True,
            date_format="%Y-%m-%d",
            float_format="%.12g",
            lineterminator="\n",
        )
        digest.update(rendered.encode("utf-8"))
    return digest.hexdigest()


def prepare_market_data(
    raw: pd.DataFrame,
    *,
    requested_start: str,
    requested_end_exclusive: str,
    tickers: Sequence[str] = RESEARCH_TICKERS,
    source: str = "provided adjusted OHLCV",
) -> MarketData:
    """Validate an adjusted OHLCV response without filling or dropping gaps."""
    opens = _extract_field(raw, "Open", tickers)
    closes = _extract_field(raw, "Close", tickers)
    volumes = _extract_field(raw, "Volume", tickers)
    if not opens.index.equals(closes.index) or not opens.index.equals(volumes.index):
        raise RuntimeError("Open, Close, and Volume indexes differ")

    final_session = _last_session_before(requested_end_exclusive)
    today_new_york = pd.Timestamp.now(tz=engine.NEW_YORK).tz_localize(None).normalize()
    if final_session >= today_new_york:
        completed = engine.expected_completed_session()
        if final_session > completed:
            raise RuntimeError(
                "Research data includes a session that is not yet a completed "
                f"production session: requested={final_session.date().isoformat()}, "
                f"completed={completed.date().isoformat()}"
            )
    expected = _calendar_sessions(pd.Timestamp(requested_start), final_session)
    received = opens.index
    missing_sessions = expected.difference(received)
    unexpected_sessions = received.difference(expected)
    if len(missing_sessions) or len(unexpected_sessions):
        missing = [item.date().isoformat() for item in missing_sessions[:5]]
        unexpected = [item.date().isoformat() for item in unexpected_sessions[:5]]
        raise RuntimeError(
            "Research market-data session continuity failed: "
            f"missing={missing}, non_sessions={unexpected}"
        )
    if len(received) == 0 or received[-1] != final_session:
        raise RuntimeError(
            f"Research data is stale: expected {final_session.date().isoformat()}"
        )

    first_valid_dates: list[pd.Timestamp] = []
    # Every traded/signal security needs executable adjusted prices.  Only QQQ
    # volume enters a signal.  Zero volume in a thin ETF is retained as an
    # unavailable ADV observation rather than treated as a missing price bar.
    inception_fields = (
        ("Open", opens, tickers),
        ("Close", closes, tickers),
        ("Volume", volumes, (engine.MARKET_INDEX,)),
    )
    for field_name, frame, required_tickers in inception_fields:
        for ticker in required_tickers:
            values = frame[ticker].to_numpy(dtype=float)
            valid = np.isfinite(values) & (values > 0)
            if not valid.any():
                raise RuntimeError(f"{ticker} has no valid adjusted {field_name} history")
            first_valid_dates.append(pd.Timestamp(frame.index[np.argmax(valid)]))

    common_start = max(first_valid_dates).normalize()
    try:
        common_position = received.get_loc(common_start)
    except KeyError as exc:
        raise RuntimeError("Common inception is not an XNYS data session") from exc
    required_rows = max(engine.required_signal_rows(), 64)
    if not isinstance(common_position, (int, np.integer)) or common_position < required_rows - 1:
        raise RuntimeError(
            "Insufficient actual QQQ/leader warm-up before common ETF inception"
        )

    score_slice = slice(common_position, None)
    _finite_positive_frame(opens.iloc[score_slice], "Scored adjusted opens", allow_preinception_missing=False)
    _finite_positive_frame(closes.iloc[score_slice], "Scored adjusted closes", allow_preinception_missing=False)
    _finite_positive_frame(
        volumes.loc[:, [engine.MARKET_INDEX]].iloc[score_slice],
        "Scored QQQ volumes",
        allow_preinception_missing=False,
    )

    warmup_slice = slice(common_position - required_rows + 1, None)
    _finite_positive_frame(
        closes.loc[:, [engine.MARKET_INDEX, *engine.LEADER_CANDIDATES]].iloc[warmup_slice],
        "Indicator warm-up closes",
        allow_preinception_missing=False,
    )
    _finite_positive_frame(
        volumes.loc[:, [engine.MARKET_INDEX]].iloc[warmup_slice],
        "Indicator warm-up volume",
        allow_preinception_missing=False,
    )

    scoring_dates = received[common_position:]
    return MarketData(
        opens=opens,
        closes=closes,
        volumes=volumes,
        scoring_dates=scoring_dates,
        common_start=common_start,
        final_session=final_session,
        fingerprint=_full_data_fingerprint(opens, closes, volumes),
        requested_start=requested_start,
        requested_end_exclusive=requested_end_exclusive,
        source=source,
    )


def download_market_data(
    *,
    requested_start: str,
    requested_end_exclusive: str,
    tickers: Sequence[str] = RESEARCH_TICKERS,
) -> MarketData:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("yfinance is required for a live research download") from exc

    raw = yf.download(
        list(tickers),
        start=requested_start,
        end=requested_end_exclusive,
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
        threads=False,
    )
    return prepare_market_data(
        raw,
        requested_start=requested_start,
        requested_end_exclusive=requested_end_exclusive,
        tickers=tickers,
        source="Yahoo Finance via yfinance; auto_adjust=True",
    )


def save_market_snapshot(data: MarketData, path: Path) -> None:
    """Save the exact validated adjusted inputs in a deterministic long CSV."""
    rows: list[pd.DataFrame] = []
    for field_name, frame in (
        ("Open", data.opens),
        ("Close", data.closes),
        ("Volume", data.volumes),
    ):
        stacked = frame.stack(future_stack=True).rename("Value").reset_index()
        stacked.columns = ["Session", "Ticker", "Value"]
        stacked.insert(2, "Field", field_name)
        rows.append(stacked)
    snapshot = pd.concat(rows, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    snapshot.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.12g",
        lineterminator="\n",
    )
    os.replace(temporary, path)


def load_market_snapshot(
    path: Path,
    *,
    requested_start: str,
    requested_end_exclusive: str,
    tickers: Sequence[str] = RESEARCH_TICKERS,
) -> MarketData:
    """Load and revalidate a frozen long-form snapshot created by this module."""
    try:
        snapshot = pd.read_csv(path, parse_dates=["Session"])
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Unable to read research snapshot: {path}") from exc
    required = {"Session", "Ticker", "Field", "Value"}
    if set(snapshot.columns) != required:
        raise RuntimeError(
            f"Research snapshot columns must be exactly {sorted(required)}"
        )
    if snapshot.duplicated(["Session", "Ticker", "Field"]).any():
        raise RuntimeError("Research snapshot contains duplicate observations")
    if set(snapshot["Field"]) != {"Open", "Close", "Volume"}:
        raise RuntimeError("Research snapshot must contain Open, Close, and Volume")
    raw = snapshot.pivot(
        index="Session",
        columns=["Field", "Ticker"],
        values="Value",
    )
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    return prepare_market_data(
        raw,
        requested_start=requested_start,
        requested_end_exclusive=requested_end_exclusive,
        tickers=tickers,
        source=f"frozen local snapshot: {path.name}",
    )


def restrict_scoring_start(
    data: MarketData,
    scoring_start: str,
) -> MarketData:
    """Start a labeled sensitivity run later without altering warm-up data."""
    requested = pd.Timestamp(scoring_start).normalize()
    if requested < data.common_start:
        raise ValueError(
            "Scoring start cannot precede the actual common ETF inception"
        )
    if requested > data.final_session or requested not in data.opens.index:
        raise ValueError("Scoring start must be a validated XNYS data session")
    return replace(
        data,
        scoring_dates=data.opens.index[data.opens.index >= requested],
    )


def solve_post_cost_target(
    *,
    current_values: Mapping[str, float],
    cash: float,
    target_weights: Mapping[str, float],
    cost_rate: float,
) -> tuple[float, float]:
    """Solve post-cost NAV and security gross notional exactly by bisection."""
    if not np.isfinite(cost_rate) or not (0 <= cost_rate < 1):
        raise ValueError("Transaction-cost rate must be finite and in [0, 1)")
    target = dict(target_weights)
    if any(not np.isfinite(value) or value < 0 for value in target.values()):
        raise ValueError("Target weights must be finite and nonnegative")
    if not np.isclose(sum(target.values()), 1.0, atol=1e-12):
        raise ValueError("Target weights must sum to one")
    if any(not np.isfinite(value) or value < 0 for value in current_values.values()):
        raise ValueError("Current security values must be finite and nonnegative")
    if not np.isfinite(cash) or cash < 0:
        raise ValueError("Cash must be finite and nonnegative")

    securities = (set(current_values) | set(target)) - {engine.CASH_ASSET}
    pretrade_nav = float(cash + sum(current_values.values()))
    if pretrade_nav <= 0:
        raise ValueError("Pretrade NAV must be positive")

    def gross(posttrade_nav: float) -> float:
        return float(
            sum(
                abs(
                    target.get(ticker, 0.0) * posttrade_nav
                    - current_values.get(ticker, 0.0)
                )
                for ticker in securities
            )
        )

    if cost_rate == 0:
        return pretrade_nav, gross(pretrade_nav)

    low = 0.0
    high = pretrade_nav
    for _ in range(100):
        middle = (low + high) / 2.0
        residual = middle + cost_rate * gross(middle) - pretrade_nav
        if residual > 0:
            high = middle
        else:
            low = middle
    posttrade_nav = (low + high) / 2.0
    gross_notional = gross(posttrade_nav)
    if not np.isclose(
        posttrade_nav + cost_rate * gross_notional,
        pretrade_nav,
        rtol=0,
        atol=max(1e-9, pretrade_nav * 1e-12),
    ):
        raise RuntimeError("Post-cost target solver did not converge")
    return posttrade_nav, gross_notional


def execute_target_at_open(
    *,
    shares: Mapping[str, float],
    cash: float,
    open_prices: pd.Series,
    target_weights: Mapping[str, float],
    cost_bps: float,
) -> FillResult:
    """Execute fractional shares to exact post-cost weights at adjusted Open."""
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Transaction costs must be finite and nonnegative")
    target = dict(target_weights)
    target.setdefault(engine.CASH_ASSET, 0.0)
    if not np.isclose(sum(target.values()), 1.0, atol=1e-12):
        raise ValueError("Execution target must sum to one")

    tickers = (set(shares) | set(target)) - {engine.CASH_ASSET}
    current_values: dict[str, float] = {}
    for ticker in tickers:
        try:
            price = float(open_prices[ticker])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Missing execution Open for {ticker}") from exc
        if not np.isfinite(price) or price <= 0:
            raise RuntimeError(f"Invalid execution Open for {ticker}")
        units = float(shares.get(ticker, 0.0))
        if not np.isfinite(units) or units < 0:
            raise RuntimeError(f"Invalid current shares for {ticker}")
        current_values[ticker] = units * price

    pretrade_nav = float(cash + sum(current_values.values()))
    posttrade_nav, gross_notional = solve_post_cost_target(
        current_values=current_values,
        cash=float(cash),
        target_weights=target,
        cost_rate=cost_bps / 10_000.0,
    )
    cost = pretrade_nav - posttrade_nav

    next_shares: dict[str, float] = {}
    trade_notional: dict[str, float] = {}
    for ticker in sorted(tickers):
        price = float(open_prices[ticker])
        target_value = target.get(ticker, 0.0) * posttrade_nav
        delta = target_value - current_values.get(ticker, 0.0)
        trade_notional[ticker] = float(delta)
        if target_value > 1e-12:
            next_shares[ticker] = target_value / price

    next_cash = target.get(engine.CASH_ASSET, 0.0) * posttrade_nav
    accounting_cash = (
        cash
        + sum(current_values.get(ticker, 0.0) for ticker in tickers)
        - sum(
            target.get(ticker, 0.0) * posttrade_nav
            for ticker in tickers
        )
        - cost
    )
    if not np.isclose(
        next_cash,
        accounting_cash,
        rtol=0,
        atol=max(1e-8, pretrade_nav * 1e-11),
    ):
        raise RuntimeError("Execution cash accounting does not balance")
    next_cash = max(0.0, float(next_cash))

    pre_weights = {
        ticker: value / pretrade_nav for ticker, value in current_values.items()
    }
    pre_weights[engine.CASH_ASSET] = cash / pretrade_nav
    components = set(pre_weights) | set(target)
    one_way = 0.5 * sum(
        abs(target.get(component, 0.0) - pre_weights.get(component, 0.0))
        for component in components
    )
    return FillResult(
        shares=next_shares,
        cash=next_cash,
        pretrade_nav=pretrade_nav,
        posttrade_nav=posttrade_nav,
        gross_notional=gross_notional,
        gross_fraction=gross_notional / pretrade_nav,
        one_way_fraction=float(one_way),
        cost=cost,
        cost_fraction=cost / pretrade_nav,
        security_orders=sum(abs(value) > pretrade_nav * 1e-12 for value in trade_notional.values()),
        trade_notional=trade_notional,
    )


def _volatility_hysteresis_decision(
    annualized_volatility: float,
    state: engine.PortfolioState,
    *,
    bullish: bool,
) -> engine.VolatilityTierDecision:
    """Candidate C3: production de-risk thresholds plus buffered re-risking."""
    if not bullish:
        return engine.VolatilityTierDecision(
            tier="N/A",
            raw_tier="N/A",
            transition="BEAR",
            processed_sessions=1,
        )
    raw = engine.classify_volatility(annualized_volatility)
    levels = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    existing = state.volatility_tier
    if existing not in levels:
        return engine.VolatilityTierDecision(
            tier=raw,
            raw_tier=raw,
            transition="INITIAL",
            processed_sessions=1,
        )

    if levels[raw] > levels[existing]:
        return engine.VolatilityTierDecision(
            tier=raw,
            raw_tier=raw,
            transition="DE_RISK",
            processed_sessions=1,
        )

    if existing == "HIGH":
        desired = (
            "LOW"
            if annualized_volatility < 0.13
            else "MODERATE"
            if annualized_volatility <= 0.20
            else "HIGH"
        )
    elif existing == "MODERATE":
        desired = (
            "LOW"
            if annualized_volatility < 0.13
            else "HIGH"
            if annualized_volatility > engine.MODERATE_VOL_THRESHOLD
            else "MODERATE"
        )
    else:
        desired = (
            "HIGH"
            if annualized_volatility > engine.MODERATE_VOL_THRESHOLD
            else "MODERATE"
            if annualized_volatility >= engine.LOW_VOL_THRESHOLD
            else "LOW"
        )

    if desired == existing:
        return engine.VolatilityTierDecision(
            tier=existing,
            raw_tier=raw,
            transition="UNCHANGED",
            processed_sessions=1,
        )
    if levels[desired] > levels[existing]:
        return engine.VolatilityTierDecision(
            tier=desired,
            raw_tier=raw,
            transition="DE_RISK",
            processed_sessions=1,
        )

    pending_days = state.pending_volatility_days + 1 if state.pending_volatility_tier == desired else 1
    if pending_days >= engine.VOLATILITY_RERISK_PERSISTENCE:
        return engine.VolatilityTierDecision(
            tier=desired,
            raw_tier=raw,
            transition="RE_RISK",
            processed_sessions=1,
        )
    return engine.VolatilityTierDecision(
        tier=existing,
        raw_tier=raw,
        pending_tier=desired,
        pending_days=pending_days,
        transition="DELAY_RE_RISK",
        processed_sessions=1,
    )


def _static_result(
    weights: Mapping[str, float],
    *,
    regime: str,
) -> engine.StrategyResult:
    return engine.StrategyResult(
        target_weights=dict(weights),
        regime=regime,
        leader=engine.LEVERAGED_SEMICONDUCTOR,
        volatility_tier="N/A",
        annualized_volatility=0.0,
        raw_volatility_tier="N/A",
    )


def _equal_leader_result(
    result: engine.StrategyResult,
) -> engine.StrategyResult:
    if result.regime != "BULL" or result.volatility_tier == "HIGH":
        return replace(result, leader=engine.LEVERAGED_SEMICONDUCTOR)
    if result.volatility_tier == "LOW":
        target = {
            engine.LEVERAGED_SEMICONDUCTOR: 0.30,
            engine.LEVERAGED_TECH: 0.30,
            engine.SEMICONDUCTOR_ETF: 0.25,
            engine.LEVERAGED_INDEX: 0.15,
        }
    elif result.volatility_tier == "MODERATE":
        target = {
            engine.LEVERAGED_SEMICONDUCTOR: 0.175,
            engine.LEVERAGED_TECH: 0.175,
            engine.SEMICONDUCTOR_ETF: 0.45,
            engine.LEVERAGED_INDEX: 0.20,
        }
    else:
        raise RuntimeError("Unexpected bull volatility tier")
    return replace(
        result,
        target_weights=target,
        leader=engine.LEVERAGED_SEMICONDUCTOR,
    )


def _median_dollar_volume20_at_signal(
    data: MarketData,
    signal_position: int,
    tickers: Iterable[str],
) -> dict[str, float]:
    start = max(0, signal_position - 19)
    result: dict[str, float] = {}
    for ticker in tickers:
        if ticker == engine.CASH_ASSET:
            continue
        values = (
            data.closes[ticker].iloc[start : signal_position + 1]
            * data.volumes[ticker].iloc[start : signal_position + 1]
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if len(values):
            result[ticker] = float(np.median(values))
    return result


def _mark_close(
    state: engine.PortfolioState,
    close_prices: pd.Series,
) -> float:
    total = float(state.cash_balance)
    for ticker, units in state.shares.items():
        price = float(close_prices[ticker])
        if not np.isfinite(price) or price <= 0:
            raise RuntimeError(f"Invalid adjusted Close for held ticker {ticker}")
        total += units * price
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Marked-to-market research NAV is invalid")
    return total


def _actual_weights(
    state: engine.PortfolioState,
    close_prices: pd.Series,
) -> dict[str, float]:
    frame = pd.DataFrame([close_prices], index=[close_prices.name])
    return engine.existing_weights(state, frame)


def _build_buy_hold_plan(
    existing: Mapping[str, float],
    result: engine.StrategyResult,
    state: engine.PortfolioState,
) -> engine.RebalancePlan:
    target = dict(result.target_weights)
    target.setdefault(engine.CASH_ASSET, 0.0)
    initial = state.executed_regime == "UNKNOWN" or not state.shares
    if initial:
        components = set(existing) | set(target)
        one_way = 0.5 * sum(
            abs(target.get(component, 0.0) - existing.get(component, 0.0))
            for component in components
        )
        return engine.RebalancePlan(
            execution_weights=target,
            rebalance_due=True,
            full_transition=True,
            reason="INITIAL_ALLOCATION",
            one_way_turnover=one_way,
            individual_orders=1,
        )
    return engine.RebalancePlan(
        execution_weights=target,
        rebalance_due=False,
        full_transition=False,
        reason="HOLD",
        one_way_turnover=0.0,
        individual_orders=0,
    )


def _apply_fill_metadata(
    state: engine.PortfolioState,
    pending: PendingOrder,
    fill: FillResult,
) -> None:
    state.shares = dict(fill.shares)
    state.cash_balance = float(fill.cash)
    state.target_weights = dict(pending.execution_weights)
    state.executed_leader = pending.result.leader
    state.executed_regime = pending.result.regime
    state.executed_volatility_tier = pending.result.volatility_tier
    # build_rebalance_plan compares this exact production sentinel.  Candidate
    # identity belongs in research metadata, not in live execution state.
    state.executed_strategy_fingerprint = engine.STRATEGY_FINGERPRINT


def _advance_signal_state(
    state: engine.PortfolioState,
    *,
    signal_date: pd.Timestamp,
    result: engine.StrategyResult,
    tier_decision: engine.VolatilityTierDecision,
    sector_due: bool,
    plan: engine.RebalancePlan,
) -> None:
    state.portfolio_value = _safe_float(state.portfolio_value)
    state.regime = result.regime
    state.volatility_tier = result.volatility_tier
    state.leader = result.leader
    state.pending_volatility_tier = tier_decision.pending_tier
    state.pending_volatility_days = tier_decision.pending_days
    state.last_processed_signal_date = signal_date.date().isoformat()
    if sector_due:
        state.last_sector_rebalance = signal_date.date().isoformat()
    if plan.reason == "CONFIRMED_TARGET_STATE":
        state.target_weights = dict(plan.execution_weights)
        state.executed_leader = result.leader
        state.executed_regime = result.regime
        state.executed_volatility_tier = result.volatility_tier
        state.executed_strategy_fingerprint = engine.STRATEGY_FINGERPRINT


def _safe_float(value: object) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError("Research calculation produced a nonfinite value")
    return result


def simulate_variant(
    data: MarketData,
    variant: Variant,
    *,
    cost_bps: float,
    starting_cash: float = STARTING_CASH,
) -> BacktestResult:
    """Run one variant/cost path with actual shares and queued next-open fills."""
    if variant.name not in VARIANTS:
        raise ValueError(f"Unregistered research variant: {variant.name}")
    if variant != VARIANTS[variant.name]:
        raise ValueError(
            f"Variant {variant.name} does not match its locked registry definition"
        )
    if not np.isfinite(starting_cash) or starting_cash <= 0:
        raise ValueError("Starting cash must be positive and finite")

    runtime = RuntimeState(
        portfolio=engine.PortfolioState(cash_balance=float(starting_cash))
    )
    indicators = engine.calculate_indicators(data.closes, data.volumes)
    if variant.momentum_window != engine.MOMENTUM_WINDOW:
        indicators = indicators.copy()
        indicators["soxl_momentum"] = data.closes[
            engine.LEVERAGED_SEMICONDUCTOR
        ].pct_change(variant.momentum_window, fill_method=None)
        indicators["tecl_momentum"] = data.closes[
            engine.LEVERAGED_TECH
        ].pct_change(variant.momentum_window, fill_method=None)

    rows: list[dict[str, object]] = []
    previous_close_nav: float | None = None
    date_positions = {session: index for index, session in enumerate(data.opens.index)}

    for signal_date in data.scoring_dates:
        full_position = date_positions[signal_date]
        fill: FillResult | None = None
        filled_reason = ""
        initial_deployment = False
        adv_ratio = 0.0
        if runtime.pending_order is not None:
            pending = runtime.pending_order
            fill = execute_target_at_open(
                shares=runtime.portfolio.shares,
                cash=runtime.portfolio.cash_balance,
                open_prices=data.opens.loc[signal_date],
                target_weights=pending.execution_weights,
                cost_bps=cost_bps,
            )
            initial_deployment = not runtime.has_executed_trade
            runtime.has_executed_trade = True
            _apply_fill_metadata(runtime.portfolio, pending, fill)
            filled_reason = pending.reason
            ratios: list[float] = []
            missing_trade_adv = False
            for ticker, notional in fill.trade_notional.items():
                if abs(notional) <= fill.pretrade_nav * 1e-12:
                    continue
                adv = pending.median_dollar_volume20.get(ticker)
                if adv is None or not np.isfinite(adv) or adv <= 0:
                    missing_trade_adv = True
                    continue
                ratios.append(abs(notional) / adv)
            adv_ratio = math.inf if missing_trade_adv else max(ratios, default=0.0)
            runtime.pending_order = None

        close_prices = data.closes.loc[signal_date]
        close_nav = _mark_close(runtime.portfolio, close_prices)
        runtime.portfolio.portfolio_value = close_nav
        daily_return = (
            0.0
            if previous_close_nav is None
            else close_nav / previous_close_nav - 1.0
        )
        if daily_return <= -1 or not np.isfinite(daily_return):
            raise RuntimeError("Research daily return is invalid or terminal")
        previous_close_nav = close_nav
        current_weights = _actual_weights(runtime.portfolio, close_prices)

        latest = indicators.loc[signal_date].copy()
        if variant.kind == "original":
            engine.validate_latest_indicators(latest)
            bullish = int(latest["bullish_consensus"]) == 1
            runtime.bullish_streak = (
                runtime.bullish_streak + 1 if bullish else 0
            )
            if variant.volatility_hysteresis:
                tier_decision = _volatility_hysteresis_decision(
                    float(latest["annualized_volatility"]),
                    runtime.portfolio,
                    bullish=bullish,
                )
            else:
                tier_decision = engine.replay_volatility_state(
                    indicators.iloc[: full_position + 1],
                    runtime.portfolio,
                )
            sector_due = engine.sector_review_due(
                runtime.portfolio.last_sector_rebalance,
                signal_date,
                data.closes.index[: full_position + 1],
            )
            result = engine.determine_target_allocation(
                latest,
                runtime.portfolio.leader,
                sector_due,
                tier_decision,
            )
            if (
                bullish
                and runtime.bullish_streak < variant.bullish_reentry_closes
            ):
                result = engine.StrategyResult(
                    target_weights=dict(engine.BEAR_ALLOCATION),
                    regime="BEAR",
                    leader=result.leader,
                    volatility_tier="N/A",
                    annualized_volatility=float(latest["annualized_volatility"]),
                    raw_volatility_tier="N/A",
                )
            if variant.equal_leaders:
                result = _equal_leader_result(result)
            plan = engine.build_rebalance_plan(
                current_weights,
                result,
                runtime.portfolio,
                rebalance_band=variant.rebalance_band,
                rebalance_destination=variant.rebalance_destination,
            )
        else:
            tier_decision = engine.VolatilityTierDecision(
                tier="N/A",
                raw_tier="N/A",
                transition="STATIC",
                processed_sessions=1,
            )
            sector_due = False
            if variant.kind == "static_buffered":
                result = _static_result(
                    {
                        engine.DEFENSIVE_EQUITY: 0.50,
                        engine.SEMICONDUCTOR_ETF: 0.50,
                    },
                    regime="STATIC",
                )
                plan = engine.build_rebalance_plan(
                    current_weights,
                    result,
                    runtime.portfolio,
                    rebalance_band=variant.rebalance_band,
                    rebalance_destination=variant.rebalance_destination,
                )
            elif variant.kind == "buy_hold_qld":
                result = _static_result(
                    {engine.LEVERAGED_INDEX: 1.0},
                    regime="BUY_HOLD",
                )
                plan = _build_buy_hold_plan(current_weights, result, runtime.portfolio)
            elif variant.kind == "buy_hold_spy":
                result = _static_result({SPY: 1.0}, regime="BUY_HOLD")
                plan = _build_buy_hold_plan(current_weights, result, runtime.portfolio)
            else:
                raise ValueError(f"Unknown variant kind: {variant.kind}")

        if plan.rebalance_due:
            components = (
                set(runtime.portfolio.shares)
                | set(plan.execution_weights)
            )
            runtime.pending_order = PendingOrder(
                signal_date=signal_date,
                execution_weights=dict(plan.execution_weights),
                result=result,
                reason=plan.reason,
                median_dollar_volume20=_median_dollar_volume20_at_signal(
                    data,
                    full_position,
                    components,
                ),
            )

        _advance_signal_state(
            runtime.portfolio,
            signal_date=signal_date,
            result=result,
            tier_decision=tier_decision,
            sector_due=sector_due,
            plan=plan,
        )

        close_exposure_multipliers = dict(engine.ADVERTISED_DAILY_MULTIPLIERS)
        close_exposure_multipliers[SPY] = 1.0
        exposure = sum(
            current_weights.get(ticker, 0.0) * multiplier
            for ticker, multiplier in close_exposure_multipliers.items()
        )
        rows.append(
            {
                "session": signal_date,
                "nav": close_nav,
                "daily_return": daily_return,
                "cash": runtime.portfolio.cash_balance,
                "regime": result.regime,
                "volatility_tier": result.volatility_tier,
                "raw_volatility_tier": result.raw_volatility_tier,
                "leader": result.leader,
                "bullish_consensus": (
                    int(latest["bullish_consensus"])
                    if "bullish_consensus" in latest
                    else -1
                ),
                "annualized_volatility": (
                    float(latest["annualized_volatility"])
                    if "annualized_volatility" in latest
                    else 0.0
                ),
                "signal_reason": plan.reason,
                "signal_rebalance_due": bool(plan.rebalance_due),
                "filled_reason": filled_reason,
                "initial_deployment": initial_deployment,
                "gross_trade_fraction": fill.gross_fraction if fill else 0.0,
                "ongoing_gross_trade_fraction": (
                    0.0 if fill is None or initial_deployment else fill.gross_fraction
                ),
                "one_way_turnover": (
                    0.0 if fill is None or initial_deployment else fill.one_way_fraction
                ),
                "transaction_cost": fill.cost if fill else 0.0,
                "transaction_cost_fraction": fill.cost_fraction if fill else 0.0,
                "security_orders": fill.security_orders if fill else 0,
                "max_trade_to_median_dollar_volume20": adv_ratio,
                "advertised_daily_exposure": exposure,
                "weights": json.dumps(
                    current_weights,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "shares": json.dumps(
                    runtime.portfolio.shares,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "target": json.dumps(
                    result.target_weights,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "queued_target": json.dumps(
                    plan.execution_weights if plan.rebalance_due else {},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )

    ledger = pd.DataFrame(rows).set_index("session")
    metrics = calculate_metrics(ledger)
    return BacktestResult(
        variant=variant,
        cost_bps=float(cost_bps),
        ledger=ledger,
        metrics=metrics,
        unfilled_final_order=runtime.pending_order is not None,
    )


def _drawdown_from_nav(nav: pd.Series) -> tuple[float, int]:
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    max_drawdown = float(drawdown.min())
    longest = 0
    current = 0
    for value in drawdown.to_numpy(dtype=float):
        if value < -1e-12:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return max_drawdown, longest


def _worst_rolling(nav: pd.Series, sessions: int) -> float | None:
    if len(nav) <= sessions:
        return None
    values = nav / nav.shift(sessions) - 1.0
    return float(values.dropna().min())


def _series_checksum(series: pd.Series) -> str:
    rendered = series.to_csv(
        index=True,
        date_format="%Y-%m-%d",
        float_format="%.15g",
        lineterminator="\n",
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def calculate_metrics(ledger: pd.DataFrame) -> dict[str, object]:
    if len(ledger) < 2:
        raise RuntimeError("At least two scored sessions are required")
    nav = ledger["nav"].astype(float)
    returns = nav.pct_change(fill_method=None).iloc[1:]
    if not np.isfinite(returns.to_numpy()).all() or (returns <= -1).any():
        raise RuntimeError("Metric return series is invalid")
    years = len(returns) / 252.0
    terminal_wealth = float(nav.iloc[-1] / nav.iloc[0])
    cagr = terminal_wealth ** (1.0 / years) - 1.0
    volatility = float(returns.std(ddof=1) * math.sqrt(252))
    mean_return = float(returns.mean())
    sharpe = (
        mean_return / float(returns.std(ddof=1)) * math.sqrt(252)
        if float(returns.std(ddof=1)) > 0
        else 0.0
    )
    downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)))
    sortino = (
        mean_return / downside_deviation * math.sqrt(252)
        if downside_deviation > 0
        else 0.0
    )
    max_drawdown, underwater = _drawdown_from_nav(nav)
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0
    tail_count = max(1, int(math.ceil(0.05 * len(returns))))
    expected_shortfall = float(np.sort(returns.to_numpy())[:tail_count].mean())
    without_best = returns.drop(returns.nlargest(min(5, len(returns))).index)
    cagr_without_best_5 = (
        float(np.prod(1.0 + without_best.to_numpy()) ** (252.0 / len(returns)) - 1.0)
        if len(without_best)
        else None
    )
    annualizer = 1.0 / years
    return {
        "observations": int(len(returns)),
        "years_252": years,
        "start": ledger.index[0].date().isoformat(),
        "end": ledger.index[-1].date().isoformat(),
        "starting_nav": float(nav.iloc[0]),
        "ending_nav": float(nav.iloc[-1]),
        "terminal_wealth_multiple": terminal_wealth,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_rf": sharpe,
        "sortino_zero_mar": sortino,
        "calmar": calmar,
        "maximum_drawdown": max_drawdown,
        "maximum_time_underwater_sessions": underwater,
        "expected_shortfall_95_daily": expected_shortfall,
        "worst_rolling_1y": _worst_rolling(nav, 252),
        "worst_rolling_3y": _worst_rolling(nav, 756),
        "worst_rolling_5y": _worst_rolling(nav, 1260),
        "cagr_without_best_5_sessions": cagr_without_best_5,
        "annual_gross_turnover": float(
            ledger["ongoing_gross_trade_fraction"].sum() * annualizer
        ),
        "annual_one_way_turnover": float(
            ledger["one_way_turnover"].sum() * annualizer
        ),
        "ongoing_security_orders_annual": float(
            ledger.loc[~ledger["initial_deployment"], "security_orders"].sum()
            * annualizer
        ),
        "total_modeled_cost": float(ledger["transaction_cost"].sum()),
        "annual_modeled_cost_fraction": float(
            ledger["transaction_cost_fraction"].sum() * annualizer
        ),
        "maximum_trade_to_median_dollar_volume20": float(
            ledger["max_trade_to_median_dollar_volume20"].max()
        ),
        "unmeasurable_liquidity_trade_sessions": int(
            np.isinf(
                ledger["max_trade_to_median_dollar_volume20"].to_numpy(
                    dtype=float
                )
            ).sum()
        ),
        "mean_advertised_daily_exposure": float(
            ledger["advertised_daily_exposure"].mean()
        ),
        "return_checksum": _series_checksum(returns),
    }


def _compounded_return(returns: pd.Series) -> float:
    return float(np.prod(1.0 + returns.to_numpy(dtype=float)) - 1.0)


def _drawdown_from_returns(returns: pd.Series) -> float:
    nav = pd.Series(
        np.concatenate(([1.0], np.cumprod(1.0 + returns.to_numpy(dtype=float))))
    )
    return _drawdown_from_nav(nav)[0]


def chronological_folds(
    candidate: BacktestResult,
    baseline: BacktestResult,
    *,
    folds: int = 4,
) -> list[dict[str, object]]:
    candidate_returns = candidate.ledger["nav"].pct_change(fill_method=None).iloc[1:]
    baseline_returns = baseline.ledger["nav"].pct_change(fill_method=None).iloc[1:]
    if not candidate_returns.index.equals(baseline_returns.index):
        raise RuntimeError("Fold return indexes differ")
    positions = np.array_split(np.arange(len(candidate_returns)), folds)
    output: list[dict[str, object]] = []
    for number, fold_positions in enumerate(positions, start=1):
        if len(fold_positions) == 0:
            raise RuntimeError("Chronological fold is empty")
        c = candidate_returns.iloc[fold_positions]
        b = baseline_returns.iloc[fold_positions]
        candidate_implementation = candidate.ledger.iloc[fold_positions + 1]
        baseline_implementation = baseline.ledger.iloc[fold_positions + 1]
        fold_annualizer = 252.0 / len(fold_positions)
        log_difference = np.log1p(c) - np.log1p(b)
        output.append(
            {
                "fold": number,
                "start": c.index[0].date().isoformat(),
                "end": c.index[-1].date().isoformat(),
                "observations": int(len(c)),
                "underpowered": bool(len(c) < 126),
                "candidate_compounded_return": _compounded_return(c),
                "baseline_compounded_return": _compounded_return(b),
                "candidate_annualized_log_growth": float(np.log1p(c).mean() * 252),
                "baseline_annualized_log_growth": float(np.log1p(b).mean() * 252),
                "annualized_excess_geometric_growth": float(
                    np.expm1(log_difference.mean() * 252)
                ),
                "relative_terminal_wealth": float(np.expm1(log_difference.sum())),
                "candidate_maximum_drawdown": _drawdown_from_returns(c),
                "baseline_maximum_drawdown": _drawdown_from_returns(b),
                "candidate_annual_gross_turnover": float(
                    candidate_implementation[
                        "ongoing_gross_trade_fraction"
                    ].sum()
                    * fold_annualizer
                ),
                "baseline_annual_gross_turnover": float(
                    baseline_implementation[
                        "ongoing_gross_trade_fraction"
                    ].sum()
                    * fold_annualizer
                ),
                "candidate_annual_one_way_turnover": float(
                    candidate_implementation["one_way_turnover"].sum()
                    * fold_annualizer
                ),
                "baseline_annual_one_way_turnover": float(
                    baseline_implementation["one_way_turnover"].sum()
                    * fold_annualizer
                ),
                "candidate_modeled_cost": float(
                    candidate_implementation["transaction_cost"].sum()
                ),
                "baseline_modeled_cost": float(
                    baseline_implementation["transaction_cost"].sum()
                ),
                "positive_excess": bool(log_difference.mean() > 0),
            }
        )
    return output


def bootstrap_block_starts(
    observations: int,
    *,
    samples: int,
    block_length: int,
    seed: int = MASTER_SEED,
) -> np.ndarray:
    if observations < 2:
        raise ValueError("Bootstrap requires at least two observations")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    if not (1 <= block_length <= observations):
        raise ValueError("Bootstrap block length is invalid")
    blocks_needed = int(math.ceil(observations / block_length))
    rng = np.random.default_rng(seed)
    return rng.integers(
        0,
        observations - block_length + 1,
        size=(samples, blocks_needed),
        endpoint=False,
    )


def _block_resample_means(
    values: np.ndarray,
    starts: np.ndarray,
    block_length: int,
) -> np.ndarray:
    observations = len(values)
    full_blocks = observations // block_length
    remainder = observations % block_length
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    block_sums = cumulative[block_length:] - cumulative[:-block_length]
    totals = np.zeros(starts.shape[0], dtype=float)
    if full_blocks:
        totals += block_sums[starts[:, :full_blocks]].sum(axis=1)
    if remainder:
        partial_starts = starts[:, full_blocks]
        partial_sums = (
            cumulative[partial_starts + remainder]
            - cumulative[partial_starts]
        )
        totals += partial_sums
    return totals / observations


def paired_moving_block_bootstrap(
    candidate: BacktestResult,
    baseline: BacktestResult,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    block_length: int | None = None,
    seed: int = MASTER_SEED,
    starts: np.ndarray | None = None,
) -> dict[str, object]:
    c = candidate.ledger["nav"].pct_change(fill_method=None).iloc[1:]
    b = baseline.ledger["nav"].pct_change(fill_method=None).iloc[1:]
    if not c.index.equals(b.index):
        raise RuntimeError("Bootstrap return indexes differ")
    difference = (np.log1p(c) - np.log1p(b)).to_numpy(dtype=float)
    if not np.isfinite(difference).all():
        raise RuntimeError("Paired log-growth difference is nonfinite")
    observations = len(difference)
    selected_length = (
        max(5, int(round(observations ** (1.0 / 3.0))))
        if block_length is None
        else int(block_length)
    )
    if selected_length > observations // 4:
        raise RuntimeError("Bootstrap block length exceeds one quarter of sample")
    if starts is None:
        starts = bootstrap_block_starts(
            observations,
            samples=samples,
            block_length=selected_length,
            seed=seed,
        )
    expected_columns = int(math.ceil(observations / selected_length))
    if starts.shape != (samples, expected_columns):
        raise ValueError("Shared bootstrap start matrix has the wrong shape")
    if not np.issubdtype(starts.dtype, np.integer):
        raise ValueError("Shared bootstrap starts must use an integer dtype")
    if starts.size and (
        int(starts.min()) < 0
        or int(starts.max()) > observations - selected_length
    ):
        raise ValueError("Shared bootstrap starts are outside valid block bounds")

    observed_mean = float(difference.mean())
    if np.allclose(difference, 0.0, rtol=0, atol=1e-15):
        return {
            "observations": observations,
            "samples": samples,
            "block_length": selected_length,
            "seed": seed,
            "annualized_log_excess": 0.0,
            "annualized_excess_geometric_growth": 0.0,
            "ci95_excess_geometric_growth": [0.0, 0.0],
            "one_sided_p_value": 1.0,
            "degenerate": True,
            "inferential_sample_count_adequate": bool(
                samples >= MIN_INFERENTIAL_BOOTSTRAP_SAMPLES
            ),
            "p_value_resolution": 1.0 / (samples + 1.0),
        }

    resampled_means = _block_resample_means(
        difference,
        starts,
        selected_length,
    )
    centered_means = resampled_means - observed_mean
    observed_annualized = observed_mean * 252.0
    lower, upper = np.quantile(resampled_means * 252.0, [0.025, 0.975])
    p_value = (
        1.0
        + float(np.count_nonzero(centered_means * 252.0 >= observed_annualized))
    ) / (samples + 1.0)
    return {
        "observations": observations,
        "samples": samples,
        "block_length": selected_length,
        "seed": seed,
        "annualized_log_excess": observed_annualized,
        "annualized_excess_geometric_growth": float(np.expm1(observed_annualized)),
        "ci95_excess_geometric_growth": [
            float(np.expm1(lower)),
            float(np.expm1(upper)),
        ],
        "one_sided_p_value": p_value,
        "degenerate": bool(np.std(difference, ddof=1) <= 1e-15),
        "inferential_sample_count_adequate": bool(
            samples >= MIN_INFERENTIAL_BOOTSTRAP_SAMPLES
        ),
        "p_value_resolution": 1.0 / (samples + 1.0),
        "caveat": (
            "Moving-block bootstrap inference assumes short-range dependence "
            "and approximate stability."
        ),
    }


def benjamini_hochberg(
    p_values: Mapping[str, float],
) -> dict[str, float]:
    if not p_values:
        return {}
    ordered = sorted(
        ((name, float(value)) for name, value in p_values.items()),
        key=lambda item: (item[1], item[0]),
    )
    if any(not np.isfinite(value) or not (0 <= value <= 1) for _, value in ordered):
        raise ValueError("BH p-values must be finite and in [0, 1]")
    total = len(ordered)
    adjusted_ordered = [0.0] * total
    running = 1.0
    for index in range(total - 1, -1, -1):
        rank = index + 1
        running = min(running, ordered[index][1] * total / rank)
        adjusted_ordered[index] = min(1.0, running)
    return {
        name: adjusted
        for (name, _), adjusted in zip(ordered, adjusted_ordered)
    }


def pbo_diagnostic(
    results: Mapping[str, BacktestResult],
    *,
    blocks: int = 8,
) -> dict[str, object]:
    if blocks <= 0 or blocks % 2:
        raise ValueError("PBO block count must be positive and even")
    names = sorted(results)
    if len(names) < 3:
        return {
            "available": False,
            "reason": "At least three complete strategies are required",
        }
    returns = {
        name: results[name].ledger["nav"].pct_change(fill_method=None).iloc[1:]
        for name in names
    }
    reference_index = returns[names[0]].index
    if any(not series.index.equals(reference_index) for series in returns.values()):
        raise RuntimeError("PBO return indexes differ")
    if len(reference_index) < 63 * blocks:
        return {
            "available": False,
            "reason": f"At least {63 * blocks} common sessions are required",
        }

    observations = len(reference_index)
    block_positions = np.array_split(np.arange(observations), blocks)
    matrix = np.column_stack(
        [np.log1p(returns[name].to_numpy(dtype=float)) for name in names]
    )
    logits: list[float] = []
    ranks: list[float] = []
    tie_count = 0
    selections: dict[str, int] = {name: 0 for name in names}
    all_blocks = set(range(blocks))
    for in_sample_tuple in itertools.combinations(range(blocks), blocks // 2):
        in_sample = set(in_sample_tuple)
        out_sample = all_blocks - in_sample
        in_positions = np.concatenate([block_positions[item] for item in sorted(in_sample)])
        out_positions = np.concatenate([block_positions[item] for item in sorted(out_sample)])
        in_scores = matrix[in_positions].mean(axis=0)
        best_score = float(np.max(in_scores))
        tied = [
            index
            for index, value in enumerate(in_scores)
            if np.isclose(value, best_score, rtol=0, atol=1e-15)
        ]
        if len(tied) > 1:
            tie_count += 1
        selected_index = min(tied, key=lambda item: names[item])
        selections[names[selected_index]] += 1

        out_scores = matrix[out_positions].mean(axis=0)
        selected_score = float(out_scores[selected_index])
        less = int(np.count_nonzero(out_scores < selected_score - 1e-15))
        equal = int(np.count_nonzero(np.isclose(out_scores, selected_score, rtol=0, atol=1e-15)))
        rank = 1.0 + less + (equal - 1.0) / 2.0
        normalized = rank / (len(names) + 1.0)
        logit = math.log(normalized / (1.0 - normalized))
        ranks.append(rank)
        logits.append(logit)

    failures = sum(value <= 0 for value in logits)
    return {
        "available": True,
        "label": "CSCV-style selection-instability diagnostic",
        "strategies": names,
        "blocks": blocks,
        "usable_observations": observations,
        "splits": len(logits),
        "failures": failures,
        "pbo": failures / len(logits),
        "mean_oos_rank": float(np.mean(ranks)),
        "is_tie_splits": tie_count,
        "is_selections": selections,
        "caveat": (
            "Interleaved dependent splits diagnose instability in this candidate "
            "matrix; they are not chronological walk-forward proof."
        ),
    }


def _year_dependence(
    candidate: BacktestResult,
    baseline: BacktestResult,
) -> dict[str, object]:
    c = candidate.ledger["nav"].pct_change(fill_method=None).iloc[1:]
    b = baseline.ledger["nav"].pct_change(fill_method=None).iloc[1:]
    difference = np.log1p(c) - np.log1p(b)
    segment_rows: dict[str, dict[str, object]] = {}
    complete_values: dict[int, float] = {}
    for year, segment in difference.groupby(difference.index.year):
        expected = _calendar_sessions(
            pd.Timestamp(f"{int(year)}-01-01"),
            pd.Timestamp(f"{int(year)}-12-31"),
        )
        actual = pd.DatetimeIndex(segment.index).normalize()
        complete = actual.equals(expected)
        value = float(segment.sum())
        segment_rows[str(int(year))] = {
            "start": actual[0].date().isoformat(),
            "end": actual[-1].date().isoformat(),
            "observations": len(actual),
            "complete_calendar_year": complete,
            "excess_log_growth": value,
        }
        if complete:
            complete_values[int(year)] = value

    complete_series = pd.Series(complete_values, dtype=float)
    total = float(complete_series.sum())
    positive = complete_series.clip(lower=0.0)
    positive_total = float(positive.sum())
    contribution = (
        float(positive.max() / positive_total)
        if positive_total > 0
        else None
    )
    leave_one_out = [total - float(value) for value in complete_series]
    positive_fraction = (
        sum(value > 0 for value in leave_one_out) / len(leave_one_out)
        if leave_one_out
        else 0.0
    )
    return {
        "calendar_segments": segment_rows,
        "complete_calendar_years": [
            int(year) for year in complete_series.index
        ],
        "complete_year_count": len(complete_series),
        "largest_positive_complete_year_share_of_positive_excess": contribution,
        "leave_one_complete_year_out_positive_fraction": positive_fraction,
        "descriptive_only": True,
    }


def evaluate_candidates(
    all_results: Mapping[tuple[str, float], BacktestResult],
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    baseline_primary = all_results[("original", PRIMARY_COST_BPS)]
    candidate_names = sorted(
        name for name in DECLARED_CANDIDATE_FAMILY
        if (name, PRIMARY_COST_BPS) in all_results
    )
    family_complete = (
        tuple(candidate_names) == DECLARED_CANDIDATE_FAMILY
        and all((name, 25.0) in all_results for name in ("original", *candidate_names))
    )
    observations = int(baseline_primary.metrics["observations"])
    block_length = max(5, int(round(observations ** (1.0 / 3.0))))
    shared_starts = bootstrap_block_starts(
        observations,
        samples=bootstrap_samples,
        block_length=block_length,
        seed=MASTER_SEED,
    )

    comparisons: dict[str, dict[str, object]] = {}
    raw_p_values: dict[str, float] = {}
    for name in candidate_names:
        candidate = all_results[(name, PRIMARY_COST_BPS)]
        folds = chronological_folds(candidate, baseline_primary)
        bootstrap = paired_moving_block_bootstrap(
            candidate,
            baseline_primary,
            samples=bootstrap_samples,
            block_length=block_length,
            seed=MASTER_SEED,
            starts=shared_starts,
        )
        sensitivity: dict[str, object] = {}
        for sensitivity_length in (5, 10, 21):
            sensitivity_starts = bootstrap_block_starts(
                observations,
                samples=bootstrap_samples,
                block_length=sensitivity_length,
                seed=MASTER_SEED,
            )
            sensitivity[str(sensitivity_length)] = paired_moving_block_bootstrap(
                candidate,
                baseline_primary,
                samples=bootstrap_samples,
                block_length=sensitivity_length,
                seed=MASTER_SEED,
                starts=sensitivity_starts,
            )
        raw_p_values[name] = float(bootstrap["one_sided_p_value"])
        year_dependence = _year_dependence(candidate, baseline_primary)
        folds_adequate = not any(bool(item["underpowered"]) for item in folds)
        comparisons[name] = {
            "description": VARIANTS[name].description,
            "primary_cost_bps": PRIMARY_COST_BPS,
            "descriptive_cagr_difference_at_10bps": (
                float(candidate.metrics["cagr"])
                - float(baseline_primary.metrics["cagr"])
            ),
            "folds": folds,
            "four_fold_evidence_available": folds_adequate,
            "fold_win_rate": sum(bool(item["positive_excess"]) for item in folds)
            / len(folds),
            "median_fold_excess_geometric_growth": float(
                np.median(
                    [
                        float(item["annualized_excess_geometric_growth"])
                        for item in folds
                    ]
                )
            ),
            "bootstrap": bootstrap,
            "bootstrap_block_length_sensitivity": sensitivity,
            "year_dependence": year_dependence,
        }

    adjusted = (
        benjamini_hochberg(raw_p_values)
        if family_complete
        else {name: None for name in candidate_names}
    )
    primary_candidate_results = {
        name: all_results[(name, PRIMARY_COST_BPS)]
        for name in ["original", *candidate_names]
    }
    pbo = pbo_diagnostic(primary_candidate_results)
    pbo["scope"] = (
        "complete_declared_candidate_matrix"
        if family_complete
        else "exploratory_subset"
    )
    pbo["historical_selection_warning"] = (
        "This prospective matrix cannot measure selection of Original over the "
        "unavailable historical A/B daily paths."
    )

    for name, comparison in comparisons.items():
        comparison["bh_adjusted_p_value"] = adjusted[name]
        comparison["bh_reject_at_prespecified_fdr"] = bool(
            family_complete
            and adjusted[name] is not None
            and float(adjusted[name]) <= FDR_Q
        )
        candidate_10 = all_results[(name, PRIMARY_COST_BPS)]
        baseline_10 = baseline_primary
        candidate_25 = all_results.get((name, 25.0))
        baseline_25 = all_results.get(("original", 25.0))
        excess_25 = (
            float(candidate_25.metrics["cagr"]) - float(baseline_25.metrics["cagr"])
            if candidate_25 is not None and baseline_25 is not None
            else None
        )
        paired_excess_25: float | None = None
        if candidate_25 is not None and baseline_25 is not None:
            candidate_returns_25 = (
                candidate_25.ledger["nav"].pct_change(fill_method=None).iloc[1:]
            )
            baseline_returns_25 = (
                baseline_25.ledger["nav"].pct_change(fill_method=None).iloc[1:]
            )
            paired_excess_25 = float(
                np.expm1(
                    (
                        np.log1p(candidate_returns_25)
                        - np.log1p(baseline_returns_25)
                    ).mean()
                    * 252.0
                )
            )
        dd_worsening = abs(min(0.0, float(candidate_10.metrics["maximum_drawdown"]))) - abs(
            min(0.0, float(baseline_10.metrics["maximum_drawdown"]))
        )
        candidate_es_raw = float(
            candidate_10.metrics["expected_shortfall_95_daily"]
        )
        baseline_es_raw = float(
            baseline_10.metrics["expected_shortfall_95_daily"]
        )
        candidate_loss_es = max(0.0, -candidate_es_raw)
        baseline_loss_es = max(0.0, -baseline_es_raw)
        es_tolerance = 1e-15
        es_pass = (
            candidate_loss_es <= es_tolerance
            if baseline_loss_es <= es_tolerance
            else candidate_loss_es <= baseline_loss_es * 1.10
        )
        turnover_ratio = (
            float(candidate_10.metrics["annual_gross_turnover"])
            / float(baseline_10.metrics["annual_gross_turnover"])
            if float(baseline_10.metrics["annual_gross_turnover"]) > 0
            else math.inf
        )
        year_info = comparison["year_dependence"]
        largest_year_share = year_info[
            "largest_positive_complete_year_share_of_positive_excess"
        ]
        candidate_liquidity = float(
            candidate_10.metrics[
                "maximum_trade_to_median_dollar_volume20"
            ]
        )
        baseline_liquidity = float(
            baseline_10.metrics[
                "maximum_trade_to_median_dollar_volume20"
            ]
        )
        liquidity_measurable = (
            int(candidate_10.metrics["unmeasurable_liquidity_trade_sessions"]) == 0
            and int(baseline_10.metrics["unmeasurable_liquidity_trade_sessions"]) == 0
        )
        paired_point = float(
            comparison["bootstrap"]["annualized_excess_geometric_growth"]
        )
        ci_lower = float(
            comparison["bootstrap"]["ci95_excess_geometric_growth"][0]
        )
        statistical_checks = {
            "complete_prespecified_candidate_family": family_complete,
            "at_least_9999_bootstrap_samples": bool(
                bootstrap_samples >= MIN_INFERENTIAL_BOOTSTRAP_SAMPLES
            ),
            "bh_adjusted_one_sided_p_at_most_prespecified_fdr": bool(
                family_complete
                and adjusted[name] is not None
                and float(adjusted[name]) <= FDR_Q
            ),
            "bootstrap_ci_lower_bound_positive": bool(ci_lower > 0),
        }
        economic_checks = {
            "paired_excess_geometric_growth_at_10bps_at_least_1pp": bool(
                paired_point >= 0.01
            ),
            "positive_paired_excess_geometric_growth_at_25bps": bool(
                paired_excess_25 is not None and paired_excess_25 > 0
            ),
        }
        risk_implementation_checks = {
            "drawdown_not_worse_by_more_than_3pp": bool(dd_worsening <= 0.03),
            "expected_shortfall_loss_not_worse_by_more_than_10pct": bool(es_pass),
            "turnover_within_25pct_or_25bp_hurdle": bool(
                turnover_ratio <= 1.25
                or (paired_excess_25 is not None and paired_excess_25 >= 0.01)
            ),
            "trade_size_at_most_0_1pct_of_20_session_median_dollar_volume": bool(
                liquidity_measurable
                and candidate_liquidity <= 0.001
                and baseline_liquidity <= 0.001
            ),
        }
        stability_checks = {
            "four_chronological_slices_have_at_least_126_sessions": bool(
                comparison["four_fold_evidence_available"]
            ),
            "median_fold_excess_positive": bool(
                comparison["four_fold_evidence_available"]
                and
                comparison["median_fold_excess_geometric_growth"] > 0
            ),
            "positive_in_at_least_3_of_4_chronological_slices": bool(
                comparison["four_fold_evidence_available"]
                and comparison["fold_win_rate"] >= 0.75
            ),
            "at_least_5_complete_calendar_years_for_descriptive_stability": bool(
                year_info["complete_year_count"] >= 5
            ),
            "no_single_complete_year_over_half_of_positive_excess": bool(
                largest_year_share is not None and largest_year_share <= 0.50
            ),
            "leave_one_complete_year_out_positive_at_least_80pct": bool(
                year_info["leave_one_complete_year_out_positive_fraction"] >= 0.80
            ),
        }
        grouped_checks = {
            "statistical_evidence": statistical_checks,
            "economic_materiality": economic_checks,
            "risk_and_implementation": risk_implementation_checks,
            "descriptive_stability": stability_checks,
        }
        comparison["descriptive_cagr_difference_at_25bps"] = excess_25
        comparison[
            "paired_excess_geometric_growth_at_25bps"
        ] = paired_excess_25
        comparison["drawdown_worsening"] = dd_worsening
        comparison["expected_shortfall_loss_ratio"] = (
            candidate_loss_es / baseline_loss_es
            if baseline_loss_es > es_tolerance
            else None
        )
        comparison["gross_turnover_ratio"] = turnover_ratio
        comparison["screening_checks"] = grouped_checks
        comparison["passes_exploratory_screening_checklist"] = all(
            value
            for group in grouped_checks.values()
            for value in group.values()
        )
        comparison["eligible_for_live_promotion"] = False
        comparison["promotion_blockers"] = [
            "No untouched historical holdout remains",
            "Prior A/B daily histories and full cumulative trial count are unavailable",
            "Deflated-Sharpe selection correction cannot be computed honestly",
            "A 252-session live shadow period has not been completed",
        ]
        if not risk_implementation_checks[
            "trade_size_at_most_0_1pct_of_20_session_median_dollar_volume"
        ]:
            comparison["promotion_blockers"].append(
                "Historical simulated trade size exceeds the locked liquidity limit"
            )

    return {
        "comparisons": comparisons,
        "declared_candidate_family": list(DECLARED_CANDIDATE_FAMILY),
        "declared_candidate_family_fingerprint": DECLARED_FAMILY_FINGERPRINT,
        "candidate_family_complete": family_complete,
        "benjamini_hochberg": {
            "prespecified_fdr": FDR_Q,
            "family": list(DECLARED_CANDIDATE_FAMILY),
            "family_complete": family_complete,
            "adjusted_values": adjusted,
            "dependence_caveat": (
                "BH is prespecified multiplicity control; formal guarantees "
                "require independence or suitable positive dependence."
            ),
        },
        "prospective_candidate_matrix_selection_instability": pbo,
        "statistical_caveat": (
            "The old A/B summary lacks daily paths and cannot enter paired inference, "
            "BH, DSR, or historical-selection PBO. These results are exploratory."
        ),
    }


def _json_clean(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def build_report(
    data: MarketData,
    results: Mapping[tuple[str, float], BacktestResult],
    evaluation: Mapping[str, object],
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    summaries: dict[str, dict[str, object]] = {}
    for (name, cost), result in sorted(results.items()):
        summaries.setdefault(name, {})[f"{cost:g}bps"] = {
            "metrics": result.metrics,
            "unfilled_final_order": result.unfilled_final_order,
        }
    report = {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "research_only": True,
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "production_strategy_revision": engine.STRATEGY_REVISION,
        "production_strategy_fingerprint": engine.STRATEGY_FINGERPRINT,
        "production_implementation_fingerprint": (
            engine.calculate_implementation_fingerprint()
        ),
        "data": {
            "source": data.source,
            "adjustment_semantics": (
                "Adjusted Open/Close are used together; adjusted prices embed "
                "distributions and split history."
            ),
            "requested_start": data.requested_start,
            "requested_end_exclusive": data.requested_end_exclusive,
            "common_actual_inception": data.common_start.date().isoformat(),
            "scored_start": data.scoring_dates[0].date().isoformat(),
            "final_session": data.final_session.date().isoformat(),
            "scored_sessions": len(data.scoring_dates),
            "fingerprint_sha256": data.fingerprint,
            "missing_data_policy": "fail_closed_no_fill_drop_or_substitution",
        },
        "execution": {
            "signal": "completed adjusted Close t",
            "fill": "adjusted Open of next validated XNYS session",
            "holdings_truth": "actual fractional shares and cash",
            "cost_basis": "basis points on gross security notional",
            "costs_bps": sorted({cost for _, cost in results}),
            "initial_deployment": (
                "cost included in wealth; excluded from ongoing turnover"
            ),
            "one_way_turnover_definition": (
                "one-half L1 distance between pretrade and post-cost target "
                "weights, including cash"
            ),
            "liquidity_diagnostic": (
                "absolute trade notional divided by trailing 20-session median "
                "adjusted-close dollar volume"
            ),
            "locked_liquidity_limit": 0.001,
            "cash_return": 0.0,
        },
        "variants": {
            name: asdict(VARIANTS[name])
            for name in sorted({name for name, _ in results})
        },
        "summaries": summaries,
        "evaluation": evaluation,
        "inference": {
            "seed": MASTER_SEED,
            "bootstrap_samples": bootstrap_samples,
            "objective": "paired annualized daily log-growth difference",
            "folds": 4,
            "pbo_blocks": 8,
            "prespecified_fdr": FDR_Q,
            "declared_candidate_family": list(DECLARED_CANDIDATE_FAMILY),
            "declared_candidate_family_fingerprint": (
                DECLARED_FAMILY_FINGERPRINT
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return _json_clean(report)


def save_json_atomic(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_ledgers(
    results: Mapping[tuple[str, float], BacktestResult],
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for (name, cost), result in sorted(results.items()):
        path = directory / f"{name}_{cost:g}bps.csv"
        temporary = path.with_name(path.name + ".tmp")
        result.ledger.to_csv(
            temporary,
            index=True,
            date_format="%Y-%m-%d",
            float_format="%.12g",
            lineterminator="\n",
        )
        os.replace(temporary, path)


def print_human_summary(
    data: MarketData,
    results: Mapping[tuple[str, float], BacktestResult],
    evaluation: Mapping[str, object],
) -> None:
    print(
        "Research-only production-parity backtest "
        f"{data.scoring_dates[0].date()} through {data.final_session.date()}"
    )
    print(f"Data SHA-256: {data.fingerprint}")
    print(
        f"{'Variant':<24} {'Cost':>6} {'CAGR':>9} {'MaxDD':>9} "
        f"{'Sharpe':>8} {'GrossTO':>9} {'End NAV':>12}"
    )
    for (name, cost), result in sorted(results.items(), key=lambda item: (item[0][1], item[0][0])):
        metrics = result.metrics
        print(
            f"{name:<24} {cost:>5g}bp "
            f"{float(metrics['cagr']):>8.2%} "
            f"{float(metrics['maximum_drawdown']):>8.2%} "
            f"{float(metrics['sharpe_zero_rf']):>8.2f} "
            f"{float(metrics['annual_gross_turnover']):>8.2%} "
            f"{float(metrics['ending_nav']):>12,.2f}"
        )
    comparisons = evaluation.get("comparisons", {})
    if comparisons:
        print("\nCandidate evidence at 10bp versus Original:")
        for name, comparison in comparisons.items():
            ci = comparison["bootstrap"]["ci95_excess_geometric_growth"]
            adjusted = comparison["bh_adjusted_p_value"]
            adjusted_text = (
                f"{float(adjusted):.4f}"
                if adjusted is not None
                else "N/A (incomplete family)"
            )
            print(
                f"- {name}: descriptive CAGR difference "
                f"{float(comparison['descriptive_cagr_difference_at_10bps']):+.2%}; "
                f"fold wins {float(comparison['fold_win_rate']):.0%}; "
                f"bootstrap excess CI [{float(ci[0]):+.2%}, {float(ci[1]):+.2%}]; "
                f"BH-adjusted one-sided p={adjusted_text}; "
                "exploratory checklist="
                f"{comparison['passes_exploratory_screening_checklist']}"
            )
    print(
        "\nNo candidate is eligible for live promotion from this inspected sample; "
        "see the JSON promotion blockers."
    )


def default_end_exclusive() -> str:
    completed = engine.expected_completed_session(
        datetime.now(engine.NEW_YORK)
    )
    return (completed.date() + timedelta(days=1)).isoformat()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated production-parity portfolio research"
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=default_end_exclusive())
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANT_NAMES),
        choices=sorted(VARIANTS),
    )
    parser.add_argument(
        "--cost-bps",
        nargs="+",
        type=float,
        default=list(DEFAULT_COSTS_BPS),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument(
        "--score-start",
        help="Optional later XNYS scoring start; warm-up and fingerprint remain intact",
    )
    parser.add_argument(
        "--input-snapshot",
        type=Path,
        help="Replay a frozen long-form snapshot instead of downloading",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ledger-dir", type=Path)
    parser.add_argument("--snapshot", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if "original" not in args.variants:
        raise SystemExit("--variants must include original for paired evaluation")
    if PRIMARY_COST_BPS not in args.cost_bps:
        raise SystemExit("--cost-bps must include the primary 10bp scenario")
    if any(value < 0 or not np.isfinite(value) for value in args.cost_bps):
        raise SystemExit("--cost-bps values must be finite and nonnegative")
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")

    data = (
        load_market_snapshot(
            args.input_snapshot,
            requested_start=args.start,
            requested_end_exclusive=args.end,
        )
        if args.input_snapshot
        else download_market_data(
            requested_start=args.start,
            requested_end_exclusive=args.end,
        )
    )
    if args.score_start:
        data = restrict_scoring_start(data, args.score_start)
    if args.snapshot:
        save_market_snapshot(data, args.snapshot)

    results: dict[tuple[str, float], BacktestResult] = {}
    for cost_bps in sorted(set(float(value) for value in args.cost_bps)):
        for name in args.variants:
            result = simulate_variant(
                data,
                VARIANTS[name],
                cost_bps=cost_bps,
                starting_cash=args.starting_cash,
            )
            results[(name, cost_bps)] = result

    evaluation = evaluate_candidates(
        results,
        bootstrap_samples=args.bootstrap_samples,
    )
    report = build_report(
        data,
        results,
        evaluation,
        bootstrap_samples=args.bootstrap_samples,
    )
    if args.output:
        save_json_atomic(report, args.output)
    if args.ledger_dir:
        save_ledgers(results, args.ledger_dir)
    print_human_summary(data, results, evaluation)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Frozen, research-only comparator for the Original ROTH IRA Barbell.

The strategy in this module is a causal port of the implementation at Git
commit ``fa9b3245acfb3d6d17f4a1bc6c17cd96e27137e3``.  It is intentionally
isolated from the live engine: there is no production-state access, network
download, notification, email, or file-writing path.

Signals are formed from completed adjusted Close/Volume observations at
session ``t``.  A queued target is executed with fractional shares at the
adjusted Open of the next validated XNYS session.  Actual shares and cash,
not a previously calculated target, drive every later drift decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping

import exchange_calendars as xcals
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Frozen source provenance
# ---------------------------------------------------------------------------
LEGACY_SOURCE_COMMIT = "fa9b3245acfb3d6d17f4a1bc6c17cd96e27137e3"
LEGACY_PORT12_BLOB_SHA = "82c7392046d0168350c2d391f7fe4a95823ee0e3"
LEGACY_RESEARCH_BLOB_SHA = "17f85d79db93fa59ae8d0e0ea75f8a1b8b1183ac"
LEGACY_TESTS_BLOB_SHA = "c43c3bb2c0f67654fd36e549dbcb67ce0472629b"
LEGACY_STRATEGY_FINGERPRINT = (
    "7cd1ac4e2eb950b1836c923eb218df9e1500319d4700a9a6a5caaf573366c872"
)
LEGACY_IMPLEMENTATION_FINGERPRINT = (
    "fc40b2fd547b8d55cf8cff868f8ce94f9070d88fc526c5329856d0c0cd41c2f9"
)
LEGACY_REFERENCE_RETURN_CHECKSUM_10BPS = (
    "6f535053ec673548b92f0cb3e231801f16d00d6dcf6972579436be1e99c81907"
)
LEGACY_REFERENCE_COMMON_START = "2015-10-12"


# ---------------------------------------------------------------------------
# Frozen Original configuration
# ---------------------------------------------------------------------------
MARKET_INDEX = "QQQ"
LEVERAGED_SEMICONDUCTOR = "SOXL"
LEVERAGED_TECH = "TECL"
SEMICONDUCTOR_ETF = "SMH"
LEVERAGED_INDEX = "QLD"
DEFENSIVE_EQUITY = "SPMO"
HEDGE_ASSET = "GLD"
CASH_ASSET = "CASH"
SPY = "SPY"

LEADER_CANDIDATES = (LEVERAGED_SEMICONDUCTOR, LEVERAGED_TECH)
LEVERAGED_SECTOR_ETFS = frozenset(LEADER_CANDIDATES)
LEGACY_TICKERS = (
    MARKET_INDEX,
    LEVERAGED_SEMICONDUCTOR,
    LEVERAGED_TECH,
    SEMICONDUCTOR_ETF,
    LEVERAGED_INDEX,
    DEFENSIVE_EQUITY,
    HEDGE_ASSET,
)
TRADED_TICKERS = tuple(sorted(set(LEGACY_TICKERS) - {MARKET_INDEX}))
PAIRED_TICKERS = (
    MARKET_INDEX,
    SEMICONDUCTOR_ETF,
    LEVERAGED_INDEX,
    LEVERAGED_SEMICONDUCTOR,
    SPY,
    LEVERAGED_TECH,
    DEFENSIVE_EQUITY,
    HEDGE_ASSET,
)

SMA_WINDOW = 200
DONCHIAN_WINDOW = 50
VWMA_WINDOW = 50
VOLATILITY_FAST_WINDOW = 10
VOLATILITY_SLOW_WINDOW = 30
MOMENTUM_WINDOW = 15

LOW_VOL_THRESHOLD = 0.15
MODERATE_VOL_THRESHOLD = 0.22
VOLATILITY_RERISK_PERSISTENCE = 2
REBALANCE_BAND = 0.05
REBALANCE_DESTINATION = 0.025
MAX_LEVERAGED_POSITION = 0.45
LEADER_SWITCH_THRESHOLD = 0.05
SECTOR_REBALANCE_DAYS = 21
STRATEGY_REVISION = "original-barbell-v1"
STARTING_CASH = 10_000.0

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

ADVERTISED_DAILY_MULTIPLIERS = {
    LEVERAGED_SEMICONDUCTOR: 3.0,
    LEVERAGED_TECH: 3.0,
    SEMICONDUCTOR_ETF: 1.0,
    LEVERAGED_INDEX: 2.0,
    DEFENSIVE_EQUITY: 1.0,
    HEDGE_ASSET: 1.0,
    CASH_ASSET: 0.0,
}


def _canonical_sha256(payload: object) -> str:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def strategy_manifest() -> dict[str, object]:
    """Return the exact reviewed strategy manifest from the source commit."""

    return {
        "revision": STRATEGY_REVISION,
        "universe": {
            "signal": MARKET_INDEX,
            "leader_candidates": list(LEADER_CANDIDATES),
            "traded": sorted(TRADED_TICKERS),
        },
        "windows": {
            "sma": SMA_WINDOW,
            "donchian": DONCHIAN_WINDOW,
            "vwma": VWMA_WINDOW,
            "volatility_fast": VOLATILITY_FAST_WINDOW,
            "volatility_slow": VOLATILITY_SLOW_WINDOW,
            "momentum": MOMENTUM_WINDOW,
        },
        "signal_rules": {
            "trend_votes_required": 2,
            "trend_comparison": "close_greater_or_equal",
            "volatility_tiers": {
                "low": "volatility < low_threshold",
                "moderate": (
                    "low_threshold <= volatility <= moderate_threshold"
                ),
                "high": "volatility > moderate_threshold",
            },
            "volatility_estimator": (
                "maximum_sample_standard_deviation_ddof_1_of_daily_returns"
            ),
            "volatility_annualization_sessions": 252,
            "leader_momentum_measure": "price_pct_change",
            "leader_initial_tie_break": LEVERAGED_SEMICONDUCTOR,
            "leader_switch_comparison": (
                "challenger_minus_incumbent_strictly_greater_than_threshold"
            ),
            "leader_review_policy": (
                "initially_and_after_completed_session_cadence"
            ),
            "market_data_auto_adjust": True,
        },
        "state_transition_rules": {
            "bearish_trend": "immediate_exact_target_transition",
            "more_defensive_volatility_tier": "immediate_exact_target_transition",
            "less_defensive_volatility_tier": (
                "two_distinct_completed_closes_at_same_raw_tier"
            ),
            "strategy_regime_tier_or_leader_change": (
                "exact_target_transition"
            ),
            "drift_trigger": (
                "any_absolute_component_drift_greater_or_equal_to_band"
            ),
            "drift_destination": (
                "project_all_components_inside_destination_band"
            ),
            "same_date_run": "does_not_advance_stateful_counts",
            "confirmed_shares_and_cash": "sole_current_weight_source",
        },
        "thresholds": {
            "low_volatility": LOW_VOL_THRESHOLD,
            "moderate_volatility": MODERATE_VOL_THRESHOLD,
            "volatility_rerisk_closes": VOLATILITY_RERISK_PERSISTENCE,
            "leader_switch": LEADER_SWITCH_THRESHOLD,
            "leader_review_sessions": SECTOR_REBALANCE_DAYS,
            "rebalance_band": REBALANCE_BAND,
            "rebalance_destination": REBALANCE_DESTINATION,
            "leveraged_position_cap": MAX_LEVERAGED_POSITION,
        },
        "allocations": {
            "low": dict(LOW_VOL_ALLOCATION),
            "moderate": dict(MODERATE_VOL_ALLOCATION),
            "high": dict(HIGH_VOL_ALLOCATION),
            "bear": dict(BEAR_ALLOCATION),
        },
    }


CALCULATED_LEGACY_STRATEGY_FINGERPRINT = _canonical_sha256(strategy_manifest())

COMPARATOR_IMPLEMENTATION_FINGERPRINT = _canonical_sha256(
    {
        "schema": 1,
        "strategy_source_commit": LEGACY_SOURCE_COMMIT,
        "strategy_blob": LEGACY_PORT12_BLOB_SHA,
        "research_blob": LEGACY_RESEARCH_BLOB_SHA,
        "execution": "completed close t -> adjusted open t+1",
        "portfolio_source": "fractional confirmed shares plus cash",
        "comparison_boundary": "reset starting cash at requested execution start",
    }
)


@dataclass(frozen=True)
class LegacyMarketData:
    """Validated union data used by the legacy comparator."""

    opens: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame
    sessions: pd.DatetimeIndex
    scoring_dates: pd.DatetimeIndex
    common_start: pd.Timestamp
    final_session: pd.Timestamp
    fingerprint: str
    source: str


@dataclass(frozen=True)
class VolatilityTierDecision:
    tier: str
    raw_tier: str
    pending_tier: str = ""
    pending_days: int = 0
    transition: str = ""


@dataclass(frozen=True)
class LegacyStrategyResult:
    target_weights: dict[str, float]
    regime: str
    leader: str
    volatility_tier: str
    annualized_volatility: float
    raw_volatility_tier: str


@dataclass(frozen=True)
class LegacyTargetSchedule:
    """Stateful strategic target for every completed signal session."""

    frame: pd.DataFrame
    common_start: pd.Timestamp
    data_fingerprint: str
    strategy_fingerprint: str
    fingerprint: str


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
class RebalancePlan:
    execution_weights: dict[str, float]
    rebalance_due: bool
    full_transition: bool
    reason: str
    one_way_turnover: float
    individual_orders: int


@dataclass(frozen=True)
class ExecutionMetadata:
    regime: str = "UNKNOWN"
    volatility_tier: str = "N/A"
    leader: str = ""


@dataclass(frozen=True)
class LegacyCandidate:
    name: str
    description: str
    kind: str = "legacy_original"


LEGACY_CANDIDATE = LegacyCandidate(
    name="legacy_original",
    description="Frozen Original ROTH IRA Barbell from fa9b324",
)
LEGACY_CANDIDATE_FINGERPRINT = _canonical_sha256(
    {
        "candidate": {
            "name": LEGACY_CANDIDATE.name,
            "description": LEGACY_CANDIDATE.description,
            "kind": LEGACY_CANDIDATE.kind,
        },
        "strategy_fingerprint": LEGACY_STRATEGY_FINGERPRINT,
        "source_commit": LEGACY_SOURCE_COMMIT,
    }
)


@dataclass(frozen=True)
class LegacyBacktestResult:
    """Duck-type-compatible result for paired research diagnostics."""

    candidate: LegacyCandidate
    cost_bps: float
    ledger: pd.DataFrame
    metrics: dict[str, object]
    data_fingerprint: str
    family_fingerprint: str
    candidate_fingerprint: str
    software_fingerprint: str
    schedule_fingerprint: str
    execution_start: pd.Timestamp
    unfilled_final_order: bool


@dataclass(frozen=True)
class _QueuedOrder:
    signal_date: pd.Timestamp
    execution_weights: dict[str, float]
    result: LegacyStrategyResult
    reason: str
    median_dollar_volume20: dict[str, float]


def _normalize_frame(frame: pd.DataFrame, field_name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{field_name} must be a pandas DataFrame")
    if frame.empty:
        raise RuntimeError(f"{field_name} is empty")
    result = frame.copy()
    try:
        index = pd.DatetimeIndex(pd.to_datetime(result.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} index is not datetime-like") from exc
    if index.tz is not None:
        index = index.tz_convert(None)
    index = index.normalize()
    if index.has_duplicates:
        raise RuntimeError(f"{field_name} contains duplicate sessions")
    if not index.is_monotonic_increasing:
        raise RuntimeError(f"{field_name} sessions are not increasing")
    result.index = index
    result.columns = [str(column).upper() for column in result.columns]
    if result.columns.duplicated().any():
        raise RuntimeError(f"{field_name} contains duplicate ticker columns")
    missing = sorted(set(PAIRED_TICKERS) - set(result.columns))
    if missing:
        raise RuntimeError(
            f"{field_name} is missing union tickers: {missing}"
        )
    try:
        result = result.loc[:, list(PAIRED_TICKERS)].astype(float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} contains nonnumeric observations") from exc
    result.index.name = "Session"
    result.columns.name = "Ticker"
    return result


def _calendar_sessions(
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar("XNYS")
    sessions = pd.DatetimeIndex(
        calendar.sessions_in_range(
            pd.Timestamp(start).normalize(),
            pd.Timestamp(end_inclusive).normalize(),
        )
    )
    if sessions.tz is not None:
        sessions = sessions.tz_convert(None)
    return sessions.normalize()


def _require_finite_positive(
    values: pd.DataFrame | pd.Series,
    description: str,
) -> None:
    array = values.to_numpy(dtype=float)
    valid = np.isfinite(array) & (array > 0)
    if not valid.all():
        locations = np.argwhere(~valid)
        first = locations[0].tolist() if len(locations) else []
        raise RuntimeError(
            f"{description} contains a missing, zero, or invalid observation "
            f"(first offset={first})"
        )


def _first_valid_session(series: pd.Series, description: str) -> pd.Timestamp:
    values = series.to_numpy(dtype=float)
    valid = np.isfinite(values) & (values > 0)
    if not valid.any():
        raise RuntimeError(f"{description} has no valid history")
    return pd.Timestamp(series.index[int(np.argmax(valid))]).normalize()


def _full_data_fingerprint(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
) -> str:
    digest = hashlib.sha256()
    for field_name, frame in (
        ("Open", opens),
        ("Close", closes),
        ("Volume", volumes),
    ):
        digest.update(field_name.encode("utf-8"))
        digest.update(b"\n")
        rendered = frame.to_csv(
            index=True,
            date_format="%Y-%m-%d",
            float_format="%.17g",
            lineterminator="\n",
        )
        digest.update(rendered.encode("utf-8"))
    return digest.hexdigest()


def validate_union_market_data(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    source: str = "provided adjusted union Open/Close/Volume",
) -> LegacyMarketData:
    """Validate the paired union without fill, deletion, or substitution.

    Pre-inception missing values are retained.  Scoring begins only when every
    union security has a real adjusted Open and Close and QQQ has real Volume.
    The 200-session signal warm-up must be actual QQQ/SOXL/TECL history.
    """

    normalized_opens = _normalize_frame(opens, "Open")
    normalized_closes = _normalize_frame(closes, "Close")
    normalized_volumes = _normalize_frame(volumes, "Volume")
    if not normalized_opens.index.equals(normalized_closes.index):
        raise RuntimeError("Open and Close indexes differ")
    if not normalized_opens.index.equals(normalized_volumes.index):
        raise RuntimeError("Open and Volume indexes differ")

    sessions = normalized_opens.index
    expected = _calendar_sessions(sessions[0], sessions[-1])
    missing_sessions = expected.difference(sessions)
    unexpected_sessions = sessions.difference(expected)
    if len(missing_sessions) or len(unexpected_sessions):
        missing = [item.date().isoformat() for item in missing_sessions[:5]]
        unexpected = [
            item.date().isoformat() for item in unexpected_sessions[:5]
        ]
        raise RuntimeError(
            "Union market-data session continuity failed: "
            f"missing={missing}, non_sessions={unexpected}"
        )

    first_valid_dates: list[pd.Timestamp] = []
    for field_name, frame in (
        ("Open", normalized_opens),
        ("Close", normalized_closes),
    ):
        for ticker in PAIRED_TICKERS:
            first_valid_dates.append(
                _first_valid_session(
                    frame[ticker],
                    f"{ticker} adjusted {field_name}",
                )
            )
    first_valid_dates.append(
        _first_valid_session(
            normalized_volumes[MARKET_INDEX],
            f"{MARKET_INDEX} adjusted Volume",
        )
    )
    common_start = max(first_valid_dates).normalize()
    location = sessions.get_loc(common_start)
    if not isinstance(location, (int, np.integer)):
        raise RuntimeError("Common inception is not a unique XNYS session")
    common_position = int(location)
    required_rows = max(
        SMA_WINDOW,
        DONCHIAN_WINDOW,
        VWMA_WINDOW,
        VOLATILITY_SLOW_WINDOW + 1,
        MOMENTUM_WINDOW + 1,
    )
    if common_position < required_rows - 1:
        raise RuntimeError(
            "Insufficient actual QQQ/leader warm-up before common ETF inception"
        )

    scored_slice = slice(common_position, None)
    _require_finite_positive(
        normalized_opens.iloc[scored_slice],
        "Scored adjusted Opens",
    )
    _require_finite_positive(
        normalized_closes.iloc[scored_slice],
        "Scored adjusted Closes",
    )
    _require_finite_positive(
        normalized_volumes.loc[:, [MARKET_INDEX]].iloc[scored_slice],
        "Scored QQQ Volume",
    )
    warmup_slice = slice(common_position - required_rows + 1, None)
    _require_finite_positive(
        normalized_closes.loc[
            :, [MARKET_INDEX, *LEADER_CANDIDATES]
        ].iloc[warmup_slice],
        "Original indicator warm-up Closes",
    )
    _require_finite_positive(
        normalized_volumes.loc[:, [MARKET_INDEX]].iloc[warmup_slice],
        "Original indicator warm-up QQQ Volume",
    )

    scoring_dates = sessions[common_position:]
    if len(scoring_dates) < 2:
        raise RuntimeError("At least two common sessions are required")
    return LegacyMarketData(
        opens=normalized_opens,
        closes=normalized_closes,
        volumes=normalized_volumes,
        sessions=sessions,
        scoring_dates=scoring_dates,
        common_start=common_start,
        final_session=pd.Timestamp(sessions[-1]),
        fingerprint=_full_data_fingerprint(
            normalized_opens,
            normalized_closes,
            normalized_volumes,
        ),
        source=str(source),
    )


def calculate_indicators(
    close_data: pd.DataFrame,
    volume_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the Original indicators with their historical boundaries."""

    indicators = pd.DataFrame(index=close_data.index)
    index_close = close_data[MARKET_INDEX]
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
    indicators["vwma_50"] = price_times_volume.rolling(
        VWMA_WINDOW,
        min_periods=VWMA_WINDOW,
    ).sum() / volume_sum

    indicators["sma_signal"] = (
        index_close >= indicators["sma_200"]
    ).astype(int)
    indicators["donchian_signal"] = (
        index_close >= indicators["donchian_mid"]
    ).astype(int)
    indicators["vwma_signal"] = (
        index_close >= indicators["vwma_50"]
    ).astype(int)
    indicators["bullish_consensus"] = (
        indicators[
            ["sma_signal", "donchian_signal", "vwma_signal"]
        ].sum(axis=1)
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
    indicators["soxl_momentum"] = close_data[
        LEVERAGED_SEMICONDUCTOR
    ].pct_change(MOMENTUM_WINDOW, fill_method=None)
    indicators["tecl_momentum"] = close_data[
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
    template: Mapping[str, float],
    leader: str,
) -> dict[str, float]:
    follower = (
        LEVERAGED_TECH
        if leader == LEVERAGED_SEMICONDUCTOR
        else LEVERAGED_SEMICONDUCTOR
    )
    target: dict[str, float] = {}
    for slot, original_weight in template.items():
        ticker = (
            leader
            if slot == "Leader"
            else follower if slot == "Follower" else slot
        )
        weight = float(original_weight)
        if ticker in LEVERAGED_SECTOR_ETFS and weight > MAX_LEVERAGED_POSITION:
            excess = weight - MAX_LEVERAGED_POSITION
            weight = MAX_LEVERAGED_POSITION
            target[SEMICONDUCTOR_ETF] = (
                target.get(SEMICONDUCTOR_ETF, 0.0) + excess
            )
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
) -> LegacyStrategyResult:
    annualized_volatility = float(latest["annualized_volatility"])
    leader = existing_leader or LEVERAGED_SEMICONDUCTOR
    if allow_leader_review:
        leader = select_leader(
            float(latest["soxl_momentum"]),
            float(latest["tecl_momentum"]),
            existing_leader,
        )
    if int(latest["bullish_consensus"]) != 1:
        return LegacyStrategyResult(
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
    return LegacyStrategyResult(
        target_weights=apply_allocation_template(template, leader),
        regime="BULL",
        leader=leader,
        volatility_tier=tier_decision.tier,
        annualized_volatility=annualized_volatility,
        raw_volatility_tier=tier_decision.raw_tier,
    )


def _schedule_fingerprint(
    data_fingerprint: str,
    frame: pd.DataFrame,
) -> str:
    rows: list[dict[str, object]] = []
    for date, row in frame.iterrows():
        rows.append(
            {
                "session": pd.Timestamp(date).date().isoformat(),
                "regime": str(row["regime"]),
                "leader": str(row["leader"]),
                "volatility_tier": str(row["volatility_tier"]),
                "raw_volatility_tier": str(row["raw_volatility_tier"]),
                "annualized_volatility": float(row["annualized_volatility"]),
                "bullish_consensus": int(row["bullish_consensus"]),
                "sector_review_due": bool(row["sector_review_due"]),
                "tier_transition": str(row["tier_transition"]),
                "pending_volatility_tier": str(
                    row["pending_volatility_tier"]
                ),
                "pending_volatility_days": int(
                    row["pending_volatility_days"]
                ),
                "target_weights": dict(row["target_weights"]),
            }
        )
    return _canonical_sha256(
        {
            "data_fingerprint": data_fingerprint,
            "strategy_fingerprint": LEGACY_STRATEGY_FINGERPRINT,
            "rows": rows,
        }
    )


def build_original_target_schedule(
    data: LegacyMarketData,
) -> LegacyTargetSchedule:
    """Build the complete stateful Original target schedule once, causally."""

    if data.fingerprint != _full_data_fingerprint(
        data.opens,
        data.closes,
        data.volumes,
    ):
        raise ValueError("LegacyMarketData fingerprint does not match its frames")
    indicators = calculate_indicators(data.closes, data.volumes)
    current_tier = "N/A"
    pending_tier = ""
    pending_days = 0
    leader: str | None = None
    last_sector_review: pd.Timestamp | None = None
    rows: list[dict[str, object]] = []

    for signal_date in data.scoring_dates:
        signal_date = pd.Timestamp(signal_date)
        latest = indicators.loc[signal_date]
        validate_latest_indicators(latest)
        bullish = int(latest["bullish_consensus"]) == 1
        if not bullish:
            tier_decision = VolatilityTierDecision(
                tier="N/A",
                raw_tier="N/A",
                transition="BEAR",
            )
        else:
            tier_decision = classify_volatility_with_persistence(
                float(latest["annualized_volatility"]),
                current_tier,
                pending_tier,
                pending_days,
            )

        sector_due = (
            last_sector_review is None
            or int(
                (
                    data.sessions[
                        (data.sessions > last_sector_review)
                        & (data.sessions <= signal_date)
                    ]
                ).size
            )
            >= SECTOR_REBALANCE_DAYS
        )
        result = determine_target_allocation(
            latest,
            leader,
            sector_due,
            tier_decision,
        )

        current_tier = result.volatility_tier
        pending_tier = tier_decision.pending_tier
        pending_days = tier_decision.pending_days
        leader = result.leader
        if sector_due:
            last_sector_review = signal_date

        rows.append(
            {
                "session": signal_date,
                "regime": result.regime,
                "leader": result.leader,
                "volatility_tier": result.volatility_tier,
                "raw_volatility_tier": result.raw_volatility_tier,
                "annualized_volatility": result.annualized_volatility,
                "bullish_consensus": int(latest["bullish_consensus"]),
                "sector_review_due": sector_due,
                "tier_transition": tier_decision.transition,
                "pending_volatility_tier": pending_tier,
                "pending_volatility_days": pending_days,
                "target_weights": dict(result.target_weights),
                "target_json": json.dumps(
                    result.target_weights,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )

    frame = pd.DataFrame(rows).set_index("session")
    fingerprint = _schedule_fingerprint(data.fingerprint, frame)
    return LegacyTargetSchedule(
        frame=frame,
        common_start=data.common_start,
        data_fingerprint=data.fingerprint,
        strategy_fingerprint=LEGACY_STRATEGY_FINGERPRINT,
        fingerprint=fingerprint,
    )


def should_rebalance(
    existing: Mapping[str, float],
    target: Mapping[str, float],
    band: float = REBALANCE_BAND,
) -> bool:
    if not (0 < band < 1):
        raise ValueError("Rebalance band must be between zero and one")
    if not existing:
        return True
    tickers = set(existing) | set(target)
    return any(
        abs(target.get(ticker, 0.0) - existing.get(ticker, 0.0))
        >= band - 1e-12
        for ticker in tickers
    )


def inner_band_rebalance_weights(
    existing: Mapping[str, float],
    target: Mapping[str, float],
    destination: float = REBALANCE_DESTINATION,
    *,
    trigger_band: float = REBALANCE_BAND,
) -> dict[str, float]:
    if not existing:
        return dict(target)
    if not (0 < destination < trigger_band < 1):
        raise ValueError("Destination must be positive and inside the trigger band")
    tickers = sorted(set(existing) | set(target))
    current = np.array(
        [existing.get(ticker, 0.0) for ticker in tickers],
        dtype=float,
    )
    desired = np.array(
        [target.get(ticker, 0.0) for ticker in tickers],
        dtype=float,
    )
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


def _with_cash_target(weights: Mapping[str, float]) -> dict[str, float]:
    result = dict(weights)
    result.setdefault(CASH_ASSET, 0.0)
    return result


def build_rebalance_plan(
    existing: Mapping[str, float],
    result: LegacyStrategyResult,
    executed: ExecutionMetadata,
    *,
    rebalance_band: float = REBALANCE_BAND,
    rebalance_destination: float = REBALANCE_DESTINATION,
) -> RebalancePlan:
    """Apply the Original exact-transition and buffered-drift rules."""

    if not (0 < rebalance_destination < rebalance_band < 1):
        raise ValueError(
            "Rebalance destination must be positive and inside the trigger band"
        )
    target = _with_cash_target(result.target_weights)
    initial_allocation = executed.regime == "UNKNOWN" or not existing
    regime_changed = (
        not initial_allocation and executed.regime != result.regime
    )
    tier_changed = (
        not initial_allocation
        and result.regime == "BULL"
        and (
            executed.regime != "BULL"
            or executed.volatility_tier != result.volatility_tier
        )
    )
    leader_changed = (
        not initial_allocation
        and result.regime == "BULL"
        and executed.leader != result.leader
    )
    full_transition = (
        initial_allocation or regime_changed or tier_changed or leader_changed
    )
    drift_exceeded = should_rebalance(
        existing,
        target,
        band=rebalance_band,
    )
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
        else inner_band_rebalance_weights(
            existing,
            target,
            destination=rebalance_destination,
            trigger_band=rebalance_band,
        )
    )
    if rebalance_due:
        components = set(existing) | set(execution)
        one_way_turnover = 0.5 * sum(
            abs(
                execution.get(component, 0.0)
                - existing.get(component, 0.0)
            )
            for component in components
        )
        individual_orders = sum(
            component != CASH_ASSET
            and abs(
                execution.get(component, 0.0)
                - existing.get(component, 0.0)
            )
            > 1e-9
            for component in components
        )
    else:
        one_way_turnover = 0.0
        individual_orders = 0
    if rebalance_due and individual_orders == 0 and one_way_turnover <= 1e-12:
        return RebalancePlan(
            execution_weights=execution,
            rebalance_due=False,
            full_transition=False,
            reason="CONFIRMED_TARGET_STATE",
            one_way_turnover=0.0,
            individual_orders=0,
        )
    return RebalancePlan(
        execution_weights=execution,
        rebalance_due=rebalance_due,
        full_transition=full_transition,
        reason=reason,
        one_way_turnover=float(one_way_turnover),
        individual_orders=int(individual_orders),
    )


def solve_post_cost_target(
    *,
    current_values: Mapping[str, float],
    cash: float,
    target_weights: Mapping[str, float],
    cost_rate: float,
) -> tuple[float, float]:
    """Solve exact post-cost NAV and security gross notional by bisection."""

    if not np.isfinite(cost_rate) or not (0 <= cost_rate < 1):
        raise ValueError("Transaction-cost rate must be finite and in [0, 1)")
    target = dict(target_weights)
    if any(not np.isfinite(value) or value < 0 for value in target.values()):
        raise ValueError("Target weights must be finite and nonnegative")
    if not np.isclose(sum(target.values()), 1.0, atol=1e-12):
        raise ValueError("Target weights must sum to one")
    if any(
        not np.isfinite(value) or value < 0
        for value in current_values.values()
    ):
        raise ValueError("Current security values must be finite and nonnegative")
    if not np.isfinite(cash) or cash < 0:
        raise ValueError("Cash must be finite and nonnegative")
    securities = (set(current_values) | set(target)) - {CASH_ASSET}
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
    """Execute fractional shares to exact post-cost adjusted-Open weights."""

    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Transaction costs must be finite and nonnegative")
    target = _with_cash_target(target_weights)
    if not np.isclose(sum(target.values()), 1.0, atol=1e-12):
        raise ValueError("Execution target must sum to one")
    tickers = (set(shares) | set(target)) - {CASH_ASSET}
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

    next_cash = target.get(CASH_ASSET, 0.0) * posttrade_nav
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
        ticker: value / pretrade_nav
        for ticker, value in current_values.items()
    }
    pre_weights[CASH_ASSET] = cash / pretrade_nav
    components = set(pre_weights) | set(target)
    one_way = 0.5 * sum(
        abs(
            target.get(component, 0.0)
            - pre_weights.get(component, 0.0)
        )
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
        security_orders=sum(
            abs(value) > pretrade_nav * 1e-12
            for value in trade_notional.values()
        ),
        trade_notional=trade_notional,
    )


def _strategy_from_schedule_row(row: pd.Series) -> LegacyStrategyResult:
    return LegacyStrategyResult(
        target_weights=dict(row["target_weights"]),
        regime=str(row["regime"]),
        leader=str(row["leader"]),
        volatility_tier=str(row["volatility_tier"]),
        annualized_volatility=float(row["annualized_volatility"]),
        raw_volatility_tier=str(row["raw_volatility_tier"]),
    )


def _mark_close(
    shares: Mapping[str, float],
    cash: float,
    close_prices: pd.Series,
) -> float:
    total = float(cash)
    for ticker, units in shares.items():
        price = float(close_prices[ticker])
        if not np.isfinite(price) or price <= 0:
            raise RuntimeError(f"Invalid adjusted Close for held ticker {ticker}")
        total += float(units) * price
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Marked-to-market research NAV is invalid")
    return total


def _actual_weights(
    shares: Mapping[str, float],
    cash: float,
    close_prices: pd.Series,
) -> tuple[dict[str, float], float]:
    nav = _mark_close(shares, cash, close_prices)
    weights = {
        ticker: float(units) * float(close_prices[ticker]) / nav
        for ticker, units in shares.items()
        if float(units) > 0
    }
    if cash > 0:
        weights[CASH_ASSET] = float(cash) / nav
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
        raise RuntimeError("Actual portfolio weights do not sum to one")
    return weights, nav


def _median_dollar_volume20(
    data: LegacyMarketData,
    signal_date: pd.Timestamp,
    tickers: Mapping[str, float] | set[str],
) -> dict[str, float]:
    position = data.sessions.get_loc(signal_date)
    if not isinstance(position, (int, np.integer)):
        raise RuntimeError("Signal date is not a unique data session")
    start = max(0, int(position) - 19)
    result: dict[str, float] = {}
    for ticker in tickers:
        if ticker == CASH_ASSET:
            continue
        values = (
            data.closes[ticker].iloc[start : int(position) + 1]
            * data.volumes[ticker].iloc[start : int(position) + 1]
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if len(values):
            result[ticker] = float(np.median(values))
    return result


def _advertised_exposure(weights: Mapping[str, float]) -> float:
    return float(
        sum(
            weights.get(ticker, 0.0) * multiplier
            for ticker, multiplier in ADVERTISED_DAILY_MULTIPLIERS.items()
        )
    )


def _series_checksum(series: pd.Series) -> str:
    rendered = series.to_csv(
        index=True,
        date_format="%Y-%m-%d",
        float_format="%.15g",
        lineterminator="\n",
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _log_returns_from_ledger(ledger: pd.DataFrame) -> pd.Series:
    nav = ledger["nav"].astype(float)
    starting_capital = float(ledger["starting_capital"].iloc[0])
    result = pd.concat(
        [
            pd.Series(
                [math.log(float(nav.iloc[0] / starting_capital))],
                index=pd.DatetimeIndex([nav.index[0]]),
            ),
            np.log1p(nav.pct_change(fill_method=None).iloc[1:]),
        ]
    )
    result.name = "legacy_original_log_return"
    return result


def _drawdown_diagnostics(
    nav: pd.Series,
    starting_capital: float,
) -> tuple[float, int]:
    initial = pd.Series(
        [float(starting_capital)],
        index=pd.DatetimeIndex(
            [pd.Timestamp(nav.index[0]) - pd.Timedelta(nanoseconds=1)]
        ),
    )
    complete = pd.concat([initial, nav.astype(float)])
    drawdown = complete / complete.cummax() - 1.0
    longest = 0
    current = 0
    for value in drawdown.to_numpy(dtype=float):
        if value < -1e-12:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(drawdown.min()), longest


def _worst_rolling(nav: pd.Series, sessions: int) -> float | None:
    if len(nav) <= sessions:
        return None
    values = nav / nav.shift(sessions) - 1.0
    return float(values.dropna().min())


def calculate_metrics(ledger: pd.DataFrame) -> dict[str, object]:
    """Calculate comparison metrics including the initial deployment return."""

    if len(ledger) < 2:
        raise RuntimeError("At least two execution sessions are required")
    logs = _log_returns_from_ledger(ledger)
    returns = np.expm1(logs)
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise RuntimeError("Metric return series is nonfinite")
    if (returns <= -1).any():
        raise RuntimeError("Metric return series is terminal")
    nav = ledger["nav"].astype(float)
    starting_capital = float(ledger["starting_capital"].iloc[0])
    years = len(returns) / 252.0
    terminal_wealth = float(nav.iloc[-1] / starting_capital)
    cagr = terminal_wealth ** (1.0 / years) - 1.0
    standard_deviation = float(returns.std(ddof=1))
    annualized_volatility = standard_deviation * math.sqrt(252)
    mean_return = float(returns.mean())
    sharpe = (
        mean_return / standard_deviation * math.sqrt(252)
        if standard_deviation > 0
        else 0.0
    )
    downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)))
    sortino = (
        mean_return / downside_deviation * math.sqrt(252)
        if downside_deviation > 0
        else 0.0
    )
    max_drawdown, underwater = _drawdown_diagnostics(nav, starting_capital)
    tail_count = max(1, int(math.ceil(0.05 * len(returns))))
    expected_shortfall = float(
        np.sort(returns.to_numpy(dtype=float))[:tail_count].mean()
    )
    annualizer = 1.0 / years
    return {
        "observations": int(len(returns)),
        "years_252": years,
        "start": ledger.index[0].date().isoformat(),
        "end": ledger.index[-1].date().isoformat(),
        "starting_nav": starting_capital,
        "ending_nav": float(nav.iloc[-1]),
        "terminal_wealth_multiple": terminal_wealth,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe_zero_rf": sharpe,
        "sortino_zero_mar": sortino,
        "calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0,
        "maximum_drawdown": max_drawdown,
        "maximum_time_underwater_sessions": underwater,
        "expected_shortfall_95_daily": expected_shortfall,
        "worst_rolling_1y": _worst_rolling(nav, 252),
        "worst_rolling_3y": _worst_rolling(nav, 756),
        "worst_rolling_5y": _worst_rolling(nav, 1260),
        "annual_gross_turnover": float(
            ledger["ongoing_gross_trade_fraction"].sum() * annualizer
        ),
        "annual_one_way_turnover": float(
            ledger["one_way_turnover"].sum() * annualizer
        ),
        "ongoing_security_orders_annual": float(
            ledger.loc[
                ~ledger["initial_deployment"], "security_orders"
            ].sum()
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
                ledger[
                    "max_trade_to_median_dollar_volume20"
                ].to_numpy(dtype=float)
            ).sum()
        ),
        "mean_advertised_daily_exposure": float(
            ledger["advertised_daily_exposure"].mean()
        ),
        "return_checksum": _series_checksum(returns),
    }


def simulate_original(
    data: LegacyMarketData,
    schedule: LegacyTargetSchedule,
    *,
    cost_bps: float = 10.0,
    execution_start: pd.Timestamp | str | None = None,
    starting_cash: float = STARTING_CASH,
) -> LegacyBacktestResult:
    """Simulate the frozen Original from a common reset-cash boundary."""

    if schedule.data_fingerprint != data.fingerprint:
        raise ValueError("Target schedule and union data fingerprints differ")
    if schedule.strategy_fingerprint != LEGACY_STRATEGY_FINGERPRINT:
        raise ValueError("Target schedule is not the frozen Original strategy")
    if schedule.fingerprint != _schedule_fingerprint(
        data.fingerprint,
        schedule.frame,
    ):
        raise ValueError("Target schedule fingerprint does not match its frame")
    if not np.isfinite(starting_cash) or starting_cash <= 0:
        raise ValueError("Starting cash must be positive and finite")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Transaction costs must be finite and nonnegative")

    start = (
        pd.Timestamp(data.scoring_dates[1])
        if execution_start is None
        else pd.Timestamp(execution_start).normalize()
    )
    if start not in data.sessions:
        raise ValueError("Execution start is not an actual XNYS data session")
    start_location = data.sessions.get_loc(start)
    if not isinstance(start_location, (int, np.integer)):
        raise ValueError("Execution start is not a unique session")
    start_position = int(start_location)
    if start_position < 1:
        raise ValueError("Execution start needs a preceding signal session")
    prior_signal_date = pd.Timestamp(data.sessions[start_position - 1])
    if prior_signal_date not in schedule.frame.index:
        raise ValueError(
            "Execution start must follow a covered Original signal session"
        )
    scoring_dates = data.sessions[start_position:]
    if len(scoring_dates) < 2:
        raise ValueError("At least two execution sessions are required")

    prior_result = _strategy_from_schedule_row(
        schedule.frame.loc[prior_signal_date]
    )
    pending: _QueuedOrder | None = _QueuedOrder(
        signal_date=prior_signal_date,
        execution_weights=_with_cash_target(prior_result.target_weights),
        result=prior_result,
        reason="INITIAL_DEPLOYMENT",
        median_dollar_volume20=_median_dollar_volume20(
            data,
            prior_signal_date,
            set(prior_result.target_weights),
        ),
    )
    shares: dict[str, float] = {}
    cash = float(starting_cash)
    executed = ExecutionMetadata()
    has_executed = False
    rows: list[dict[str, object]] = []

    for signal_date in scoring_dates:
        signal_date = pd.Timestamp(signal_date)
        fill: FillResult | None = None
        fill_reason = ""
        fill_signal_date: pd.Timestamp | None = None
        initial_deployment = False
        max_trade_to_adv = 0.0
        if pending is not None:
            fill = execute_target_at_open(
                shares=shares,
                cash=cash,
                open_prices=data.opens.loc[signal_date],
                target_weights=pending.execution_weights,
                cost_bps=cost_bps,
            )
            shares = dict(fill.shares)
            cash = float(fill.cash)
            executed = ExecutionMetadata(
                regime=pending.result.regime,
                volatility_tier=pending.result.volatility_tier,
                leader=pending.result.leader,
            )
            fill_reason = pending.reason
            fill_signal_date = pending.signal_date
            initial_deployment = not has_executed
            has_executed = True
            ratios: list[float] = []
            missing_trade_adv = False
            for ticker, notional in fill.trade_notional.items():
                if abs(notional) <= fill.pretrade_nav * 1e-12:
                    continue
                adv = pending.median_dollar_volume20.get(ticker)
                if adv is None or not np.isfinite(adv) or adv <= 0:
                    missing_trade_adv = True
                else:
                    ratios.append(abs(notional) / adv)
            max_trade_to_adv = (
                math.inf
                if missing_trade_adv
                else max(ratios, default=0.0)
            )
            pending = None

        close_prices = data.closes.loc[signal_date]
        actual_weights, nav = _actual_weights(shares, cash, close_prices)
        schedule_row = schedule.frame.loc[signal_date]
        result = _strategy_from_schedule_row(schedule_row)
        plan = build_rebalance_plan(actual_weights, result, executed)
        if plan.rebalance_due:
            pending = _QueuedOrder(
                signal_date=signal_date,
                execution_weights=dict(plan.execution_weights),
                result=result,
                reason=plan.reason,
                median_dollar_volume20=_median_dollar_volume20(
                    data,
                    signal_date,
                    set(shares) | set(plan.execution_weights),
                ),
            )
        elif plan.reason == "CONFIRMED_TARGET_STATE":
            executed = ExecutionMetadata(
                regime=result.regime,
                volatility_tier=result.volatility_tier,
                leader=result.leader,
            )

        gross_fraction = fill.gross_fraction if fill is not None else 0.0
        one_way = fill.one_way_fraction if fill is not None else 0.0
        cost = fill.cost if fill is not None else 0.0
        cost_fraction = fill.cost_fraction if fill is not None else 0.0
        orders = fill.security_orders if fill is not None else 0
        rows.append(
            {
                "session": signal_date,
                "nav": nav,
                "starting_capital": float(starting_cash),
                "cash": cash,
                "regime": result.regime,
                "volatility_tier": result.volatility_tier,
                "raw_volatility_tier": result.raw_volatility_tier,
                "leader": result.leader,
                "bullish_consensus": int(
                    schedule_row["bullish_consensus"]
                ),
                "annualized_volatility": result.annualized_volatility,
                "sector_review_due": bool(
                    schedule_row["sector_review_due"]
                ),
                "tier_transition": str(schedule_row["tier_transition"]),
                "signal_reason": plan.reason,
                "signal_rebalance_due": bool(plan.rebalance_due),
                "queued_reason": plan.reason if plan.rebalance_due else "",
                "filled_reason": fill_reason,
                "fill_signal_date": fill_signal_date,
                "initial_deployment": initial_deployment,
                "gross_trade_fraction": gross_fraction,
                "ongoing_gross_trade_fraction": (
                    0.0 if initial_deployment else gross_fraction
                ),
                "one_way_turnover": (
                    0.0 if initial_deployment else one_way
                ),
                "transaction_cost": cost,
                "transaction_cost_fraction": cost_fraction,
                "security_orders": orders,
                "max_trade_to_median_dollar_volume20": max_trade_to_adv,
                "advertised_daily_exposure": _advertised_exposure(
                    actual_weights
                ),
                "weights": json.dumps(
                    actual_weights,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "shares": json.dumps(
                    shares,
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
                **{
                    f"{ticker.lower()}_shares": float(shares.get(ticker, 0.0))
                    for ticker in TRADED_TICKERS
                },
                **{
                    f"{ticker.lower()}_weight": float(
                        actual_weights.get(ticker, 0.0)
                    )
                    for ticker in TRADED_TICKERS
                },
                "cash_weight": float(actual_weights.get(CASH_ASSET, 0.0)),
            }
        )

    ledger = pd.DataFrame(rows).set_index("session")
    metrics = calculate_metrics(ledger)
    return LegacyBacktestResult(
        candidate=LEGACY_CANDIDATE,
        cost_bps=float(cost_bps),
        ledger=ledger,
        metrics=metrics,
        data_fingerprint=data.fingerprint,
        family_fingerprint=LEGACY_STRATEGY_FINGERPRINT,
        candidate_fingerprint=LEGACY_CANDIDATE_FINGERPRINT,
        software_fingerprint=COMPARATOR_IMPLEMENTATION_FINGERPRINT,
        schedule_fingerprint=schedule.fingerprint,
        execution_start=start,
        unfilled_final_order=pending is not None,
    )


def result_log_returns(result: LegacyBacktestResult) -> pd.Series:
    """Return the causal log-return stream expected by paired diagnostics."""

    return _log_returns_from_ledger(result.ledger)


def adapter_record(result: LegacyBacktestResult) -> dict[str, object]:
    """Return a compact, serialization-friendly integration record."""

    return {
        "name": result.candidate.name,
        "cost_bps": result.cost_bps,
        "ledger": result.ledger,
        "metrics": result.metrics,
        "log_returns": result_log_returns(result),
        "data_fingerprint": result.data_fingerprint,
        "family_fingerprint": result.family_fingerprint,
        "candidate_fingerprint": result.candidate_fingerprint,
        "software_fingerprint": result.software_fingerprint,
        "schedule_fingerprint": result.schedule_fingerprint,
        "execution_start": result.execution_start,
        "unfilled_final_order": result.unfilled_final_order,
    }


__all__ = [
    "BEAR_ALLOCATION",
    "CALCULATED_LEGACY_STRATEGY_FINGERPRINT",
    "CASH_ASSET",
    "COMPARATOR_IMPLEMENTATION_FINGERPRINT",
    "ExecutionMetadata",
    "HIGH_VOL_ALLOCATION",
    "LEGACY_CANDIDATE",
    "LEGACY_CANDIDATE_FINGERPRINT",
    "LEGACY_IMPLEMENTATION_FINGERPRINT",
    "LEGACY_PORT12_BLOB_SHA",
    "LEGACY_REFERENCE_COMMON_START",
    "LEGACY_REFERENCE_RETURN_CHECKSUM_10BPS",
    "LEGACY_RESEARCH_BLOB_SHA",
    "LEGACY_SOURCE_COMMIT",
    "LEGACY_STRATEGY_FINGERPRINT",
    "LEGACY_TESTS_BLOB_SHA",
    "LEGACY_TICKERS",
    "LOW_VOL_ALLOCATION",
    "LegacyBacktestResult",
    "LegacyMarketData",
    "LegacyStrategyResult",
    "LegacyTargetSchedule",
    "MODERATE_VOL_ALLOCATION",
    "PAIRED_TICKERS",
    "RebalancePlan",
    "VolatilityTierDecision",
    "adapter_record",
    "apply_allocation_template",
    "build_original_target_schedule",
    "build_rebalance_plan",
    "calculate_indicators",
    "calculate_metrics",
    "classify_volatility",
    "classify_volatility_with_persistence",
    "determine_target_allocation",
    "execute_target_at_open",
    "inner_band_rebalance_weights",
    "result_log_returns",
    "select_leader",
    "should_rebalance",
    "simulate_original",
    "solve_post_cost_target",
    "strategy_manifest",
    "validate_latest_indicators",
    "validate_union_market_data",
]

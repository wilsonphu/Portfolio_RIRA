#!/usr/bin/env python3
"""Frozen, research-only alpha harness for the QLD/SOXL redesign.

This module implements :mod:`ALPHA_RESEARCH_PROTOCOL.md`.  It is intentionally
isolated from the production engine: it never reads or writes broker state,
never sends notifications, and never substitutes one security for another.
Signals use a completed close and all ordinary orders fill at the next
validated session's adjusted open.

The sample used to design this family has already been inspected.  Results
from this harness are therefore contamination-aware research, not a claim of
untouched out-of-sample alpha.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd

import alpha_core as core


ALPHA_RESEARCH_SCHEMA_VERSION = 1
TICKERS = ("QQQ", "SMH", "QLD", "SOXL", "SPY")
TRADED_TICKERS = ("QLD", "SOXL")
CASH = core.CASH
NEW_YORK = ZoneInfo("America/New_York")
DEFAULT_START = "2010-01-01"
MODEL_HISTORY_START = core.MODEL_HISTORY_START
MARKET_CLOSE_BUFFER_MINUTES = 15
DEFAULT_COSTS_BPS = (0.0, 10.0, 25.0, 50.0)
PRIMARY_COST_BPS = 10.0
STARTING_CASH = 10_000.0
MASTER_SEED = 20_260_806

ESTIMATION_RETURNS = core.BETA_WINDOW
SCORING_RETURNS = core.RESIDUAL_SCORE_WINDOW
TREND_SESSIONS = core.SMA_WINDOW
REVIEW_SESSIONS = core.ALPHA_REVIEW_SESSIONS
REENTRY_CLOSES = core.BULLISH_REENTRY_CLOSES
VOL_UPSHIFT_CLOSES = core.VOLATILITY_UPSHIFT_CLOSES
DRIFT_TRIGGER = 0.05
DRIFT_DESTINATION = 0.025

HAR_WINDOWS = (
    core.VARIANCE_FAST_WINDOW,
    core.VARIANCE_MONTH_WINDOW,
    core.VARIANCE_QUARTER_WINDOW,
)
HAR_HORIZON = core.VARIANCE_HORIZON
HAR_ALPHA = core.RIDGE_ALPHA
RIDGE_ALPHA = core.RIDGE_ALPHA
MIN_LABELED_OBSERVATIONS = core.VARIANCE_MIN_TRAINING
VOLATILITY_BUDGET = core.VOLATILITY_BUDGET
SOXL_WEIGHT_GRID = core.SOXL_WEIGHT_GRID

HAR_FEATURES = (
    "log_var_5",
    "log_var_21",
    "log_var_63",
    "log_downside_var_21",
)
RIDGE_FEATURES = (
    "qqq_log_trend_gap",
    "residual_z",
    "qld_forecast_to_vol63_log",
    "qqq_drawdown252",
    "qqq_downside_ratio21",
)


@dataclass(frozen=True)
class Candidate:
    """A predeclared path in the frozen candidate/ablation family."""

    name: str
    description: str
    kind: str = "dynamic"
    gate: str = "trend_residual"
    volatility_sizing: bool = False
    fixed_soxl_weight: float = 0.35
    ridge_confirmation: bool = False
    production_eligible: bool = True


CANDIDATES: dict[str, Candidate] = {
    "qld_buy_hold": Candidate(
        "qld_buy_hold",
        "100% QLD buy and hold",
        kind="qld_buy_hold",
        gate="none",
        fixed_soxl_weight=0.0,
    ),
    "static_65_35": Candidate(
        "static_65_35",
        "Initial 65% QLD / 35% SOXL, then hold",
        kind="static",
        gate="none",
    ),
    "residual_35": Candidate(
        "residual_35",
        "QQQ trend plus separated-window residual momentum; fixed 35% SOXL",
    ),
    "residual_vol55": Candidate(
        "residual_vol55",
        "Transparent residual candidate with a 55% volatility budget",
        volatility_sizing=True,
    ),
    "trend_vol55": Candidate(
        "trend_vol55",
        "Trend-only volatility-sized ablation",
        gate="trend",
        volatility_sizing=True,
    ),
    "residual_vol55_no_trend": Candidate(
        "residual_vol55_no_trend",
        "Residual-only volatility-sized ablation",
        gate="residual",
        volatility_sizing=True,
    ),
    "ridge_shadow": Candidate(
        "ridge_shadow",
        "Ridge confirmation layered below residual_vol55",
        volatility_sizing=True,
        ridge_confirmation=True,
        production_eligible=False,
    ),
}

FROZEN_CANDIDATE_NAMES = tuple(CANDIDATES)


def _canonical_sha256(payload: object) -> str:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


FROZEN_CONFIGURATION = {
    "schema_version": ALPHA_RESEARCH_SCHEMA_VERSION,
    "decision_semantic_revision": core.DECISION_SEMANTIC_REVISION,
    "tickers": TICKERS,
    "model_history_start": MODEL_HISTORY_START,
    "market_close_buffer_minutes": MARKET_CLOSE_BUFFER_MINUTES,
    "execution": "completed close t -> adjusted open t+1",
    "estimation_returns": ESTIMATION_RETURNS,
    "scoring_returns": SCORING_RETURNS,
    "trend_sessions": TREND_SESSIONS,
    "review_sessions": REVIEW_SESSIONS,
    "reentry_closes": REENTRY_CLOSES,
    "vol_upshift_closes": VOL_UPSHIFT_CLOSES,
    "drift_trigger": DRIFT_TRIGGER,
    "drift_destination": DRIFT_DESTINATION,
    "har_windows": HAR_WINDOWS,
    "har_horizon": HAR_HORIZON,
    "har_alpha": HAR_ALPHA,
    "ridge_alpha": RIDGE_ALPHA,
    "minimum_labeled_observations": MIN_LABELED_OBSERVATIONS,
    "volatility_budget": VOLATILITY_BUDGET,
    "soxl_weight_grid": SOXL_WEIGHT_GRID,
    "har_features": HAR_FEATURES,
    "ridge_features": RIDGE_FEATURES,
    "cost_cases_bps": DEFAULT_COSTS_BPS,
    "candidate_order": FROZEN_CANDIDATE_NAMES,
    "candidates": {
        name: asdict(candidate) for name, candidate in CANDIDATES.items()
    },
}
FROZEN_FAMILY_FINGERPRINT = _canonical_sha256(FROZEN_CONFIGURATION)
RIDGE_CHALLENGER_FINGERPRINT = _canonical_sha256(
    {
        "family_fingerprint": FROZEN_FAMILY_FINGERPRINT,
        "features": RIDGE_FEATURES,
        "alpha": RIDGE_ALPHA,
        "minimum_labeled_observations": MIN_LABELED_OBSERVATIONS,
        "horizon": REVIEW_SESSIONS,
        "training": "expanding, chronological, train-only standardization",
    }
)


def candidate_fingerprint(candidate: Candidate | str) -> str:
    selected = CANDIDATES[candidate] if isinstance(candidate, str) else candidate
    return _canonical_sha256(
        {
            "family_fingerprint": FROZEN_FAMILY_FINGERPRINT,
            "candidate": asdict(selected),
        }
    )


@dataclass(frozen=True)
class AlphaMarketData:
    opens: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame
    sessions: pd.DatetimeIndex
    common_start: pd.Timestamp
    final_session: pd.Timestamp
    fingerprint: str
    requested_start: str
    requested_end_exclusive: str
    source: str


@dataclass(frozen=True)
class ResidualEstimate:
    available: bool
    signal_date: pd.Timestamp
    estimation_start: pd.Timestamp | None = None
    estimation_end: pd.Timestamp | None = None
    scoring_start: pd.Timestamp | None = None
    scoring_end: pd.Timestamp | None = None
    intercept: float | None = None
    beta: float | None = None
    residual_standard_deviation: float | None = None
    residual_momentum: float | None = None
    z_score: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class RidgeEstimate:
    available: bool
    signal_date: pd.Timestamp
    feature_names: tuple[str, ...]
    training_cutoff: pd.Timestamp | None = None
    last_label_origin: pd.Timestamp | None = None
    sample_count: int = 0
    feature_means: tuple[float, ...] = ()
    feature_standard_deviations: tuple[float, ...] = ()
    coefficients: tuple[float, ...] = ()
    intercept: float | None = None
    current_features: tuple[float, ...] = ()
    prediction: float | None = None
    smearing_factor: float | None = None
    training_fingerprint: str | None = None
    challenger_fingerprint: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class VolatilityEstimate:
    available: bool
    signal_date: pd.Timestamp
    qld_model_volatility: float | None = None
    soxl_model_volatility: float | None = None
    qld_trailing21_volatility: float | None = None
    qld_trailing63_volatility: float | None = None
    soxl_trailing21_volatility: float | None = None
    soxl_trailing63_volatility: float | None = None
    correlation21: float | None = None
    correlation63: float | None = None
    conservative_correlation: float | None = None
    qld_forecast_volatility: float | None = None
    soxl_forecast_volatility: float | None = None
    raw_soxl_weight: float = 0.0
    qld_model: RidgeEstimate | None = None
    soxl_model: RidgeEstimate | None = None
    reason: str = ""


@dataclass(frozen=True)
class SignalPanel:
    """Causal signal rows and audit metadata used by every candidate."""

    frame: pd.DataFrame
    ridge_details: Mapping[pd.Timestamp, RidgeEstimate]
    data_fingerprint: str
    family_fingerprint: str = FROZEN_FAMILY_FINGERPRINT

    @property
    def first_ridge_date(self) -> pd.Timestamp | None:
        available = self.frame.index[self.frame["ridge_available"].astype(bool)]
        return pd.Timestamp(available[0]) if len(available) else None


@dataclass(frozen=True)
class AlphaBacktestResult:
    candidate: Candidate
    cost_bps: float
    ledger: pd.DataFrame
    metrics: dict[str, object]
    data_fingerprint: str
    family_fingerprint: str
    candidate_fingerprint: str
    software_fingerprint: str
    unfilled_final_order: bool


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
class _QueuedOrder:
    signal_date: pd.Timestamp
    target_weights: dict[str, float]
    reason: str
    strategic_soxl_weight: float
    median_dollar_volume20: dict[str, float]


@dataclass
class _SleeveState:
    active: bool = False
    eligibility_streak: int = 0
    strategic_soxl_weight: float = 0.0
    raw_upshift_weight: float = 0.0
    raw_upshift_streak: int = 0
    ridge_permitted: bool = True


def software_fingerprint() -> str:
    """Hash the executing source/configuration and relevant runtime versions."""

    return _canonical_sha256(
        {
            "alpha_research_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "alpha_core_sha256": hashlib.sha256(
                Path(core.__file__).read_bytes()
            ).hexdigest(),
            "family_fingerprint": FROZEN_FAMILY_FINGERPRINT,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
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


def _normalize_market_frame(
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
            raise RuntimeError(f"Market response is missing {field_name}")
    else:
        if len(tickers) != 1 or field_name not in raw.columns:
            raise RuntimeError(f"Market response is missing {field_name}")
        frame = pd.DataFrame({tickers[0]: raw[field_name]}, index=raw.index)
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=tickers[0])
    return _normalize_frame(frame, tickers)


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


def _last_session_before(exclusive_end: str) -> pd.Timestamp:
    end = pd.Timestamp(exclusive_end).normalize()
    sessions = _calendar_sessions(
        end - pd.Timedelta(days=14),
        end - pd.Timedelta(days=1),
    )
    if len(sessions) == 0:
        raise RuntimeError("XNYS calendar returned no session before requested end")
    return pd.Timestamp(sessions[-1]).normalize()


def _expected_completed_session(
    now_new_york: pd.Timestamp | datetime | None = None,
) -> pd.Timestamp:
    now = (
        pd.Timestamp(datetime.now(NEW_YORK))
        if now_new_york is None
        else pd.Timestamp(now_new_york)
    )
    if now.tzinfo is None:
        now = now.tz_localize(NEW_YORK)
    else:
        now = now.tz_convert(NEW_YORK)
    calendar = xcals.get_calendar("XNYS")
    candidates = calendar.sessions_in_range(
        (now - pd.Timedelta(days=14)).date(),
        now.date(),
    )
    for session in reversed(candidates):
        finalized = (
            calendar.session_close(session).tz_convert(NEW_YORK)
            + pd.Timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES)
        )
        if now >= finalized:
            completed = pd.Timestamp(session)
            if completed.tz is not None:
                completed = completed.tz_convert(None)
            return completed.normalize()
    raise RuntimeError("Unable to identify the latest completed XNYS session")


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
        canonical = frame.copy()
        canonical.index.name = "Session"
        canonical.columns.name = "Ticker"
        rendered = canonical.to_csv(
            index=True,
            date_format="%Y-%m-%d",
            float_format="%.17g",
            lineterminator="\n",
        )
        digest.update(rendered.encode("utf-8"))
    return digest.hexdigest()


def prepare_market_data(
    raw: pd.DataFrame,
    *,
    requested_start: str,
    requested_end_exclusive: str,
    tickers: Sequence[str] = TICKERS,
    source: str = "provided adjusted OHLCV",
) -> AlphaMarketData:
    """Validate complete adjusted OHLCV without fill, deletion, or substitution."""

    selected = tuple(tickers)
    if selected != TICKERS:
        raise ValueError(f"Frozen alpha universe must be exactly {list(TICKERS)}")
    opens = _normalize_market_frame(raw, "Open", selected)
    closes = _normalize_market_frame(raw, "Close", selected)
    volumes = _normalize_market_frame(raw, "Volume", selected)
    if not opens.index.equals(closes.index) or not opens.index.equals(volumes.index):
        raise RuntimeError("Open, Close, and Volume indexes differ")

    final_session = _last_session_before(requested_end_exclusive)
    today_ny = pd.Timestamp.now(tz=NEW_YORK).tz_localize(None).normalize()
    if final_session >= today_ny:
        completed = _expected_completed_session()
        if final_session > completed:
            raise RuntimeError(
                "Alpha data includes an incomplete XNYS session: "
                f"requested={final_session.date().isoformat()}, "
                f"completed={completed.date().isoformat()}"
            )
    expected = _calendar_sessions(
        pd.Timestamp(requested_start),
        final_session,
    )
    if not opens.index.equals(expected):
        missing = expected.difference(opens.index)
        unexpected = opens.index.difference(expected)
        raise RuntimeError(
            "Alpha market-data session continuity failed: "
            f"missing={[item.date().isoformat() for item in missing[:5]]}, "
            f"non_sessions={[item.date().isoformat() for item in unexpected[:5]]}"
        )

    first_valid: list[pd.Timestamp] = []
    for field_name, frame in (
        ("Open", opens),
        ("Close", closes),
        ("Volume", volumes),
    ):
        for ticker in selected:
            values = frame[ticker].to_numpy(dtype=float)
            valid = np.isfinite(values) & (values > 0)
            if not valid.any():
                raise RuntimeError(
                    f"{ticker} has no valid adjusted {field_name} history"
                )
            first_valid.append(pd.Timestamp(frame.index[np.argmax(valid)]))
    common_start = max(first_valid).normalize()
    expected_model_start = pd.Timestamp(MODEL_HISTORY_START)
    if common_start != expected_model_start:
        raise RuntimeError(
            "Frozen model history must begin at the actual SOXL inception "
            f"{MODEL_HISTORY_START}; received "
            f"{common_start.date().isoformat()}"
        )
    common_position = opens.index.get_loc(common_start)
    if not isinstance(common_position, (int, np.integer)):
        raise RuntimeError("Common inception does not resolve to one session")

    for field_name, frame in (
        ("Open", opens),
        ("Close", closes),
        ("Volume", volumes),
    ):
        scored = frame.iloc[int(common_position) :].to_numpy(dtype=float)
        invalid = ~np.isfinite(scored) | (scored <= 0)
        if invalid.any():
            row, column = np.argwhere(invalid)[0]
            absolute_row = int(common_position) + int(row)
            raise RuntimeError(
                f"Scored adjusted {field_name} contains missing or invalid data "
                f"at {frame.index[absolute_row].date().isoformat()} / "
                f"{frame.columns[int(column)]}"
            )

    return AlphaMarketData(
        opens=opens,
        closes=closes,
        volumes=volumes,
        sessions=opens.index[int(common_position) :],
        common_start=common_start,
        final_session=final_session,
        fingerprint=_full_data_fingerprint(
            opens,
            closes,
            volumes,
        ),
        requested_start=requested_start,
        requested_end_exclusive=requested_end_exclusive,
        source=source,
    )


def validate_frozen_history_boundary(data: AlphaMarketData) -> None:
    """Require the fixed inception that anchors training and review cadence."""

    expected = pd.Timestamp(MODEL_HISTORY_START)
    if data.common_start != expected:
        raise RuntimeError(
            "Frozen alpha data has a shifted common inception: "
            f"expected {MODEL_HISTORY_START}, received "
            f"{data.common_start.date().isoformat()}"
        )
    if not len(data.sessions) or pd.Timestamp(data.sessions[0]) != expected:
        raise RuntimeError("Frozen alpha sessions do not start at model inception")


def download_market_data(
    *,
    requested_start: str,
    requested_end_exclusive: str,
) -> AlphaMarketData:
    """Download the frozen five-ticker adjusted snapshot through yfinance."""

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required for a live research download") from exc
    raw = yf.download(
        list(TICKERS),
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
        source="Yahoo Finance via yfinance; auto_adjust=True",
    )


def save_market_snapshot(data: AlphaMarketData, path: Path) -> None:
    """Atomically save exact validated inputs as deterministic long-form CSV."""

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
        float_format="%.17g",
        lineterminator="\n",
    )
    os.replace(temporary, path)


def load_market_snapshot(
    path: Path,
    *,
    requested_start: str,
    requested_end_exclusive: str,
) -> AlphaMarketData:
    """Load and revalidate a snapshot produced by :func:`save_market_snapshot`."""

    try:
        snapshot = pd.read_csv(
            path,
            parse_dates=["Session"],
            float_precision="round_trip",
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Unable to read alpha snapshot: {path}") from exc
    required = {"Session", "Ticker", "Field", "Value"}
    if set(snapshot.columns) != required:
        raise RuntimeError(
            f"Alpha snapshot columns must be exactly {sorted(required)}"
        )
    if snapshot.duplicated(["Session", "Ticker", "Field"]).any():
        raise RuntimeError("Alpha snapshot contains duplicate observations")
    if set(snapshot["Ticker"]) != set(TICKERS):
        raise RuntimeError("Alpha snapshot does not contain the frozen universe")
    if set(snapshot["Field"]) != {"Open", "Close", "Volume"}:
        raise RuntimeError("Alpha snapshot must contain Open, Close, and Volume")
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
        source=f"frozen local snapshot: {path.name}",
    )


def _positive_log_returns(close: pd.Series) -> pd.Series:
    values = close.astype(float)
    if (~np.isfinite(values.to_numpy())).any() or (values <= 0).any():
        raise RuntimeError("Adjusted close series contains an invalid price")
    result = np.log(values).diff()
    result.name = str(close.name)
    return result


def residual_estimate_at(
    closes: pd.DataFrame,
    signal_date: pd.Timestamp | str,
) -> ResidualEstimate:
    """Fit separated-window SMH-on-QQQ OLS and score the following 63 returns."""

    date = pd.Timestamp(signal_date).normalize()
    try:
        close_position = closes.index.get_loc(date)
    except KeyError:
        return ResidualEstimate(False, date, reason="signal date is absent")
    if not isinstance(close_position, (int, np.integer)):
        return ResidualEstimate(False, date, reason="signal date is duplicated")
    needed_returns = ESTIMATION_RETURNS + SCORING_RETURNS
    if int(close_position) < needed_returns:
        return ResidualEstimate(False, date, reason="insufficient separated history")

    required_closes = needed_returns + 1
    window = closes.loc[
        :date,
        ["QQQ", "SMH"],
    ].iloc[-required_closes:]
    try:
        signal = core.calculate_residual_signal(
            window,
            beta_window=ESTIMATION_RETURNS,
            score_window=SCORING_RETURNS,
        )
    except ValueError as exc:
        return ResidualEstimate(False, date, reason=str(exc))

    return ResidualEstimate(
        available=True,
        signal_date=date,
        estimation_start=signal.estimation_start,
        estimation_end=signal.estimation_end,
        scoring_start=signal.scoring_start,
        scoring_end=signal.scoring_end,
        intercept=signal.intercept,
        beta=signal.beta,
        residual_standard_deviation=signal.residual_sigma,
        residual_momentum=signal.residual_momentum,
        z_score=signal.residual_z,
    )


def build_residual_frame(closes: pd.DataFrame) -> pd.DataFrame:
    """Calculate every causal residual score without overlapping fit/score windows."""

    rows: list[dict[str, object]] = []
    for date in closes.index:
        estimate = residual_estimate_at(closes, date)
        rows.append(
            {
                "session": pd.Timestamp(date),
                "residual_available": estimate.available,
                "residual_momentum": estimate.residual_momentum,
                "residual_z": estimate.z_score,
                "residual_beta": estimate.beta,
                "residual_intercept": estimate.intercept,
                "residual_sigma": estimate.residual_standard_deviation,
                "residual_estimation_end": estimate.estimation_end,
                "residual_scoring_start": estimate.scoring_start,
                "residual_reason": estimate.reason,
            }
        )
    return pd.DataFrame(rows).set_index("session")


def _forward_mean(series: pd.Series, horizon: int) -> pd.Series:
    """Mean of observations ``t+1`` through ``t+horizon``, aligned at ``t``."""

    return series.rolling(horizon, min_periods=horizon).mean().shift(-horizon)


def har_design(close: pd.Series) -> pd.DataFrame:
    """Create causal HAR-style features and future annualized variance labels."""

    return core.build_variance_learning_frame(close)


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    canonical.index.name = "Session"
    rendered = canonical.to_csv(
        index=True,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _ridge_at(
    design: pd.DataFrame,
    signal_date: pd.Timestamp | str,
    *,
    feature_names: Sequence[str],
    target_name: str,
    alpha: float,
    minimum_observations: int,
    label_horizon: int,
    challenger_fingerprint: str | None,
    include_training_fingerprint: bool = True,
) -> RidgeEstimate:
    """Fit a fixed-penalty expanding ridge with an explicit label cutoff."""

    date = pd.Timestamp(signal_date).normalize()
    names = tuple(feature_names)
    try:
        signal_position = design.index.get_loc(date)
    except KeyError:
        return RidgeEstimate(
            False,
            date,
            names,
            reason="signal date is absent from ridge design",
            challenger_fingerprint=challenger_fingerprint,
        )
    if not isinstance(signal_position, (int, np.integer)):
        return RidgeEstimate(
            False,
            date,
            names,
            reason="signal date is duplicated",
            challenger_fingerprint=challenger_fingerprint,
        )
    cutoff_position = int(signal_position) - int(label_horizon)
    if cutoff_position < 0:
        return RidgeEstimate(
            False,
            date,
            names,
            reason="no fully observed future labels",
            challenger_fingerprint=challenger_fingerprint,
        )
    training = design.iloc[: cutoff_position + 1][[*names, target_name]].copy()
    values = training.to_numpy(dtype=float)
    usable = np.isfinite(values).all(axis=1)
    training = training.loc[usable]
    sample_count = len(training)
    training_cutoff = (
        pd.Timestamp(training.index[-1]) if sample_count else None
    )
    if sample_count < minimum_observations:
        return RidgeEstimate(
            False,
            date,
            names,
            training_cutoff=training_cutoff,
            sample_count=sample_count,
            challenger_fingerprint=challenger_fingerprint,
            reason=(
                f"requires {minimum_observations} fully labeled rows; "
                f"found {sample_count}"
            ),
        )
    current = design.loc[date, list(names)].to_numpy(dtype=float)
    if not np.isfinite(current).all():
        return RidgeEstimate(
            False,
            date,
            names,
            training_cutoff=training_cutoff,
            sample_count=sample_count,
            challenger_fingerprint=challenger_fingerprint,
            reason="current feature row is nonfinite",
        )

    try:
        audited = core.standardized_ridge_fit_predict(
            training.loc[:, list(names)],
            training[target_name],
            pd.Series(current, index=list(names), dtype=float, name=date),
            alpha=alpha,
            min_samples=minimum_observations,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return RidgeEstimate(
            False,
            date,
            names,
            training_cutoff=training_cutoff,
            sample_count=sample_count,
            challenger_fingerprint=challenger_fingerprint,
            reason=str(exc),
        )
    prediction = float(audited.prediction)
    if not np.isfinite(prediction):
        return RidgeEstimate(
            False,
            date,
            names,
            training_cutoff=training_cutoff,
            sample_count=sample_count,
            challenger_fingerprint=challenger_fingerprint,
            reason="ridge prediction is nonfinite",
        )
    return RidgeEstimate(
        available=True,
        signal_date=date,
        feature_names=names,
        training_cutoff=audited.training_end,
        last_label_origin=audited.last_label_origin,
        sample_count=audited.sample_count,
        feature_means=audited.feature_means,
        feature_standard_deviations=audited.feature_scales,
        coefficients=audited.coefficients,
        intercept=audited.intercept,
        current_features=audited.current_features,
        prediction=prediction,
        smearing_factor=audited.smearing_factor,
        training_fingerprint=(
            _frame_fingerprint(training) if include_training_fingerprint else None
        ),
        challenger_fingerprint=challenger_fingerprint,
    )


def har_forecast_at(
    design: pd.DataFrame,
    signal_date: pd.Timestamp | str,
    *,
    minimum_observations: int = MIN_LABELED_OBSERVATIONS,
) -> RidgeEstimate:
    """Forecast log annualized variance with fixed-alpha HAR-style ridge."""

    return _ridge_at(
        design,
        signal_date,
        feature_names=HAR_FEATURES,
        target_name="future_log_var_21",
        alpha=HAR_ALPHA,
        minimum_observations=minimum_observations,
        label_horizon=HAR_HORIZON,
        challenger_fingerprint=_canonical_sha256(
            {
                "model": "daily-bar HAR-style log variance ridge",
                "features": HAR_FEATURES,
                "alpha": HAR_ALPHA,
                "horizon": HAR_HORIZON,
                "minimum": minimum_observations,
            }
        ),
        include_training_fingerprint=False,
    )


def _safe_annualized_volatility(variance: object) -> float | None:
    try:
        value = float(variance)
    except (TypeError, ValueError):
        return None
    annualized_variance = value * 252.0
    if not np.isfinite(annualized_variance) or annualized_variance <= 0:
        return None
    return math.sqrt(annualized_variance)


def choose_volatility_weight(
    qld_volatility: float,
    soxl_volatility: float,
    correlation: float,
    *,
    budget: float = VOLATILITY_BUDGET,
    grid: Sequence[float] = SOXL_WEIGHT_GRID,
) -> float:
    """Choose the largest frozen SOXL weight within the portfolio vol budget."""

    try:
        weight, _ = core.choose_soxl_weight(
            qld_volatility,
            soxl_volatility,
            correlation,
            budget=budget,
            grid=grid,
        )
    except ValueError:
        # Research signal construction fails closed: invalid estimates never
        # authorize the incremental leveraged sleeve.
        return 0.0
    return float(weight)


def volatility_estimate_at(
    data: AlphaMarketData,
    signal_date: pd.Timestamp | str,
    *,
    qld_design: pd.DataFrame | None = None,
    soxl_design: pd.DataFrame | None = None,
    minimum_observations: int = MIN_LABELED_OBSERVATIONS,
) -> VolatilityEstimate:
    """Build causal QLD/SOXL forecasts, conservative correlation, and raw weight."""

    date = pd.Timestamp(signal_date).normalize()
    common_closes = data.closes.loc[data.common_start :]
    try:
        position = common_closes.index.get_loc(date)
    except KeyError:
        return VolatilityEstimate(False, date, reason="signal date is absent")
    if not isinstance(position, (int, np.integer)):
        return VolatilityEstimate(False, date, reason="signal date is duplicated")
    qld_design = (
        qld_design
        if qld_design is not None
        else har_design(common_closes["QLD"])
    )
    soxl_design = (
        soxl_design
        if soxl_design is not None
        else har_design(common_closes["SOXL"])
    )
    qld_model = har_forecast_at(
        qld_design,
        date,
        minimum_observations=minimum_observations,
    )
    soxl_model = har_forecast_at(
        soxl_design,
        date,
        minimum_observations=minimum_observations,
    )
    if not qld_model.available or not soxl_model.available:
        return VolatilityEstimate(
            False,
            date,
            qld_model=qld_model,
            soxl_model=soxl_model,
            reason=(
                "HAR forecast unavailable: "
                f"QLD={qld_model.reason or 'ok'}; "
                f"SOXL={soxl_model.reason or 'ok'}"
            ),
        )
    assert qld_model.prediction is not None
    assert soxl_model.prediction is not None
    try:
        if qld_model.smearing_factor is None or soxl_model.smearing_factor is None:
            raise ValueError("missing train-only smearing correction")
        qld_model_volatility = math.sqrt(
            math.exp(qld_model.prediction) * qld_model.smearing_factor
        )
        soxl_model_volatility = math.sqrt(
            math.exp(soxl_model.prediction) * soxl_model.smearing_factor
        )
    except (OverflowError, ValueError):
        return VolatilityEstimate(
            False,
            date,
            qld_model=qld_model,
            soxl_model=soxl_model,
            reason="HAR variance forecast cannot be converted to volatility",
        )

    qld_returns = _positive_log_returns(common_closes["QLD"])
    soxl_returns = _positive_log_returns(common_closes["SOXL"])
    qld_history = qld_returns.iloc[: int(position) + 1].dropna()
    soxl_history = soxl_returns.iloc[: int(position) + 1].dropna()
    qld21 = float(qld_history.tail(21).std(ddof=1) * math.sqrt(252.0))
    qld63 = float(qld_history.tail(63).std(ddof=1) * math.sqrt(252.0))
    soxl21 = float(soxl_history.tail(21).std(ddof=1) * math.sqrt(252.0))
    soxl63 = float(soxl_history.tail(63).std(ddof=1) * math.sqrt(252.0))
    paired = pd.concat([qld_returns, soxl_returns], axis=1)
    corr21 = float(paired.iloc[: int(position) + 1].tail(21).corr().iloc[0, 1])
    corr63 = float(paired.iloc[: int(position) + 1].tail(63).corr().iloc[0, 1])
    required = (
        qld_model_volatility,
        soxl_model_volatility,
        qld21,
        qld63,
        soxl21,
        soxl63,
    )
    if (
        not all(value is not None and np.isfinite(value) and value > 0 for value in required)
    ):
        return VolatilityEstimate(
            False,
            date,
            qld_model=qld_model,
            soxl_model=soxl_model,
            reason="zero/nonfinite volatility or correlation fails closed",
        )
    try:
        conservative_correlation = core.conservative_finite_correlation(
            corr21,
            corr63,
        )
    except ValueError:
        return VolatilityEstimate(
            False,
            date,
            qld_model=qld_model,
            soxl_model=soxl_model,
            reason="zero/nonfinite volatility or correlation fails closed",
        )
    qld_forecast = max(qld_model_volatility, float(qld21), float(qld63))
    soxl_forecast = max(soxl_model_volatility, float(soxl21), float(soxl63))
    raw_weight = choose_volatility_weight(
        qld_forecast,
        soxl_forecast,
        conservative_correlation,
    )
    return VolatilityEstimate(
        available=True,
        signal_date=date,
        qld_model_volatility=qld_model_volatility,
        soxl_model_volatility=soxl_model_volatility,
        qld_trailing21_volatility=qld21,
        qld_trailing63_volatility=qld63,
        soxl_trailing21_volatility=soxl21,
        soxl_trailing63_volatility=soxl63,
        correlation21=corr21 if np.isfinite(corr21) else None,
        correlation63=corr63 if np.isfinite(corr63) else None,
        conservative_correlation=conservative_correlation,
        qld_forecast_volatility=qld_forecast,
        soxl_forecast_volatility=soxl_forecast,
        raw_soxl_weight=raw_weight,
        qld_model=qld_model,
        soxl_model=soxl_model,
    )


def ridge_relative_label_frame(opens: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen next-open, 21-holding-interval relative log label.

    A row formed after close ``t`` enters at open ``t+1`` and is measured at
    open ``t+22``.  Those endpoints contain exactly 21 open-to-open holding
    intervals.  The 65/35 sleeve is bought once and not rebalanced inside the
    label window.
    """

    required = opens.loc[:, ["QLD", "SOXL"]].astype(float)
    values = required.to_numpy(dtype=float)
    if (~np.isfinite(values)).any() or (values <= 0).any():
        raise ValueError("Ridge label opens contain invalid prices")
    entry_qld = required["QLD"].shift(-1)
    entry_soxl = required["SOXL"].shift(-1)
    exit_qld = required["QLD"].shift(-(REVIEW_SESSIONS + 1))
    exit_soxl = required["SOXL"].shift(-(REVIEW_SESSIONS + 1))
    qld_growth = exit_qld / entry_qld
    sleeve_growth = (
        0.65 * qld_growth
        + 0.35 * (exit_soxl / entry_soxl)
    )
    label = np.log(sleeve_growth) - np.log(qld_growth)
    frame = pd.DataFrame(
        {
            "ridge_relative_log_return_21": label,
            "ridge_label_entry_date": pd.Series(
                opens.index,
                index=opens.index,
            ).shift(-1),
            "ridge_label_exit_date": pd.Series(
                opens.index,
                index=opens.index,
            ).shift(-(REVIEW_SESSIONS + 1)),
        },
        index=opens.index,
    )
    return frame


def ridge_shadow_estimate_at(
    design: pd.DataFrame,
    signal_date: pd.Timestamp | str,
    *,
    minimum_observations: int = MIN_LABELED_OBSERVATIONS,
    include_training_fingerprint: bool = True,
) -> RidgeEstimate:
    """Fit the frozen expanding ridge shadow without an overlapping label."""

    return _ridge_at(
        design,
        signal_date,
        feature_names=RIDGE_FEATURES,
        target_name="ridge_relative_log_return_21",
        alpha=RIDGE_ALPHA,
        minimum_observations=minimum_observations,
        # A t close enters at t+1 open and exits at t+22 open.  At close T,
        # T's open is known, so the newest admissible feature row is T-22.
        label_horizon=REVIEW_SESSIONS + 1,
        challenger_fingerprint=RIDGE_CHALLENGER_FINGERPRINT,
        include_training_fingerprint=include_training_fingerprint,
    )


def build_signal_panel(
    data: AlphaMarketData,
    *,
    minimum_observations: int = MIN_LABELED_OBSERVATIONS,
) -> SignalPanel:
    """Precompute the complete causal signal panel for the frozen family."""

    common_closes = data.closes.loc[data.common_start :]
    common_opens = data.opens.loc[data.common_start :]
    qld_design = har_design(common_closes["QLD"])
    soxl_design = har_design(common_closes["SOXL"])
    residual = build_residual_frame(data.closes)
    qqq_close = data.closes["QQQ"].astype(float)
    qqq_sma = qqq_close.rolling(
        TREND_SESSIONS,
        min_periods=TREND_SESSIONS,
    ).mean()
    qqq_returns = _positive_log_returns(qqq_close)
    qqq_squared = qqq_returns.pow(2)
    qqq_downside_squared = qqq_returns.clip(upper=0.0).pow(2)
    downside_ratio = (
        qqq_downside_squared.rolling(21, min_periods=21).sum()
        / qqq_squared.rolling(21, min_periods=21).sum()
    )
    drawdown = qqq_close / qqq_close.rolling(
        252,
        min_periods=252,
    ).max() - 1.0

    rows: list[dict[str, object]] = []
    for date in data.sessions:
        volatility = volatility_estimate_at(
            data,
            date,
            qld_design=qld_design,
            soxl_design=soxl_design,
            minimum_observations=minimum_observations,
        )
        residual_row = residual.loc[date]
        sma = float(qqq_sma.loc[date]) if pd.notna(qqq_sma.loc[date]) else np.nan
        qqq_price = float(qqq_close.loc[date])
        trend_gap = (
            math.log(qqq_price / sma)
            if np.isfinite(sma) and sma > 0 and qqq_price > 0
            else np.nan
        )
        qld_model_volatility = volatility.qld_model_volatility
        qld_volatility63 = volatility.qld_trailing63_volatility
        forecast_ratio = (
            math.log(float(qld_model_volatility) / float(qld_volatility63))
            if qld_model_volatility is not None
            and qld_volatility63 is not None
            and np.isfinite(qld_model_volatility)
            and np.isfinite(qld_volatility63)
            and qld_model_volatility > 0
            and qld_volatility63 > 0
            else np.nan
        )
        rows.append(
            {
                "session": pd.Timestamp(date),
                "trend_positive": bool(
                    np.isfinite(sma) and qqq_price > sma
                ),
                "qqq_log_trend_gap": trend_gap,
                "residual_available": bool(residual_row["residual_available"]),
                "residual_momentum": residual_row["residual_momentum"],
                "residual_z": residual_row["residual_z"],
                "residual_positive": bool(
                    residual_row["residual_available"]
                    and np.isfinite(float(residual_row["residual_momentum"]))
                    and float(residual_row["residual_momentum"]) > 0
                ),
                "qld_model_volatility": qld_model_volatility,
                "soxl_model_volatility": volatility.soxl_model_volatility,
                "qld_forecast_volatility": volatility.qld_forecast_volatility,
                "soxl_forecast_volatility": volatility.soxl_forecast_volatility,
                "qld_trailing21_volatility": volatility.qld_trailing21_volatility,
                "qld_trailing63_volatility": qld_volatility63,
                "soxl_trailing21_volatility": volatility.soxl_trailing21_volatility,
                "soxl_trailing63_volatility": volatility.soxl_trailing63_volatility,
                "correlation21": volatility.correlation21,
                "correlation63": volatility.correlation63,
                "sizing_correlation": volatility.conservative_correlation,
                "volatility_available": volatility.available,
                "raw_soxl_weight": (
                    volatility.raw_soxl_weight if volatility.available else 0.0
                ),
                "qld_forecast_to_vol63_log": forecast_ratio,
                "qqq_drawdown252": drawdown.loc[date],
                "qqq_downside_ratio21": downside_ratio.loc[date],
            }
        )
    frame = pd.DataFrame(rows).set_index("session")
    labels = ridge_relative_label_frame(common_opens)
    frame = frame.join(labels[["ridge_relative_log_return_21"]], how="left")

    ridge_details: dict[pd.Timestamp, RidgeEstimate] = {}
    frame["ridge_available"] = False
    frame["ridge_prediction"] = np.nan
    frame["ridge_training_cutoff"] = pd.NaT
    frame["ridge_sample_count"] = 0
    # Ridge is only a 21-session review challenger, so fitting it between
    # reviews adds no decision information.  The anchor is the first complete
    # signal row; every strategy records the same review phase.
    for position in range(0, len(frame), REVIEW_SESSIONS):
        date = pd.Timestamp(frame.index[position])
        estimate = ridge_shadow_estimate_at(
            frame,
            date,
            minimum_observations=minimum_observations,
            include_training_fingerprint=True,
        )
        ridge_details[date] = estimate
        frame.loc[date, "ridge_available"] = estimate.available
        frame.loc[date, "ridge_prediction"] = estimate.prediction
        frame.loc[date, "ridge_training_cutoff"] = estimate.training_cutoff
        frame.loc[date, "ridge_sample_count"] = estimate.sample_count
    frame["ridge_available"] = frame["ridge_available"].astype(bool)
    frame["ridge_sample_count"] = frame["ridge_sample_count"].astype(int)
    return SignalPanel(
        frame=frame,
        ridge_details=ridge_details,
        data_fingerprint=data.fingerprint,
    )


def build_target_schedule(
    panel: SignalPanel,
    candidate: Candidate | str,
) -> pd.DataFrame:
    """Apply the frozen state machine to causal signal rows."""

    selected = CANDIDATES[candidate] if isinstance(candidate, str) else candidate
    if selected.name not in CANDIDATES or CANDIDATES[selected.name] != selected:
        raise ValueError("Candidate is not in the frozen registry")
    frame = panel.frame
    if frame.empty:
        raise ValueError("Signal panel is empty")

    if selected.kind in {"qld_buy_hold", "static"}:
        soxl_weight = 0.0 if selected.kind == "qld_buy_hold" else 0.35
        return pd.DataFrame(
            {
                "strategic_soxl_weight": soxl_weight,
                "overlay_active": soxl_weight > 0,
                "eligibility_streak": 0,
                "alpha_review_due": False,
                "alpha_reviewed": False,
                "structural_change": False,
                "decision_reason": (
                    "QLD_BUY_HOLD" if soxl_weight == 0 else "STATIC_BUY_HOLD"
                ),
            },
            index=frame.index,
        )

    state = core.OverlayState()
    rows: list[dict[str, object]] = []
    for position, (date, signal) in enumerate(frame.iterrows()):
        review_due = position % REVIEW_SESSIONS == 0
        trend = bool(signal["trend_positive"])
        residual_available = bool(signal["residual_available"])
        residual = bool(signal["residual_positive"])
        if selected.gate == "trend":
            residual = True
        elif selected.gate == "residual":
            # A valid residual row removes the broad QQQ gate for this
            # ablation; an invalid row still fails closed immediately.
            trend = residual_available
        elif selected.gate != "trend_residual":
            raise ValueError(f"Unsupported dynamic gate: {selected.gate}")
        else:
            trend = trend and residual_available

        raw_weight = (
            float(signal["raw_soxl_weight"])
            if selected.volatility_sizing
            and bool(signal["volatility_available"])
            and np.isfinite(float(signal["raw_soxl_weight"]))
            else selected.fixed_soxl_weight
            if not selected.volatility_sizing
            else 0.0
        )
        ridge_permitted = True
        if selected.ridge_confirmation and review_due:
            ridge_prediction = signal["ridge_prediction"]
            ridge_permitted = bool(
                signal["ridge_available"]
                and pd.notna(ridge_prediction)
                and np.isfinite(float(ridge_prediction))
                and float(ridge_prediction) > 0
            )
            residual = residual and ridge_permitted

        transition = core.advance_overlay_state(
            state,
            signal_date=pd.Timestamp(date),
            trend_positive=trend,
            residual_positive=residual,
            raw_soxl_weight=raw_weight,
            alpha_review_due=review_due,
            reentry_closes=REENTRY_CLOSES,
            upshift_closes=VOL_UPSHIFT_CLOSES,
        )
        state = transition.state
        reason = transition.reason
        if selected.ridge_confirmation and review_due and not ridge_permitted:
            reason = f"RIDGE_BLOCK/{reason}"
        rows.append(
            {
                "session": pd.Timestamp(date),
                "strategic_soxl_weight": state.soxl_weight,
                "overlay_active": state.overlay_active,
                "eligibility_streak": state.eligible_streak,
                "pending_scale_weight": state.pending_soxl_weight,
                "pending_scale_days": state.pending_scale_days,
                "alpha_review_due": review_due,
                "alpha_reviewed": transition.alpha_reviewed,
                "structural_change": transition.structural_change,
                "decision_reason": reason,
                "ridge_permitted": ridge_permitted,
            }
        )
    return pd.DataFrame(rows).set_index("session")


def common_scoring_start(
    data: AlphaMarketData,
    panel: SignalPanel,
) -> pd.Timestamp:
    """Return the first execution date shared by every frozen candidate."""

    ridge_date = panel.first_ridge_date
    if ridge_date is None:
        raise RuntimeError(
            "No ridge-shadow prediction is available for a common family interval"
        )
    position = data.sessions.get_loc(ridge_date)
    if not isinstance(position, (int, np.integer)) or int(position) + 1 >= len(
        data.sessions
    ):
        raise RuntimeError("No executable session follows the first ridge signal")
    return pd.Timestamp(data.sessions[int(position) + 1])


def _median_dollar_volume20(
    data: AlphaMarketData,
    signal_date: pd.Timestamp,
) -> dict[str, float]:
    close = data.closes.loc[:signal_date, list(TRADED_TICKERS)].tail(20)
    volume = data.volumes.loc[:signal_date, list(TRADED_TICKERS)].tail(20)
    dollar_volume = close * volume
    result: dict[str, float] = {}
    for ticker in TRADED_TICKERS:
        value = float(dollar_volume[ticker].median())
        result[ticker] = value if np.isfinite(value) and value > 0 else 0.0
    return result


def _weights_from_close(
    shares: Mapping[str, float],
    cash: float,
    close_prices: pd.Series,
) -> tuple[dict[str, float], float]:
    values = {
        ticker: float(shares.get(ticker, 0.0)) * float(close_prices[ticker])
        for ticker in TRADED_TICKERS
    }
    nav = float(cash + sum(values.values()))
    if not np.isfinite(nav) or nav <= 0:
        raise RuntimeError("Portfolio close NAV is invalid")
    weights = {ticker: value / nav for ticker, value in values.items()}
    weights[CASH] = float(cash) / nav
    return weights, nav


def _buffered_drift_target(
    actual_soxl_weight: float,
    strategic_soxl_weight: float,
) -> float | None:
    return core.buffered_soxl_rebalance_weight(
        actual_soxl_weight,
        strategic_soxl_weight,
        trigger=DRIFT_TRIGGER,
        destination=DRIFT_DESTINATION,
    )


def _execution_target(soxl_weight: float) -> dict[str, float]:
    target = core.target_weights(soxl_weight)
    target[CASH] = 0.0
    return target


def solve_post_cost_target(
    *,
    current_values: Mapping[str, float],
    cash: float,
    target_weights: Mapping[str, float],
    cost_rate: float,
) -> tuple[float, float]:
    """Solve exact post-cost NAV for a fractional-share target by bisection."""

    if not np.isfinite(cost_rate) or not (0 <= cost_rate < 1):
        raise ValueError("Transaction-cost rate must be finite and in [0, 1)")
    target = dict(target_weights)
    if any(not np.isfinite(value) or value < 0 for value in target.values()):
        raise ValueError("Target weights must be finite and nonnegative")
    if not np.isclose(sum(target.values()), 1.0, atol=1e-12):
        raise ValueError("Target weights must sum to one")
    if any(not np.isfinite(value) or value < 0 for value in current_values.values()):
        raise ValueError("Current values must be finite and nonnegative")
    if not np.isfinite(cash) or cash < 0:
        raise ValueError("Cash must be finite and nonnegative")
    securities = (set(current_values) | set(target)) - {CASH}
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
    low, high = 0.0, pretrade_nav
    for _ in range(100):
        middle = (low + high) / 2.0
        if middle + cost_rate * gross(middle) > pretrade_nav:
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
    """Execute exact fractional target weights at adjusted open with costs."""

    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Transaction costs must be finite and nonnegative")
    target = dict(target_weights)
    target.setdefault(CASH, 0.0)
    if not np.isclose(sum(target.values()), 1.0, atol=1e-12):
        raise ValueError("Execution target must sum to one")
    tickers = (set(shares) | set(target)) - {CASH}
    current_values: dict[str, float] = {}
    for ticker in tickers:
        try:
            price = float(open_prices[ticker])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Missing execution Open for {ticker}") from exc
        units = float(shares.get(ticker, 0.0))
        if not np.isfinite(price) or price <= 0:
            raise RuntimeError(f"Invalid execution Open for {ticker}")
        if not np.isfinite(units) or units < 0:
            raise RuntimeError(f"Invalid shares for {ticker}")
        current_values[ticker] = units * price
    pretrade_nav = float(cash + sum(current_values.values()))
    posttrade_nav, gross_notional = solve_post_cost_target(
        current_values=current_values,
        cash=float(cash),
        target_weights=target,
        cost_rate=float(cost_bps) / 10_000.0,
    )
    cost = pretrade_nav - posttrade_nav
    next_shares: dict[str, float] = {}
    trade_notional: dict[str, float] = {}
    for ticker in sorted(tickers):
        price = float(open_prices[ticker])
        target_value = target.get(ticker, 0.0) * posttrade_nav
        trade_notional[ticker] = target_value - current_values.get(ticker, 0.0)
        if target_value > 1e-12:
            next_shares[ticker] = target_value / price
    next_cash = float(target.get(CASH, 0.0) * posttrade_nav)
    pre_weights = {
        ticker: value / pretrade_nav for ticker, value in current_values.items()
    }
    pre_weights[CASH] = float(cash) / pretrade_nav
    components = set(pre_weights) | set(target)
    one_way = 0.5 * sum(
        abs(target.get(item, 0.0) - pre_weights.get(item, 0.0))
        for item in components
    )
    return FillResult(
        shares=next_shares,
        cash=max(0.0, next_cash),
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


def simulate_candidate(
    data: AlphaMarketData,
    panel: SignalPanel,
    candidate: Candidate | str,
    *,
    cost_bps: float = PRIMARY_COST_BPS,
    scoring_start: pd.Timestamp | str | None = None,
    starting_cash: float = STARTING_CASH,
) -> AlphaBacktestResult:
    """Simulate one frozen path with fractional shares and next-open fills."""

    selected = CANDIDATES[candidate] if isinstance(candidate, str) else candidate
    schedule = build_target_schedule(panel, selected)
    start = (
        common_scoring_start(data, panel)
        if scoring_start is None
        else pd.Timestamp(scoring_start).normalize()
    )
    if start not in data.sessions:
        raise ValueError("Scoring start is not an actual common XNYS session")
    start_position = data.sessions.get_loc(start)
    if not isinstance(start_position, (int, np.integer)) or int(start_position) < 1:
        raise ValueError("Scoring start needs a preceding signal session")
    scoring_dates = data.sessions[int(start_position) :]
    prior_signal_date = pd.Timestamp(data.sessions[int(start_position) - 1])
    if prior_signal_date not in schedule.index:
        raise ValueError("Target schedule does not cover the initial signal date")

    initial_soxl = float(
        schedule.loc[prior_signal_date, "strategic_soxl_weight"]
    )
    pending: _QueuedOrder | None = _QueuedOrder(
        signal_date=prior_signal_date,
        target_weights=_execution_target(initial_soxl),
        reason="INITIAL_DEPLOYMENT",
        strategic_soxl_weight=initial_soxl,
        median_dollar_volume20=_median_dollar_volume20(data, prior_signal_date),
    )
    shares: dict[str, float] = {}
    cash = float(starting_cash)
    if not np.isfinite(cash) or cash <= 0:
        raise ValueError("Starting cash must be positive and finite")
    executed_strategic_soxl = initial_soxl
    has_executed = False
    previous_close: pd.Series | None = None
    rows: list[dict[str, object]] = []

    for date in scoring_dates:
        date = pd.Timestamp(date)
        open_prices = data.opens.loc[date, list(TRADED_TICKERS)]
        close_prices = data.closes.loc[date, list(TRADED_TICKERS)]
        old_shares = dict(shares)
        overnight_pnl = {
            ticker: (
                old_shares.get(ticker, 0.0)
                * (
                    float(open_prices[ticker])
                    - float(previous_close[ticker])
                )
                if previous_close is not None
                else 0.0
            )
            for ticker in TRADED_TICKERS
        }

        fill: FillResult | None = None
        fill_reason = ""
        fill_signal_date: pd.Timestamp | None = None
        initial_deployment = False
        max_trade_to_adv = 0.0
        if pending is not None:
            fill = execute_target_at_open(
                shares=shares,
                cash=cash,
                open_prices=open_prices,
                target_weights=pending.target_weights,
                cost_bps=cost_bps,
            )
            shares = fill.shares
            cash = fill.cash
            fill_reason = pending.reason
            fill_signal_date = pending.signal_date
            executed_strategic_soxl = pending.strategic_soxl_weight
            initial_deployment = not has_executed
            has_executed = True
            ratios = []
            for ticker, notional in fill.trade_notional.items():
                if abs(notional) <= fill.pretrade_nav * 1e-12:
                    continue
                adv = pending.median_dollar_volume20.get(ticker, 0.0)
                ratios.append(abs(notional) / adv if adv > 0 else math.inf)
            max_trade_to_adv = max(ratios, default=0.0)
            pending = None

        intraday_pnl = {
            ticker: float(shares.get(ticker, 0.0))
            * (float(close_prices[ticker]) - float(open_prices[ticker]))
            for ticker in TRADED_TICKERS
        }
        component_pnl = {
            ticker: overnight_pnl[ticker] + intraday_pnl[ticker]
            for ticker in TRADED_TICKERS
        }
        actual_weights, nav = _weights_from_close(shares, cash, close_prices)
        signal = panel.frame.loc[date]
        state_row = schedule.loc[date]
        desired_soxl = float(state_row["strategic_soxl_weight"])
        order_reason = ""
        queued_weight: float | None = None
        if selected.kind not in {"qld_buy_hold", "static"}:
            if abs(desired_soxl - executed_strategic_soxl) > 1e-12:
                queued_weight = desired_soxl
                order_reason = f"STRUCTURAL/{state_row['decision_reason']}"
            else:
                buffered = _buffered_drift_target(
                    actual_weights["SOXL"],
                    desired_soxl,
                )
                if buffered is not None:
                    queued_weight = buffered
                    order_reason = "ORDINARY_DRIFT"
        if queued_weight is not None:
            pending = _QueuedOrder(
                signal_date=date,
                target_weights=_execution_target(queued_weight),
                reason=order_reason,
                # A drift fill does not change the strategic destination.
                strategic_soxl_weight=desired_soxl,
                median_dollar_volume20=_median_dollar_volume20(data, date),
            )

        gross_fraction = fill.gross_fraction if fill is not None else 0.0
        one_way = fill.one_way_fraction if fill is not None else 0.0
        transaction_cost = fill.cost if fill is not None else 0.0
        cost_fraction = fill.cost_fraction if fill is not None else 0.0
        security_orders = fill.security_orders if fill is not None else 0
        rows.append(
            {
                "session": date,
                "nav": nav,
                "starting_capital": float(starting_cash),
                "cash": cash,
                "qld_shares": float(shares.get("QLD", 0.0)),
                "soxl_shares": float(shares.get("SOXL", 0.0)),
                "qld_weight": actual_weights["QLD"],
                "soxl_weight": actual_weights["SOXL"],
                "cash_weight": actual_weights[CASH],
                "strategic_soxl_weight": desired_soxl,
                "executed_strategic_soxl_weight": executed_strategic_soxl,
                "advertised_daily_exposure": core.advertised_daily_exposure(
                    desired_soxl
                ),
                "overlay_active": bool(state_row["overlay_active"]),
                "decision_reason": state_row["decision_reason"],
                "alpha_review_due": bool(state_row["alpha_review_due"]),
                "trend_positive": bool(signal["trend_positive"]),
                "residual_positive": bool(signal["residual_positive"]),
                "residual_z": signal["residual_z"],
                "raw_soxl_weight": signal["raw_soxl_weight"],
                "ridge_prediction": signal["ridge_prediction"],
                "queued_soxl_weight": queued_weight,
                "queued_reason": order_reason,
                "fill_signal_date": fill_signal_date,
                "fill_reason": fill_reason,
                "initial_deployment": initial_deployment,
                "gross_trade_fraction": gross_fraction,
                "ongoing_gross_trade_fraction": (
                    0.0 if initial_deployment else gross_fraction
                ),
                "one_way_turnover": 0.0 if initial_deployment else one_way,
                "transaction_cost": transaction_cost,
                "transaction_cost_fraction": cost_fraction,
                "security_orders": security_orders,
                "max_trade_to_median_dollar_volume20": max_trade_to_adv,
                "qld_pnl": component_pnl["QLD"],
                "soxl_pnl": component_pnl["SOXL"],
                "cost_pnl": -transaction_cost,
            }
        )
        previous_close = close_prices

    ledger = pd.DataFrame(rows).set_index("session")
    metrics = calculate_metrics(ledger)
    return AlphaBacktestResult(
        candidate=selected,
        cost_bps=float(cost_bps),
        ledger=ledger,
        metrics=metrics,
        data_fingerprint=data.fingerprint,
        family_fingerprint=FROZEN_FAMILY_FINGERPRINT,
        candidate_fingerprint=candidate_fingerprint(selected),
        software_fingerprint=software_fingerprint(),
        unfilled_final_order=pending is not None,
    )


def _drawdown_diagnostics(
    nav: pd.Series,
    *,
    starting_capital: float,
    starting_date: pd.Timestamp,
) -> dict[str, object]:
    initial = pd.Series(
        [float(starting_capital)],
        index=pd.DatetimeIndex([pd.Timestamp(starting_date)]),
        dtype=float,
    )
    complete_nav = pd.concat([initial, nav.astype(float)])
    if complete_nav.index.has_duplicates:
        raise RuntimeError("Drawdown baseline date overlaps the scored ledger")
    running_peak = complete_nav.cummax()
    drawdown = complete_nav / running_peak - 1.0
    trough_date = pd.Timestamp(drawdown.idxmin())
    peak_value = float(running_peak.loc[trough_date])
    peak_candidates = complete_nav.loc[:trough_date]
    peak_date = pd.Timestamp(
        peak_candidates.index[
            np.flatnonzero(
                np.isclose(
                    peak_candidates.to_numpy(dtype=float),
                    peak_value,
                    rtol=0,
                    atol=max(1e-12, peak_value * 1e-12),
                )
            )[-1]
        ]
    )
    after_trough = complete_nav.loc[trough_date:]
    recovered = after_trough[after_trough >= peak_value * (1.0 - 1e-12)]
    recovery_date = pd.Timestamp(recovered.index[0]) if len(recovered) else None
    max_underwater = 0
    current = 0
    for value in drawdown.to_numpy(dtype=float):
        if value < -1e-12:
            current += 1
            max_underwater = max(max_underwater, current)
        else:
            current = 0
    return {
        "maximum_drawdown": float(drawdown.min()),
        "maximum_drawdown_peak": peak_date.date().isoformat(),
        "maximum_drawdown_trough": trough_date.date().isoformat(),
        "maximum_drawdown_recovery": (
            recovery_date.date().isoformat() if recovery_date is not None else None
        ),
        "maximum_drawdown_recovery_sessions": (
            int(
                complete_nav.index.get_loc(recovery_date)
                - complete_nav.index.get_loc(trough_date)
            )
            if recovery_date is not None
            else None
        ),
        "maximum_time_underwater_sessions": max_underwater,
        "fraction_sessions_underwater": float(
            (drawdown.iloc[1:] < -1e-12).mean()
        ),
    }


def _worst_rolling_return(nav: pd.Series, sessions: int) -> float | None:
    if len(nav) <= sessions:
        return None
    return float((nav / nav.shift(sessions) - 1.0).dropna().min())


def calculate_metrics(ledger: pd.DataFrame) -> dict[str, object]:
    """Calculate the frozen growth, tail, drawdown, cost, and turnover metrics."""

    if len(ledger) < 2:
        raise RuntimeError("At least two scored sessions are required")
    nav = ledger["nav"].astype(float)
    starting_capital = float(ledger["starting_capital"].iloc[0])
    later_returns = nav.pct_change(fill_method=None).iloc[1:]
    returns = pd.concat(
        [
            pd.Series(
                [float(nav.iloc[0] / starting_capital - 1.0)],
                index=[nav.index[0]],
            ),
            later_returns,
        ]
    )
    if (
        not np.isfinite(nav.to_numpy()).all()
        or (nav <= 0).any()
        or not np.isfinite(returns.to_numpy()).all()
        or (returns <= -1).any()
    ):
        raise RuntimeError("Metric NAV/return series is invalid")
    log_returns = np.log1p(returns)
    observations = len(returns)
    years = observations / 252.0
    terminal_multiple = float(nav.iloc[-1] / starting_capital)
    annualized_log_growth = float(log_returns.mean() * 252.0)
    cagr = float(math.exp(annualized_log_growth) - 1.0)
    annualized_volatility = float(returns.std(ddof=1) * math.sqrt(252.0))
    downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
    downside_deviation = float(
        math.sqrt(float(np.mean(downside**2))) * math.sqrt(252.0)
    )
    tail_count = max(1, int(math.ceil(0.05 * observations)))
    expected_shortfall = float(
        np.sort(returns.to_numpy(dtype=float))[:tail_count].mean()
    )
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    annualizer = 1.0 / years
    fills = ledger["security_orders"].astype(float) > 0
    ongoing_fills = fills & ~ledger["initial_deployment"].astype(bool)
    result = {
        "observations": observations,
        "years_252": years,
        "start": nav.index[0].date().isoformat(),
        "end": nav.index[-1].date().isoformat(),
        "starting_nav": starting_capital,
        "ending_nav": float(nav.iloc[-1]),
        "terminal_wealth_multiple": terminal_multiple,
        "cagr": cagr,
        "annualized_log_growth": annualized_log_growth,
        "annualized_volatility": annualized_volatility,
        "downside_deviation": downside_deviation,
        "worst_month": float(monthly.min()) if len(monthly) else None,
        "expected_shortfall_95_daily": expected_shortfall,
        "worst_rolling_3y": _worst_rolling_return(nav, 756),
        "worst_rolling_5y": _worst_rolling_return(nav, 1260),
        "annual_gross_turnover": float(
            ledger["ongoing_gross_trade_fraction"].sum() * annualizer
        ),
        "annual_one_way_turnover": float(
            ledger["one_way_turnover"].sum() * annualizer
        ),
        "ongoing_rebalance_events": int(ongoing_fills.sum()),
        "ongoing_security_orders": int(
            ledger.loc[ongoing_fills, "security_orders"].sum()
        ),
        "total_modeled_cost": float(ledger["transaction_cost"].sum()),
        "annual_modeled_cost_fraction": float(
            ledger["transaction_cost_fraction"].sum() * annualizer
        ),
        "maximum_trade_to_median_dollar_volume20": float(
            ledger["max_trade_to_median_dollar_volume20"].max()
        ),
        "mean_advertised_daily_exposure": float(
            ledger["advertised_daily_exposure"].mean()
        ),
        "qld_dollar_pnl": float(ledger["qld_pnl"].sum()),
        "soxl_dollar_pnl": float(ledger["soxl_pnl"].sum()),
        "cost_dollar_pnl": float(ledger["cost_pnl"].sum()),
    }
    initial_signal_date = ledger["fill_signal_date"].iloc[0]
    starting_date = (
        pd.Timestamp(initial_signal_date)
        if pd.notna(initial_signal_date)
        else pd.Timestamp(nav.index[0]) - pd.Timedelta(nanoseconds=1)
    )
    result.update(
        _drawdown_diagnostics(
            nav,
            starting_capital=starting_capital,
            starting_date=starting_date,
        )
    )
    return result


def _result_log_returns(result: AlphaBacktestResult) -> pd.Series:
    nav = result.ledger["nav"].astype(float)
    starting_capital = float(result.ledger["starting_capital"].iloc[0])
    return pd.concat(
        [
            pd.Series(
                [math.log(float(nav.iloc[0] / starting_capital))],
                index=[nav.index[0]],
            ),
            np.log1p(nav.pct_change(fill_method=None).iloc[1:]),
        ]
    )


def rolling_win_rates(
    candidate: AlphaBacktestResult,
    benchmark: AlphaBacktestResult,
) -> dict[str, object]:
    """Compare all overlapping three- and five-year endpoint windows."""

    c = _result_log_returns(candidate)
    b = _result_log_returns(benchmark)
    if not c.index.equals(b.index):
        raise RuntimeError("Rolling comparison indexes differ")
    difference = c - b
    output: dict[str, object] = {}
    for label, window in (("3y", 756), ("5y", 1260)):
        rolling = difference.rolling(window, min_periods=window).sum().dropna()
        output[label] = {
            "windows": len(rolling),
            "win_rate": float((rolling > 0).mean()) if len(rolling) else None,
            "median_annualized_log_excess": (
                float(rolling.median() * 252.0 / window)
                if len(rolling)
                else None
            ),
            "worst_annualized_log_excess": (
                float(rolling.min() * 252.0 / window)
                if len(rolling)
                else None
            ),
        }
    return output


def chronological_slices(
    candidate: AlphaBacktestResult,
    benchmark: AlphaBacktestResult,
    *,
    slices: int = 4,
) -> list[dict[str, object]]:
    """Return equal-length chronological excess-growth slices."""

    if slices < 2:
        raise ValueError("At least two chronological slices are required")
    c = _result_log_returns(candidate)
    b = _result_log_returns(benchmark)
    if not c.index.equals(b.index):
        raise RuntimeError("Chronological comparison indexes differ")
    rows: list[dict[str, object]] = []
    for number, positions in enumerate(
        np.array_split(np.arange(len(c)), slices),
        start=1,
    ):
        if not len(positions):
            continue
        c_slice = c.iloc[positions]
        b_slice = b.iloc[positions]
        rows.append(
            {
                "slice": number,
                "start": c_slice.index[0].date().isoformat(),
                "end": c_slice.index[-1].date().isoformat(),
                "observations": len(c_slice),
                "candidate_annualized_log_growth": float(c_slice.mean() * 252),
                "benchmark_annualized_log_growth": float(b_slice.mean() * 252),
                "annualized_log_excess": float(
                    (c_slice - b_slice).mean() * 252
                ),
                "positive_excess": bool((c_slice - b_slice).mean() > 0),
            }
        )
    return rows


def _moving_block_means(
    values: np.ndarray,
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    observations = len(values)
    if observations < 2 or not (1 <= block_length <= observations):
        raise ValueError("Moving-block bootstrap dimensions are invalid")
    blocks = int(math.ceil(observations / block_length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0,
        observations - block_length + 1,
        size=(samples, blocks),
    )
    full_blocks, remainder = divmod(observations, block_length)
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    block_sums = cumulative[block_length:] - cumulative[:-block_length]
    totals = np.zeros(samples, dtype=float)
    if full_blocks:
        totals += block_sums[starts[:, :full_blocks]].sum(axis=1)
    if remainder:
        partial_starts = starts[:, full_blocks]
        totals += (
            cumulative[partial_starts + remainder]
            - cumulative[partial_starts]
        )
    return totals / observations


def moving_block_bootstrap(
    candidate: AlphaBacktestResult,
    benchmark: AlphaBacktestResult,
    *,
    samples: int = 20_000,
    block_length: int | None = None,
    seed: int = MASTER_SEED,
) -> dict[str, object]:
    """Deterministic paired interval for annualized relative log growth."""

    c = _result_log_returns(candidate)
    b = _result_log_returns(benchmark)
    if not c.index.equals(b.index):
        raise RuntimeError("Bootstrap comparison indexes differ")
    difference = (c - b).to_numpy(dtype=float)
    selected = (
        max(5, int(round(len(difference) ** (1.0 / 3.0))))
        if block_length is None
        else int(block_length)
    )
    resampled = _moving_block_means(
        difference,
        samples=samples,
        block_length=selected,
        seed=seed,
    )
    observed = float(difference.mean() * 252.0)
    low, high = np.quantile(resampled * 252.0, [0.025, 0.975])
    return {
        "observations": len(difference),
        "samples": samples,
        "block_length": selected,
        "seed": seed,
        "annualized_log_excess": observed,
        "annualized_geometric_excess": float(math.expm1(observed)),
        "ci95_annualized_log_excess": [float(low), float(high)],
        "probability_nonpositive": float(
            (1 + np.count_nonzero(resampled <= 0.0)) / (samples + 1)
        ),
    }


def white_reality_check(
    results: Mapping[str, AlphaBacktestResult],
    *,
    baseline_name: str = "qld_buy_hold",
    samples: int = 10_000,
    block_length: int | None = None,
    seed: int = MASTER_SEED,
) -> dict[str, object]:
    """White-style max-stat bootstrap across the honest frozen trial family."""

    if set(results) != set(FROZEN_CANDIDATE_NAMES):
        raise ValueError(
            "Reality Check requires the complete frozen candidate registry"
        )
    baseline = _result_log_returns(results[baseline_name])
    challenger_names = [
        name for name in FROZEN_CANDIDATE_NAMES if name != baseline_name
    ]
    differences: list[np.ndarray] = []
    for name in challenger_names:
        returns = _result_log_returns(results[name])
        if not returns.index.equals(baseline.index):
            raise RuntimeError("Reality Check return indexes differ")
        differences.append(
            (
                returns - baseline
            ).to_numpy(dtype=float)
        )
    matrix = np.column_stack(differences)
    observations = len(matrix)
    selected = (
        max(5, int(round(observations ** (1.0 / 3.0))))
        if block_length is None
        else int(block_length)
    )
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(observations / selected))
    starts = rng.integers(
        0,
        observations - selected + 1,
        size=(samples, blocks),
    )
    observed_means = matrix.mean(axis=0)
    centered = matrix - observed_means
    bootstrap_max = np.full(samples, -np.inf, dtype=float)
    for column in range(centered.shape[1]):
        cumulative = np.concatenate(
            ([0.0], np.cumsum(centered[:, column], dtype=float))
        )
        block_sums = cumulative[selected:] - cumulative[:-selected]
        full_blocks, remainder = divmod(observations, selected)
        totals = np.zeros(samples, dtype=float)
        if full_blocks:
            totals += block_sums[starts[:, :full_blocks]].sum(axis=1)
        if remainder:
            partial = starts[:, full_blocks]
            totals += cumulative[partial + remainder] - cumulative[partial]
        column_means = totals / observations
        bootstrap_max = np.maximum(bootstrap_max, column_means)
    observed_max = float(observed_means.max())
    best_index = int(np.argmax(observed_means))
    p_value = float(
        (1 + np.count_nonzero(bootstrap_max >= observed_max))
        / (samples + 1)
    )
    return {
        "label": "White Reality Check style centered moving-block max statistic",
        "honest_trial_count": len(FROZEN_CANDIDATE_NAMES),
        "baseline": baseline_name,
        "challengers": challenger_names,
        "best_challenger": challenger_names[best_index],
        "best_annualized_log_excess": observed_max * 252.0,
        "samples": samples,
        "block_length": selected,
        "seed": seed,
        "one_sided_p_value": p_value,
    }


def cscv_pbo(
    results: Mapping[str, AlphaBacktestResult],
    *,
    blocks: int = 8,
) -> dict[str, object]:
    """CSCV-style probability-of-backtest-overfitting instability diagnostic."""

    if blocks <= 0 or blocks % 2:
        raise ValueError("CSCV block count must be positive and even")
    if set(results) != set(FROZEN_CANDIDATE_NAMES):
        raise ValueError("PBO requires the complete honest candidate family")
    names = list(FROZEN_CANDIDATE_NAMES)
    returns = {name: _result_log_returns(results[name]) for name in names}
    reference = returns[names[0]].index
    if any(not values.index.equals(reference) for values in returns.values()):
        raise RuntimeError("PBO return indexes differ")
    if len(reference) < 63 * blocks:
        return {
            "available": False,
            "reason": f"At least {63 * blocks} common observations are required",
        }
    block_positions = np.array_split(np.arange(len(reference)), blocks)
    matrix = np.column_stack([returns[name].to_numpy(dtype=float) for name in names])
    all_blocks = set(range(blocks))
    logits: list[float] = []
    selections = {name: 0 for name in names}
    for selected_blocks in itertools.combinations(range(blocks), blocks // 2):
        in_blocks = set(selected_blocks)
        out_blocks = all_blocks - in_blocks
        in_positions = np.concatenate(
            [block_positions[item] for item in sorted(in_blocks)]
        )
        out_positions = np.concatenate(
            [block_positions[item] for item in sorted(out_blocks)]
        )
        in_scores = matrix[in_positions].mean(axis=0)
        selected_index = int(np.argmax(in_scores))
        selections[names[selected_index]] += 1
        out_scores = matrix[out_positions].mean(axis=0)
        selected_score = float(out_scores[selected_index])
        less = int(np.count_nonzero(out_scores < selected_score - 1e-15))
        equal = int(
            np.count_nonzero(
                np.isclose(out_scores, selected_score, rtol=0, atol=1e-15)
            )
        )
        rank = 1.0 + less + (equal - 1.0) / 2.0
        normalized = rank / (len(names) + 1.0)
        logits.append(math.log(normalized / (1.0 - normalized)))
    failures = sum(value <= 0 for value in logits)
    return {
        "available": True,
        "label": "CSCV-style selection-instability diagnostic",
        "honest_trial_count": len(names),
        "blocks": blocks,
        "splits": len(logits),
        "failures": failures,
        "pbo": failures / len(logits),
        "in_sample_selections": selections,
        "caveat": (
            "Dependent interleaved splits diagnose historical instability; "
            "they are not chronological out-of-sample proof."
        ),
    }


def deflated_sharpe_diagnostic(
    results: Mapping[str, AlphaBacktestResult],
) -> dict[str, object]:
    """Approximate non-normal deflated-Sharpe probabilities for honest trials."""

    if set(results) != set(FROZEN_CANDIDATE_NAMES):
        raise ValueError("Deflated Sharpe requires the honest frozen family")
    trial_count = len(FROZEN_CANDIDATE_NAMES)
    normal = NormalDist()
    gamma = 0.5772156649015329
    rows: dict[str, object] = {}
    for name in FROZEN_CANDIDATE_NAMES:
        values = np.expm1(
            _result_log_returns(results[name]).to_numpy(dtype=float)
        )
        observations = len(values)
        standard_deviation = float(values.std(ddof=1))
        if observations < 3 or standard_deviation <= 0:
            rows[name] = {"available": False}
            continue
        sharpe = float(values.mean() / standard_deviation)
        centered = (values - values.mean()) / standard_deviation
        skewness = float(np.mean(centered**3))
        kurtosis = float(np.mean(centered**4))
        variance = (
            1.0
            - skewness * sharpe
            + ((kurtosis - 1.0) / 4.0) * sharpe**2
        ) / max(1, observations - 1)
        sharpe_error = math.sqrt(max(variance, 1e-18))
        expected_max = sharpe_error * (
            (1.0 - gamma) * normal.inv_cdf(1.0 - 1.0 / trial_count)
            + gamma
            * normal.inv_cdf(
                1.0 - 1.0 / (trial_count * math.e)
            )
        )
        probability = normal.cdf((sharpe - expected_max) / sharpe_error)
        rows[name] = {
            "available": True,
            "daily_sharpe": sharpe,
            "annualized_sharpe": sharpe * math.sqrt(252.0),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "estimated_sharpe_error": sharpe_error,
            "expected_maximum_daily_sharpe_under_trials": expected_max,
            "deflated_sharpe_probability": probability,
        }
    return {
        "label": "Approximate non-normal deflated-Sharpe diagnostic",
        "honest_trial_count": trial_count,
        "strategies": rows,
        "caveat": (
            "This diagnostic depends on moment and independent-trial "
            "approximations; the frozen strategies are correlated."
        ),
    }


def _project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto nonnegative weights summing to one."""

    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    candidates = ordered - (cumulative - 1.0) / (
        np.arange(len(values), dtype=float) + 1.0
    )
    positive = np.flatnonzero(candidates > 0)
    if not len(positive):
        return np.full(len(values), 1.0 / len(values))
    rho = int(positive[-1])
    theta = (cumulative[rho] - 1.0) / (rho + 1.0)
    return np.maximum(values - theta, 0.0)


def _constrained_style_fit(
    y: np.ndarray,
    x: np.ndarray,
) -> tuple[float, np.ndarray, float, float]:
    centered_x = x - x.mean(axis=0)
    centered_y = y - y.mean()
    spectral = float(np.linalg.norm(centered_x, ord=2) ** 2)
    step = 1.0 / max(spectral, 1e-12)
    weights = np.full(x.shape[1], 1.0 / x.shape[1])
    for _ in range(2_000):
        gradient = centered_x.T @ (centered_x @ weights - centered_y)
        revised = _project_simplex(weights - step * gradient)
        if np.max(np.abs(revised - weights)) <= 1e-12:
            weights = revised
            break
        weights = revised
    intercept = float(y.mean() - x.mean(axis=0) @ weights)
    residual = y - intercept - x @ weights
    residual_volatility = float(residual.std(ddof=x.shape[1] + 1) * math.sqrt(252))
    total = float((y - y.mean()) @ (y - y.mean()))
    r_squared = (
        float(1.0 - (residual @ residual) / total) if total > 0 else 0.0
    )
    return intercept, weights, residual_volatility, r_squared


def rolling_style_attribution(
    result: AlphaBacktestResult,
    data: AlphaMarketData,
    *,
    window: int = 756,
    step: int = 21,
) -> pd.DataFrame:
    """Rolling constrained return-based style attribution to QLD/SOXL/SPY."""

    strategy = result.ledger["nav"].pct_change(fill_method=None)
    factors = data.closes.loc[
        strategy.index,
        ["QLD", "SOXL", "SPY"],
    ].pct_change(fill_method=None)
    joined = pd.concat([strategy.rename("strategy"), factors], axis=1).dropna()
    rows: list[dict[str, object]] = []
    for end in range(window, len(joined) + 1, step):
        sample = joined.iloc[end - window : end]
        intercept, weights, residual_vol, r_squared = _constrained_style_fit(
            sample["strategy"].to_numpy(dtype=float),
            sample[["QLD", "SOXL", "SPY"]].to_numpy(dtype=float),
        )
        rows.append(
            {
                "session": pd.Timestamp(sample.index[-1]),
                "window_start": sample.index[0],
                "observations": len(sample),
                "annualized_intercept": intercept * 252.0,
                "qld_exposure": float(weights[0]),
                "soxl_exposure": float(weights[1]),
                "spy_exposure": float(weights[2]),
                "residual_volatility": residual_vol,
                "r_squared": r_squared,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=(
                "window_start",
                "observations",
                "annualized_intercept",
                "qld_exposure",
                "soxl_exposure",
                "spy_exposure",
                "residual_volatility",
                "r_squared",
            )
        )
    return pd.DataFrame(rows).set_index("session")


def return_contribution(
    candidate: AlphaBacktestResult,
    static_benchmark: AlphaBacktestResult,
) -> dict[str, object]:
    """Report exact dollar components plus timing's relative log-growth effect."""

    c = _result_log_returns(candidate)
    static = _result_log_returns(static_benchmark)
    if not c.index.equals(static.index):
        raise RuntimeError("Contribution comparison indexes differ")
    ledger = candidate.ledger
    return {
        "qld_core_dollar_pnl": float(ledger["qld_pnl"].sum()),
        "soxl_sleeve_dollar_pnl": float(ledger["soxl_pnl"].sum()),
        "modeled_cost_dollar_pnl": float(ledger["cost_pnl"].sum()),
        "timing_annualized_log_growth_vs_static_65_35": float(
            (c - static).mean() * 252.0
        ),
        "identity_caveat": (
            "Dollar P&L components reconcile holdings and costs; timing is a "
            "counterfactual log-growth comparison and is not added to them."
        ),
    }


def evaluate_frozen_family(
    data: AlphaMarketData,
    *,
    costs_bps: Sequence[float] = DEFAULT_COSTS_BPS,
    bootstrap_samples: int = 10_000,
) -> tuple[
    SignalPanel,
    dict[tuple[str, float], AlphaBacktestResult],
    dict[str, object],
]:
    """Run every declared candidate/cost case and build the audit report."""

    validate_frozen_history_boundary(data)
    if tuple(float(value) for value in costs_bps) != DEFAULT_COSTS_BPS:
        raise ValueError(
            f"Confirmatory costs must be exactly {list(DEFAULT_COSTS_BPS)}"
        )
    panel = build_signal_panel(data)
    scoring_start = common_scoring_start(data, panel)
    results: dict[tuple[str, float], AlphaBacktestResult] = {}
    for name in FROZEN_CANDIDATE_NAMES:
        for cost_bps in DEFAULT_COSTS_BPS:
            results[(name, cost_bps)] = simulate_candidate(
                data,
                panel,
                name,
                cost_bps=cost_bps,
                scoring_start=scoring_start,
            )
    primary = {
        name: results[(name, PRIMARY_COST_BPS)]
        for name in FROZEN_CANDIDATE_NAMES
    }
    qld = primary["qld_buy_hold"]
    static = primary["static_65_35"]
    comparisons: dict[str, object] = {}
    for name in FROZEN_CANDIDATE_NAMES:
        if name in {"qld_buy_hold", "static_65_35"}:
            continue
        candidate = primary[name]
        comparisons[name] = {
            "rolling_vs_qld": rolling_win_rates(candidate, qld),
            "rolling_vs_static": rolling_win_rates(candidate, static),
            "slices_vs_qld": chronological_slices(candidate, qld),
            "slices_vs_static": chronological_slices(candidate, static),
            "bootstrap_vs_qld": moving_block_bootstrap(
                candidate,
                qld,
                samples=bootstrap_samples,
                seed=MASTER_SEED,
            ),
            "bootstrap_vs_static": moving_block_bootstrap(
                candidate,
                static,
                samples=bootstrap_samples,
                seed=MASTER_SEED + 1,
            ),
            "return_contribution": return_contribution(candidate, static),
        }
    reality_vs_qld = white_reality_check(
        primary,
        baseline_name="qld_buy_hold",
        samples=bootstrap_samples,
        seed=MASTER_SEED + 2,
    )
    reality_vs_static = white_reality_check(
        primary,
        baseline_name="static_65_35",
        samples=bootstrap_samples,
        seed=MASTER_SEED + 3,
    )
    residual_available_dates = panel.frame.index[
        panel.frame["residual_available"].astype(bool)
    ]
    if not len(residual_available_dates):
        raise RuntimeError("Residual candidate has no valid 189-return warm-up")
    residual_signal_start = pd.Timestamp(residual_available_dates[0])
    residual_signal_position = data.sessions.get_loc(residual_signal_start)
    if (
        not isinstance(residual_signal_position, (int, np.integer))
        or int(residual_signal_position) + 1 >= len(data.sessions)
    ):
        raise RuntimeError("No execution session follows residual warm-up")
    residual_sensitivity_start = pd.Timestamp(
        data.sessions[int(residual_signal_position) + 1]
    )
    sensitivity_results: dict[tuple[str, float], AlphaBacktestResult] = {}
    sensitivity_names = ("qld_buy_hold", "static_65_35", "residual_35")
    for name in sensitivity_names:
        for cost_bps in DEFAULT_COSTS_BPS:
            sensitivity_results[(name, cost_bps)] = simulate_candidate(
                data,
                panel,
                name,
                cost_bps=cost_bps,
                scoring_start=residual_sensitivity_start,
            )
    sensitivity_residual = sensitivity_results[
        ("residual_35", PRIMARY_COST_BPS)
    ]
    sensitivity_qld = sensitivity_results[
        ("qld_buy_hold", PRIMARY_COST_BPS)
    ]
    sensitivity_static = sensitivity_results[
        ("static_65_35", PRIMARY_COST_BPS)
    ]
    earlier_inception_sensitivity: dict[str, object] = {
        "label": (
            "Per-strategy earlier-inception sensitivity; not the frozen-family "
            "common scoring interval"
        ),
        "residual_signal_warmup": (
            f"{ESTIMATION_RETURNS}+{SCORING_RETURNS} daily log returns"
        ),
        "trend_warmup": (
            f"{TREND_SESSIONS} closes; unavailable trend fails closed"
        ),
        "first_residual_signal_date": residual_signal_start.date().isoformat(),
        "first_execution_date": residual_sensitivity_start.date().isoformat(),
        "results": {
            name: {
                f"{cost:g}bps": sensitivity_results[(name, cost)].metrics
                for cost in DEFAULT_COSTS_BPS
            }
            for name in sensitivity_names
        },
        "residual_35_vs_qld_at_10bps": {
            "rolling": rolling_win_rates(
                sensitivity_residual,
                sensitivity_qld,
            ),
            "chronological_slices": chronological_slices(
                sensitivity_residual,
                sensitivity_qld,
            ),
            "moving_block_bootstrap": moving_block_bootstrap(
                sensitivity_residual,
                sensitivity_qld,
                samples=bootstrap_samples,
                seed=MASTER_SEED + 10,
            ),
        },
        "residual_35_vs_static_at_10bps": {
            "rolling": rolling_win_rates(
                sensitivity_residual,
                sensitivity_static,
            ),
            "chronological_slices": chronological_slices(
                sensitivity_residual,
                sensitivity_static,
            ),
            "moving_block_bootstrap": moving_block_bootstrap(
                sensitivity_residual,
                sensitivity_static,
                samples=bootstrap_samples,
                seed=MASTER_SEED + 11,
            ),
        },
    }
    style: dict[str, object] = {}
    for name, result in primary.items():
        attribution = rolling_style_attribution(result, data)
        style[name] = {
            "windows": len(attribution),
            "latest": (
                {
                    key: (
                        value.isoformat()
                        if isinstance(value, pd.Timestamp)
                        else float(value)
                        if isinstance(value, (float, np.floating))
                        else int(value)
                        if isinstance(value, (int, np.integer))
                        else value
                    )
                    for key, value in attribution.iloc[-1].to_dict().items()
                }
                if len(attribution)
                else None
            ),
        }
    promotion_evaluation: dict[str, object] = {}
    for name in FROZEN_CANDIDATE_NAMES:
        if name in {"qld_buy_hold", "static_65_35", "ridge_shadow"}:
            continue
        at_10 = results[(name, 10.0)].metrics["annualized_log_growth"]
        qld_10 = results[("qld_buy_hold", 10.0)].metrics[
            "annualized_log_growth"
        ]
        static_10 = results[("static_65_35", 10.0)].metrics[
            "annualized_log_growth"
        ]
        at_25 = results[(name, 25.0)].metrics["annualized_log_growth"]
        qld_25 = results[("qld_buy_hold", 25.0)].metrics[
            "annualized_log_growth"
        ]
        static_25 = results[("static_65_35", 25.0)].metrics[
            "annualized_log_growth"
        ]
        slices_vs_static = comparisons[name]["slices_vs_static"]
        positive_slices = sum(
            bool(item["positive_excess"]) for item in slices_vs_static
        )
        historical_numeric_gates = bool(
            at_10 > qld_10
            and at_10 > static_10
            and at_25 > qld_25
            and at_25 > static_25
        )
        promotion_evaluation[name] = {
            "positive_10bps_log_growth_vs_qld": bool(at_10 > qld_10),
            "positive_10bps_log_growth_vs_static_65_35": bool(
                at_10 > static_10
            ),
            "no_reversal_at_25bps_vs_qld": bool(at_25 > qld_25),
            "no_reversal_at_25bps_vs_static_65_35": bool(at_25 > static_25),
            "positive_chronological_slices_vs_static": positive_slices,
            "historical_numeric_gates_met": historical_numeric_gates,
            "future_shadow_requirement_met": False,
            "promotion_complete": False,
        }
    report: dict[str, object] = {
        "schema_version": ALPHA_RESEARCH_SCHEMA_VERSION,
        "research_only": True,
        "contamination_warning": (
            "The residual-momentum idea and portions of this history were "
            "viewed before the protocol was frozen."
        ),
        "data": {
            "source": data.source,
            "requested_start": data.requested_start,
            "requested_end_exclusive": data.requested_end_exclusive,
            "actual_common_start": data.common_start.date().isoformat(),
            "final_session": data.final_session.date().isoformat(),
            "common_scoring_start": scoring_start.date().isoformat(),
            "sha256": data.fingerprint,
        },
        "configuration": FROZEN_CONFIGURATION,
        "family_fingerprint": FROZEN_FAMILY_FINGERPRINT,
        "candidate_fingerprints": {
            name: candidate_fingerprint(name)
            for name in FROZEN_CANDIDATE_NAMES
        },
        "ridge_challenger_fingerprint": RIDGE_CHALLENGER_FINGERPRINT,
        "software_fingerprint": software_fingerprint(),
        "results": {
            name: {
                f"{cost:g}bps": results[(name, cost)].metrics
                for cost in DEFAULT_COSTS_BPS
            }
            for name in FROZEN_CANDIDATE_NAMES
        },
        "comparisons_at_10bps": comparisons,
        "white_reality_checks": {
            "vs_qld_buy_hold": reality_vs_qld,
            "vs_static_65_35": reality_vs_static,
        },
        "cscv_pbo": cscv_pbo(primary),
        "deflated_sharpe": deflated_sharpe_diagnostic(primary),
        "rolling_style_attribution": style,
        "earlier_inception_sensitivity": earlier_inception_sensitivity,
        "ridge_audit_latest": (
            asdict(panel.ridge_details[max(panel.ridge_details)])
            if panel.ridge_details
            else None
        ),
        "ridge_audit_history": {
            date.date().isoformat(): asdict(estimate)
            for date, estimate in sorted(panel.ridge_details.items())
        },
        "promotion_rule": {
            "positive_10bps_growth_vs_qld_and_static": "required",
            "no_reversal_at_25bps": "required",
            "acceptable_liquidity": "required",
            "no_single_chronological_slice_carries_conclusion": "required",
            "future_shadow_sessions_required": 252,
            "future_structural_decisions_required": 3,
        },
        "promotion_evaluation": promotion_evaluation,
    }
    return panel, results, report


def _json_clean(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json_atomic(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_clean(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _default_end_exclusive() -> str:
    return (
        pd.Timestamp.now(tz=NEW_YORK).normalize() + pd.Timedelta(days=1)
    ).date().isoformat()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen, research-only QLD/SOXL alpha protocol. "
            "This command never touches production state or email."
        )
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=_default_end_exclusive())
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Load an exact five-ticker long-form snapshot instead of downloading",
    )
    parser.add_argument(
        "--save-snapshot",
        type=Path,
        help="Atomically preserve the validated adjusted input snapshot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research_outputs") / "alpha_results.json",
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        help="Optional directory for per-candidate/cost research ledgers",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap-samples must be at least 100")
    if args.snapshot is not None:
        data = load_market_snapshot(
            args.snapshot,
            requested_start=args.start,
            requested_end_exclusive=args.end,
        )
    else:
        data = download_market_data(
            requested_start=args.start,
            requested_end_exclusive=args.end,
        )
    if args.save_snapshot is not None:
        save_market_snapshot(data, args.save_snapshot)
    _, results, report = evaluate_frozen_family(
        data,
        bootstrap_samples=args.bootstrap_samples,
    )
    save_json_atomic(report, args.output)
    if args.ledger_dir is not None:
        args.ledger_dir.mkdir(parents=True, exist_ok=True)
        for (name, cost), result in results.items():
            path = args.ledger_dir / f"{name}_{cost:g}bps.csv"
            result.ledger.to_csv(
                path,
                index=True,
                date_format="%Y-%m-%d",
                float_format="%.12g",
                lineterminator="\n",
            )
    primary = report["results"]
    for name in FROZEN_CANDIDATE_NAMES:
        metrics = primary[name]["10bps"]
        print(
            f"{name:26s} CAGR={metrics['cagr']:8.2%} "
            f"log-growth={metrics['annualized_log_growth']:8.2%} "
            f"maxDD={metrics['maximum_drawdown']:8.2%}"
        )
    print(f"Saved research report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

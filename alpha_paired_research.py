#!/usr/bin/env python3
"""Paired research orchestration for the frozen alpha family and Original.

Both engines receive views of one exact adjusted Open/Close/Volume snapshot.
The current seven-candidate family remains the complete confirmatory trial
family.  The frozen Original ROTH IRA Barbell is an external historical
baseline and is never added to White Reality Check, CSCV/PBO, or deflated
Sharpe trial counts.

This module is research-only.  It has no production-state, broker, email, or
notification integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd

import alpha_research as alpha
import legacy_original_research as legacy


PAIRED_RESEARCH_SCHEMA_VERSION = 1
UNION_TICKERS = (
    "QQQ",
    "SMH",
    "QLD",
    "SOXL",
    "SPY",
    "TECL",
    "SPMO",
    "GLD",
)
CONFIRMATORY_COSTS_BPS = (0.0, 10.0, 25.0, 50.0)
PAIRWISE_COSTS_BPS = (10.0, 25.0)
DEFAULT_START = alpha.DEFAULT_START
MASTER_SEED = alpha.MASTER_SEED + 50_000
NEW_YORK = ZoneInfo("America/New_York")
MARKET_CLOSE_BUFFER_MINUTES = alpha.MARKET_CLOSE_BUFFER_MINUTES

if UNION_TICKERS != legacy.PAIRED_TICKERS:
    raise RuntimeError("Paired union and frozen legacy union definitions differ")
if CONFIRMATORY_COSTS_BPS != alpha.DEFAULT_COSTS_BPS:
    raise RuntimeError("Paired and frozen-family cost grids differ")


def _canonical_sha256(payload: object) -> str:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


PAIRED_CONFIGURATION = {
    "schema_version": PAIRED_RESEARCH_SCHEMA_VERSION,
    "union_tickers": UNION_TICKERS,
    "confirmatory_costs_bps": CONFIRMATORY_COSTS_BPS,
    "pairwise_costs_bps": PAIRWISE_COSTS_BPS,
    "current_family": alpha.FROZEN_CANDIDATE_NAMES,
    "external_baseline": legacy.LEGACY_CANDIDATE.name,
    "external_baseline_in_trial_count": False,
    "comparison_direction": "current candidate minus Original",
    "execution": "completed close t -> adjusted open t+1",
    "market_close_buffer_minutes": MARKET_CLOSE_BUFFER_MINUTES,
    "snapshot_float_format": "%.17g",
    "snapshot_parser": "pandas float_precision=round_trip",
}
PAIRED_CONFIGURATION_FINGERPRINT = _canonical_sha256(PAIRED_CONFIGURATION)


@dataclass(frozen=True)
class UnionMarketData:
    """One exact adjusted union snapshot supplied to both frozen engines."""

    opens: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame
    sessions: pd.DatetimeIndex
    final_session: pd.Timestamp
    fingerprint: str
    requested_start: str
    requested_end_exclusive: str
    source: str


@dataclass(frozen=True)
class PairedEngineViews:
    """Validated bit-identical projections of the exact union snapshot."""

    union: UnionMarketData
    current: alpha.AlphaMarketData
    original: legacy.LegacyMarketData
    verification_fingerprint: str


@dataclass(frozen=True)
class PairedResearchResult:
    """Complete in-memory output of the paired research run."""

    views: PairedEngineViews
    panel: alpha.SignalPanel
    current_results: dict[tuple[str, float], alpha.AlphaBacktestResult]
    original_own_history: dict[float, legacy.LegacyBacktestResult]
    original_shared_interval: dict[float, legacy.LegacyBacktestResult]
    report: dict[str, object]


def _normalize_union_frame(
    frame: pd.DataFrame,
    field_name: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError(f"{field_name} union frame is empty")
    result = frame.copy()
    try:
        index = pd.DatetimeIndex(pd.to_datetime(result.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} index is not datetime-like") from exc
    if index.tz is not None:
        index = index.tz_convert(None)
    index = index.normalize()
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise RuntimeError(
            f"{field_name} union sessions are duplicated or unsorted"
        )
    result.index = index
    result.columns = [str(column).upper() for column in result.columns]
    if result.columns.duplicated().any():
        raise RuntimeError(f"{field_name} has duplicate ticker columns")
    missing = sorted(set(UNION_TICKERS) - set(result.columns))
    if missing:
        raise RuntimeError(
            f"{field_name} is missing paired union tickers: {missing}"
        )
    try:
        result = result.loc[:, list(UNION_TICKERS)].astype(float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{field_name} contains nonnumeric union observations"
        ) from exc
    result.index.name = "Session"
    result.columns.name = "Ticker"
    return result


def _extract_union_field(raw: pd.DataFrame, field_name: str) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise RuntimeError("Union market response is empty")
    if not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError(
            "Union market response must have Field/Ticker MultiIndex columns"
        )
    first = set(str(value) for value in raw.columns.get_level_values(0))
    second = set(str(value) for value in raw.columns.get_level_values(1))
    if field_name in first:
        frame = raw[field_name]
    elif field_name in second:
        frame = raw.xs(field_name, axis=1, level=1)
    else:
        raise RuntimeError(f"Union market response is missing {field_name}")
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    return _normalize_union_frame(frame, field_name)


def _calendar_sessions(
    start: pd.Timestamp | str,
    end_inclusive: pd.Timestamp | str,
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
    if not len(sessions):
        raise RuntimeError("XNYS calendar returned no session before paired end")
    return pd.Timestamp(sessions[-1]).normalize()


def _expected_completed_session() -> pd.Timestamp:
    return alpha._expected_completed_session()


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


def validate_union_frames(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    requested_start: str,
    requested_end_exclusive: str,
    source: str = "provided adjusted paired union",
) -> UnionMarketData:
    """Validate exact union frames without filling, dropping, or substitution."""

    normalized_opens = _normalize_union_frame(opens, "Open")
    normalized_closes = _normalize_union_frame(closes, "Close")
    normalized_volumes = _normalize_union_frame(volumes, "Volume")
    if not normalized_opens.index.equals(normalized_closes.index):
        raise RuntimeError("Union Open and Close indexes differ")
    if not normalized_opens.index.equals(normalized_volumes.index):
        raise RuntimeError("Union Open and Volume indexes differ")

    final_session = _last_session_before(requested_end_exclusive)
    today = pd.Timestamp.now(tz=NEW_YORK).tz_localize(None).normalize()
    if final_session >= today:
        completed = _expected_completed_session()
        if final_session > completed:
            raise RuntimeError(
                "Paired union includes an incomplete XNYS session: "
                f"requested={final_session.date().isoformat()}, "
                f"completed={completed.date().isoformat()}"
            )
    expected = _calendar_sessions(requested_start, final_session)
    if not normalized_opens.index.equals(expected):
        missing = expected.difference(normalized_opens.index)
        unexpected = normalized_opens.index.difference(expected)
        raise RuntimeError(
            "Paired union session continuity failed: "
            f"missing={[value.date().isoformat() for value in missing[:5]]}, "
            f"non_sessions="
            f"{[value.date().isoformat() for value in unexpected[:5]]}"
        )

    # This performs the frozen Original's inception, warm-up, and scored-data
    # validation against these exact arrays.  It does not mutate them.
    legacy.validate_union_market_data(
        normalized_opens,
        normalized_closes,
        normalized_volumes,
        source=source,
    )
    return UnionMarketData(
        opens=normalized_opens,
        closes=normalized_closes,
        volumes=normalized_volumes,
        sessions=normalized_opens.index,
        final_session=final_session,
        fingerprint=_full_data_fingerprint(
            normalized_opens,
            normalized_closes,
            normalized_volumes,
        ),
        requested_start=str(requested_start),
        requested_end_exclusive=str(requested_end_exclusive),
        source=str(source),
    )


def prepare_union_market_data(
    raw: pd.DataFrame,
    *,
    requested_start: str,
    requested_end_exclusive: str,
    source: str = "provided adjusted paired OHLCV",
) -> UnionMarketData:
    """Extract and validate one exact eight-ticker adjusted response."""

    return validate_union_frames(
        _extract_union_field(raw, "Open"),
        _extract_union_field(raw, "Close"),
        _extract_union_field(raw, "Volume"),
        requested_start=requested_start,
        requested_end_exclusive=requested_end_exclusive,
        source=source,
    )


def download_union_market_data(
    *,
    requested_start: str,
    requested_end_exclusive: str,
) -> UnionMarketData:
    """Download the exact paired union once through yfinance."""

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required for a live paired download") from exc
    raw = yf.download(
        list(UNION_TICKERS),
        start=requested_start,
        end=requested_end_exclusive,
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
        threads=False,
    )
    return prepare_union_market_data(
        raw,
        requested_start=requested_start,
        requested_end_exclusive=requested_end_exclusive,
        source="Yahoo Finance via yfinance; auto_adjust=True; one union request",
    )


def save_union_snapshot(data: UnionMarketData, path: Path) -> None:
    """Atomically save every union value with ``%.17g`` precision."""

    if data.fingerprint != _full_data_fingerprint(
        data.opens,
        data.closes,
        data.volumes,
    ):
        raise ValueError("Union fingerprint does not match the snapshot frames")
    rows: list[pd.DataFrame] = []
    session_values = data.sessions.to_numpy()
    ticker_values = np.asarray(UNION_TICKERS, dtype=object)
    for field_name, frame in (
        ("Open", data.opens),
        ("Close", data.closes),
        ("Volume", data.volumes),
    ):
        rows.append(
            pd.DataFrame(
                {
                    "Session": np.repeat(session_values, len(UNION_TICKERS)),
                    "Ticker": np.tile(ticker_values, len(data.sessions)),
                    "Field": field_name,
                    "Value": frame.loc[
                        data.sessions, list(UNION_TICKERS)
                    ].to_numpy(dtype=float).reshape(-1),
                }
            )
        )
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


def load_union_snapshot(
    path: Path,
    *,
    requested_start: str,
    requested_end_exclusive: str,
) -> UnionMarketData:
    """Load with round-trip float parsing and revalidate the exact union."""

    try:
        snapshot = pd.read_csv(
            path,
            parse_dates=["Session"],
            float_precision="round_trip",
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Unable to read paired union snapshot: {path}") from exc
    required = {"Session", "Ticker", "Field", "Value"}
    if set(snapshot.columns) != required:
        raise RuntimeError(
            f"Paired snapshot columns must be exactly {sorted(required)}"
        )
    if snapshot.duplicated(["Session", "Ticker", "Field"]).any():
        raise RuntimeError("Paired snapshot contains duplicate observations")
    if set(snapshot["Ticker"]) != set(UNION_TICKERS):
        raise RuntimeError("Paired snapshot does not contain the exact union")
    if set(snapshot["Field"]) != {"Open", "Close", "Volume"}:
        raise RuntimeError("Paired snapshot must contain Open, Close, and Volume")

    frames: dict[str, pd.DataFrame] = {}
    for field_name in ("Open", "Close", "Volume"):
        selected = snapshot.loc[
            snapshot["Field"] == field_name,
            ["Session", "Ticker", "Value"],
        ]
        frame = selected.pivot(
            index="Session",
            columns="Ticker",
            values="Value",
        )
        frames[field_name] = frame.reindex(columns=list(UNION_TICKERS))
    return validate_union_frames(
        frames["Open"],
        frames["Close"],
        frames["Volume"],
        requested_start=requested_start,
        requested_end_exclusive=requested_end_exclusive,
        source=f"frozen paired union snapshot: {path.name}",
    )


def _alpha_raw_view(data: UnionMarketData) -> pd.DataFrame:
    fields = {
        field_name: frame.loc[:, list(alpha.TICKERS)].copy()
        for field_name, frame in (
            ("Open", data.opens),
            ("Close", data.closes),
            ("Volume", data.volumes),
        )
    }
    raw = pd.concat(fields, axis=1)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    return raw


def _assert_float_frames_bit_identical(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    description: str,
) -> None:
    if not expected.index.equals(actual.index):
        raise RuntimeError(f"{description} indexes differ")
    if list(expected.columns) != list(actual.columns):
        raise RuntimeError(f"{description} columns differ")
    left = expected.to_numpy(dtype=np.float64)
    right = actual.to_numpy(dtype=np.float64)
    left_nan = np.isnan(left)
    right_nan = np.isnan(right)
    if not np.array_equal(left_nan, right_nan):
        raise RuntimeError(f"{description} missing-value masks differ")
    finite = ~left_nan
    if not np.array_equal(
        left[finite].view(np.uint64),
        right[finite].view(np.uint64),
    ):
        raise RuntimeError(f"{description} finite float bits differ")


def create_engine_views(data: UnionMarketData) -> PairedEngineViews:
    """Create both validated views and prove their overlap is identical."""

    if data.fingerprint != _full_data_fingerprint(
        data.opens,
        data.closes,
        data.volumes,
    ):
        raise ValueError("Union fingerprint does not match its frames")
    current = alpha.prepare_market_data(
        _alpha_raw_view(data),
        requested_start=data.requested_start,
        requested_end_exclusive=data.requested_end_exclusive,
        source=f"{data.source}; exact current five-ticker projection",
    )
    original = legacy.validate_union_market_data(
        data.opens,
        data.closes,
        data.volumes,
        source=f"{data.source}; exact Original union projection",
    )

    for field_name, union_frame, current_frame, original_frame in (
        ("Open", data.opens, current.opens, original.opens),
        ("Close", data.closes, current.closes, original.closes),
        ("Volume", data.volumes, current.volumes, original.volumes),
    ):
        _assert_float_frames_bit_identical(
            union_frame.loc[:, list(alpha.TICKERS)],
            current_frame,
            description=f"Union/current {field_name}",
        )
        _assert_float_frames_bit_identical(
            union_frame.loc[:, list(UNION_TICKERS)],
            original_frame,
            description=f"Union/Original {field_name}",
        )
        _assert_float_frames_bit_identical(
            current_frame,
            original_frame.loc[:, list(alpha.TICKERS)],
            description=f"Current/Original overlapping {field_name}",
        )

    verification_fingerprint = _canonical_sha256(
        {
            "union_fingerprint": data.fingerprint,
            "current_view_fingerprint": current.fingerprint,
            "original_view_fingerprint": original.fingerprint,
            "current_columns": alpha.TICKERS,
            "original_columns": UNION_TICKERS,
            "fields": ("Open", "Close", "Volume"),
            "verification": "IEEE-754 finite bits and NaN masks identical",
        }
    )
    return PairedEngineViews(
        union=data,
        current=current,
        original=original,
        verification_fingerprint=verification_fingerprint,
    )


def external_baseline_trial_audit(
    frozen_family_report: Mapping[str, object],
) -> dict[str, object]:
    """Verify that Original was not inserted into multiplicity diagnostics."""

    expected = len(alpha.FROZEN_CANDIDATE_NAMES)
    reality = frozen_family_report.get("white_reality_checks")
    if not isinstance(reality, Mapping) or not reality:
        raise RuntimeError("Frozen-family report lacks White Reality Checks")
    observed: dict[str, int] = {}
    for label, payload in reality.items():
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Reality Check {label} is malformed")
        count = int(payload.get("honest_trial_count", -1))
        if count != expected:
            raise RuntimeError(
                f"Reality Check {label} trial count is {count}; expected {expected}"
            )
        challengers = set(str(value) for value in payload.get("challengers", ()))
        if legacy.LEGACY_CANDIDATE.name in challengers:
            raise RuntimeError("Original was included in Reality Check challengers")
        observed[str(label)] = count

    for label in ("cscv_pbo", "deflated_sharpe"):
        payload = frozen_family_report.get(label)
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Frozen-family report lacks {label}")
        if bool(payload.get("available", True)):
            count = int(payload.get("honest_trial_count", -1))
            if count != expected:
                raise RuntimeError(
                    f"{label} trial count is {count}; expected {expected}"
                )
            observed[label] = count
    return {
        "frozen_current_family_trial_count": expected,
        "frozen_current_candidates": list(alpha.FROZEN_CANDIDATE_NAMES),
        "external_baselines": [legacy.LEGACY_CANDIDATE.name],
        "external_baseline_in_trial_count": False,
        "external_baseline_in_reality_check": False,
        "verified_trial_counts": observed,
        "interpretation": (
            "Original is a historical external comparator, not an additional "
            "candidate selected from the frozen current-family search."
        ),
    }


def verify_shared_date_alignment(
    current_results: Mapping[
        tuple[str, float],
        alpha.AlphaBacktestResult,
    ],
    original_results: Mapping[float, legacy.LegacyBacktestResult],
    *,
    costs_bps: Sequence[float] = CONFIRMATORY_COSTS_BPS,
) -> dict[str, object]:
    """Require every shared result to have the exact same execution dates."""

    selected_costs = tuple(float(value) for value in costs_bps)
    reference_index: pd.DatetimeIndex | None = None
    for cost in selected_costs:
        try:
            original = original_results[cost]
        except KeyError as exc:
            raise RuntimeError(f"Missing shared Original result at {cost:g}bps") from exc
        original_index = pd.DatetimeIndex(original.ledger.index)
        if reference_index is None:
            reference_index = original_index
        elif not original_index.equals(reference_index):
            raise RuntimeError("Shared Original cost cases use different dates")
        for name in alpha.FROZEN_CANDIDATE_NAMES:
            try:
                current = current_results[(name, cost)]
            except KeyError as exc:
                raise RuntimeError(
                    f"Missing current result {name} at {cost:g}bps"
                ) from exc
            if not pd.DatetimeIndex(current.ledger.index).equals(original_index):
                raise RuntimeError(
                    f"Shared-date alignment failed for {name} at {cost:g}bps"
                )
    assert reference_index is not None
    return {
        "aligned": True,
        "start": reference_index[0].date().isoformat(),
        "end": reference_index[-1].date().isoformat(),
        "observations": len(reference_index),
        "costs_bps": list(selected_costs),
        "current_candidates": list(alpha.FROZEN_CANDIDATE_NAMES),
        "external_baseline": legacy.LEGACY_CANDIDATE.name,
    }


def _paired_comparisons(
    current_results: Mapping[
        tuple[str, float],
        alpha.AlphaBacktestResult,
    ],
    original_shared: Mapping[float, legacy.LegacyBacktestResult],
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    for cost_offset, cost in enumerate(PAIRWISE_COSTS_BPS):
        original = original_shared[cost]
        cost_rows: dict[str, object] = {}
        for candidate_offset, name in enumerate(alpha.FROZEN_CANDIDATE_NAMES):
            candidate = current_results[(name, cost)]
            candidate_growth = float(
                candidate.metrics["annualized_log_growth"]
            )
            original_growth = float(
                legacy.result_log_returns(original).mean() * 252.0
            )
            cost_rows[name] = {
                "direction": "current candidate minus Original",
                "current_annualized_log_growth": candidate_growth,
                "original_annualized_log_growth": original_growth,
                "annualized_log_growth_difference": (
                    candidate_growth - original_growth
                ),
                "rolling_win_rates": alpha.rolling_win_rates(
                    candidate,
                    original,
                ),
                "chronological_slices": alpha.chronological_slices(
                    candidate,
                    original,
                ),
                "moving_block_bootstrap": alpha.moving_block_bootstrap(
                    candidate,
                    original,
                    samples=bootstrap_samples,
                    seed=(
                        MASTER_SEED
                        + cost_offset * 100
                        + candidate_offset
                    ),
                ),
            }
        comparisons[f"{cost:g}bps"] = cost_rows
    return comparisons


def _expected_current_result_keys() -> set[tuple[str, float]]:
    return {
        (name, cost)
        for name in alpha.FROZEN_CANDIDATE_NAMES
        for cost in CONFIRMATORY_COSTS_BPS
    }


def _original_metrics(
    result: legacy.LegacyBacktestResult,
) -> dict[str, object]:
    metrics = dict(result.metrics)
    metrics["annualized_log_growth"] = float(
        legacy.result_log_returns(result).mean() * 252.0
    )
    return metrics


def evaluate_paired_research(
    data: UnionMarketData,
    *,
    costs_bps: Sequence[float] = CONFIRMATORY_COSTS_BPS,
    bootstrap_samples: int = 10_000,
) -> PairedResearchResult:
    """Run both frozen engines and build the paired audit report."""

    selected_costs = tuple(float(value) for value in costs_bps)
    if selected_costs != CONFIRMATORY_COSTS_BPS:
        raise ValueError(
            "Paired confirmatory costs must be exactly "
            f"{list(CONFIRMATORY_COSTS_BPS)}"
        )
    if bootstrap_samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    views = create_engine_views(data)
    panel, current_results, frozen_report = alpha.evaluate_frozen_family(
        views.current,
        costs_bps=selected_costs,
        bootstrap_samples=bootstrap_samples,
    )
    if set(current_results) != _expected_current_result_keys():
        raise RuntimeError("Frozen current-family result grid is incomplete")
    shared_start = alpha.common_scoring_start(views.current, panel)
    original_schedule = legacy.build_original_target_schedule(views.original)

    original_own: dict[float, legacy.LegacyBacktestResult] = {}
    original_shared: dict[float, legacy.LegacyBacktestResult] = {}
    for cost in CONFIRMATORY_COSTS_BPS:
        original_own[cost] = legacy.simulate_original(
            views.original,
            original_schedule,
            cost_bps=cost,
        )
        original_shared[cost] = legacy.simulate_original(
            views.original,
            original_schedule,
            cost_bps=cost,
            execution_start=shared_start,
        )

    alignment = verify_shared_date_alignment(
        current_results,
        original_shared,
    )
    trial_audit = external_baseline_trial_audit(frozen_report)
    comparisons = _paired_comparisons(
        current_results,
        original_shared,
        bootstrap_samples=bootstrap_samples,
    )
    report: dict[str, object] = {
        "schema_version": PAIRED_RESEARCH_SCHEMA_VERSION,
        "research_only": True,
        "configuration": PAIRED_CONFIGURATION,
        "configuration_fingerprint": PAIRED_CONFIGURATION_FINGERPRINT,
        "data": {
            "source": data.source,
            "requested_start": data.requested_start,
            "requested_end_exclusive": data.requested_end_exclusive,
            "final_session": data.final_session.date().isoformat(),
            "union_tickers": list(UNION_TICKERS),
            "union_sha256": data.fingerprint,
            "current_view_sha256": views.current.fingerprint,
            "original_view_sha256": views.original.fingerprint,
            "bit_identity_verification_sha256": (
                views.verification_fingerprint
            ),
            "current_actual_common_start": (
                views.current.common_start.date().isoformat()
            ),
            "original_actual_common_start": (
                views.original.common_start.date().isoformat()
            ),
            "shared_execution_start": shared_start.date().isoformat(),
        },
        "provenance": {
            "current_family_fingerprint": alpha.FROZEN_FAMILY_FINGERPRINT,
            "current_software_fingerprint": alpha.software_fingerprint(),
            "current_candidate_fingerprints": {
                name: alpha.candidate_fingerprint(name)
                for name in alpha.FROZEN_CANDIDATE_NAMES
            },
            "original_source_commit": legacy.LEGACY_SOURCE_COMMIT,
            "original_strategy_blob": legacy.LEGACY_PORT12_BLOB_SHA,
            "original_research_blob": legacy.LEGACY_RESEARCH_BLOB_SHA,
            "original_tests_blob": legacy.LEGACY_TESTS_BLOB_SHA,
            "original_strategy_fingerprint": (
                legacy.LEGACY_STRATEGY_FINGERPRINT
            ),
            "original_historical_implementation_fingerprint": (
                legacy.LEGACY_IMPLEMENTATION_FINGERPRINT
            ),
            "original_comparator_implementation_fingerprint": (
                legacy.COMPARATOR_IMPLEMENTATION_FINGERPRINT
            ),
            "original_candidate_fingerprint": (
                legacy.LEGACY_CANDIDATE_FINGERPRINT
            ),
            "original_schedule_fingerprint": original_schedule.fingerprint,
        },
        "shared_date_alignment": alignment,
        "multiplicity": trial_audit,
        "current_frozen_family_report": frozen_report,
        "original_results": {
            "own_executable_history": {
                f"{cost:g}bps": _original_metrics(original_own[cost])
                for cost in CONFIRMATORY_COSTS_BPS
            },
            "shared_current_family_interval": {
                f"{cost:g}bps": _original_metrics(original_shared[cost])
                for cost in CONFIRMATORY_COSTS_BPS
            },
        },
        "current_vs_original": comparisons,
        "interpretation": {
            "comparison_direction": "current candidate minus Original",
            "own_history_is_not_paired": (
                "Original own-history metrics use its first executable session "
                "and are descriptive only."
            ),
            "shared_interval_is_paired": (
                "All inferential comparisons reset both strategies to equal "
                "cash and use identical execution dates."
            ),
            "multiplicity": (
                "Original is external and does not change the frozen seven-path "
                "current-family trial count."
            ),
        },
    }
    return PairedResearchResult(
        views=views,
        panel=panel,
        current_results=dict(current_results),
        original_own_history=original_own,
        original_shared_interval=original_shared,
        report=report,
    )


def _default_end_exclusive() -> str:
    return (
        pd.Timestamp.now(tz=NEW_YORK).normalize() + pd.Timedelta(days=1)
    ).date().isoformat()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paired frozen alpha-family versus Original research. "
            "This command never touches production state or email."
        )
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=_default_end_exclusive())
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Load an exact eight-ticker union snapshot instead of downloading",
    )
    parser.add_argument(
        "--save-snapshot",
        type=Path,
        help="Atomically preserve the exact validated union snapshot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research_outputs") / "alpha_paired_results.json",
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        help="Optional directory for current and Original research ledgers",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )
    return parser


def _save_ledgers(
    result: PairedResearchResult,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for (name, cost), current in result.current_results.items():
        current.ledger.to_csv(
            directory / f"current_{name}_{cost:g}bps.csv",
            index=True,
            date_format="%Y-%m-%d",
            float_format="%.17g",
            lineterminator="\n",
        )
    for cost, original in result.original_own_history.items():
        original.ledger.to_csv(
            directory / f"original_own_history_{cost:g}bps.csv",
            index=True,
            date_format="%Y-%m-%d",
            float_format="%.17g",
            lineterminator="\n",
        )
    for cost, original in result.original_shared_interval.items():
        original.ledger.to_csv(
            directory / f"original_shared_{cost:g}bps.csv",
            index=True,
            date_format="%Y-%m-%d",
            float_format="%.17g",
            lineterminator="\n",
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap-samples must be at least 100")
    if args.snapshot is not None:
        data = load_union_snapshot(
            args.snapshot,
            requested_start=args.start,
            requested_end_exclusive=args.end,
        )
    else:
        data = download_union_market_data(
            requested_start=args.start,
            requested_end_exclusive=args.end,
        )
    if args.save_snapshot is not None:
        save_union_snapshot(data, args.save_snapshot)
    result = evaluate_paired_research(
        data,
        bootstrap_samples=args.bootstrap_samples,
    )
    alpha.save_json_atomic(result.report, args.output)
    if args.ledger_dir is not None:
        _save_ledgers(result, args.ledger_dir)

    original = result.original_shared_interval[10.0]
    original_log_growth = float(
        legacy.result_log_returns(original).mean() * 252.0
    )
    print(
        f"{'legacy_original':26s} "
        f"CAGR={original.metrics['cagr']:8.2%} "
        f"log-growth={original_log_growth:8.2%} "
        f"maxDD={original.metrics['maximum_drawdown']:8.2%}"
    )
    for name in alpha.FROZEN_CANDIDATE_NAMES:
        metrics = result.current_results[(name, 10.0)].metrics
        print(
            f"{name:26s} CAGR={metrics['cagr']:8.2%} "
            f"log-growth={metrics['annualized_log_growth']:8.2%} "
            f"maxDD={metrics['maximum_drawdown']:8.2%}"
        )
    print(f"Saved paired research report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

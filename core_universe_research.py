#!/usr/bin/env python3
"""Evaluate the frozen leveraged-core and satellite universe.

This module is research-only. It has no production-state, portfolio, or
notification authority. The complete family is frozen in
``CORE_UNIVERSE_PROTOCOL.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import alpha_core as core
import alpha_research as research


SIGNAL_TICKERS = ("QQQ", "SPY", "SMH", "XLK")
TRADED_TICKERS = ("QLD", "SSO", "UPRO", "SOXL", "USD", "TECL", "GLD", "IEF")
REQUIRED_TICKERS = (*SIGNAL_TICKERS, *TRADED_TICKERS)
EQUITY_TICKERS = ("QLD", "SSO", "UPRO", "SOXL", "USD", "TECL")
LEVERAGE = {
    "QLD": 2.0,
    "SSO": 2.0,
    "UPRO": 3.0,
    "SOXL": 3.0,
    "USD": 2.0,
    "TECL": 3.0,
    "GLD": 1.0,
    "IEF": 1.0,
}

STARTING_CASH = 10_000.0
COSTS_BPS = (10.0, 25.0, 50.0)
VOLATILITY_BUDGET = 0.55
SATELLITE_GRID = core.SOXL_WEIGHT_GRID
DRIFT_TRIGGER = 0.05
DRIFT_DESTINATION = 0.025
BOOTSTRAP_BLOCK = 21
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260809
MAXIMUM_DRAWDOWN_BOUNDARY = -0.70
RISK_EFFICIENCY_CAGR_SACRIFICE = 0.03
RISK_EFFICIENCY_DRAWDOWN_IMPROVEMENT = 0.05


@dataclass(frozen=True)
class Candidate:
    name: str
    base: Mapping[str, float] | None
    satellite: str
    signal: str
    router: bool = False


CANDIDATES = (
    Candidate("live_qld_soxl", {"QLD": 1.0}, "SOXL", "semiconductor"),
    Candidate("sso_soxl", {"SSO": 1.0}, "SOXL", "semiconductor"),
    Candidate(
        "qld_sso_equal_soxl",
        {"QLD": 0.5, "SSO": 0.5},
        "SOXL",
        "semiconductor",
    ),
    Candidate(
        "qld_sso_router_soxl",
        None,
        "SOXL",
        "semiconductor",
        router=True,
    ),
    Candidate(
        "qld80_gld20_soxl",
        {"QLD": 0.8, "GLD": 0.2},
        "SOXL",
        "semiconductor",
    ),
    Candidate(
        "upro60_gld20_ief20_soxl",
        {"UPRO": 0.6, "GLD": 0.2, "IEF": 0.2},
        "SOXL",
        "semiconductor",
    ),
    Candidate("qld_usd", {"QLD": 1.0}, "USD", "semiconductor"),
    Candidate("sso_tecl", {"SSO": 1.0}, "TECL", "technology"),
)
CANDIDATE_NAMES = tuple(item.name for item in CANDIDATES)
BENCHMARK = CANDIDATE_NAMES[0]


@dataclass(frozen=True)
class UniverseData:
    opens: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame
    sessions: pd.DatetimeIndex
    common_start: pd.Timestamp
    fingerprint: str
    source: str


@dataclass(frozen=True)
class PathResult:
    candidate: str
    cost_bps: float
    ledger: pd.DataFrame
    metrics: dict[str, object]
    unfilled_final_order: bool


@dataclass(frozen=True)
class PendingOrder:
    signal_date: pd.Timestamp
    execution_target: dict[str, float]
    strategic_target: dict[str, float]
    reason: str


def _normalize_field(
    raw: pd.DataFrame,
    field: str,
    tickers: Sequence[str],
) -> pd.DataFrame:
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Universe market response is empty or malformed")
    first = set(str(item) for item in raw.columns.get_level_values(0))
    second = set(str(item) for item in raw.columns.get_level_values(1))
    if field in first:
        frame = raw[field]
    elif field in second:
        frame = raw.xs(field, axis=1, level=1)
    else:
        raise RuntimeError(f"Universe market response is missing {field}")
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    result = frame.copy()
    if result.index.tz is not None:
        result.index = result.index.tz_convert(None)
    result.index = pd.DatetimeIndex(result.index).normalize()
    result.index.name = None
    result.columns = [str(item) for item in result.columns]
    result.columns.name = None
    missing = set(tickers) - set(result.columns)
    if missing:
        raise RuntimeError(f"Universe market data is missing tickers: {sorted(missing)}")
    result = result.loc[:, list(tickers)].apply(pd.to_numeric, errors="coerce")
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise RuntimeError("Universe market dates are duplicated or unsorted")
    return result


def _fingerprint_frames(*frames: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        canonical = frame.copy()
        canonical.index.name = "Session"
        canonical.columns.name = "Ticker"
        digest.update(
            canonical.to_csv(
                index=True,
                date_format="%Y-%m-%d",
                float_format="%.17g",
                lineterminator="\n",
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def strategy_fingerprint() -> str:
    return _canonical_hash(
        {
            "candidate_names": CANDIDATE_NAMES,
            "candidates": [
                {
                    "name": item.name,
                    "base": dict(item.base) if item.base is not None else None,
                    "satellite": item.satellite,
                    "signal": item.signal,
                    "router": item.router,
                }
                for item in CANDIDATES
            ],
            "costs_bps": COSTS_BPS,
            "volatility_budget": VOLATILITY_BUDGET,
            "satellite_grid": SATELLITE_GRID,
            "drift_trigger": DRIFT_TRIGGER,
            "drift_destination": DRIFT_DESTINATION,
            "drawdown_boundary": MAXIMUM_DRAWDOWN_BOUNDARY,
            "risk_efficiency_cagr_sacrifice": RISK_EFFICIENCY_CAGR_SACRIFICE,
            "risk_efficiency_drawdown_improvement": (
                RISK_EFFICIENCY_DRAWDOWN_IMPROVEMENT
            ),
        }
    )


def software_fingerprint() -> str:
    return _canonical_hash(
        {
            "core_universe_research": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "alpha_core": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
            "alpha_research": hashlib.sha256(
                Path(research.__file__).read_bytes()
            ).hexdigest(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
    )


def prepare_universe_data(
    raw: pd.DataFrame,
    *,
    requested_start: str,
    requested_end_exclusive: str,
    source: str,
) -> UniverseData:
    final_session = research._last_session_before(requested_end_exclusive)
    sessions = research._calendar_sessions(pd.Timestamp(requested_start), final_session)
    opens = _normalize_field(raw, "Open", REQUIRED_TICKERS).reindex(sessions)
    closes = _normalize_field(raw, "Close", REQUIRED_TICKERS).reindex(sessions)
    volumes = _normalize_field(raw, "Volume", REQUIRED_TICKERS).reindex(sessions)
    joint = np.isfinite(opens.to_numpy(dtype=float)).all(axis=1)
    joint &= np.isfinite(closes.to_numpy(dtype=float)).all(axis=1)
    joint &= np.isfinite(volumes.to_numpy(dtype=float)).all(axis=1)
    joint &= (opens.to_numpy(dtype=float) > 0).all(axis=1)
    joint &= (closes.to_numpy(dtype=float) > 0).all(axis=1)
    joint &= (volumes.to_numpy(dtype=float) >= 0).all(axis=1)
    if not joint.any():
        raise RuntimeError("Universe has no common actual-history session")
    first = int(np.flatnonzero(joint)[0])
    common_start = pd.Timestamp(sessions[first])
    for field, frame in (("Open", opens), ("Close", closes)):
        scored = frame.loc[common_start:]
        values = scored.to_numpy(dtype=float)
        invalid = ~np.isfinite(values) | (values <= 0)
        if invalid.any():
            row, column = np.argwhere(invalid)[0]
            raise RuntimeError(
                f"Universe adjusted {field} is invalid at "
                f"{scored.index[int(row)].date().isoformat()} / "
                f"{scored.columns[int(column)]}"
            )
    scored_volumes = volumes.loc[common_start:]
    volume_values = scored_volumes.to_numpy(dtype=float)
    invalid_volume = ~np.isfinite(volume_values) | (volume_values < 0)
    if invalid_volume.any():
        row, column = np.argwhere(invalid_volume)[0]
        raise RuntimeError(
            "Universe adjusted Volume is invalid at "
            f"{scored_volumes.index[int(row)].date().isoformat()} / "
            f"{scored_volumes.columns[int(column)]}"
        )
    return UniverseData(
        opens=opens,
        closes=closes,
        volumes=volumes,
        sessions=sessions,
        common_start=common_start,
        fingerprint=_fingerprint_frames(opens, closes, volumes),
        source=source,
    )


def download_universe_data(
    *,
    requested_start: str,
    requested_end_exclusive: str,
) -> UniverseData:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required for universe research") from exc
    raw = yf.download(
        list(REQUIRED_TICKERS),
        start=requested_start,
        end=requested_end_exclusive,
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
        threads=False,
    )
    return prepare_universe_data(
        raw,
        requested_start=requested_start,
        requested_end_exclusive=requested_end_exclusive,
        source="Yahoo Finance via yfinance; auto_adjust=True",
    )


def save_snapshot(data: UniverseData, path: Path) -> None:
    rows = []
    for field, frame in (
        ("Open", data.opens),
        ("Close", data.closes),
        ("Volume", data.volumes),
    ):
        stacked = frame.stack(future_stack=True).rename("Value").reset_index()
        stacked.columns = ["Session", "Ticker", "Value"]
        stacked.insert(2, "Field", field)
        rows.append(stacked)
    output = pd.concat(rows, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    output.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    os.replace(temporary, path)


def load_snapshot(
    path: Path,
    *,
    requested_start: str,
    requested_end_exclusive: str,
) -> UniverseData:
    snapshot = pd.read_csv(path, parse_dates=["Session"], float_precision="round_trip")
    if set(snapshot.columns) != {"Session", "Ticker", "Field", "Value"}:
        raise RuntimeError("Universe snapshot columns are invalid")
    if snapshot.duplicated(["Session", "Ticker", "Field"]).any():
        raise RuntimeError("Universe snapshot contains duplicate observations")
    if set(snapshot["Ticker"]) != set(REQUIRED_TICKERS):
        raise RuntimeError("Universe snapshot ticker set differs")
    if set(snapshot["Field"]) != {"Open", "Close", "Volume"}:
        raise RuntimeError("Universe snapshot fields differ")
    raw = snapshot.pivot(
        index="Session",
        columns=["Field", "Ticker"],
        values="Value",
    )
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    return prepare_universe_data(
        raw,
        requested_start=requested_start,
        requested_end_exclusive=requested_end_exclusive,
        source=f"frozen local snapshot: {path.name}",
    )


def generalized_residual_frame(
    dependent_close: pd.Series,
    regressor_close: pd.Series,
) -> pd.DataFrame:
    dependent = pd.to_numeric(dependent_close, errors="coerce")
    regressor = pd.to_numeric(regressor_close, errors="coerce")
    if not dependent.index.equals(regressor.index):
        raise ValueError("Residual series dates differ")
    prices = pd.concat([regressor.rename("x"), dependent.rename("y")], axis=1)
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Residual series contains invalid prices")
    returns = np.log(prices).diff()
    required = core.BETA_WINDOW + core.RESIDUAL_SCORE_WINDOW
    output = pd.DataFrame(
        {
            "available": False,
            "momentum": np.nan,
            "z": np.nan,
        },
        index=prices.index,
    )
    for position in range(required, len(prices)):
        window = returns.iloc[position - required + 1 : position + 1]
        estimation = window.iloc[: core.BETA_WINDOW]
        scoring = window.iloc[core.BETA_WINDOW :]
        x = estimation["x"].to_numpy(dtype=float)
        y = estimation["y"].to_numpy(dtype=float)
        x_centered = x - x.mean()
        denominator = float(x_centered @ x_centered)
        if not np.isfinite(denominator) or denominator <= 1e-12:
            continue
        beta = float(x_centered @ (y - y.mean()) / denominator)
        intercept = float(y.mean() - beta * x.mean())
        residuals = y - (intercept + beta * x)
        variance = float(residuals @ residuals / (core.BETA_WINDOW - 2))
        if not np.isfinite(variance) or variance <= 1e-12:
            continue
        score = scoring["y"].to_numpy(dtype=float) - (
            intercept + beta * scoring["x"].to_numpy(dtype=float)
        )
        momentum = float(score.sum())
        z = float(momentum / (math.sqrt(variance) * math.sqrt(len(score))))
        if np.isfinite((momentum, z)).all():
            output.loc[prices.index[position]] = (True, momentum, z)
    output["available"] = output["available"].astype(bool)
    return output


def expanding_har_volatility(
    close: pd.Series,
    *,
    minimum_observations: int = core.VARIANCE_MIN_TRAINING,
) -> pd.DataFrame:
    """Calculate the frozen expanding HAR forecast with efficient sufficient sums."""
    design = core.build_variance_learning_frame(close)
    feature_names = (
        "log_var_5",
        "log_var_21",
        "log_var_63",
        "log_downside_var_21",
    )
    x = design.loc[:, list(feature_names)].to_numpy(dtype=float)
    y = design["future_log_var_21"].to_numpy(dtype=float)
    usable = np.isfinite(x).all(axis=1) & np.isfinite(y)
    clean_x = np.where(usable[:, None], x, 0.0)
    clean_y = np.where(usable, y, 0.0)
    counts = np.cumsum(usable.astype(int))
    sum_x = np.cumsum(clean_x, axis=0)
    sum_y = np.cumsum(clean_y)
    sum_xx = np.cumsum(
        clean_x[:, :, None] * clean_x[:, None, :],
        axis=0,
    )
    sum_xy = np.cumsum(clean_x * clean_y[:, None], axis=0)
    returns = np.log(pd.to_numeric(close, errors="coerce")).diff()
    result = pd.DataFrame(
        {
            "model": np.nan,
            "trailing21": np.nan,
            "trailing63": np.nan,
            "sizing": np.nan,
            "sample_count": 0,
            "available": False,
        },
        index=design.index,
    )
    for position in range(len(design)):
        cutoff = position - core.VARIANCE_HORIZON
        if cutoff < 0 or not np.isfinite(x[position]).all():
            continue
        count = int(counts[cutoff])
        result.iloc[position, result.columns.get_loc("sample_count")] = count
        if count < minimum_observations:
            continue
        sx = sum_x[cutoff]
        sy = float(sum_y[cutoff])
        means = sx / count
        centered_xx = sum_xx[cutoff] - np.outer(sx, sx) / count
        scales = np.sqrt(np.diag(centered_xx) / (count - 1))
        if not np.isfinite(scales).all() or (scales <= 1e-12).any():
            continue
        gram = centered_xx / np.outer(scales, scales)
        centered_xy = sum_xy[cutoff] - sx * sy / count
        rhs = centered_xy / scales
        try:
            coefficients = np.linalg.solve(
                gram + core.RIDGE_ALPHA * np.eye(len(feature_names)),
                rhs,
            )
        except np.linalg.LinAlgError:
            continue
        y_mean = sy / count
        prediction = float(y_mean + ((x[position] - means) / scales) @ coefficients)
        training_mask = usable[: cutoff + 1]
        training_x = x[: cutoff + 1][training_mask]
        training_y = y[: cutoff + 1][training_mask]
        fitted = y_mean + ((training_x - means) / scales) @ coefficients
        smearing = float(np.mean(np.exp(training_y - fitted)))
        history = returns.iloc[: position + 1].dropna()
        trailing21 = float(history.tail(21).std(ddof=1) * math.sqrt(252.0))
        trailing63 = float(history.tail(63).std(ddof=1) * math.sqrt(252.0))
        try:
            model = float(math.sqrt(math.exp(prediction) * smearing))
        except (OverflowError, ValueError):
            continue
        values = (model, trailing21, trailing63)
        if not np.isfinite(values).all() or min(values) <= 0:
            continue
        result.loc[design.index[position], ["model", "trailing21", "trailing63"]] = values
        result.loc[design.index[position], "sizing"] = max(values)
        result.loc[design.index[position], "available"] = True
    result["sample_count"] = result["sample_count"].astype(int)
    result["available"] = result["available"].astype(bool)
    return result


def _complete_weights(weights: Mapping[str, float]) -> dict[str, float]:
    result = {ticker: 0.0 for ticker in TRADED_TICKERS}
    for ticker, value in weights.items():
        if ticker not in result:
            raise ValueError(f"Unknown traded ticker: {ticker}")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0:
            raise ValueError("Target weights must be finite and nonnegative")
        result[ticker] = numeric
    if not np.isclose(sum(result.values()), 1.0, atol=1e-12):
        raise ValueError("Base weights must sum to one")
    result[core.CASH] = 0.0
    return result


def target_with_satellite(
    base: Mapping[str, float],
    satellite: str,
    satellite_weight: float,
) -> dict[str, float]:
    completed = _complete_weights(base)
    weight = float(satellite_weight)
    if not np.isfinite(weight) or weight < 0 or weight > 0.35 + 1e-12:
        raise ValueError("Satellite weight is outside the frozen range")
    result = {
        ticker: (1.0 - weight) * completed[ticker]
        for ticker in TRADED_TICKERS
    }
    result[satellite] += weight
    result[core.CASH] = 0.0
    if not np.isclose(sum(result.values()), 1.0, atol=1e-12):
        raise RuntimeError("Satellite target does not sum to one")
    return result


def forecast_weight_volatility(
    weights: Mapping[str, float],
    sizing_volatility: pd.Series,
    correlations: Sequence[pd.DataFrame],
) -> float:
    tickers = [ticker for ticker in TRADED_TICKERS if float(weights.get(ticker, 0.0)) > 0]
    if not tickers:
        raise ValueError("Portfolio has no risky holdings")
    vector = np.array([float(weights[ticker]) for ticker in tickers], dtype=float)
    vol = sizing_volatility.reindex(tickers).to_numpy(dtype=float)
    if not np.isfinite(vol).all() or (vol <= 0).any():
        raise ValueError("Portfolio sizing volatility is invalid")
    estimates = []
    for correlation in correlations:
        matrix = correlation.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("Portfolio correlation is invalid")
        covariance = np.outer(vol, vol) * matrix
        variance = float(vector @ covariance @ vector)
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("Portfolio forecast variance is invalid")
        estimates.append(math.sqrt(variance))
    return float(max(estimates))


def choose_satellite_weight(
    base: Mapping[str, float],
    satellite: str,
    sizing_volatility: pd.Series,
    correlations: Sequence[pd.DataFrame],
) -> tuple[float, float]:
    base_target = target_with_satellite(base, satellite, 0.0)
    base_volatility = forecast_weight_volatility(
        base_target,
        sizing_volatility,
        correlations,
    )
    if base_volatility > VOLATILITY_BUDGET + 1e-12:
        return 0.0, base_volatility
    selected = 0.0
    selected_volatility = base_volatility
    for weight in SATELLITE_GRID[1:]:
        target = target_with_satellite(base, satellite, weight)
        volatility = forecast_weight_volatility(
            target,
            sizing_volatility,
            correlations,
        )
        if volatility <= VOLATILITY_BUDGET + 1e-12:
            selected = float(weight)
            selected_volatility = volatility
    return selected, selected_volatility


def build_schedules(
    data: UniverseData,
    volatility: Mapping[str, pd.DataFrame],
) -> tuple[pd.Timestamp, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    closes = data.closes.loc[data.common_start:]
    sessions = closes.index
    returns = np.log(closes.loc[:, list(TRADED_TICKERS)]).diff()
    semiconductor = generalized_residual_frame(closes["SMH"], closes["QQQ"])
    technology = generalized_residual_frame(closes["XLK"], closes["SPY"])
    router_signal = generalized_residual_frame(closes["QQQ"], closes["SPY"])
    qqq_sma = closes["QQQ"].rolling(core.SMA_WINDOW, min_periods=core.SMA_WINDOW).mean()
    spy_sma = closes["SPY"].rolling(core.SMA_WINDOW, min_periods=core.SMA_WINDOW).mean()
    qqq_trend = closes["QQQ"] > qqq_sma
    spy_trend = closes["SPY"] > spy_sma
    states = {item.name: core.OverlayState() for item in CANDIDATES}
    router_ticker = "SSO"
    sizing_frame = pd.DataFrame(
        {
            ticker: volatility[ticker]["sizing"].reindex(sessions)
            for ticker in TRADED_TICKERS
        },
        index=sessions,
    )
    target_rows: dict[str, list[dict[str, object]]] = {item.name: [] for item in CANDIDATES}
    diagnostic_rows: dict[str, list[dict[str, object]]] = {
        item.name: [] for item in CANDIDATES
    }
    first_available: pd.Timestamp | None = None

    for position, date in enumerate(sessions):
        date = pd.Timestamp(date)
        review_due = position % core.ALPHA_REVIEW_SESSIONS == 0
        if review_due:
            router_row = router_signal.loc[date]
            router_ticker = (
                "QLD"
                if bool(router_row["available"])
                and float(router_row["momentum"]) > 0
                else "SSO"
            )
        sizing = sizing_frame.loc[date].astype(float)
        paired = returns.loc[:date]
        corr21 = paired.tail(21).corr()
        corr63 = paired.tail(63).corr()
        all_volatility_available = np.isfinite(sizing.to_numpy()).all()
        correlations_available = (
            np.isfinite(corr21.to_numpy(dtype=float)).all()
            and np.isfinite(corr63.to_numpy(dtype=float)).all()
        )
        if all_volatility_available and correlations_available and first_available is None:
            first_available = date

        for candidate in CANDIDATES:
            base = (
                {router_ticker: 1.0}
                if candidate.router
                else dict(candidate.base or {})
            )
            signal_frame = semiconductor if candidate.signal == "semiconductor" else technology
            signal_row = signal_frame.loc[date]
            trend = bool(qqq_trend.loc[date]) if candidate.signal == "semiconductor" else bool(spy_trend.loc[date])
            available = bool(signal_row["available"])
            residual_positive = bool(
                available
                and np.isfinite(float(signal_row["momentum"]))
                and float(signal_row["momentum"]) > 0
            )
            trend = bool(trend and available)
            raw_weight = 0.0
            raw_volatility = math.nan
            if all_volatility_available and correlations_available:
                try:
                    raw_weight, raw_volatility = choose_satellite_weight(
                        base,
                        candidate.satellite,
                        sizing,
                        (corr21, corr63),
                    )
                except ValueError:
                    raw_weight = 0.0
                    raw_volatility = math.nan
            transition = core.advance_overlay_state(
                states[candidate.name],
                signal_date=date,
                trend_positive=trend,
                residual_positive=residual_positive,
                raw_soxl_weight=raw_weight,
                alpha_review_due=review_due,
            )
            states[candidate.name] = transition.state
            target = target_with_satellite(
                base,
                candidate.satellite,
                transition.state.soxl_weight,
            )
            target_rows[candidate.name].append({"session": date, **target})
            diagnostic_rows[candidate.name].append(
                {
                    "session": date,
                    "review_due": review_due,
                    "trend_positive": trend,
                    "residual_available": available,
                    "residual_z": signal_row["z"],
                    "residual_positive": residual_positive,
                    "raw_satellite_weight": raw_weight,
                    "strategic_satellite_weight": transition.state.soxl_weight,
                    "raw_portfolio_volatility": raw_volatility,
                    "base_router": router_ticker if candidate.router else "STATIC",
                    "decision_reason": transition.reason,
                }
            )
    if first_available is None:
        raise RuntimeError("No common volatility signal is available")
    targets = {
        name: pd.DataFrame(rows).set_index("session")
        for name, rows in target_rows.items()
    }
    diagnostics = {
        name: pd.DataFrame(rows).set_index("session")
        for name, rows in diagnostic_rows.items()
    }
    return first_available, targets, diagnostics


def _actual_weights(
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
        raise RuntimeError("Universe portfolio NAV is invalid")
    weights = {ticker: values[ticker] / nav for ticker in TRADED_TICKERS}
    weights[core.CASH] = float(cash) / nav
    return weights, nav


def drift_triggered(actual: Mapping[str, float], strategic: Mapping[str, float]) -> bool:
    individual = any(
        abs(float(actual.get(ticker, 0.0)) - float(strategic.get(ticker, 0.0)))
        >= DRIFT_TRIGGER - 1e-12
        for ticker in TRADED_TICKERS
    )
    actual_equity = sum(float(actual.get(ticker, 0.0)) for ticker in EQUITY_TICKERS)
    strategic_equity = sum(
        float(strategic.get(ticker, 0.0)) for ticker in EQUITY_TICKERS
    )
    return bool(
        individual
        or abs(actual_equity - strategic_equity) >= DRIFT_TRIGGER - 1e-12
    )


def inner_band_target(
    actual: Mapping[str, float],
    strategic: Mapping[str, float],
) -> dict[str, float]:
    """Project actual weights to the closest fully invested 2.5pp box."""
    actual_vector = np.array(
        [float(actual.get(ticker, 0.0)) for ticker in TRADED_TICKERS],
        dtype=float,
    )
    strategic_vector = np.array(
        [float(strategic.get(ticker, 0.0)) for ticker in TRADED_TICKERS],
        dtype=float,
    )
    if not np.isfinite(actual_vector).all() or not np.isfinite(strategic_vector).all():
        raise ValueError("Drift weights are invalid")
    lower = np.maximum(0.0, strategic_vector - DRIFT_DESTINATION)
    upper = np.minimum(1.0, strategic_vector + DRIFT_DESTINATION)
    # Ordinary drift may rebalance an authorized sleeve but must never create
    # exposure to a ticker whose strategic weight is zero.
    upper = np.where(strategic_vector > 1e-12, upper, 0.0)
    if lower.sum() > 1.0 + 1e-12 or upper.sum() < 1.0 - 1e-12:
        raise RuntimeError("Inner drift box does not contain a portfolio")
    low = float(np.min(actual_vector - upper) - 1.0)
    high = float(np.max(actual_vector - lower) + 1.0)
    for _ in range(100):
        middle = (low + high) / 2.0
        projected = np.clip(actual_vector - middle, lower, upper)
        if projected.sum() > 1.0:
            low = middle
        else:
            high = middle
    projected = np.clip(actual_vector - (low + high) / 2.0, lower, upper)
    projected /= projected.sum()
    if not np.isclose(projected.sum(), 1.0, atol=1e-12):
        raise RuntimeError("Inner drift target does not sum to one")
    result = {
        ticker: float(projected[index])
        for index, ticker in enumerate(TRADED_TICKERS)
    }
    result[core.CASH] = 0.0
    return result


def _strategic_changed(
    desired: Mapping[str, float],
    executed: Mapping[str, float],
) -> bool:
    return any(
        abs(float(desired.get(item, 0.0)) - float(executed.get(item, 0.0)))
        > 1e-12
        for item in (*TRADED_TICKERS, core.CASH)
    )


def simulate_schedule(
    data: UniverseData,
    targets: pd.DataFrame,
    *,
    candidate: str,
    cost_bps: float,
    first_signal_date: pd.Timestamp,
) -> PathResult:
    if candidate not in CANDIDATE_NAMES:
        raise ValueError("Candidate is not in the frozen universe")
    sessions = data.sessions[data.sessions.get_loc(data.common_start) :]
    if not targets.index.equals(sessions):
        raise ValueError("Target schedule dates differ from universe sessions")
    expected_columns = set((*TRADED_TICKERS, core.CASH))
    if set(targets.columns) != expected_columns:
        raise ValueError("Target schedule columns differ from the universe")
    signal = pd.Timestamp(first_signal_date)
    signal_position = sessions.get_loc(signal)
    if not isinstance(signal_position, (int, np.integer)) or int(signal_position) + 1 >= len(sessions):
        raise ValueError("First signal has no next-open execution")
    scoring_dates = sessions[int(signal_position) + 1 :]
    initial_target = {
        item: float(targets.loc[signal, item]) for item in expected_columns
    }
    pending: PendingOrder | None = PendingOrder(
        signal,
        initial_target,
        initial_target,
        "INITIAL_DEPLOYMENT",
    )
    executed_strategic = {ticker: 0.0 for ticker in TRADED_TICKERS}
    executed_strategic[core.CASH] = 1.0
    shares: dict[str, float] = {}
    cash = STARTING_CASH
    previous_close: pd.Series | None = None
    rows: list[dict[str, object]] = []

    for date in scoring_dates:
        date = pd.Timestamp(date)
        open_prices = data.opens.loc[date, list(TRADED_TICKERS)]
        close_prices = data.closes.loc[date, list(TRADED_TICKERS)]
        old_shares = dict(shares)
        fill = None
        fill_signal_date = pd.NaT
        fill_reason = ""
        initial_deployment = False
        if pending is not None:
            fill = research.execute_target_at_open(
                shares=shares,
                cash=cash,
                open_prices=open_prices,
                target_weights=pending.execution_target,
                cost_bps=cost_bps,
            )
            shares = fill.shares
            cash = fill.cash
            fill_signal_date = pending.signal_date
            fill_reason = pending.reason
            initial_deployment = fill_reason == "INITIAL_DEPLOYMENT"
            executed_strategic = dict(pending.strategic_target)
            pending = None
        overnight = {
            ticker: (
                old_shares.get(ticker, 0.0)
                * (float(open_prices[ticker]) - float(previous_close[ticker]))
                if previous_close is not None
                else 0.0
            )
            for ticker in TRADED_TICKERS
        }
        intraday = {
            ticker: float(shares.get(ticker, 0.0))
            * (float(close_prices[ticker]) - float(open_prices[ticker]))
            for ticker in TRADED_TICKERS
        }
        actual, nav = _actual_weights(shares, cash, close_prices)
        desired = {
            item: float(targets.loc[date, item]) for item in expected_columns
        }
        if _strategic_changed(desired, executed_strategic):
            pending = PendingOrder(date, desired, desired, "STRUCTURAL_CHANGE")
        elif drift_triggered(actual, desired):
            pending = PendingOrder(
                date,
                inner_band_target(actual, desired),
                desired,
                "ORDINARY_DRIFT",
            )
        row: dict[str, object] = {
            "session": date,
            "nav": nav,
            "cash": cash,
            "fill_signal_date": fill_signal_date,
            "fill_reason": fill_reason,
            "initial_deployment": initial_deployment,
            "transaction_cost": fill.cost if fill is not None else 0.0,
            "transaction_cost_fraction": (
                fill.cost_fraction if fill is not None else 0.0
            ),
            "gross_trade_fraction": (
                fill.gross_fraction
                if fill is not None and not initial_deployment
                else 0.0
            ),
            "one_way_turnover": (
                fill.one_way_fraction
                if fill is not None and not initial_deployment
                else 0.0
            ),
            "security_orders": (
                fill.security_orders
                if fill is not None and not initial_deployment
                else 0
            ),
            "advertised_daily_exposure": sum(
                actual[ticker] * LEVERAGE[ticker] for ticker in TRADED_TICKERS
            ),
            "cost_pnl": -(fill.cost if fill is not None else 0.0),
        }
        for ticker in TRADED_TICKERS:
            row[f"{ticker.lower()}_weight"] = actual[ticker]
            row[f"{ticker.lower()}_shares"] = float(shares.get(ticker, 0.0))
            row[f"{ticker.lower()}_pnl"] = overnight[ticker] + intraday[ticker]
        row["cash_weight"] = actual[core.CASH]
        rows.append(row)
        previous_close = close_prices
    ledger = pd.DataFrame(rows).set_index("session")
    return PathResult(
        candidate=candidate,
        cost_bps=float(cost_bps),
        ledger=ledger,
        metrics=calculate_metrics(ledger),
        unfilled_final_order=pending is not None,
    )


def _worst_rolling(nav: pd.Series, sessions: int) -> float | None:
    values = (nav / nav.shift(sessions) - 1.0).dropna()
    return float(values.min()) if len(values) else None


def drawdown_diagnostics(nav: pd.Series) -> dict[str, object]:
    initial_date = pd.Timestamp(nav.index[0]) - pd.Timedelta(nanoseconds=1)
    complete = pd.concat(
        [pd.Series([STARTING_CASH], index=[initial_date]), nav.astype(float)]
    )
    peaks = complete.cummax()
    drawdown = complete / peaks - 1.0
    trough_position = int(np.argmin(drawdown.to_numpy(dtype=float)))
    trough_date = pd.Timestamp(drawdown.index[trough_position])
    peak_position = int(
        np.argmax(complete.iloc[: trough_position + 1].to_numpy(dtype=float))
    )
    peak_date = pd.Timestamp(complete.index[peak_position])
    peak_value = float(complete.iloc[peak_position])
    recovery_date: pd.Timestamp | None = None
    recovery_position: int | None = None
    for position in range(trough_position + 1, len(complete)):
        if float(complete.iloc[position]) >= peak_value - 1e-12:
            recovery_date = pd.Timestamp(complete.index[position])
            recovery_position = position
            break
    underwater = drawdown.to_numpy(dtype=float) < -1e-12
    maximum_underwater = 0
    current_underwater = 0
    for value in underwater:
        current_underwater = current_underwater + 1 if value else 0
        maximum_underwater = max(maximum_underwater, current_underwater)
    return {
        "maximum_drawdown": float(drawdown.min()),
        "drawdown_peak_date": peak_date.date().isoformat(),
        "drawdown_trough_date": trough_date.date().isoformat(),
        "drawdown_recovery_date": (
            recovery_date.date().isoformat() if recovery_date is not None else None
        ),
        "peak_to_trough_sessions": trough_position - peak_position,
        "trough_to_recovery_sessions": (
            recovery_position - trough_position
            if recovery_position is not None
            else None
        ),
        "maximum_time_underwater_sessions": maximum_underwater,
        "fraction_sessions_underwater": float(underwater[1:].mean()),
    }


def calculate_metrics(ledger: pd.DataFrame) -> dict[str, object]:
    if len(ledger) < 2:
        raise RuntimeError("Universe path has too few observations")
    nav = ledger["nav"].astype(float)
    returns = pd.concat(
        [
            pd.Series([float(nav.iloc[0] / STARTING_CASH - 1.0)], index=[nav.index[0]]),
            nav.pct_change(fill_method=None).iloc[1:],
        ]
    )
    if not np.isfinite(returns.to_numpy()).all() or (returns <= -1.0).any():
        raise RuntimeError("Universe path returns are invalid")
    log_returns = np.log1p(returns)
    annual_log_growth = float(log_returns.mean() * 252.0)
    cagr = float(math.exp(annual_log_growth) - 1.0)
    volatility = float(returns.std(ddof=1) * math.sqrt(252.0))
    downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
    downside_deviation = float(math.sqrt(np.mean(downside**2)) * math.sqrt(252.0))
    sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(252.0))
    drawdown_metrics = drawdown_diagnostics(nav)
    maximum_drawdown = float(drawdown_metrics["maximum_drawdown"])
    years = len(returns) / 252.0
    result: dict[str, object] = {
        "observations": len(returns),
        "start": nav.index[0].date().isoformat(),
        "end": nav.index[-1].date().isoformat(),
        "ending_nav": float(nav.iloc[-1]),
        "cagr": cagr,
        "annualized_log_growth": annual_log_growth,
        "annualized_volatility": volatility,
        "downside_deviation": downside_deviation,
        "sharpe_zero_cash_rate": sharpe,
        "maximum_drawdown": maximum_drawdown,
        "calmar": float(cagr / abs(maximum_drawdown)) if maximum_drawdown < 0 else math.inf,
        "worst_rolling_3y": _worst_rolling(nav, 756),
        "worst_rolling_5y": _worst_rolling(nav, 1260),
        "annual_gross_turnover": float(ledger["gross_trade_fraction"].sum() / years),
        "annual_one_way_turnover": float(ledger["one_way_turnover"].sum() / years),
        "total_modeled_cost": float(ledger["transaction_cost"].sum()),
        "rebalance_events": int((ledger["security_orders"] > 0).sum()),
        "mean_advertised_daily_exposure": float(ledger["advertised_daily_exposure"].mean()),
    }
    for ticker in TRADED_TICKERS:
        result[f"mean_{ticker.lower()}_weight"] = float(
            ledger[f"{ticker.lower()}_weight"].mean()
        )
        result[f"{ticker.lower()}_dollar_pnl"] = float(
            ledger[f"{ticker.lower()}_pnl"].sum()
        )
    result["cost_dollar_pnl"] = float(ledger["cost_pnl"].sum())
    result.update(drawdown_metrics)
    return result


def _log_returns(result: PathResult) -> pd.Series:
    nav = result.ledger["nav"].astype(float)
    return pd.concat(
        [
            pd.Series([math.log(float(nav.iloc[0] / STARTING_CASH))], index=[nav.index[0]]),
            np.log(nav).diff().iloc[1:],
        ]
    )


def chronological_differences(
    candidate: PathResult,
    benchmark: PathResult,
    *,
    slices: int = 4,
) -> list[float]:
    left = _log_returns(candidate)
    right = _log_returns(benchmark)
    if not left.index.equals(right.index):
        raise RuntimeError("Universe comparison dates differ")
    difference = (left - right).to_numpy(dtype=float)
    return [
        float(part.mean() * 252.0)
        for part in np.array_split(difference, slices)
    ]


def moving_block_difference(
    candidate: PathResult,
    benchmark: PathResult,
    *,
    samples: int,
) -> dict[str, float]:
    left = _log_returns(candidate)
    right = _log_returns(benchmark)
    if not left.index.equals(right.index):
        raise RuntimeError("Universe comparison dates differ")
    difference = (left - right).to_numpy(dtype=float)
    if len(difference) < BOOTSTRAP_BLOCK:
        raise RuntimeError("Universe comparison is too short")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    starts = np.arange(len(difference) - BOOTSTRAP_BLOCK + 1)
    blocks = int(math.ceil(len(difference) / BOOTSTRAP_BLOCK))
    estimates = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected = rng.choice(starts, size=blocks, replace=True)
        draw = np.concatenate(
            [difference[start : start + BOOTSTRAP_BLOCK] for start in selected]
        )[: len(difference)]
        estimates[sample] = float(draw.mean() * 252.0)
    return {
        "median": float(np.median(estimates)),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "probability_nonpositive": float(np.mean(estimates <= 0.0)),
    }


def familywise_reality_check(
    results: Mapping[str, PathResult],
    *,
    samples: int,
) -> dict[str, float]:
    benchmark = _log_returns(results[BENCHMARK])
    names = [name for name in CANDIDATE_NAMES if name != BENCHMARK]
    differences = []
    for name in names:
        candidate = _log_returns(results[name])
        if not candidate.index.equals(benchmark.index):
            raise RuntimeError("Familywise comparison dates differ")
        differences.append((candidate - benchmark).to_numpy(dtype=float))
    matrix = np.column_stack(differences)
    observed = float(np.max(matrix.mean(axis=0) * 252.0))
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    starts = np.arange(len(matrix) - BOOTSTRAP_BLOCK + 1)
    blocks = int(math.ceil(len(matrix) / BOOTSTRAP_BLOCK))
    maxima = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected = rng.choice(starts, size=blocks, replace=True)
        draw = np.concatenate(
            [centered[start : start + BOOTSTRAP_BLOCK] for start in selected],
            axis=0,
        )[: len(matrix)]
        maxima[sample] = float(np.max(draw.mean(axis=0) * 252.0))
    return {
        "observed_best_annualized_log_growth_difference": observed,
        "p_value": float(np.mean(maxima >= max(0.0, observed))),
        "trial_count": len(CANDIDATE_NAMES),
    }


def pareto_frontier(results: Mapping[str, PathResult]) -> list[str]:
    frontier = []
    for name, result in results.items():
        metrics = result.metrics
        dominated = False
        for other_name, other in results.items():
            if other_name == name:
                continue
            other_metrics = other.metrics
            at_least = (
                float(other_metrics["annualized_log_growth"])
                >= float(metrics["annualized_log_growth"])
                and float(other_metrics["maximum_drawdown"])
                >= float(metrics["maximum_drawdown"])
                and float(other_metrics["sharpe_zero_cash_rate"])
                >= float(metrics["sharpe_zero_cash_rate"])
                and float(other_metrics["calmar"]) >= float(metrics["calmar"])
            )
            strict = (
                float(other_metrics["annualized_log_growth"])
                > float(metrics["annualized_log_growth"]) + 1e-12
                or float(other_metrics["maximum_drawdown"])
                > float(metrics["maximum_drawdown"]) + 1e-12
                or float(other_metrics["sharpe_zero_cash_rate"])
                > float(metrics["sharpe_zero_cash_rate"]) + 1e-12
                or float(other_metrics["calmar"]) > float(metrics["calmar"]) + 1e-12
            )
            if at_least and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return frontier


def evaluate(
    data: UniverseData,
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    closes = data.closes.loc[data.common_start:, list(TRADED_TICKERS)]
    volatility = {
        ticker: expanding_har_volatility(closes[ticker])
        for ticker in TRADED_TICKERS
    }
    first_signal, targets, diagnostics = build_schedules(data, volatility)
    results: dict[tuple[str, float], PathResult] = {}
    for cost in COSTS_BPS:
        for name in CANDIDATE_NAMES:
            results[(name, cost)] = simulate_schedule(
                data,
                targets[name],
                candidate=name,
                cost_bps=cost,
                first_signal_date=first_signal,
            )
    report: dict[str, object] = {
        "protocol": "CORE_UNIVERSE_PROTOCOL.md",
        "status": "exploratory_post_selection_no_live_authority",
        "fingerprints": {
            "strategy": strategy_fingerprint(),
            "software": software_fingerprint(),
            "protocol": hashlib.sha256(
                Path("CORE_UNIVERSE_PROTOCOL.md").read_bytes()
            ).hexdigest(),
        },
        "data": {
            "source": data.source,
            "fingerprint": data.fingerprint,
            "common_start": data.common_start.date().isoformat(),
            "end": data.sessions[-1].date().isoformat(),
            "first_signal": first_signal.date().isoformat(),
            "first_execution": results[(BENCHMARK, COSTS_BPS[0])].metrics["start"],
        },
        "candidate_names": list(CANDIDATE_NAMES),
        "results": {},
        "comparisons_vs_live": {},
        "familywise_reality_check": {},
        "pareto_frontier": {},
        "latest_targets": {
            name: {
                ticker: float(targets[name].iloc[-1][ticker])
                for ticker in (*TRADED_TICKERS, core.CASH)
            }
            for name in CANDIDATE_NAMES
        },
        "latest_diagnostics": {
            name: {
                key: (
                    value.item() if isinstance(value, np.generic) else value
                )
                for key, value in diagnostics[name].iloc[-1].to_dict().items()
            }
            for name in CANDIDATE_NAMES
        },
    }
    for cost in COSTS_BPS:
        key = str(int(cost))
        cost_results = {name: results[(name, cost)] for name in CANDIDATE_NAMES}
        report["results"][key] = {
            name: {
                **path.metrics,
                "unfilled_final_order": path.unfilled_final_order,
            }
            for name, path in cost_results.items()
        }
        benchmark = cost_results[BENCHMARK]
        report["comparisons_vs_live"][key] = {}
        for name in CANDIDATE_NAMES[1:]:
            path = cost_results[name]
            slices = chronological_differences(path, benchmark)
            bootstrap = moving_block_difference(
                path,
                benchmark,
                samples=bootstrap_samples,
            )
            live_metrics = benchmark.metrics
            metrics = path.metrics
            report["comparisons_vs_live"][key][name] = {
                "annualized_log_growth_difference": float(
                    metrics["annualized_log_growth"]
                    - live_metrics["annualized_log_growth"]
                ),
                "cagr_difference": float(metrics["cagr"] - live_metrics["cagr"]),
                "drawdown_improvement": float(
                    metrics["maximum_drawdown"]
                    - live_metrics["maximum_drawdown"]
                ),
                "sharpe_difference": float(
                    metrics["sharpe_zero_cash_rate"]
                    - live_metrics["sharpe_zero_cash_rate"]
                ),
                "calmar_difference": float(metrics["calmar"] - live_metrics["calmar"]),
                "chronological_slice_differences": slices,
                "positive_slices": int(sum(value > 0 for value in slices)),
                "moving_block_bootstrap": bootstrap,
            }
        report["familywise_reality_check"][key] = familywise_reality_check(
            cost_results,
            samples=bootstrap_samples,
        )
        report["pareto_frontier"][key] = pareto_frontier(cost_results)

    primary = report["results"]["25"]
    primary_comparisons = report["comparisons_vs_live"]["25"]
    decisions = {}
    live = primary[BENCHMARK]
    for name in CANDIDATE_NAMES[1:]:
        metrics = primary[name]
        comparison = primary_comparisons[name]
        beats_growth = all(
            report["comparisons_vs_live"][cost][name][
                "annualized_log_growth_difference"
            ]
            > 0
            for cost in ("10", "25")
        )
        historical_growth_challenger = bool(
            beats_growth
            and metrics["maximum_drawdown"] >= MAXIMUM_DRAWDOWN_BOUNDARY
            and all(
                report["comparisons_vs_live"][cost][name]["positive_slices"] >= 3
                for cost in ("10", "25")
            )
            and all(
                report["comparisons_vs_live"][cost][name][
                    "moving_block_bootstrap"
                ]["median"]
                > 0
                for cost in ("10", "25")
            )
            and report["comparisons_vs_live"]["50"][name][
                "annualized_log_growth_difference"
            ]
            > 0
        )
        risk_efficiency = bool(
            metrics["sharpe_zero_cash_rate"] > live["sharpe_zero_cash_rate"]
            and metrics["calmar"] > live["calmar"]
            and metrics["maximum_drawdown"] - live["maximum_drawdown"]
            >= RISK_EFFICIENCY_DRAWDOWN_IMPROVEMENT
            and metrics["cagr"] >= live["cagr"] - RISK_EFFICIENCY_CAGR_SACRIFICE
        )
        decisions[name] = {
            "historical_growth_challenger": historical_growth_challenger,
            "risk_efficiency_alternative": risk_efficiency,
            "live_authority": False,
        }
    report["decisions"] = decisions
    eligible = [
        (name, primary[name])
        for name in CANDIDATE_NAMES
        if primary[name]["maximum_drawdown"] >= MAXIMUM_DRAWDOWN_BOUNDARY
    ]
    report["growth_winner_under_70pct_drawdown"] = max(
        eligible,
        key=lambda item: item[1]["annualized_log_growth"],
    )[0]
    return report


def _json_default(value: object):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen leveraged-core universe comparison"
    )
    parser.add_argument("--start", default=research.DEFAULT_START)
    parser.add_argument("--end", default=research._default_end_exclusive())
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--save-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap-samples must be at least 100")
    data = (
        load_snapshot(
            args.snapshot,
            requested_start=args.start,
            requested_end_exclusive=args.end,
        )
        if args.snapshot
        else download_universe_data(
            requested_start=args.start,
            requested_end_exclusive=args.end,
        )
    )
    if args.save_snapshot:
        save_snapshot(data, args.save_snapshot)
    report = evaluate(data, bootstrap_samples=args.bootstrap_samples)
    rendered = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

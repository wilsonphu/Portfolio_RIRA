#!/usr/bin/env python3
"""Run the preregistered TCN and graph-risk shadow comparison.

The command is research-only.  It never reads or writes portfolio state,
changes live target weights, or sends a notification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import alpha_core as core
import alpha_research as research
import tcn_shadow as shadow


EXTENDED_TICKERS = shadow.REQUIRED_TICKERS
PRIMARY_COST_BPS = 10.0
COSTS_BPS = (10.0, 25.0)
STARTING_CASH = 10_000.0
DRIFT_TRIGGER = 0.05
DRIFT_DESTINATION = 0.025
BOOTSTRAP_BLOCK = 21
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260809

PATHS = (
    "live_residual_vol55",
    "graph_haircut",
    "tcn_modifier",
    "tcn_graph_hybrid",
    "cash_gated_qld",
)


@dataclass(frozen=True)
class ExtendedMarketData:
    opens: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame
    base: research.AlphaMarketData
    fingerprint: str
    source: str


@dataclass(frozen=True)
class PathResult:
    name: str
    cost_bps: float
    ledger: pd.DataFrame
    metrics: dict[str, object]
    unfilled_final_order: bool


def _normalize_field(
    raw: pd.DataFrame,
    field: str,
    tickers: Sequence[str],
) -> pd.DataFrame:
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Extended market response is empty or malformed")
    first = set(str(item) for item in raw.columns.get_level_values(0))
    second = set(str(item) for item in raw.columns.get_level_values(1))
    if field in first:
        frame = raw[field]
    elif field in second:
        frame = raw.xs(field, axis=1, level=1)
    else:
        raise RuntimeError(f"Extended market response is missing {field}")
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
        raise RuntimeError(f"Extended market data is missing tickers: {sorted(missing)}")
    result = result.loc[:, list(tickers)].apply(pd.to_numeric, errors="coerce")
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise RuntimeError("Extended market dates are duplicated or unsorted")
    return result


def _select_base_raw(raw: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for field in ("Open", "Close", "Volume"):
        frame = _normalize_field(raw, field, research.TICKERS)
        frame.columns = pd.MultiIndex.from_product([[field], frame.columns])
        selected.append(frame)
    return pd.concat(selected, axis=1).sort_index(axis=1)


def _fingerprint_frames(*frames: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        canonical = frame.copy()
        canonical.index = pd.DatetimeIndex(canonical.index).normalize()
        canonical.columns = [str(item) for item in canonical.columns]
        digest.update(
            canonical.to_csv(
                index=True,
                date_format="%Y-%m-%d",
                float_format="%.17g",
                lineterminator="\n",
            ).encode("utf-8")
        )
    return digest.hexdigest()


def prepare_extended_market_data(
    raw: pd.DataFrame,
    *,
    requested_start: str,
    requested_end_exclusive: str,
    source: str,
) -> ExtendedMarketData:
    final_session = research._last_session_before(requested_end_exclusive)
    expected_sessions = research._calendar_sessions(
        pd.Timestamp(requested_start),
        final_session,
    )
    # Context series such as ^VIX can publish observations on dates when XNYS
    # is closed.  They are outside the frozen decision clock and are excluded
    # by an explicit calendar reindex; missing XNYS observations remain NaN and
    # fail validation below.
    base_raw = _select_base_raw(raw).reindex(expected_sessions)
    base = research.prepare_market_data(
        base_raw,
        requested_start=requested_start,
        requested_end_exclusive=requested_end_exclusive,
    )
    opens = _normalize_field(raw, "Open", EXTENDED_TICKERS).reindex(expected_sessions)
    closes = _normalize_field(raw, "Close", EXTENDED_TICKERS).reindex(expected_sessions)
    volumes = _normalize_field(raw, "Volume", EXTENDED_TICKERS).reindex(expected_sessions)
    if not opens.index.equals(expected_sessions) or not closes.index.equals(
        expected_sessions
    ):
        raise RuntimeError("Extended data does not align to the frozen XNYS sessions")
    for field, frame in (("Open", opens), ("Close", closes)):
        scored = frame.loc[base.common_start :]
        values = scored.to_numpy(dtype=float)
        invalid = ~np.isfinite(values) | (values <= 0)
        if invalid.any():
            row, column = np.argwhere(invalid)[0]
            raise RuntimeError(
                f"Extended adjusted {field} is invalid at "
                f"{scored.index[int(row)].date().isoformat()} / "
                f"{scored.columns[int(column)]}"
            )
    return ExtendedMarketData(
        opens=opens,
        closes=closes,
        volumes=volumes,
        base=base,
        fingerprint=_fingerprint_frames(opens, closes, volumes),
        source=source,
    )


def download_extended_market_data(
    *,
    requested_start: str,
    requested_end_exclusive: str,
) -> ExtendedMarketData:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required for a live shadow download") from exc
    raw = yf.download(
        list(EXTENDED_TICKERS),
        start=requested_start,
        end=requested_end_exclusive,
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
        threads=False,
    )
    return prepare_extended_market_data(
        raw,
        requested_start=requested_start,
        requested_end_exclusive=requested_end_exclusive,
        source="Yahoo Finance via yfinance; auto_adjust=True",
    )


def save_extended_snapshot(data: ExtendedMarketData, path: Path) -> None:
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


def load_extended_snapshot(
    path: Path,
    *,
    requested_start: str,
    requested_end_exclusive: str,
) -> ExtendedMarketData:
    snapshot = pd.read_csv(
        path,
        parse_dates=["Session"],
        float_precision="round_trip",
    )
    required = {"Session", "Ticker", "Field", "Value"}
    if set(snapshot.columns) != required:
        raise RuntimeError("Extended snapshot columns are invalid")
    if snapshot.duplicated(["Session", "Ticker", "Field"]).any():
        raise RuntimeError("Extended snapshot contains duplicate observations")
    if set(snapshot["Ticker"]) != set(EXTENDED_TICKERS):
        raise RuntimeError("Extended snapshot ticker universe differs")
    if set(snapshot["Field"]) != {"Open", "Close", "Volume"}:
        raise RuntimeError("Extended snapshot fields differ")
    raw = snapshot.pivot(
        index="Session",
        columns=["Field", "Ticker"],
        values="Value",
    )
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    return prepare_extended_market_data(
        raw,
        requested_start=requested_start,
        requested_end_exclusive=requested_end_exclusive,
        source=f"frozen local snapshot: {path.name}",
    )


def _target(soxl_weight: float) -> dict[str, float]:
    weight = float(np.clip(soxl_weight, 0.0, core.MAX_SOXL_WEIGHT))
    return {"QLD": 1.0 - weight, "SOXL": weight, "CASH": 0.0}


def _cash_gate_target(trend_positive: bool) -> dict[str, float]:
    return (
        {"QLD": 1.0, "SOXL": 0.0, "CASH": 0.0}
        if bool(trend_positive)
        else {"QLD": 0.0, "SOXL": 0.0, "CASH": 1.0}
    )


def _actual_close_weights(
    shares: Mapping[str, float],
    cash: float,
    close_prices: pd.Series,
) -> tuple[dict[str, float], float]:
    values = {
        ticker: float(shares.get(ticker, 0.0)) * float(close_prices[ticker])
        for ticker in research.TRADED_TICKERS
    }
    nav = float(cash + sum(values.values()))
    if not np.isfinite(nav) or nav <= 0:
        raise RuntimeError("Shadow portfolio NAV is invalid")
    return {
        "QLD": values["QLD"] / nav,
        "SOXL": values["SOXL"] / nav,
        "CASH": float(cash) / nav,
    }, nav


def _queue_target(
    *,
    mode: str,
    desired: dict[str, float],
    executed: dict[str, float],
    actual: dict[str, float],
) -> tuple[dict[str, float] | None, str]:
    if mode == "exact":
        if any(abs(desired[item] - executed[item]) > 1e-12 for item in desired):
            return desired, "EXACT_SIGNAL_CHANGE"
    elif mode == "modifier":
        desired_soxl = desired["SOXL"]
        executed_soxl = executed["SOXL"]
        if desired_soxl < executed_soxl - 1e-12:
            return desired, "EXACT_RISK_REDUCTION"
    else:
        raise ValueError("Unknown shadow execution mode")
    desired_soxl = desired["SOXL"]
    difference = actual["SOXL"] - desired_soxl
    if abs(difference) >= DRIFT_TRIGGER - 1e-12:
        buffered = core.buffered_soxl_rebalance_weight(
            actual["SOXL"],
            desired_soxl,
            trigger=DRIFT_TRIGGER,
            destination=DRIFT_DESTINATION,
        )
        if buffered is None:
            raise RuntimeError("Shadow drift trigger failed to produce a destination")
        return _target(buffered), "DRIFT_REBALANCE"
    return None, ""


def simulate_target_schedule(
    data: research.AlphaMarketData,
    targets: pd.DataFrame,
    *,
    name: str,
    cost_bps: float,
    first_signal_date: pd.Timestamp,
    mode: str,
) -> PathResult:
    if name not in PATHS:
        raise ValueError("Shadow path is not preregistered")
    signal = pd.Timestamp(first_signal_date).normalize()
    if signal not in data.sessions:
        raise ValueError("Initial shadow signal is absent from market sessions")
    signal_position = data.sessions.get_loc(signal)
    if not isinstance(signal_position, (int, np.integer)):
        raise RuntimeError("Initial shadow signal is duplicated")
    if int(signal_position) + 1 >= len(data.sessions):
        raise ValueError("Initial shadow signal has no executable next session")
    scoring_dates = data.sessions[int(signal_position) + 1 :]
    required = {"QLD", "SOXL", "CASH"}
    if set(targets.columns) != required or not targets.index.equals(data.sessions):
        raise ValueError("Shadow target schedule shape differs from market data")

    initial_target = {
        item: float(targets.loc[signal, item]) for item in required
    }
    pending: tuple[
        pd.Timestamp,
        dict[str, float],
        str,
        dict[str, float],
    ] | None = (
        signal,
        initial_target,
        "INITIAL_DEPLOYMENT",
        initial_target,
    )
    executed = {"QLD": 0.0, "SOXL": 0.0, "CASH": 1.0}
    shares: dict[str, float] = {}
    cash = STARTING_CASH
    previous_close: pd.Series | None = None
    rows: list[dict[str, object]] = []

    for date in scoring_dates:
        date = pd.Timestamp(date)
        open_prices = data.opens.loc[date, list(research.TRADED_TICKERS)]
        close_prices = data.closes.loc[date, list(research.TRADED_TICKERS)]
        old_shares = dict(shares)
        fill = None
        fill_signal_date = pd.NaT
        fill_reason = ""
        initial_deployment = False
        if pending is not None:
            (
                fill_signal_date,
                execution_target,
                fill_reason,
                strategic_target,
            ) = pending
            fill = research.execute_target_at_open(
                shares=shares,
                cash=cash,
                open_prices=open_prices,
                target_weights=execution_target,
                cost_bps=cost_bps,
            )
            shares = fill.shares
            cash = fill.cash
            executed = dict(strategic_target)
            initial_deployment = fill_reason == "INITIAL_DEPLOYMENT"
            pending = None

        overnight_pnl = {
            ticker: (
                old_shares.get(ticker, 0.0)
                * (float(open_prices[ticker]) - float(previous_close[ticker]))
                if previous_close is not None
                else 0.0
            )
            for ticker in research.TRADED_TICKERS
        }
        intraday_pnl = {
            ticker: float(shares.get(ticker, 0.0))
            * (float(close_prices[ticker]) - float(open_prices[ticker]))
            for ticker in research.TRADED_TICKERS
        }
        actual, nav = _actual_close_weights(shares, cash, close_prices)
        desired = {
            item: float(targets.loc[date, item]) for item in required
        }
        next_target, next_reason = _queue_target(
            mode=mode,
            desired=desired,
            executed=executed,
            actual=actual,
        )
        if next_target is not None:
            # A buffered drift fill changes actual holdings without changing
            # the strategic destination used for later risk comparisons.
            pending = (date, next_target, next_reason, desired)

        rows.append(
            {
                "session": date,
                "nav": nav,
                "target_soxl_weight": desired["SOXL"],
                "executed_soxl_weight": executed["SOXL"],
                "actual_soxl_weight": actual["SOXL"],
                "actual_qld_weight": actual["QLD"],
                "actual_cash_weight": actual["CASH"],
                "fill_signal_date": fill_signal_date,
                "fill_reason": fill_reason,
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
                "qld_pnl": overnight_pnl["QLD"] + intraday_pnl["QLD"],
                "soxl_pnl": overnight_pnl["SOXL"] + intraday_pnl["SOXL"],
                "cost_pnl": -(fill.cost if fill is not None else 0.0),
            }
        )
        previous_close = close_prices
    ledger = pd.DataFrame(rows).set_index("session")
    metrics = calculate_path_metrics(ledger)
    return PathResult(
        name=name,
        cost_bps=float(cost_bps),
        ledger=ledger,
        metrics=metrics,
        unfilled_final_order=pending is not None,
    )


def calculate_path_metrics(ledger: pd.DataFrame) -> dict[str, object]:
    if len(ledger) < 2:
        raise RuntimeError("Shadow path has too few sessions")
    nav = ledger["nav"].astype(float)
    returns = pd.concat(
        [
            pd.Series(
                [float(nav.iloc[0] / STARTING_CASH - 1.0)],
                index=[nav.index[0]],
            ),
            nav.pct_change(fill_method=None).iloc[1:],
        ]
    )
    if not np.isfinite(returns.to_numpy()).all() or (returns <= -1).any():
        raise RuntimeError("Shadow path returns are invalid")
    log_returns = np.log1p(returns)
    annual_log_growth = float(log_returns.mean() * 252.0)
    cagr = float(math.exp(annual_log_growth) - 1.0)
    volatility = float(returns.std(ddof=1) * math.sqrt(252.0))
    sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(252.0))
    running_peak = pd.concat(
        [pd.Series([STARTING_CASH]), nav],
        ignore_index=True,
    ).cummax().iloc[1:].set_axis(nav.index)
    drawdown = nav / running_peak - 1.0
    maximum_drawdown = float(drawdown.min())
    calmar = float(cagr / abs(maximum_drawdown)) if maximum_drawdown < 0 else math.inf
    years = len(returns) / 252.0
    return {
        "observations": len(returns),
        "start": nav.index[0].date().isoformat(),
        "end": nav.index[-1].date().isoformat(),
        "ending_nav": float(nav.iloc[-1]),
        "cagr": cagr,
        "annualized_log_growth": annual_log_growth,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": sharpe,
        "maximum_drawdown": maximum_drawdown,
        "calmar": calmar,
        "annual_gross_turnover": float(
            ledger["gross_trade_fraction"].sum() / years
        ),
        "annual_one_way_turnover": float(
            ledger["one_way_turnover"].sum() / years
        ),
        "total_modeled_cost": float(ledger["transaction_cost"].sum()),
        "rebalance_events": int((ledger["security_orders"] > 0).sum()),
        "mean_soxl_weight": float(ledger["actual_soxl_weight"].mean()),
        "mean_cash_weight": float(ledger["actual_cash_weight"].mean()),
    }


def _path_log_returns(result: PathResult) -> pd.Series:
    nav = result.ledger["nav"].astype(float)
    return pd.concat(
        [
            pd.Series(
                [math.log(float(nav.iloc[0] / STARTING_CASH))],
                index=[nav.index[0]],
            ),
            np.log(nav).diff().iloc[1:],
        ]
    )


def chronological_differences(
    candidate: PathResult,
    benchmark: PathResult,
    *,
    slices: int = 4,
) -> list[float]:
    candidate_returns = _path_log_returns(candidate)
    benchmark_returns = _path_log_returns(benchmark)
    if not candidate_returns.index.equals(benchmark_returns.index):
        raise RuntimeError("Shadow comparison dates differ")
    return [
        float(part.mean() * 252.0)
        for part in np.array_split(candidate_returns - benchmark_returns, slices)
    ]


def moving_block_difference(
    candidate: PathResult,
    benchmark: PathResult,
    *,
    samples: int,
) -> dict[str, float]:
    candidate_returns = _path_log_returns(candidate)
    benchmark_returns = _path_log_returns(benchmark)
    if not candidate_returns.index.equals(benchmark_returns.index):
        raise RuntimeError("Shadow comparison dates differ")
    difference = (candidate_returns - benchmark_returns).to_numpy(dtype=float)
    if len(difference) < BOOTSTRAP_BLOCK:
        raise RuntimeError("Shadow comparison is too short for block bootstrap")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    starts = np.arange(len(difference) - BOOTSTRAP_BLOCK + 1)
    estimates = np.empty(samples, dtype=float)
    blocks = int(math.ceil(len(difference) / BOOTSTRAP_BLOCK))
    for sample in range(samples):
        selected = rng.choice(starts, size=blocks, replace=True)
        draw = np.concatenate(
            [difference[start : start + BOOTSTRAP_BLOCK] for start in selected]
        )[: len(difference)]
        estimates[sample] = float(draw.mean() * 252.0)
    return {
        "median_annualized_log_growth_difference": float(np.median(estimates)),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "probability_nonpositive": float(np.mean(estimates <= 0.0)),
    }


def _target_frame(
    sessions: pd.DatetimeIndex,
    soxl_weights: pd.Series,
) -> pd.DataFrame:
    aligned = soxl_weights.reindex(sessions).astype(float)
    if not np.isfinite(aligned.to_numpy()).all():
        raise RuntimeError("Shadow SOXL target contains missing values")
    return pd.DataFrame(
        {
            "QLD": 1.0 - aligned,
            "SOXL": aligned,
            "CASH": 0.0,
        },
        index=sessions,
    )


def build_shadow_targets(
    data: ExtendedMarketData,
    panel: research.SignalPanel,
    tcn: shadow.WalkForwardResult,
    graph: pd.DataFrame,
) -> tuple[pd.Timestamp, dict[str, tuple[pd.DataFrame, str]]]:
    sessions = data.base.sessions
    live_schedule = research.build_target_schedule(panel, "residual_vol55")
    live = live_schedule["strategic_soxl_weight"].reindex(sessions).astype(float)
    graph_haircut = graph["graph_haircut"].reindex(sessions).fillna(0.0)
    confidence = tcn.predictions["confidence"].reindex(sessions).fillna(0.0)
    predicted = tcn.predictions["fit_date"].reindex(sessions).notna()
    if not predicted.any():
        raise RuntimeError("TCN walk-forward produced no prediction block")
    first_signal = pd.Timestamp(predicted.index[predicted][0])

    qqq = data.base.closes["QQQ"]
    trend = qqq > qqq.rolling(core.SMA_WINDOW, min_periods=core.SMA_WINDOW).mean()
    cash = pd.DataFrame(
        [
            _cash_gate_target(bool(value))
            for value in trend.reindex(sessions).fillna(False)
        ],
        index=sessions,
    )
    cash = cash.loc[:, ["QLD", "SOXL", "CASH"]]

    return first_signal, {
        "live_residual_vol55": (_target_frame(sessions, live), "exact"),
        "graph_haircut": (
            _target_frame(sessions, live * graph_haircut),
            "modifier",
        ),
        "tcn_modifier": (
            _target_frame(sessions, live * confidence),
            "modifier",
        ),
        "tcn_graph_hybrid": (
            _target_frame(sessions, live * confidence * graph_haircut),
            "modifier",
        ),
        "cash_gated_qld": (cash, "exact"),
    }


def evaluate(
    data: ExtendedMarketData,
    *,
    bootstrap_samples: int,
    progress=None,
) -> tuple[dict[str, object], shadow.WalkForwardResult]:
    panel = research.build_signal_panel(data.base)
    features = shadow.build_feature_frame(data.closes.loc[data.base.common_start :])
    labels = shadow.build_relative_label_frame(data.opens.loc[data.base.common_start :])
    tcn = shadow.walk_forward_tcn(features, labels, progress=progress)
    graph = shadow.build_graph_haircut_frame(
        data.closes.loc[data.base.common_start :]
    )
    first_signal, targets = build_shadow_targets(data, panel, tcn, graph)

    results: dict[tuple[str, float], PathResult] = {}
    for cost in COSTS_BPS:
        for name, (target, mode) in targets.items():
            results[(name, cost)] = simulate_target_schedule(
                data.base,
                target,
                name=name,
                cost_bps=cost,
                first_signal_date=first_signal,
                mode=mode,
            )

    known = tcn.predictions["probability"].notna() & labels["label"].notna()
    probabilities = tcn.predictions.loc[known, "probability"].to_numpy(dtype=float)
    observed = labels.loc[known, "label"].to_numpy(dtype=float)
    base_rate = float(observed.mean())
    brier = float(np.mean((probabilities - observed) ** 2))
    brier_baseline = float(np.mean((base_rate - observed) ** 2))
    calibration = {
        "observations": int(known.sum()),
        "base_rate": base_rate,
        "brier": brier,
        "brier_baseline": brier_baseline,
        "brier_skill": float(1.0 - brier / brier_baseline),
        "log_loss": shadow.binary_log_loss(observed, probabilities),
        "ece_10": shadow.expected_calibration_error(observed, probabilities),
        "qualified_prediction_fraction": float(
            tcn.predictions.loc[known, "qualified"].mean()
        ),
        "mean_confidence": float(tcn.predictions.loc[known, "confidence"].mean()),
    }

    report: dict[str, object] = {
        "protocol": "TCN_SHADOW_PROTOCOL.md",
        "status": "exploratory_post_selection_shadow_only",
        "live_authority": False,
        "permanent_qld_core_selected": True,
        "data": {
            "source": data.source,
            "fingerprint": data.fingerprint,
            "base_fingerprint": data.base.fingerprint,
            "start": data.base.sessions[0].date().isoformat(),
            "end": data.base.sessions[-1].date().isoformat(),
            "first_shadow_signal": first_signal.date().isoformat(),
        },
        "tcn": {
            "feature_names": list(shadow.FEATURE_NAMES),
            "parameter_count": shadow.tcn_parameter_count(),
            "fit_count": len(tcn.fits),
            "calibration": calibration,
            "fits": [
                {
                    **asdict(item),
                    "fit_date": item.fit_date.date().isoformat(),
                    "train_start": item.train_start.date().isoformat(),
                    "train_end": item.train_end.date().isoformat(),
                    "validation_start": item.validation_start.date().isoformat(),
                    "validation_end": item.validation_end.date().isoformat(),
                    "calibration_start": item.calibration_start.date().isoformat(),
                    "calibration_end": item.calibration_end.date().isoformat(),
                }
                for item in tcn.fits
            ],
        },
        "graph": {
            "available_sessions": int(graph["graph_available"].sum()),
            "haircut_sessions": int((graph["graph_haircut"] < 1.0).sum()),
            "zero_authority_sessions": int((graph["graph_haircut"] <= 0.0).sum()),
        },
        "results": {},
        "comparisons_vs_live": {},
    }
    for cost in COSTS_BPS:
        report["results"][str(int(cost))] = {
            name: {
                **results[(name, cost)].metrics,
                "unfilled_final_order": results[(name, cost)].unfilled_final_order,
            }
            for name in PATHS
        }
        live = results[("live_residual_vol55", cost)]
        report["comparisons_vs_live"][str(int(cost))] = {}
        for name in PATHS[1:]:
            candidate = results[(name, cost)]
            slices = chronological_differences(candidate, live)
            report["comparisons_vs_live"][str(int(cost))][name] = {
                "annualized_log_growth_difference": float(
                    candidate.metrics["annualized_log_growth"]
                    - live.metrics["annualized_log_growth"]
                ),
                "chronological_slice_differences": slices,
                "positive_slices": int(sum(value > 0.0 for value in slices)),
                "moving_block_bootstrap": moving_block_difference(
                    candidate,
                    live,
                    samples=bootstrap_samples,
                ),
            }
    return report, tcn


def _json_default(value: object):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _default_end_exclusive() -> str:
    return research._default_end_exclusive()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered shadow-only TCN/graph comparison"
    )
    parser.add_argument("--start", default=research.DEFAULT_START)
    parser.add_argument("--end", default=_default_end_exclusive())
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--save-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start != research.DEFAULT_START:
        raise RuntimeError(
            f"Frozen shadow history must start at {research.DEFAULT_START}"
        )
    if args.bootstrap_samples < 100:
        raise ValueError("Bootstrap sample count must be at least 100")
    data = (
        load_extended_snapshot(
            args.snapshot,
            requested_start=args.start,
            requested_end_exclusive=args.end,
        )
        if args.snapshot
        else download_extended_market_data(
            requested_start=args.start,
            requested_end_exclusive=args.end,
        )
    )
    if args.save_snapshot:
        save_extended_snapshot(data, args.save_snapshot)
    def progress(item: shadow.WalkForwardFit) -> None:
        print(
            "TCN fit "
            f"{item.fit_date.date().isoformat()} "
            f"train={item.train_count} "
            f"qualified={item.qualified} "
            f"brier_skill={item.calibration_brier_skill:+.4f}",
            flush=True,
        )

    report, tcn = evaluate(
        data,
        bootstrap_samples=args.bootstrap_samples,
        progress=progress,
    )
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
        prediction_path = args.output.with_name(args.output.stem + "_predictions.csv")
        tcn.predictions.to_csv(
            prediction_path,
            index=True,
            date_format="%Y-%m-%d",
            float_format="%.17g",
            lineterminator="\n",
        )
    if args.output:
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "data": report["data"],
                    "tcn_calibration": report["tcn"]["calibration"],
                    "results": report["results"],
                    "comparisons_vs_live": report["comparisons_vs_live"],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
        )
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

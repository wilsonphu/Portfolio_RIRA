#!/usr/bin/env python3
"""Production lifecycle Roth allocation engine.

Signals are calculated from completed, adjusted daily bars. The permanent core
starts at 65% TQQQ / 35% UGL; a volatility-sized SOXL overlay is admitted only
by the frozen QQQ-trend and SMH-residual-strength rules. As portfolio value and
investor age advance, a one-way lifecycle ratchet replaces 3x products with
2x/1x equivalents while preserving the model's risk-source proportions.
Confirmed broker shares and cash are always the source of truth.
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


# ---------------------------------------------------------------------------
# Configuration and universe
# ---------------------------------------------------------------------------
MARKET_INDEX = core.QQQ
SEMICONDUCTOR_SIGNAL = core.SMH
VOLATILITY_INDEX = core.QLD
LEVERAGED_INDEX = "TQQQ"
LEVERAGED_GOLD = "UGL"
LEVERAGED_SEMICONDUCTOR = core.SOXL
FORWARD_ANTI_BETA = "BTAL"
FORWARD_BROAD_EQUITY = "SPY"
FORWARD_LEVERAGED_BROAD_EQUITY = "SSO"
DOUBLE_SEMICONDUCTOR = "USD"
UNLEVERAGED_INDEX = "QQQM"
UNLEVERAGED_GOLD = "GLDM"
UNLEVERAGED_SEMICONDUCTOR = core.SMH
TREASURY_RESERVE = "SGOV"
CASH_ASSET = core.CASH

CORE_TQQQ_SHARE = 0.65
CORE_UGL_SHARE = 0.35
MAX_ADVERTISED_DAILY_EXPOSURE = 2.7725

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
# Portfolio milestones are stated in 2026 dollars and automatically indexed.
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

# These products can appear in confirmed pre-v8 holdings and must remain
# priceable until they are explicitly sold.  They are never strategic targets.
LEGACY_TECH = "TECL"
LEGACY_DEFENSIVE_EQUITY = "SPMO"
LEGACY_HEDGE = "GLD"
LEGACY_HOLDINGS = frozenset(
    {
        LEGACY_TECH,
        LEGACY_DEFENSIVE_EQUITY,
        LEGACY_HEDGE,
    }
)

SIGNAL_TICKERS = (MARKET_INDEX, SEMICONDUCTOR_SIGNAL)
VOLATILITY_TICKERS = (VOLATILITY_INDEX, LEVERAGED_SEMICONDUCTOR)
STRATEGIC_TICKERS = (
    LEVERAGED_INDEX,
    LEVERAGED_GOLD,
    LEVERAGED_SEMICONDUCTOR,
    VOLATILITY_INDEX,
    DOUBLE_SEMICONDUCTOR,
    UNLEVERAGED_INDEX,
    UNLEVERAGED_GOLD,
    UNLEVERAGED_SEMICONDUCTOR,
    TREASURY_RESERVE,
)
EQUITY_TICKERS = (
    LEVERAGED_INDEX,
    LEVERAGED_SEMICONDUCTOR,
    VOLATILITY_INDEX,
    DOUBLE_SEMICONDUCTOR,
    UNLEVERAGED_INDEX,
    UNLEVERAGED_SEMICONDUCTOR,
)
SEMICONDUCTOR_HOLDINGS = (
    LEVERAGED_SEMICONDUCTOR,
    DOUBLE_SEMICONDUCTOR,
    UNLEVERAGED_SEMICONDUCTOR,
)
# Newer lifecycle holdings need only a valid latest price. They are deliberately
# excluded from complete-history model validation and cannot affect signals.
MODEL_TICKERS = tuple(dict.fromkeys((*SIGNAL_TICKERS, *VOLATILITY_TICKERS)))
VALUATION_TICKERS = tuple(
    sorted(set(STRATEGIC_TICKERS) | set(LEGACY_HOLDINGS))
)
ALL_TICKERS = tuple(
    dict.fromkeys(
        (*MODEL_TICKERS, *STRATEGIC_TICKERS, *sorted(LEGACY_HOLDINGS))
    )
)
TRADED_TICKERS = frozenset(VALUATION_TICKERS)
PORTFOLIO_COMPONENTS = TRADED_TICKERS | {CASH_ASSET}

SMA_WINDOW = core.SMA_WINDOW
ALPHA_REVIEW_SESSIONS = core.ALPHA_REVIEW_SESSIONS
REBALANCE_BAND = 0.05
REBALANCE_DESTINATION = 0.025
NOTIFICATION_WEIGHT_TOLERANCE = 0.005
TRANSACTION_COST_SCENARIOS_BPS = (5, 10, 25)
MODEL_START_DATE = core.MODEL_HISTORY_START
REQUIRED_SIGNAL_ROWS = 840

STRATEGY_REVISION = "tqqq65-ugl35-soxl-delayed-lifecycle-v5"
EXPERIMENTAL_LIVE = True
STATE_VERSION = 13
DECISION_AUDIT_SCHEMA_VERSION = 4
SHADOW_LEDGER_SCHEMA_VERSION = 2
LEGACY_SHADOW_LEDGER_SCHEMA_VERSION = 1
NEW_YORK = ZoneInfo("America/New_York")
MARKET_CLOSE_BUFFER_MINUTES = 15

APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / "roth_ira_state.json"
LOG_FILE = APP_DIR / "roth_ira.log"
DECISION_AUDIT_FILE = APP_DIR / "roth_ira_decision.json"
SHADOW_LEDGER_FILE = APP_DIR / "roth_ira_shadow_ledger.jsonl"

ADVERTISED_DAILY_MULTIPLIERS = {
    VOLATILITY_INDEX: 2.0,
    LEVERAGED_INDEX: 3.0,
    LEVERAGED_GOLD: 2.0,
    LEVERAGED_SEMICONDUCTOR: 3.0,
    FORWARD_BROAD_EQUITY: 1.0,
    FORWARD_LEVERAGED_BROAD_EQUITY: 2.0,
    DOUBLE_SEMICONDUCTOR: 2.0,
    UNLEVERAGED_INDEX: 1.0,
    UNLEVERAGED_GOLD: 1.0,
    TREASURY_RESERVE: 0.0,
    SEMICONDUCTOR_SIGNAL: 1.0,
    LEGACY_TECH: 3.0,
    LEGACY_DEFENSIVE_EQUITY: 1.0,
    LEGACY_HEDGE: 1.0,
    CASH_ASSET: 0.0,
}


def aggressive_target_weights(soxl_weight: float) -> dict[str, float]:
    """Return the frozen proportional TQQQ/UGL core plus SOXL overlay."""
    # Reuse the audited core validator and grid boundary without inheriting its
    # legacy QLD allocation.
    core.target_weights(soxl_weight)
    weight = min(max(float(soxl_weight), 0.0), core.MAX_SOXL_WEIGHT)
    remaining = 1.0 - weight
    result = {
        LEVERAGED_INDEX: CORE_TQQQ_SHARE * remaining,
        LEVERAGED_GOLD: CORE_UGL_SHARE * remaining,
        LEVERAGED_SEMICONDUCTOR: weight,
    }
    if not np.isclose(sum(result.values()), 1.0, atol=1e-12):
        raise RuntimeError("Strategic target weights do not sum to 1.0")
    return result


def _risk_source_weights(soxl_weight: float) -> tuple[dict[str, float], float]:
    aggressive = aggressive_target_weights(soxl_weight)
    source_exposure = {
        "nasdaq": 3.0 * aggressive[LEVERAGED_INDEX],
        "gold": 2.0 * aggressive[LEVERAGED_GOLD],
        "semiconductors": 3.0 * aggressive[LEVERAGED_SEMICONDUCTOR],
    }
    total = float(sum(source_exposure.values()))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Aggressive source exposure is invalid")
    return (
        {name: value / total for name, value in source_exposure.items()},
        total,
    )


def _blend_weights(
    left: dict[str, float],
    right: dict[str, float],
    left_fraction: float,
) -> dict[str, float]:
    fraction = float(left_fraction)
    if not np.isfinite(fraction) or fraction < 0 or fraction > 1:
        raise ValueError("Blend fraction must be in [0, 1]")
    result = {
        ticker: fraction * left.get(ticker, 0.0)
        + (1.0 - fraction) * right.get(ticker, 0.0)
        for ticker in set(left) | set(right)
    }
    return {ticker: weight for ticker, weight in result.items() if weight > 1e-12}


def target_weights(
    soxl_weight: float,
    lifecycle_stage: str = LIFECYCLE_SPRINT,
) -> dict[str, float]:
    """Map the alpha sleeves into the delivery leverage for a lifecycle stage."""
    if lifecycle_stage not in LIFECYCLE_STAGES:
        raise ValueError(f"Unknown lifecycle stage: {lifecycle_stage!r}")
    aggressive = aggressive_target_weights(soxl_weight)
    if lifecycle_stage == LIFECYCLE_SPRINT:
        return aggressive

    sources, aggressive_exposure = _risk_source_weights(soxl_weight)
    double = {
        VOLATILITY_INDEX: sources["nasdaq"],
        LEVERAGED_GOLD: sources["gold"],
        DOUBLE_SEMICONDUCTOR: sources["semiconductors"],
    }
    single = {
        UNLEVERAGED_INDEX: sources["nasdaq"],
        UNLEVERAGED_GOLD: sources["gold"],
        UNLEVERAGED_SEMICONDUCTOR: sources["semiconductors"],
    }
    ceiling = LIFECYCLE_EXPOSURE_CEILINGS[lifecycle_stage]
    if ceiling is None:
        raise RuntimeError("Non-sprint lifecycle stage has no exposure ceiling")
    if ceiling >= 2.0:
        fraction = (ceiling - 2.0) / (aggressive_exposure - 2.0)
        result = _blend_weights(aggressive, double, fraction)
    elif ceiling >= 1.0:
        result = _blend_weights(double, single, ceiling - 1.0)
    else:
        result = {
            ticker: ceiling * weight for ticker, weight in single.items()
        }
        result[TREASURY_RESERVE] = 1.0 - ceiling
    if not np.isclose(sum(result.values()), 1.0, atol=1e-12):
        raise RuntimeError("Lifecycle target weights do not sum to 1.0")
    if not np.isclose(advertised_daily_exposure(result), ceiling, atol=1e-12):
        raise RuntimeError("Lifecycle target exposure does not match its ceiling")
    return result


def strategic_daily_exposure(
    soxl_weight: float,
    lifecycle_stage: str = LIFECYCLE_SPRINT,
) -> float:
    return advertised_daily_exposure(target_weights(soxl_weight, lifecycle_stage))


def strategy_manifest() -> dict[str, object]:
    """Return every production decision boundary in canonical form."""
    return {
        "revision": STRATEGY_REVISION,
        "governance_status": "experimental_live_user_override",
        "quantitative_core_semantic_revision": (
            core.DECISION_SEMANTIC_REVISION
        ),
        "universe": {
            "signals": list(SIGNAL_TICKERS),
            "volatility_sizing": list(VOLATILITY_TICKERS),
            "strategic_holdings": list(STRATEGIC_TICKERS),
            "legacy_valuation_only": sorted(LEGACY_HOLDINGS),
        },
        "trend": {
            "rule": "QQQ_close_strictly_above_SMA",
            "window": SMA_WINDOW,
        },
        "residual_signal": {
            "beta_window": core.BETA_WINDOW,
            "score_window": core.RESIDUAL_SCORE_WINDOW,
            "positive_comparison": "strict",
            "review_sessions": ALPHA_REVIEW_SESSIONS,
            "review_clock": (
                "fixed_21_session_phase_from_model_history_start"
            ),
            "reentry_closes": core.BULLISH_REENTRY_CLOSES,
        },
        "volatility": {
            "features": [5, 21, 63, "downside_21"],
            "history_start": MODEL_START_DATE,
            "training_window": "expanding_from_fixed_history_start",
            "forecast_horizon": core.VARIANCE_HORIZON,
            "minimum_training_rows": core.VARIANCE_MIN_TRAINING,
            "ridge_alpha": core.RIDGE_ALPHA,
            "budget": core.VOLATILITY_BUDGET,
            "soxl_grid": list(core.SOXL_WEIGHT_GRID),
            "upshift_closes": core.VOLATILITY_UPSHIFT_CLOSES,
            "downshift": "immediate",
        },
        "allocation": {
            "core": {
                "tqqq": CORE_TQQQ_SHARE,
                "ugl": CORE_UGL_SHARE,
                "application": "proportional_to_one_minus_soxl",
            },
            "soxl": "stateful_residual_momentum_overlay",
            "maximum_soxl": core.MAX_SOXL_WEIGHT,
            "maximum_advertised_daily_exposure": (
                MAX_ADVERTISED_DAILY_EXPOSURE
            ),
            "lifecycle": {
                "stages": list(LIFECYCLE_STAGES),
                "exposure_ceilings": LIFECYCLE_EXPOSURE_CEILINGS,
                "value_thresholds_2026_dollars": [
                    [threshold, stage]
                    for threshold, stage in LIFECYCLE_VALUE_THRESHOLDS_2026
                ],
                "age_thresholds": [
                    [age, stage] for age, stage in LIFECYCLE_AGE_THRESHOLDS
                ],
                "inflation_rate": LIFECYCLE_INFLATION_RATE,
                "anchor_date": LIFECYCLE_ANCHOR_DATE.isoformat(),
                "anchor_age": LIFECYCLE_ANCHOR_AGE,
                "birth_date_override": "optional_INVESTOR_BIRTH_DATE",
                "ratchet": "one_way_never_relever_after_stage_advance",
                "mapping": {
                    "nasdaq": [LEVERAGED_INDEX, VOLATILITY_INDEX, UNLEVERAGED_INDEX],
                    "gold": [LEVERAGED_GOLD, UNLEVERAGED_GOLD],
                    "semiconductors": [
                        LEVERAGED_SEMICONDUCTOR,
                        DOUBLE_SEMICONDUCTOR,
                        UNLEVERAGED_SEMICONDUCTOR,
                    ],
                    "reserve": TREASURY_RESERVE,
                },
            },
        },
        "execution": {
            "signal": "completed_close",
            "fill": "next_session",
            "missed_sessions": "replay_all_unseen_completed_sessions",
            "drift_trigger": REBALANCE_BAND,
            "non_soxl_drift_destination": REBALANCE_DESTINATION,
            "soxl_drift_destination": "exact_strategic_tier",
            "risk_off_soxl_exit": "exact_and_bypasses_drift",
            "actual_shares_and_cash": "sole_current_weight_source",
        },
    }


def operational_manifest() -> dict[str, object]:
    return {
        "data_provider": "yfinance",
        "auto_adjust": True,
        "model_start_date": MODEL_START_DATE,
        "model_history": "complete_contiguous_XNYS_history_from_fixed_start",
        "required_contiguous_sessions": REQUIRED_SIGNAL_ROWS,
        "missing_data_policy": "fail_closed_no_fill_drop_or_substitution",
        "market_close_buffer_minutes": MARKET_CLOSE_BUFFER_MINUTES,
        "notification_weight_tolerance": NOTIFICATION_WEIGHT_TOLERANCE,
        "transaction_cost_scenarios_bps": list(
            TRANSACTION_COST_SCENARIOS_BPS
        ),
    }


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calculate_strategy_fingerprint(
    manifest: dict[str, object] | None = None,
) -> str:
    return canonical_sha256(
        strategy_manifest() if manifest is None else manifest
    )


def _forward_experiment_manifest_v1() -> dict[str, object]:
    """Return the original frozen prospective comparison protocol."""
    return {
        "revision": "prospective-allocation-shadow-v1",
        "start_rule": "first_v2_shadow_observation",
        "information": "completed_adjusted_close_only",
        "execution": "next_session_close_proxy",
        "distributions": "adjusted_close_total_return",
        "contributions": "excluded",
        "transaction_cost_bps_per_one_way_turnover": 25,
        "comparators": {
            "production_v5": {
                "allocation": "live_alpha_and_lifecycle_target",
                "rebalance": "live_stateful_execution_rules",
            },
            "permanent_sprint": {
                "allocation": "live_alpha_with_sprint_delivery_forever",
                "rebalance": "same_stateful_alpha_target",
            },
            "qld_soxl_same_alpha": {
                "allocation": "QLD_one_minus_SOXL_plus_SOXL",
                "rebalance": "same_stateful_alpha_target",
            },
            "tqqq_btal_50_50_band_5pp": {
                "allocation": {
                    "TQQQ": 0.5,
                    "BTAL": 0.5,
                },
                "rebalance_trigger": (
                    "either_weight_at_least_5pp_from_target"
                ),
                "rebalance_destination": {
                    "TQQQ": 0.5,
                    "BTAL": 0.5,
                },
            },
        },
        "promotion_rule": (
            "no_production_change_without_prospective_net_log_growth_and_"
            "drawdown_review"
        ),
    }


def forward_experiment_manifest() -> dict[str, object]:
    """Return the second frozen prospective comparison protocol.

    These policies are research shadows only. They cannot alter production
    targets, broker holdings, drift decisions, or notification behavior.
    """
    manifest = copy.deepcopy(_forward_experiment_manifest_v1())
    manifest["revision"] = "prospective-allocation-shadow-v2"
    manifest["start_rule"] = (
        "first_observation_with_prospective_allocation_shadow_v2_fingerprint"
    )
    comparators = manifest["comparators"]
    if not isinstance(comparators, dict):
        raise RuntimeError("Forward comparator manifest is invalid")
    comparators.update(
        {
            "production_v5_band_2_5pp": {
                "allocation": "live_alpha_and_lifecycle_target",
                "rebalance_trigger": (
                    "either_weight_at_least_2_5pp_from_target"
                ),
                "rebalance_destination": "1_25pp_from_target",
                "status": "shadow_only_frequency_sensitivity",
            },
            "tqqq_spy_65_35_same_alpha_lifecycle": {
                "allocation": (
                    "replace_UGL_with_SPY_preserve_65_35_core_and_"
                    "source_exposure_through_lifecycle"
                ),
                "rebalance": "live_5pp_trigger_2_5pp_destination",
                "lifecycle": (
                    "shared_production_v5_stage_control_not_independent_nav"
                ),
                "status": "shadow_only_controlled_sleeve_substitution",
            },
            "tqqq_sso_65_35_same_alpha_lifecycle": {
                "allocation": (
                    "replace_UGL_with_SSO_and_GLDM_with_SPY_through_"
                    "the_same_lifecycle"
                ),
                "rebalance": "live_5pp_trigger_2_5pp_destination",
                "lifecycle": (
                    "shared_production_v5_stage_control_not_independent_nav"
                ),
                "status": "shadow_only_controlled_sleeve_substitution",
            },
        }
    )
    return manifest


LEGACY_FORWARD_EXPERIMENT_FINGERPRINT = (
    "1fa3b361e9e8884aa8241644d9738ff096ea1dd3c1567a654295998e1d7d3e59"
)
if canonical_sha256(_forward_experiment_manifest_v1()) != (
    LEGACY_FORWARD_EXPERIMENT_FINGERPRINT
):
    raise RuntimeError("Frozen v1 forward experiment manifest changed")
FORWARD_EXPERIMENT_FINGERPRINT = (
    "d629684eb029b6730cf4bba3c95af254d15a30b007925a8e3240cde3fa41d5d6"
)
if canonical_sha256(forward_experiment_manifest()) != (
    FORWARD_EXPERIMENT_FINGERPRINT
):
    raise RuntimeError("Frozen v2 forward experiment manifest changed")


_FORWARD_V1_LIFECYCLE_EXPOSURE_CEILINGS = {
    "SPRINT": None,
    "GLIDE_225": 2.25,
    "TWO_X": 2.0,
    "PHI": (1.0 + 5.0**0.5) / 2.0,
    "ONE_THREE": 1.30,
    "ONE_X": 1.0,
    "RETIREMENT": 0.75,
}
_FORWARD_V1_ADVERTISED_MULTIPLIERS = {
    "TQQQ": 3.0,
    "UGL": 2.0,
    "SOXL": 3.0,
    "QLD": 2.0,
    "USD": 2.0,
    "QQQM": 1.0,
    "GLDM": 1.0,
    "SMH": 1.0,
    "SGOV": 0.0,
    "SPY": 1.0,
    "SSO": 2.0,
}


def _forward_v1_soxl_weight(soxl_weight: float) -> float:
    weight = float(soxl_weight)
    if not any(
        abs(weight - tier) <= 1e-12 for tier in (0.0, 0.15, 0.25, 0.35)
    ):
        raise ValueError("SOXL target weight is not a frozen v1 tier")
    return weight


def _forward_v1_exposure(weights: dict[str, float]) -> float:
    try:
        return float(
            sum(
                weight * _FORWARD_V1_ADVERTISED_MULTIPLIERS[ticker]
                for ticker, weight in weights.items()
            )
        )
    except KeyError as exc:
        raise RuntimeError("Frozen forward target has an unknown ticker") from exc


def _forward_v1_target_weights(
    soxl_weight: float,
    lifecycle_stage: str,
) -> dict[str, float]:
    """Reproduce v5 targets without depending on mutable live aliases."""
    weight = _forward_v1_soxl_weight(soxl_weight)
    if lifecycle_stage not in _FORWARD_V1_LIFECYCLE_EXPOSURE_CEILINGS:
        raise ValueError(f"Unknown frozen v1 lifecycle stage: {lifecycle_stage!r}")
    remaining = 1.0 - weight
    aggressive = {
        "TQQQ": 0.65 * remaining,
        "UGL": 0.35 * remaining,
        "SOXL": weight,
    }
    if lifecycle_stage == "SPRINT":
        return aggressive

    source_exposure = {
        "nasdaq": 3.0 * aggressive["TQQQ"],
        "gold": 2.0 * aggressive["UGL"],
        "semiconductors": 3.0 * aggressive["SOXL"],
    }
    aggressive_exposure = float(sum(source_exposure.values()))
    sources = {
        name: exposure / aggressive_exposure
        for name, exposure in source_exposure.items()
    }
    double = {
        "QLD": sources["nasdaq"],
        "UGL": sources["gold"],
        "USD": sources["semiconductors"],
    }
    single = {
        "QQQM": sources["nasdaq"],
        "GLDM": sources["gold"],
        "SMH": sources["semiconductors"],
    }
    ceiling = _FORWARD_V1_LIFECYCLE_EXPOSURE_CEILINGS[lifecycle_stage]
    if ceiling is None:
        raise RuntimeError("Frozen non-sprint stage has no exposure ceiling")
    if ceiling >= 2.0:
        fraction = (ceiling - 2.0) / (aggressive_exposure - 2.0)
        left, right = aggressive, double
    elif ceiling >= 1.0:
        fraction = ceiling - 1.0
        left, right = double, single
    else:
        result = {
            ticker: ceiling * target_weight
            for ticker, target_weight in single.items()
        }
        result["SGOV"] = 1.0 - ceiling
        return result
    result = {
        ticker: fraction * left.get(ticker, 0.0)
        + (1.0 - fraction) * right.get(ticker, 0.0)
        for ticker in set(left) | set(right)
    }
    return {
        ticker: target_weight
        for ticker, target_weight in result.items()
        if target_weight > 1e-12
    }


def _legacy_forward_shadow_targets(
    soxl_weight: float,
    lifecycle_stage: str,
) -> dict[str, dict[str, float]]:
    weight = _forward_v1_soxl_weight(soxl_weight)
    return {
        "production_v5": _forward_v1_target_weights(weight, lifecycle_stage),
        "permanent_sprint": _forward_v1_target_weights(weight, "SPRINT"),
        "qld_soxl_same_alpha": {"QLD": 1.0 - weight, "SOXL": weight},
        "tqqq_btal_50_50_band_5pp": {
            "TQQQ": 0.5,
            "BTAL": 0.5,
        },
    }


def _forward_spy_target_weights(
    soxl_weight: float,
    lifecycle_stage: str,
) -> dict[str, float]:
    """Return a source-preserving SPY-core shadow target.

    SPY is a 1x sleeve, so it cannot use production's 3x/2x/1x mapper.
    The non-sprint mapping instead preserves each source's advertised
    exposure while using the least leverage needed in the Nasdaq and
    semiconductor sleeves to hit the lifecycle ceiling exactly.
    """
    weight = _forward_v1_soxl_weight(soxl_weight)
    if lifecycle_stage not in _FORWARD_V1_LIFECYCLE_EXPOSURE_CEILINGS:
        raise ValueError(
            f"Unknown frozen v2 lifecycle stage: {lifecycle_stage!r}"
        )
    remaining = 1.0 - weight
    aggressive = {
        "TQQQ": 0.65 * remaining,
        "SPY": 0.35 * remaining,
        "SOXL": weight,
    }
    if lifecycle_stage == "SPRINT":
        return aggressive

    source_exposure = {
        "nasdaq": 3.0 * aggressive["TQQQ"],
        "broad_equity": aggressive["SPY"],
        "semiconductors": 3.0 * aggressive["SOXL"],
    }
    total_exposure = float(sum(source_exposure.values()))
    sources = {
        name: exposure / total_exposure
        for name, exposure in source_exposure.items()
    }
    ceiling = _FORWARD_V1_LIFECYCLE_EXPOSURE_CEILINGS[lifecycle_stage]
    if ceiling is None:
        raise RuntimeError("Non-sprint lifecycle stage has no exposure ceiling")
    if ceiling < 1.0:
        result = {
            "QQQM": ceiling * sources["nasdaq"],
            "SPY": ceiling * sources["broad_equity"],
            "SMH": ceiling * sources["semiconductors"],
            "SGOV": 1.0 - ceiling,
        }
    else:
        broad_weight = ceiling * sources["broad_equity"]
        other_source_share = sources["nasdaq"] + sources["semiconductors"]
        other_capital = 1.0 - broad_weight
        delivery_leverage = ceiling * other_source_share / other_capital
        if delivery_leverage < 1.0 - 1e-12 or (
            delivery_leverage > 3.0 + 1e-12
        ):
            raise RuntimeError("SPY shadow delivery leverage is infeasible")
        result = {"SPY": broad_weight}
        delivery_pairs = {
            "nasdaq": ("TQQQ", "QLD", "QQQM"),
            "semiconductors": ("SOXL", "USD", "SMH"),
        }
        for source_name, (triple, double, single) in delivery_pairs.items():
            source_capital = (
                ceiling * sources[source_name] / delivery_leverage
            )
            if delivery_leverage >= 2.0:
                source_delivery = {
                    triple: delivery_leverage - 2.0,
                    double: 3.0 - delivery_leverage,
                }
            else:
                source_delivery = {
                    double: delivery_leverage - 1.0,
                    single: 2.0 - delivery_leverage,
                }
            for ticker, fraction in source_delivery.items():
                result[ticker] = (
                    result.get(ticker, 0.0) + source_capital * fraction
                )
    result = {
        ticker: value for ticker, value in result.items() if value > 1e-12
    }
    if not np.isclose(sum(result.values()), 1.0, atol=1e-12):
        raise RuntimeError("SPY shadow target weights do not sum to 1.0")
    if not np.isclose(_forward_v1_exposure(result), ceiling, atol=1e-12):
        raise RuntimeError("SPY shadow target misses lifecycle exposure ceiling")
    return result


def _forward_sso_target_weights(
    soxl_weight: float,
    lifecycle_stage: str,
) -> dict[str, float]:
    """Return the production lifecycle map with SSO/SPY replacing gold."""
    ticker_map = {"UGL": "SSO", "GLDM": "SPY"}
    result: dict[str, float] = {}
    for ticker, weight in _forward_v1_target_weights(
        soxl_weight,
        lifecycle_stage,
    ).items():
        replacement = ticker_map.get(ticker, ticker)
        result[replacement] = result.get(replacement, 0.0) + weight
    return result


def forward_shadow_targets(
    soxl_weight: float,
    lifecycle_stage: str,
    experiment_fingerprint: str | None = None,
) -> dict[str, dict[str, float]]:
    """Return exact target maps for every frozen prospective comparator."""
    fingerprint = experiment_fingerprint or FORWARD_EXPERIMENT_FINGERPRINT
    result = _legacy_forward_shadow_targets(soxl_weight, lifecycle_stage)
    if fingerprint == LEGACY_FORWARD_EXPERIMENT_FINGERPRINT:
        return result
    if fingerprint != FORWARD_EXPERIMENT_FINGERPRINT:
        raise ValueError("Unknown forward experiment fingerprint")
    result.update(
        {
            "production_v5_band_2_5pp": _forward_v1_target_weights(
                soxl_weight,
                lifecycle_stage,
            ),
            "tqqq_spy_65_35_same_alpha_lifecycle": (
                _forward_spy_target_weights(soxl_weight, lifecycle_stage)
            ),
            "tqqq_sso_65_35_same_alpha_lifecycle": (
                _forward_sso_target_weights(soxl_weight, lifecycle_stage)
            ),
        }
    )
    return result


def calculate_implementation_fingerprint() -> str:
    return canonical_sha256(
        {
            "port12_cloud_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "alpha_core_sha256": hashlib.sha256(
                Path(core.__file__).read_bytes()
            ).hexdigest(),
        }
    )


STRATEGY_FINGERPRINT = calculate_strategy_fingerprint()
EXPECTED_STRATEGY_FINGERPRINT = (
    "93f2811533f67a18cc20e8120778567c8ebde2526bb48cc6ea024c8fc64be0d7"
)

_configured_roth_amount = os.environ.get("ROTH_IRA_AMOUNT", "").strip()
try:
    ROTH_IRA_AMOUNT = (
        float(_configured_roth_amount) if _configured_roth_amount else None
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
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if persist_log:
        handlers.insert(0, logging.FileHandler(LOG_FILE))
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShadowObservation:
    signal_date: str
    data_fingerprint: str
    trend_positive: bool
    residual_positive: bool
    raw_soxl_weight: float
    overlay_active: bool
    soxl_weight: float
    transition_reason: str
    structural_change: bool
    failure_reason: str
    lifecycle_stage: str
    forward_experiment_fingerprint: str
    forward_targets: dict[str, dict[str, float]]


@dataclass(frozen=True)
class StrategyDecision:
    target_weights: dict[str, float]
    overlay_state: core.OverlayState
    transition_reason: str
    alpha_reviewed: bool
    structural_change: bool
    trend_positive: bool
    residual_positive: bool
    raw_soxl_weight: float
    qqq_close: float
    qqq_sma_200: float
    residual_signal: core.ResidualSignal | None
    portfolio_volatility: core.PortfolioVolatility | None
    failure_reason: str = ""
    processed_signal_dates: tuple[str, ...] = ()
    transition_path: tuple[str, ...] = ()
    shadow_observations: tuple[ShadowObservation, ...] = ()
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

    overlay_active: bool = False
    eligible_streak: int = 0
    soxl_weight: float = 0.0
    soxl_weight_date: str = ""
    pending_soxl_weight: float = 0.0
    pending_scale_days: int = 0
    last_alpha_review_date: str = ""
    last_processed_signal_date: str = ""
    lifecycle_stage: str = LIFECYCLE_SPRINT
    lifecycle_stage_date: str = ""
    shadow_ledger_sessions: int = 0
    shadow_ledger_last_signal_date: str = ""
    shadow_ledger_chain_hash: str = ""

    executed_overlay_active: bool = False
    executed_soxl_weight: float = 0.0
    executed_lifecycle_stage: str = LIFECYCLE_SPRINT
    executed_strategy_fingerprint: str = ""

    pending_recommendation_date: str = ""
    pending_recommendation_weights: dict[str, float] = field(default_factory=dict)
    pending_recommendation_overlay_active: bool = False
    pending_recommendation_soxl_weight: float = 0.0
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


# ---------------------------------------------------------------------------
# Generic validation
# ---------------------------------------------------------------------------
def _is_valid_number(value: object, *, allow_zero: bool = True) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not np.isfinite(value):
        return False
    return value >= 0 if allow_zero else value > 0


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


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
    return all(
        abs(left.get(ticker, 0.0) - right.get(ticker, 0.0)) <= tolerance
        for ticker in set(left) | set(right)
    )


def _with_cash_target(weights: dict[str, float]) -> dict[str, float]:
    result = dict(weights)
    result.setdefault(CASH_ASSET, 0.0)
    return result


def _lifecycle_rank(stage: str) -> int:
    try:
        return LIFECYCLE_STAGES.index(stage)
    except ValueError as exc:
        raise ValueError(f"Unknown lifecycle stage: {stage!r}") from exc


def estimated_investor_age(as_of: date) -> float:
    """Return exact configured age or the conservative dated age anchor."""
    birth_date_text = os.environ.get("INVESTOR_BIRTH_DATE", "").strip()
    if birth_date_text:
        try:
            birth_date = date.fromisoformat(birth_date_text)
        except ValueError as exc:
            raise RuntimeError(
                "INVESTOR_BIRTH_DATE must use YYYY-MM-DD"
            ) from exc
        if birth_date >= as_of:
            raise RuntimeError("INVESTOR_BIRTH_DATE must be before the signal date")
        years = as_of.year - birth_date.year
        try:
            birthday = birth_date.replace(year=as_of.year)
        except ValueError:
            birthday = date(as_of.year, 2, 28)
        if as_of < birthday:
            years -= 1
            try:
                last_birthday = birth_date.replace(year=as_of.year - 1)
            except ValueError:
                last_birthday = date(as_of.year - 1, 2, 28)
            next_birthday = birthday
        else:
            last_birthday = birthday
            try:
                next_birthday = birth_date.replace(year=as_of.year + 1)
            except ValueError:
                next_birthday = date(as_of.year + 1, 2, 28)
        fraction = (as_of - last_birthday).days / (
            next_birthday - last_birthday
        ).days
        return float(years + fraction)
    return LIFECYCLE_ANCHOR_AGE + (
        (as_of - LIFECYCLE_ANCHOR_DATE).days / 365.2425
    )


def lifecycle_inflation_factor(as_of: date) -> float:
    years = (as_of - LIFECYCLE_ANCHOR_DATE).days / 365.2425
    factor = (1.0 + LIFECYCLE_INFLATION_RATE) ** years
    if not np.isfinite(factor) or factor <= 0:
        raise RuntimeError("Lifecycle inflation factor is invalid")
    return float(factor)


def lifecycle_value_stage(portfolio_value: float, as_of: date) -> str:
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError("Lifecycle portfolio value must be positive")
    stage = LIFECYCLE_SPRINT
    factor = lifecycle_inflation_factor(as_of)
    for threshold, candidate in LIFECYCLE_VALUE_THRESHOLDS_2026:
        if portfolio_value >= threshold * factor - 1e-9:
            stage = candidate
    return stage


def lifecycle_age_stage(age: float) -> str:
    if not np.isfinite(age) or age < 0:
        raise ValueError("Investor age must be nonnegative and finite")
    stage = LIFECYCLE_SPRINT
    for threshold, candidate in LIFECYCLE_AGE_THRESHOLDS:
        if age >= threshold - 1e-12:
            stage = candidate
    return stage


def select_lifecycle_stage(
    current_stage: str,
    portfolio_value: float,
    as_of: date,
) -> LifecycleSelection:
    """Apply value/age ceilings through a one-way safety ratchet."""
    current_rank = _lifecycle_rank(current_stage)
    age = estimated_investor_age(as_of)
    value_stage = lifecycle_value_stage(portfolio_value, as_of)
    age_stage = lifecycle_age_stage(age)
    value_rank = _lifecycle_rank(value_stage)
    age_rank = _lifecycle_rank(age_stage)
    selected_rank = max(current_rank, value_rank, age_rank)
    selected = LIFECYCLE_STAGES[selected_rank]
    advanced = selected_rank > current_rank
    if not advanced:
        reason = "RATCHET_HOLD"
    elif value_rank == selected_rank and age_rank == selected_rank:
        reason = "VALUE_AND_AGE_MILESTONE"
    elif age_rank == selected_rank:
        reason = "AGE_CEILING"
    else:
        reason = "VALUE_MILESTONE"
    return LifecycleSelection(
        stage=selected,
        reason=reason,
        estimated_age=float(age),
        value_stage=value_stage,
        age_stage=age_stage,
        advanced=advanced,
    )


def apply_lifecycle_policy(
    decision: StrategyDecision,
    state: PortfolioState,
    portfolio_value: float,
    signal_date: pd.Timestamp,
) -> StrategyDecision:
    selection = select_lifecycle_stage(
        state.lifecycle_stage,
        portfolio_value,
        pd.Timestamp(signal_date).date(),
    )
    observations = decision.shadow_observations
    if observations:
        latest = observations[-1]
        observations = (
            *observations[:-1],
            replace(
                latest,
                lifecycle_stage=selection.stage,
                forward_targets=forward_shadow_targets(
                    decision.overlay_state.soxl_weight,
                    selection.stage,
                ),
            ),
        )
    return replace(
        decision,
        target_weights=target_weights(
            decision.overlay_state.soxl_weight,
            selection.stage,
        ),
        lifecycle_stage=selection.stage,
        lifecycle_reason=selection.reason,
        estimated_investor_age=selection.estimated_age,
        lifecycle_value_stage=selection.value_stage,
        lifecycle_age_stage=selection.age_stage,
        lifecycle_stage_advanced=selection.advanced,
        shadow_observations=observations,
    )


def _validate_weight_mapping(
    weights: object,
    field_name: str,
    *,
    required_total: bool,
) -> None:
    if not isinstance(weights, dict):
        raise RuntimeError(f"{field_name} must be a mapping")
    invalid = set(weights) - PORTFOLIO_COMPONENTS
    if invalid:
        raise RuntimeError(
            f"{field_name} contains unsupported components: {sorted(invalid)}"
        )
    if any(not _is_valid_number(value) for value in weights.values()):
        raise RuntimeError(f"{field_name} contains invalid weights")
    if required_total and not np.isclose(
        sum(float(value) for value in weights.values()),
        1.0,
        atol=1e-9,
    ):
        raise RuntimeError(f"{field_name} must sum to 1.0")


def _valid_soxl_weight(value: object) -> bool:
    """Accept a bounded weight stored by a current or historical decision."""
    if not _is_valid_number(value):
        return False
    numeric = float(value)
    return numeric <= core.MAX_SOXL_WEIGHT + 1e-12


def _valid_soxl_tier(value: object) -> bool:
    return _valid_soxl_weight(value) and core.is_soxl_tier(value)


def validate_configuration() -> None:
    if STRATEGY_FINGERPRINT != calculate_strategy_fingerprint():
        raise RuntimeError("Strategy fingerprint is internally inconsistent")
    if STRATEGY_FINGERPRINT != EXPECTED_STRATEGY_FINGERPRINT:
        raise RuntimeError(
            "Decision boundaries changed without a strategy revision and "
            "fingerprint review"
        )
    if REQUIRED_SIGNAL_ROWS < (
        core.VARIANCE_MIN_TRAINING
        + core.VARIANCE_HORIZON
        + core.VARIANCE_QUARTER_WINDOW
    ):
        raise RuntimeError("Configured history cannot train the variance model")
    maximum = strategic_daily_exposure(core.MAX_SOXL_WEIGHT)
    if not np.isclose(
        maximum,
        MAX_ADVERTISED_DAILY_EXPOSURE,
        atol=1e-12,
    ):
        raise RuntimeError("Advertised exposure invariant failed")
    if tuple(stage for _, stage in LIFECYCLE_VALUE_THRESHOLDS_2026) != (
        LIFECYCLE_GLIDE_225,
        LIFECYCLE_TWO_X,
        LIFECYCLE_PHI,
        LIFECYCLE_ONE_THREE,
        LIFECYCLE_ONE_X,
    ):
        raise RuntimeError("Lifecycle value-stage order is invalid")
    for soxl_weight in core.SOXL_WEIGHT_GRID:
        for stage in LIFECYCLE_STAGES:
            weights = target_weights(soxl_weight, stage)
            if not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
                raise RuntimeError("Lifecycle allocation invariant failed")
            ceiling = LIFECYCLE_EXPOSURE_CEILINGS[stage]
            if ceiling is not None and not np.isclose(
                advertised_daily_exposure(weights),
                ceiling,
                atol=1e-12,
            ):
                raise RuntimeError("Lifecycle exposure invariant failed")


def validate_state(state: PortfolioState) -> None:
    if state.state_version != STATE_VERSION:
        raise RuntimeError("Portfolio state version is invalid")
    if not isinstance(state.shares, dict):
        raise RuntimeError("shares must be a mapping")
    invalid_holdings = set(state.shares) - TRADED_TICKERS
    if invalid_holdings:
        raise RuntimeError(
            f"shares contains unsupported holdings: {sorted(invalid_holdings)}"
        )
    if any(not _is_valid_number(value) for value in state.shares.values()):
        raise RuntimeError("shares contains invalid quantities")
    for name in ("cash_balance", "portfolio_value"):
        if not _is_valid_number(getattr(state, name)):
            raise RuntimeError(f"{name} is invalid")

    _validate_weight_mapping(
        state.target_weights,
        "target_weights",
        required_total=bool(state.target_weights),
    )
    if not isinstance(state.overlay_active, bool):
        raise RuntimeError("overlay_active must be boolean")
    if (
        not isinstance(state.eligible_streak, int)
        or isinstance(state.eligible_streak, bool)
        or state.eligible_streak < 0
    ):
        raise RuntimeError("eligible_streak is invalid")
    if not _valid_soxl_tier(state.soxl_weight):
        raise RuntimeError("soxl_weight is invalid")
    if not _valid_soxl_tier(state.pending_soxl_weight):
        raise RuntimeError("pending_soxl_weight is invalid")
    if (
        not isinstance(state.pending_scale_days, int)
        or isinstance(state.pending_scale_days, bool)
        or state.pending_scale_days < 0
        or state.pending_scale_days >= core.VOLATILITY_UPSHIFT_CLOSES
    ):
        raise RuntimeError("pending_scale_days is invalid")
    if not state.overlay_active and state.soxl_weight != 0.0:
        raise RuntimeError("An inactive overlay cannot retain SOXL weight")
    if state.pending_scale_days == 0 and state.pending_soxl_weight != 0.0:
        raise RuntimeError("A pending SOXL weight requires pending scale days")
    if state.lifecycle_stage not in LIFECYCLE_STAGES:
        raise RuntimeError("lifecycle_stage is invalid")
    if (
        not isinstance(state.shadow_ledger_sessions, int)
        or isinstance(state.shadow_ledger_sessions, bool)
        or state.shadow_ledger_sessions < 0
    ):
        raise RuntimeError("shadow_ledger_sessions is invalid")
    if state.shadow_ledger_sessions:
        if not state.shadow_ledger_last_signal_date:
            raise RuntimeError("Shadow ledger anchor is missing its date")
        if not _is_sha256(state.shadow_ledger_chain_hash):
            raise RuntimeError("Shadow ledger anchor hash is invalid")
    elif (
        state.shadow_ledger_last_signal_date
        or state.shadow_ledger_chain_hash
    ):
        raise RuntimeError("Empty shadow ledger anchor is inconsistent")
    if not isinstance(state.executed_overlay_active, bool):
        raise RuntimeError("executed_overlay_active must be boolean")
    if not _valid_soxl_weight(state.executed_soxl_weight):
        raise RuntimeError("executed_soxl_weight is invalid")
    if (
        not state.executed_overlay_active
        and state.executed_soxl_weight != 0.0
    ):
        raise RuntimeError("Inactive executed metadata cannot retain SOXL weight")
    if state.executed_lifecycle_stage not in LIFECYCLE_STAGES:
        raise RuntimeError("executed_lifecycle_stage is invalid")
    if state.executed_strategy_fingerprint and not _is_sha256(
        state.executed_strategy_fingerprint
    ):
        raise RuntimeError("executed_strategy_fingerprint is invalid")

    pending_date = _parse_iso_date(
        state.pending_recommendation_date,
        "pending_recommendation_date",
    )
    _validate_weight_mapping(
        state.pending_recommendation_weights,
        "pending_recommendation_weights",
        required_total=bool(state.pending_recommendation_date),
    )
    if not isinstance(state.pending_recommendation_overlay_active, bool):
        raise RuntimeError(
            "pending_recommendation_overlay_active must be boolean"
        )
    if not _valid_soxl_weight(state.pending_recommendation_soxl_weight):
        raise RuntimeError("pending_recommendation_soxl_weight is invalid")
    if (
        not state.pending_recommendation_overlay_active
        and state.pending_recommendation_soxl_weight != 0.0
    ):
        raise RuntimeError(
            "Inactive pending metadata cannot retain SOXL weight"
        )
    if not isinstance(state.pending_recommendation_notified, bool):
        raise RuntimeError("pending_recommendation_notified must be boolean")
    if pending_date:
        if not state.pending_recommendation_weights:
            raise RuntimeError("Pending recommendation is missing weights")
        if not _is_sha256(state.pending_recommendation_fingerprint):
            raise RuntimeError("Pending recommendation fingerprint is invalid")
        if state.pending_recommendation_lifecycle_stage not in LIFECYCLE_STAGES:
            raise RuntimeError("Pending lifecycle stage is invalid")
    elif (
        state.pending_recommendation_weights
        or state.pending_recommendation_overlay_active
        or state.pending_recommendation_soxl_weight != 0.0
        or state.pending_recommendation_lifecycle_stage
        or state.pending_recommendation_notified
        or state.pending_recommendation_supersedes_date
        or state.pending_recommendation_fingerprint
    ):
        raise RuntimeError("Pending recommendation state is inconsistent")

    today = datetime.now(NEW_YORK).date()
    for name in (
        "soxl_weight_date",
        "lifecycle_stage_date",
        "last_alpha_review_date",
        "last_processed_signal_date",
        "shadow_ledger_last_signal_date",
        "pending_recommendation_supersedes_date",
        "last_delivered_signal_date",
    ):
        parsed = _parse_iso_date(getattr(state, name), name)
        if parsed and parsed > today:
            raise RuntimeError(f"{name} cannot be in the future")
    if pending_date and pending_date > today:
        raise RuntimeError("Pending recommendation date cannot be in the future")
    if state.shadow_ledger_sessions:
        if not state.last_processed_signal_date:
            raise RuntimeError(
                "A shadow ledger anchor requires a processed signal date"
            )
        if (
            pd.Timestamp(state.shadow_ledger_last_signal_date)
            > pd.Timestamp(state.last_processed_signal_date)
        ):
            raise RuntimeError(
                "Shadow ledger anchor is ahead of processed signal state"
            )
    if state.last_processed_data_fingerprint and not _is_sha256(
        state.last_processed_data_fingerprint
    ):
        raise RuntimeError("Last processed data fingerprint is invalid")
    if (
        state.last_processed_data_fingerprint
        and not state.last_processed_signal_date
    ):
        raise RuntimeError("A data fingerprint requires a signal date")
    delivered = (
        state.last_delivered_decision_hash,
        state.last_delivered_signal_date,
        state.last_delivered_notification_kind,
    )
    if any(delivered) and not all(delivered):
        raise RuntimeError("Last delivered notification evidence is incomplete")
    if all(delivered):
        if not _is_sha256(state.last_delivered_decision_hash):
            raise RuntimeError("Last delivered decision hash is invalid")
        if state.last_delivered_notification_kind not in {
            "ACTION",
            "UPDATE",
            "RETRY",
            "UPDATE_RETRY",
            "CANCELLATION",
        }:
            raise RuntimeError("Last delivered notification kind is invalid")
    if not isinstance(state.last_updated, str):
        raise RuntimeError("last_updated must be a string")
    if state.last_updated:
        try:
            datetime.fromisoformat(state.last_updated)
        except ValueError as exc:
            raise RuntimeError("last_updated is not a valid timestamp") from exc


# ---------------------------------------------------------------------------
# State migration and atomic persistence
# ---------------------------------------------------------------------------
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
    if isinstance(version, bool) or (
        not isinstance(version, int) and version is not None
    ):
        raise RuntimeError(f"Unsupported state version: {version!r}")
    if version is None and not {
        "shares",
        "target_weights",
        "portfolio_value",
    }.issubset(payload):
        raise RuntimeError(
            "Unversioned state does not match the known legacy schema"
        )
    numeric_version = 1 if version is None else int(version)
    if numeric_version < 1 or numeric_version >= STATE_VERSION:
        raise RuntimeError(f"Unsupported state version: {version!r}")
    backup = _backup_legacy_state(version) if backup_legacy else None
    migrated = asdict(PortfolioState())

    # Versions 8-12 already contain the complete production state machine and
    # outbox. Version 13 adds the monotonic lifecycle ratchet. Broker facts and
    # pending-action evidence remain exact; older actions begin in SPRINT and
    # are intentionally reconsidered under the new strategy fingerprint.
    #
    # Versions 8 and 9 fingerprinted the retired QLD/SOXL universe. That hash
    # cannot be compared with the expanded TQQQ/UGL universe on a same-date
    # deployment, so retain the processed date but begin a new data-hash
    # lineage. Versions 10 and 11 used the current universe and keep the hash.
    if numeric_version in {8, 9, 10, 11, 12}:
        for name in set(migrated) & set(payload):
            migrated[name] = payload[name]
        if numeric_version in {8, 9}:
            migrated["last_processed_data_fingerprint"] = ""
        migrated["soxl_weight"] = core.floor_soxl_tier(
            float(migrated["soxl_weight"])
        )
        migrated["pending_soxl_weight"] = core.floor_soxl_tier(
            float(migrated["pending_soxl_weight"])
        )
        if (
            migrated["pending_soxl_weight"] <= migrated["soxl_weight"]
            or migrated["pending_soxl_weight"] == 0.0
        ):
            migrated["pending_soxl_weight"] = 0.0
            migrated["pending_scale_days"] = 0
        if migrated["pending_recommendation_date"]:
            migrated["pending_recommendation_lifecycle_stage"] = (
                LIFECYCLE_SPRINT
            )
        migrated["state_version"] = STATE_VERSION
        logger.warning(
            "Migrated state version %r to version %s; backup=%s",
            version,
            STATE_VERSION,
            backup or "disabled",
        )
        return migrated

    # Holdings and cash are broker facts.  Preserve them exactly across every
    # known schema, including valuation-only products.
    for name in (
        "shares",
        "cash_balance",
        "target_weights",
        "portfolio_value",
        "last_processed_signal_date",
        "last_delivered_decision_hash",
        "last_delivered_signal_date",
        "last_delivered_notification_kind",
        "last_updated",
    ):
        if name in payload:
            migrated[name] = payload[name]

    # Preserve the v3-v7 outbox and its proof-of-delivery fields.  The old
    # fingerprint deliberately remains old so the v8 decision becomes an
    # UPDATE or CANCELLATION, never an incorrectly suppressed duplicate.
    for name in (
        "pending_recommendation_date",
        "pending_recommendation_weights",
        "pending_recommendation_notified",
        "pending_recommendation_supersedes_date",
        "pending_recommendation_fingerprint",
        "executed_strategy_fingerprint",
    ):
        if name in payload:
            migrated[name] = payload[name]
    if migrated["pending_recommendation_date"]:
        migrated["pending_recommendation_lifecycle_stage"] = LIFECYCLE_SPRINT
        if not _is_sha256(migrated["pending_recommendation_fingerprint"]):
            # Old outbox schemas could stage an action without recording its
            # strategy identity.  A fixed legacy identity preserves the action
            # while guaranteeing that v8 treats it as changed.
            migrated["pending_recommendation_fingerprint"] = "0" * 64
        if numeric_version < 5:
            migrated["pending_recommendation_notified"] = False

    # An old data hash used a different universe and canonical payload.  Keep
    # the processed date for same-date idempotence but start v8 hash lineage.
    migrated["last_processed_data_fingerprint"] = ""
    migrated["state_version"] = STATE_VERSION
    logger.warning(
        "Migrated state version %r to version %s; backup=%s",
        version,
        STATE_VERSION,
        backup or "disabled",
    )
    return migrated


def load_state(*, backup_legacy: bool = True) -> PortfolioState:
    if not STATE_FILE.exists():
        return PortfolioState()
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read {STATE_FILE}; refusing to infer holdings"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Portfolio state must be a JSON object")
    version = payload.get("state_version")
    if version != STATE_VERSION:
        payload = _migrate_state_payload(
            payload,
            version,
            backup_legacy=backup_legacy,
        )

    expected = {item.name for item in fields(PortfolioState)}
    missing = expected - set(payload)
    unexpected = set(payload) - expected
    if missing or unexpected:
        raise RuntimeError(
            "Portfolio state schema mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
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
    payload = json.dumps(asdict(state), indent=2, allow_nan=False)
    temporary = STATE_FILE.with_suffix(f"{STATE_FILE.suffix}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, STATE_FILE)
    logger.info(
        "State saved: holdings=%s overlay=%s SOXL=%.0f%% lifecycle=%s "
        "pending=%s notified=%s",
        len(state.shares),
        state.overlay_active,
        state.soxl_weight * 100.0,
        state.lifecycle_stage,
        state.pending_recommendation_date or "NONE",
        state.pending_recommendation_notified,
    )


# ---------------------------------------------------------------------------
# Completed-session market data
# ---------------------------------------------------------------------------
def _nyse_calendar() -> object:
    return xcals.get_calendar("XNYS")


def expected_completed_session(
    now_new_york: datetime | None = None,
) -> pd.Timestamp:
    now_new_york = now_new_york or datetime.now(NEW_YORK)
    calendar = _nyse_calendar()
    today = pd.Timestamp(now_new_york.date())
    sessions = calendar.sessions_in_range(
        today - pd.Timedelta(days=14),
        today + pd.Timedelta(days=1),
    )
    if len(sessions) == 0:
        raise RuntimeError("XNYS calendar returned no sessions")
    now_utc = pd.Timestamp(now_new_york).tz_convert("UTC")
    if calendar.is_session(today):
        market_open = calendar.session_open(today)
        safe_close = calendar.session_close(today) + pd.Timedelta(
            minutes=MARKET_CLOSE_BUFFER_MINUTES
        )
        if market_open <= now_utc < safe_close:
            raise RuntimeError(
                "The latest daily bar is not final; run before the session "
                f"opens or {MARKET_CLOSE_BUFFER_MINUTES} minutes after close"
            )
    completed = [
        session
        for session in sessions
        if calendar.session_close(session)
        + pd.Timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES)
        <= now_utc
    ]
    if not completed:
        raise RuntimeError("No completed XNYS session is available")
    return pd.Timestamp(completed[-1]).tz_localize(None).normalize()


def required_signal_rows() -> int:
    return REQUIRED_SIGNAL_ROWS


def required_nyse_sessions(
    ending_session: pd.Timestamp,
    count: int,
) -> pd.DatetimeIndex:
    if count <= 0:
        raise ValueError("Required session count must be positive")
    ending = pd.Timestamp(ending_session).normalize()
    sessions = pd.DatetimeIndex(
        _nyse_calendar().sessions_in_range(
            ending - pd.Timedelta(days=max(30, count * 3)),
            ending,
        )
    )
    if sessions.tz is not None:
        sessions = sessions.tz_convert(None)
    sessions = sessions.normalize()
    if len(sessions) < count:
        raise RuntimeError(
            f"XNYS calendar returned {len(sessions)} sessions; need {count}"
        )
    return sessions[-count:]


def validate_session_continuity(
    market_index: pd.DatetimeIndex,
    expected_session: pd.Timestamp,
    required_rows: int,
) -> None:
    received = pd.DatetimeIndex(market_index[-required_rows:]).normalize()
    expected = required_nyse_sessions(expected_session, required_rows)
    missing = expected.difference(received)
    unexpected = received.difference(expected)
    if len(missing) or len(unexpected):
        raise RuntimeError(
            "Market-data session continuity failed: "
            f"missing={[item.date().isoformat() for item in missing[:5]]}, "
            f"non_sessions={[item.date().isoformat() for item in unexpected[:5]]}"
        )


def _extract_yfinance_prices(
    data: pd.DataFrame,
    tickers: Iterable[str],
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise RuntimeError("yfinance returned no market data")
    requested = list(tickers)
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                prices = data["Close"].copy()
            elif "Close" in data.columns.get_level_values(1):
                prices = data.xs("Close", axis=1, level=1).copy()
            else:
                raise KeyError("Close")
        else:
            close = data["Close"]
            prices = (
                close.to_frame(name=requested[0])
                if isinstance(close, pd.Series) and len(requested) == 1
                else pd.DataFrame(close)
            )
    except KeyError as exc:
        raise RuntimeError("yfinance response is missing Close") from exc
    missing = set(requested) - set(str(item) for item in prices.columns)
    if missing:
        raise RuntimeError(
            f"Incomplete market-data universe: prices={sorted(missing)}"
        )
    prices = prices.reindex(columns=requested)
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise RuntimeError("Market data does not use a DatetimeIndex")
    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert(None)
    prices.index = prices.index.normalize()
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise RuntimeError("Market-data dates are duplicated or unsorted")
    return prices


def _require_finite_positive(
    values: pd.DataFrame | pd.Series,
    description: str,
) -> None:
    try:
        numeric = values.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{description} contains nonnumeric values") from exc
    if not np.isfinite(numeric).all() or not (numeric > 0).all():
        raise RuntimeError(f"{description} contains missing or invalid values")


def market_data_fingerprint(price_data: pd.DataFrame) -> str:
    if len(price_data) < REQUIRED_SIGNAL_ROWS:
        raise ValueError("Insufficient data for a market-data fingerprint")
    payload = {
        "sessions": [
            pd.Timestamp(item).date().isoformat()
            for item in price_data.index
        ],
        "model_closes": {
            ticker: [
                float(value)
                for value in price_data[ticker]
            ]
            for ticker in MODEL_TICKERS
        },
        "latest_valuation_prices": {
            ticker: float(price_data[ticker].iloc[-1])
            for ticker in VALUATION_TICKERS
        },
    }
    return canonical_sha256(payload)


def download_market_data(
    tickers: Iterable[str] = ALL_TICKERS,
    *,
    now_new_york: datetime | None = None,
) -> pd.DataFrame:
    expected = expected_completed_session(now_new_york)
    requested = list(tickers)
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for signal generation"
        ) from exc
    data = yf.download(
        requested,
        start=MODEL_START_DATE,
        end=(expected.date() + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    prices = _extract_yfinance_prices(data, requested)
    received = pd.Timestamp(prices.index[-1]).normalize()
    if received != expected:
        raise RuntimeError(
            "Market data is stale or misdated: "
            f"expected {expected.date()}, received {received.date()}"
        )
    if len(prices) < REQUIRED_SIGNAL_ROWS:
        raise RuntimeError(
            f"Insufficient market history: {len(prices)} rows; "
            f"need {REQUIRED_SIGNAL_ROWS}"
        )
    expected_model_sessions = pd.DatetimeIndex(
        _nyse_calendar().sessions_in_range(
            pd.Timestamp(MODEL_START_DATE),
            expected,
        )
    )
    if expected_model_sessions.tz is not None:
        expected_model_sessions = expected_model_sessions.tz_convert(None)
    expected_model_sessions = expected_model_sessions.normalize()
    if not prices.index.equals(expected_model_sessions):
        missing = expected_model_sessions.difference(prices.index)
        unexpected = prices.index.difference(expected_model_sessions)
        raise RuntimeError(
            "Full model-history continuity failed: "
            f"missing={[item.date().isoformat() for item in missing[:5]]}, "
            f"non_sessions="
            f"{[item.date().isoformat() for item in unexpected[:5]]}"
        )
    _require_finite_positive(
        prices.loc[:, list(MODEL_TICKERS)],
        "Prices in the complete quantitative model history",
    )
    _require_finite_positive(
        prices.loc[received, requested],
        "Latest prices for the full valuation universe",
    )
    return prices


# ---------------------------------------------------------------------------
# Causal signal and state transition
# ---------------------------------------------------------------------------
def alpha_review_due(
    last_review_date: str,
    signal_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
) -> bool:
    """Return the fixed 21-session review phase from model-history start."""
    current = pd.Timestamp(signal_date).normalize()
    if last_review_date:
        previous = pd.Timestamp(last_review_date).normalize()
        if previous > current:
            raise RuntimeError("Alpha review state is ahead of market data")
    completed = pd.DatetimeIndex(trading_dates).normalize()
    if completed.has_duplicates or not completed.is_monotonic_increasing:
        raise RuntimeError("Alpha review dates must be unique and increasing")
    try:
        position = completed.get_loc(current)
    except KeyError as exc:
        raise RuntimeError("Signal date is absent from alpha review history") from exc
    if not isinstance(position, (int, np.integer)):
        raise RuntimeError("Signal date is duplicated in alpha review history")
    return int(position) % ALPHA_REVIEW_SESSIONS == 0


def _overlay_state_from_portfolio(state: PortfolioState) -> core.OverlayState:
    return core.OverlayState(
        overlay_active=state.overlay_active,
        eligible_streak=state.eligible_streak,
        soxl_weight=state.soxl_weight,
        soxl_weight_date=state.soxl_weight_date,
        pending_soxl_weight=state.pending_soxl_weight,
        pending_scale_days=state.pending_scale_days,
        last_alpha_review_date=state.last_alpha_review_date,
        last_processed_signal_date=state.last_processed_signal_date,
    )


def _apply_overlay_state(
    state: PortfolioState,
    overlay: core.OverlayState,
) -> None:
    state.overlay_active = overlay.overlay_active
    state.eligible_streak = overlay.eligible_streak
    state.soxl_weight = overlay.soxl_weight
    state.soxl_weight_date = overlay.soxl_weight_date
    state.pending_soxl_weight = overlay.pending_soxl_weight
    state.pending_scale_days = overlay.pending_scale_days
    state.last_alpha_review_date = overlay.last_alpha_review_date
    state.last_processed_signal_date = overlay.last_processed_signal_date


def _apply_lifecycle_state(
    state: PortfolioState,
    decision: StrategyDecision,
    signal_date: pd.Timestamp,
) -> None:
    if decision.lifecycle_stage not in LIFECYCLE_STAGES:
        raise RuntimeError("Decision lifecycle stage is invalid")
    if _lifecycle_rank(decision.lifecycle_stage) < _lifecycle_rank(
        state.lifecycle_stage
    ):
        raise RuntimeError("Lifecycle ratchet cannot move to a riskier stage")
    if decision.lifecycle_stage != state.lifecycle_stage:
        state.lifecycle_stage = decision.lifecycle_stage
        state.lifecycle_stage_date = pd.Timestamp(signal_date).date().isoformat()


def _calculate_latest_strategy_decision(
    price_data: pd.DataFrame,
    state: PortfolioState,
) -> StrategyDecision:
    """Calculate one latest-session signal with fail-closed model errors."""
    signal_date = pd.Timestamp(price_data.index[-1]).normalize()
    qqq = pd.to_numeric(price_data[MARKET_INDEX], errors="coerce")
    qqq_close = float(qqq.iloc[-1])
    qqq_sma = float(qqq.iloc[-SMA_WINDOW:].mean())
    if (
        not np.isfinite(qqq_close)
        or not np.isfinite(qqq_sma)
        or qqq_close <= 0
        or qqq_sma <= 0
    ):
        raise RuntimeError("QQQ trend inputs are invalid")
    trend_positive = bool(qqq_close > qqq_sma)

    residual: core.ResidualSignal | None = None
    portfolio_volatility: core.PortfolioVolatility | None = None
    failures: list[str] = []
    try:
        residual = core.calculate_residual_signal(
            price_data.loc[:, list(SIGNAL_TICKERS)]
        )
        residual_positive = bool(residual.residual_momentum > 0.0)
    except ValueError as exc:
        residual_positive = False
        failures.append(f"residual={exc}")

    try:
        portfolio_volatility = core.calculate_portfolio_volatility(
            price_data.loc[:, list(VOLATILITY_TICKERS)]
        )
        raw_soxl_weight = portfolio_volatility.raw_soxl_weight
    except (ValueError, np.linalg.LinAlgError) as exc:
        raw_soxl_weight = 0.0
        failures.append(f"volatility={exc}")

    # A broken alpha statistic cannot authorize or retain incremental leverage.
    # A broken variance forecast produces a raw zero, which is an immediate
    # volatility downshift under the shared transition primitive.
    effective_trend = trend_positive and residual is not None
    review_due = alpha_review_due(
        state.last_alpha_review_date,
        signal_date,
        price_data.index,
    )
    transition = core.advance_overlay_state(
        _overlay_state_from_portfolio(state),
        signal_date=signal_date,
        trend_positive=effective_trend,
        residual_positive=residual_positive,
        raw_soxl_weight=raw_soxl_weight,
        alpha_review_due=review_due,
    )
    reason = transition.reason
    if failures and residual is None:
        reason = "SIGNAL_FAILURE"
    elif failures and portfolio_volatility is None:
        reason = (
            "VOLATILITY_FAILURE_DOWNSHIFT"
            if transition.structural_change
            else "VOLATILITY_FAILURE_BLOCK"
        )

    return StrategyDecision(
        target_weights=target_weights(transition.state.soxl_weight),
        overlay_state=transition.state,
        transition_reason=reason,
        alpha_reviewed=transition.alpha_reviewed,
        structural_change=transition.structural_change,
        trend_positive=trend_positive,
        residual_positive=residual_positive,
        raw_soxl_weight=float(raw_soxl_weight),
        qqq_close=qqq_close,
        qqq_sma_200=qqq_sma,
        residual_signal=residual,
        portfolio_volatility=portfolio_volatility,
        failure_reason="; ".join(failures),
    )


def calculate_strategy_decision(
    price_data: pd.DataFrame,
    state: PortfolioState,
) -> StrategyDecision:
    """Replay every unseen completed session before returning today's signal.

    The state machine must depend on market history, not workflow uptime. A
    fresh account starts conservatively from the latest close, while an
    existing account deterministically replays every session after its last
    persisted signal date.
    """
    if price_data.empty:
        raise RuntimeError("Strategy decision requires market data")
    latest = pd.Timestamp(price_data.index[-1]).normalize()
    if state.last_processed_signal_date:
        previous = pd.Timestamp(state.last_processed_signal_date).normalize()
        if previous > latest:
            raise RuntimeError("Portfolio state is ahead of market data")
        unseen = pd.DatetimeIndex(
            price_data.index[price_data.index > previous]
        )
        sessions = unseen if len(unseen) else pd.DatetimeIndex([latest])
    else:
        sessions = pd.DatetimeIndex([latest])

    working = copy.deepcopy(state)
    decisions: list[StrategyDecision] = []
    transition_path: list[str] = []
    processed_dates: list[str] = []
    shadow_observations: list[ShadowObservation] = []
    for session in sessions:
        history = price_data.loc[:pd.Timestamp(session)]
        if len(history) < REQUIRED_SIGNAL_ROWS:
            raise RuntimeError(
                "Persisted state requires replay before sufficient model "
                "history is available"
            )
        decision = _calculate_latest_strategy_decision(history, working)
        _apply_overlay_state(working, decision.overlay_state)
        decisions.append(decision)
        transition_path.append(decision.transition_reason)
        session_text = pd.Timestamp(session).date().isoformat()
        processed_dates.append(session_text)
        shadow_observations.append(
            ShadowObservation(
                signal_date=session_text,
                data_fingerprint=market_data_fingerprint(history),
                trend_positive=decision.trend_positive,
                residual_positive=decision.residual_positive,
                raw_soxl_weight=decision.raw_soxl_weight,
                overlay_active=decision.overlay_state.overlay_active,
                soxl_weight=decision.overlay_state.soxl_weight,
                transition_reason=decision.transition_reason,
                structural_change=decision.structural_change,
                failure_reason=decision.failure_reason,
                lifecycle_stage=working.lifecycle_stage,
                forward_experiment_fingerprint=(
                    FORWARD_EXPERIMENT_FINGERPRINT
                ),
                forward_targets=forward_shadow_targets(
                    decision.overlay_state.soxl_weight,
                    working.lifecycle_stage,
                ),
            )
        )
    final = decisions[-1]
    return replace(
        final,
        processed_signal_dates=tuple(processed_dates),
        transition_path=tuple(transition_path),
        shadow_observations=tuple(shadow_observations),
    )


def validate_same_date_data_fingerprint(
    state: PortfolioState,
    signal_date: pd.Timestamp,
    data_fingerprint: str,
) -> None:
    if not _is_sha256(data_fingerprint):
        raise ValueError("Market-data fingerprint is invalid")
    if (
        state.last_processed_signal_date
        == pd.Timestamp(signal_date).date().isoformat()
        and state.last_processed_data_fingerprint
        and state.last_processed_data_fingerprint != data_fingerprint
    ):
        raise RuntimeError(
            "Market data changed for an already processed signal date; "
            "manual review is required"
        )


# ---------------------------------------------------------------------------
# Holdings valuation and rebalance planning
# ---------------------------------------------------------------------------
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
        total += float(shares) * price
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
        ticker: float(shares) * float(price_data[ticker].iloc[-1]) / total
        for ticker, shares in state.shares.items()
        if float(shares) > 0
    }
    if state.cash_balance > 0:
        weights[CASH_ASSET] = float(state.cash_balance) / total
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
        raise RuntimeError("Current portfolio weights do not sum to 1.0")
    return weights


def resolve_portfolio_value(
    roth_amount: float | None,
    state: PortfolioState,
    price_data: pd.DataFrame,
) -> tuple[float, PortfolioState]:
    planning = copy.deepcopy(state)
    has_recorded_portfolio = bool(state.shares) or state.cash_balance > 0
    if has_recorded_portfolio:
        if roth_amount is not None:
            raise RuntimeError(
                "--roth-amount is first-run only; record contributions with "
                "--sync-holdings and a complete CASH value"
            )
        value = existing_portfolio_value(planning, price_data)
    else:
        starting_cash = roth_amount if roth_amount is not None else ROTH_IRA_AMOUNT
        if starting_cash is None:
            raise RuntimeError(
                "No portfolio value is available; explicitly initialize with "
                "--roth-amount or ROTH_IRA_AMOUNT"
            )
        if not np.isfinite(starting_cash) or starting_cash <= 0:
            raise RuntimeError("First-run Roth IRA amount must be positive and finite")
        planning.cash_balance = float(starting_cash)
        value = float(starting_cash)
    if value <= 0:
        raise RuntimeError("Portfolio value must be positive")
    return value, planning


def drift_triggers(
    existing: dict[str, float],
    target: dict[str, float],
    *,
    band: float = REBALANCE_BAND,
) -> tuple[bool, bool]:
    if not (0 < band < 1):
        raise ValueError("Rebalance band must be between zero and one")
    if not existing:
        return True, True
    desired = _with_cash_target(target)
    individual = any(
        abs(desired.get(ticker, 0.0) - existing.get(ticker, 0.0))
        >= band - 1e-12
        for ticker in set(existing) | set(desired)
    )
    existing_equity = sum(
        existing.get(ticker, 0.0) for ticker in EQUITY_TICKERS
    )
    target_equity = sum(
        desired.get(ticker, 0.0) for ticker in EQUITY_TICKERS
    )
    aggregate = (
        abs(existing_equity - target_equity) >= band - 1e-12
    )
    return individual, aggregate


def should_rebalance(
    existing: dict[str, float],
    target: dict[str, float],
    band: float = REBALANCE_BAND,
) -> bool:
    return any(drift_triggers(existing, target, band=band))


def inner_band_rebalance_weights(
    existing: dict[str, float],
    target: dict[str, float],
    destination: float = REBALANCE_DESTINATION,
    *,
    trigger_band: float = REBALANCE_BAND,
    fixed_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    if not existing:
        return dict(target)
    if not (0 < destination < trigger_band < 1):
        raise ValueError("Destination must be positive and inside the trigger band")
    desired_mapping = _with_cash_target(target)
    tickers = sorted(set(existing) | set(desired_mapping))
    current = np.array(
        [existing.get(ticker, 0.0) for ticker in tickers],
        dtype=float,
    )
    desired = np.array(
        [desired_mapping.get(ticker, 0.0) for ticker in tickers],
        dtype=float,
    )
    if not np.isclose(current.sum(), 1.0, atol=1e-9):
        raise RuntimeError("Existing portfolio weights do not sum to 1.0")
    if not np.isclose(desired.sum(), 1.0, atol=1e-9):
        raise RuntimeError("Target portfolio weights do not sum to 1.0")
    lower = np.maximum(0.0, desired - destination)
    upper = np.minimum(1.0, desired + destination)
    fixed = fixed_weights or {}
    unknown_fixed = set(fixed) - set(tickers)
    if unknown_fixed:
        raise ValueError(
            f"Fixed weights contain unknown components: {sorted(unknown_fixed)}"
        )
    for ticker, value in fixed.items():
        numeric = float(value)
        index = tickers.index(ticker)
        if (
            not np.isfinite(numeric)
            or numeric < lower[index] - 1e-12
            or numeric > upper[index] + 1e-12
        ):
            raise ValueError(f"Fixed weight for {ticker} is outside its band")
        lower[index] = numeric
        upper[index] = numeric
    soxl_index = (
        tickers.index(LEVERAGED_SEMICONDUCTOR)
        if LEVERAGED_SEMICONDUCTOR in tickers
        else None
    )
    if soxl_index is not None:
        upper[soxl_index] = min(
            upper[soxl_index],
            core.MAX_SOXL_WEIGHT,
        )
    if lower.sum() > 1.0 + 1e-12 or upper.sum() < 1.0 - 1e-12:
        raise RuntimeError("Inner-band bounds cannot form a full portfolio")

    lower_shift = float(np.min(current - upper)) - 1.0
    upper_shift = float(np.max(current - lower)) + 1.0
    for _ in range(100):
        shift = (lower_shift + upper_shift) / 2.0
        candidate = np.clip(current - shift, lower, upper)
        if candidate.sum() > 1.0:
            lower_shift = shift
        else:
            upper_shift = shift
    projected = np.clip(current - upper_shift, lower, upper)
    remainder = 1.0 - float(projected.sum())
    if abs(remainder) > 1e-10:
        slack = upper - projected if remainder > 0 else projected - lower
        for index in np.argsort(-slack):
            adjustment = min(abs(remainder), float(slack[index]))
            projected[index] += adjustment if remainder > 0 else -adjustment
            remainder += -adjustment if remainder > 0 else adjustment
            if abs(remainder) <= 1e-12:
                break
    if not np.isclose(projected.sum(), 1.0, atol=1e-9):
        raise RuntimeError("Inner-band projection did not preserve total weight")
    return {
        ticker: float(weight)
        for ticker, weight in zip(tickers, projected)
        if weight > 1e-12
    }


def build_rebalance_plan(
    existing: dict[str, float],
    decision: StrategyDecision,
    state: PortfolioState,
    *,
    rebalance_band: float = REBALANCE_BAND,
    rebalance_destination: float = REBALANCE_DESTINATION,
) -> RebalancePlan:
    if not (0 < rebalance_destination < rebalance_band < 1):
        raise ValueError(
            "Rebalance destination must be positive and inside the trigger band"
        )
    target = _with_cash_target(decision.target_weights)
    strategy_changed = (
        state.executed_strategy_fingerprint != STRATEGY_FINGERPRINT
    )
    lifecycle_changed = (
        state.executed_lifecycle_stage != decision.lifecycle_stage
    )
    structure_changed = (
        state.executed_overlay_active != decision.overlay_state.overlay_active
        or abs(
            state.executed_soxl_weight - decision.overlay_state.soxl_weight
        )
        > 1e-12
        or lifecycle_changed
    )
    legacy_exit = any(
        existing.get(ticker, 0.0) > 1e-12 for ticker in LEGACY_HOLDINGS
    )
    risk_off_residual_exit = (
        any(existing.get(ticker, 0.0) > 1e-12 for ticker in SEMICONDUCTOR_HOLDINGS)
        and not any(target.get(ticker, 0.0) > 1e-12 for ticker in SEMICONDUCTOR_HOLDINGS)
    )
    individual, aggregate = drift_triggers(
        existing,
        target,
        band=rebalance_band,
    )
    full_transition = (
        strategy_changed
        or structure_changed
        or legacy_exit
        or risk_off_residual_exit
    )
    rebalance_due = full_transition or individual or aggregate
    if strategy_changed:
        reason = "STRATEGY_REVISION_TRANSITION"
    elif legacy_exit:
        reason = "LEGACY_POSITION_EXIT"
    elif risk_off_residual_exit:
        reason = "RISK_OFF_RESIDUAL_EXIT"
    elif lifecycle_changed:
        reason = "LIFECYCLE_STAGE_ADVANCE"
    elif structure_changed:
        reason = decision.transition_reason
    elif individual:
        reason = "INDIVIDUAL_DRIFT_BAND"
    elif aggregate:
        reason = "AGGREGATE_EQUITY_DRIFT_BAND"
    else:
        reason = "HOLD"
    if full_transition or not rebalance_due:
        execution = target
    else:
        execution = inner_band_rebalance_weights(
            existing,
            target,
            destination=rebalance_destination,
            trigger_band=rebalance_band,
            fixed_weights={
                ticker: target.get(ticker, 0.0)
                for ticker in SEMICONDUCTOR_HOLDINGS
                if ticker in set(existing) | set(target)
            },
        )
    if rebalance_due:
        components = set(existing) | set(execution)
        one_way = 0.5 * sum(
            abs(execution.get(item, 0.0) - existing.get(item, 0.0))
            for item in components
        )
        orders = sum(
            item != CASH_ASSET
            and (
                (
                    item in LEGACY_HOLDINGS
                    and existing.get(item, 0.0) > 1e-12
                    and execution.get(item, 0.0) <= 1e-12
                )
                or abs(
                    execution.get(item, 0.0) - existing.get(item, 0.0)
                )
                > 1e-9
            )
            for item in components
        )
    else:
        one_way = 0.0
        orders = 0
    if rebalance_due and orders == 0 and one_way <= 1e-12:
        rebalance_due = False
        full_transition = False
        reason = "CONFIRMED_TARGET_STATE"
        one_way = 0.0
    return RebalancePlan(
        execution_weights=execution,
        rebalance_due=rebalance_due,
        full_transition=full_transition,
        reason=reason,
        one_way_turnover=float(one_way),
        individual_orders=int(orders),
        individual_drift_triggered=individual,
        aggregate_equity_drift_triggered=aggregate,
    )


def calculate_execution_table(
    price_data: pd.DataFrame,
    destination_weights: dict[str, float],
    portfolio_value: float,
    current_state: PortfolioState,
    *,
    actionable: bool,
) -> pd.DataFrame:
    components = set(destination_weights) | set(current_state.shares)
    components.add(CASH_ASSET)
    rows: list[dict[str, object]] = []
    for ticker in components:
        target_weight = float(destination_weights.get(ticker, 0.0))
        target_value = target_weight * portfolio_value
        if ticker == CASH_ASSET:
            price = 1.0
            current_units = float(current_state.cash_balance)
            estimated_units = target_value
            delta_units = estimated_units - current_units
            action = "CASH AFTER TRADES" if actionable else "HOLD"
        else:
            price = float(price_data[ticker].iloc[-1])
            if not np.isfinite(price) or price <= 0:
                raise RuntimeError(f"Invalid latest price for {ticker}")
            current_units = float(current_state.shares.get(ticker, 0.0))
            estimated_units = target_value / price
            delta_units = estimated_units - current_units
            if not actionable:
                action = "HOLD"
            elif (
                ticker in LEGACY_HOLDINGS
                and target_weight <= 1e-12
                and current_units > 0
            ):
                action = "SELL"
            elif abs(delta_units) <= 0.00005:
                action = "HOLD"
            elif delta_units > 0:
                action = "BUY"
            else:
                action = "SELL"
        rows.append(
            {
                "Ticker": ticker,
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
    order = {
        ticker: index
        for index, ticker in enumerate((*STRATEGIC_TICKERS, *VALUATION_TICKERS, CASH_ASSET))
    }
    rows.sort(key=lambda row: order.get(str(row["Ticker"]), 999))
    return pd.DataFrame(rows)


def advertised_daily_exposure(weights: dict[str, float]) -> float:
    unknown = set(weights) - set(ADVERTISED_DAILY_MULTIPLIERS)
    if unknown:
        raise ValueError(
            f"Exposure weights contain unknown components: {sorted(unknown)}"
        )
    exposure = sum(
        float(weight) * ADVERTISED_DAILY_MULTIPLIERS[ticker]
        for ticker, weight in weights.items()
    )
    if not np.isfinite(exposure):
        raise ValueError("Advertised daily exposure is invalid")
    return float(exposure)


def calculate_execution_diagnostics(
    execution_table: pd.DataFrame,
    portfolio_value: float,
    current_weights: dict[str, float],
    strategic_weights: dict[str, float],
    destination_weights: dict[str, float],
) -> ExecutionDiagnostics:
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive")
    gross_notional = 0.0
    if not execution_table.empty:
        security = execution_table["Ticker"] != CASH_ASSET
        values = execution_table.loc[security, "DeltaValue"].to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError("Execution table contains invalid trade values")
        gross_notional = float(np.abs(values).sum())
    gross_fraction = gross_notional / portfolio_value
    return ExecutionDiagnostics(
        current_daily_exposure=advertised_daily_exposure(current_weights),
        strategic_daily_exposure=advertised_daily_exposure(
            _with_cash_target(strategic_weights)
        ),
        destination_daily_exposure=advertised_daily_exposure(
            _with_cash_target(destination_weights)
        ),
        gross_security_trade_fraction=gross_fraction,
        estimated_costs={
            bps: gross_notional * bps / 10_000.0
            for bps in TRANSACTION_COST_SCENARIOS_BPS
        },
    )


# ---------------------------------------------------------------------------
# Orchestration, outbox, and confirmed execution
# ---------------------------------------------------------------------------
def preserve_pending_delivery_plan(
    plan: RebalancePlan,
    state: PortfolioState,
    current_weights: dict[str, float],
) -> RebalancePlan:
    """Retain the exact staged destination during an SMTP retry."""
    if not (
        plan.rebalance_due
        and state.pending_recommendation_date
        and not state.pending_recommendation_notified
        and state.pending_recommendation_fingerprint == STRATEGY_FINGERPRINT
        and _weights_close(
            state.pending_recommendation_weights,
            plan.execution_weights,
            tolerance=NOTIFICATION_WEIGHT_TOLERANCE,
        )
    ):
        return plan
    retry = dict(state.pending_recommendation_weights)
    components = set(retry) | set(current_weights)
    one_way = 0.5 * sum(
        abs(retry.get(item, 0.0) - current_weights.get(item, 0.0))
        for item in components
    )
    orders = sum(
        item != CASH_ASSET
        and (
            (
                item in LEGACY_HOLDINGS
                and current_weights.get(item, 0.0) > 1e-12
                and retry.get(item, 0.0) <= 1e-12
            )
            or abs(
                retry.get(item, 0.0) - current_weights.get(item, 0.0)
            )
            > 1e-9
        )
        for item in components
    )
    return RebalancePlan(
        execution_weights=retry,
        rebalance_due=plan.rebalance_due,
        full_transition=plan.full_transition,
        reason="PENDING_DELIVERY_RETRY",
        one_way_turnover=float(one_way),
        individual_orders=int(orders),
        individual_drift_triggered=plan.individual_drift_triggered,
        aggregate_equity_drift_triggered=(
            plan.aggregate_equity_drift_triggered
        ),
    )


def run_strategy(
    roth_amount: float | None,
    *,
    backup_legacy_state: bool = True,
) -> StrategyRun:
    price_data = download_market_data(ALL_TICKERS)
    signal_date = pd.Timestamp(price_data.index[-1]).normalize()
    fingerprint = market_data_fingerprint(price_data)
    state = load_state(backup_legacy=backup_legacy_state)
    validate_same_date_data_fingerprint(state, signal_date, fingerprint)
    portfolio_value, planning = resolve_portfolio_value(
        roth_amount,
        state,
        price_data,
    )
    decision = calculate_strategy_decision(price_data, state)
    decision = apply_lifecycle_policy(
        decision,
        state,
        portfolio_value,
        signal_date,
    )
    _apply_overlay_state(planning, decision.overlay_state)
    _apply_lifecycle_state(planning, decision, signal_date)
    current_weights = existing_weights(planning, price_data)
    plan = build_rebalance_plan(current_weights, decision, planning)
    plan = preserve_pending_delivery_plan(plan, state, current_weights)
    table_weights = (
        plan.execution_weights
        if plan.rebalance_due
        else _with_cash_target(decision.target_weights)
    )
    table = calculate_execution_table(
        price_data,
        table_weights,
        portfolio_value,
        planning,
        actionable=plan.rebalance_due,
    )
    diagnostics = calculate_execution_diagnostics(
        table,
        portfolio_value,
        current_weights,
        decision.target_weights,
        table_weights,
    )
    return StrategyRun(
        price_data=price_data,
        decision=decision,
        state=state,
        planning_state=planning,
        portfolio_value=portfolio_value,
        current_weights=current_weights,
        execution_table=table,
        signal_date=signal_date,
        market_data_fingerprint=fingerprint,
        rebalance_plan=plan,
        execution_diagnostics=diagnostics,
    )


def _pending_recommendation_matches(strategy_run: StrategyRun) -> bool:
    state = strategy_run.state
    overlay = strategy_run.decision.overlay_state
    return (
        bool(state.pending_recommendation_date)
        and state.pending_recommendation_fingerprint == STRATEGY_FINGERPRINT
        and state.pending_recommendation_overlay_active
        == overlay.overlay_active
        and abs(
            state.pending_recommendation_soxl_weight - overlay.soxl_weight
        )
        <= 1e-12
        and state.pending_recommendation_lifecycle_stage
        == strategy_run.decision.lifecycle_stage
        and _weights_close(
            state.pending_recommendation_weights,
            strategy_run.rebalance_plan.execution_weights,
            tolerance=NOTIFICATION_WEIGHT_TOLERANCE,
        )
    )


def decide_notification(strategy_run: StrategyRun) -> NotificationDecision:
    state = strategy_run.state
    pending_date = state.pending_recommendation_date
    supersedes_date = state.pending_recommendation_supersedes_date
    actionable = (
        strategy_run.rebalance_plan.rebalance_due
        and strategy_run.rebalance_plan.individual_orders > 0
    )
    if actionable:
        if not pending_date:
            return NotificationDecision("ACTION", "NEW_RECOMMENDATION")
        if _pending_recommendation_matches(strategy_run):
            if state.pending_recommendation_notified:
                return NotificationDecision(
                    "NONE",
                    "IDENTICAL_PENDING_RECOMMENDATION",
                    pending_date,
                )
            return NotificationDecision(
                "UPDATE_RETRY" if supersedes_date else "RETRY",
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
    if pending_date:
        return NotificationDecision(
            "CANCELLATION",
            "PENDING_ACTION_NO_LONGER_REQUIRED",
            supersedes_date or pending_date,
        )
    return NotificationDecision("NONE", "HOLD")


def persist_signal_run(strategy_run: StrategyRun) -> None:
    state = strategy_run.state
    if not state.shares and state.cash_balance == 0:
        state.cash_balance = strategy_run.planning_state.cash_balance
    _apply_overlay_state(state, strategy_run.decision.overlay_state)
    _apply_lifecycle_state(
        state,
        strategy_run.decision,
        strategy_run.signal_date,
    )
    state.last_processed_signal_date = (
        strategy_run.signal_date.date().isoformat()
    )
    state.portfolio_value = round(strategy_run.portfolio_value, 2)
    state.last_processed_data_fingerprint = (
        strategy_run.market_data_fingerprint
    )
    if strategy_run.rebalance_plan.reason == "CONFIRMED_TARGET_STATE":
        state.target_weights = dict(
            strategy_run.rebalance_plan.execution_weights
        )
        state.executed_overlay_active = (
            strategy_run.decision.overlay_state.overlay_active
        )
        state.executed_soxl_weight = (
            strategy_run.decision.overlay_state.soxl_weight
        )
        state.executed_lifecycle_stage = (
            strategy_run.decision.lifecycle_stage
        )
        state.executed_strategy_fingerprint = STRATEGY_FINGERPRINT
    save_state(state)


def prepare_notification_delivery(
    strategy_run: StrategyRun,
    notification: NotificationDecision,
) -> None:
    """Stage an exact confirmable action before attempting SMTP."""
    if not notification.should_send:
        return
    state = strategy_run.state
    if notification.kind in {"ACTION", "UPDATE"}:
        overlay = strategy_run.decision.overlay_state
        state.pending_recommendation_date = (
            strategy_run.signal_date.date().isoformat()
        )
        state.pending_recommendation_weights = dict(
            strategy_run.rebalance_plan.execution_weights
        )
        state.pending_recommendation_overlay_active = overlay.overlay_active
        state.pending_recommendation_soxl_weight = overlay.soxl_weight
        state.pending_recommendation_lifecycle_stage = (
            strategy_run.decision.lifecycle_stage
        )
        state.pending_recommendation_notified = False
        state.pending_recommendation_supersedes_date = (
            notification.previous_recommendation_date
            if notification.kind == "UPDATE"
            else ""
        )
        state.pending_recommendation_fingerprint = STRATEGY_FINGERPRINT
    elif notification.kind in {"RETRY", "UPDATE_RETRY"}:
        if not state.pending_recommendation_date:
            raise RuntimeError("Cannot retry a missing pending recommendation")
        state.pending_recommendation_notified = False
    elif notification.kind != "CANCELLATION":
        raise ValueError(f"Unsupported notification kind: {notification.kind}")


def persist_notification_delivery(
    strategy_run: StrategyRun,
    notification: NotificationDecision,
    *,
    delivered_decision_hash: str | None = None,
) -> None:
    if not notification.should_send:
        raise ValueError("A NONE notification cannot be delivered")
    state = strategy_run.state
    if delivered_decision_hash is None:
        delivered_decision_hash = str(
            build_decision_audit(
                strategy_run,
                notification,
                "DELIVERED",
            )["decision_hash"]
        )
    if not _is_sha256(delivered_decision_hash):
        raise ValueError("Delivered decision hash is invalid")
    if notification.kind in {"ACTION", "UPDATE", "RETRY", "UPDATE_RETRY"}:
        if not state.pending_recommendation_date:
            raise RuntimeError("Delivered action is missing outbox state")
        state.pending_recommendation_notified = True
        state.pending_recommendation_supersedes_date = ""
    elif notification.kind == "CANCELLATION":
        _clear_pending_recommendation(state)
    else:
        raise ValueError(f"Unsupported notification kind: {notification.kind}")
    state.last_delivered_decision_hash = delivered_decision_hash
    state.last_delivered_signal_date = (
        strategy_run.signal_date.date().isoformat()
    )
    state.last_delivered_notification_kind = notification.kind
    save_state(state)


def _clear_pending_recommendation(state: PortfolioState) -> None:
    state.pending_recommendation_date = ""
    state.pending_recommendation_weights = {}
    state.pending_recommendation_overlay_active = False
    state.pending_recommendation_soxl_weight = 0.0
    state.pending_recommendation_lifecycle_stage = ""
    state.pending_recommendation_notified = False
    state.pending_recommendation_supersedes_date = ""
    state.pending_recommendation_fingerprint = ""


def validate_execution_confirmation(
    state: PortfolioState,
    executed_signal_date: str,
) -> bool:
    if not state.pending_recommendation_date:
        raise RuntimeError("There is no pending recommendation to confirm")
    if state.pending_recommendation_date == executed_signal_date:
        return state.pending_recommendation_fingerprint == STRATEGY_FINGERPRINT
    executed = _parse_iso_date(
        executed_signal_date,
        "executed_signal_date",
    )
    pending = _parse_iso_date(
        state.pending_recommendation_date,
        "pending_recommendation_date",
    )
    if executed and pending and executed < pending:
        logger.warning(
            "Reconciling fills for superseded signal %s; current pending "
            "signal %s will be recalculated from confirmed holdings",
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
    matches = validate_execution_confirmation(state, executed_signal_date)
    state.shares = dict(executed_shares)
    state.cash_balance = float(executed_cash)
    if matches:
        state.target_weights = dict(state.pending_recommendation_weights)
        state.executed_overlay_active = (
            state.pending_recommendation_overlay_active
        )
        state.executed_soxl_weight = (
            state.pending_recommendation_soxl_weight
        )
        state.executed_lifecycle_stage = (
            state.pending_recommendation_lifecycle_stage
        )
        state.executed_strategy_fingerprint = (
            state.pending_recommendation_fingerprint
        )
    else:
        state.target_weights = {}
        state.executed_overlay_active = False
        state.executed_soxl_weight = 0.0
        state.executed_lifecycle_stage = state.lifecycle_stage
        state.executed_strategy_fingerprint = ""
    _clear_pending_recommendation(state)
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
            "Cannot sync holdings while a recommendation is pending"
        )
    state.shares = dict(executed_shares)
    state.cash_balance = float(executed_cash)
    save_state(state)
    logger.info("Holdings synchronized: holdings=%s", len(executed_shares))
    return state


def log_decision(strategy_run: StrategyRun) -> None:
    decision = strategy_run.decision
    plan = strategy_run.rebalance_plan
    logger.info(
        "decision signal_date=%s strategy=%s strategy_fp=%s data_fp=%s "
        "trend=%s residual_positive=%s raw_soxl=%.2f active=%s "
        "soxl=%.2f transition=%s lifecycle=%s lifecycle_reason=%s age=%.2f "
        "failure=%s rebalance=%s reason=%s "
        "turnover=%.6f orders=%s current=%s target=%s destination=%s",
        strategy_run.signal_date.date(),
        STRATEGY_REVISION,
        STRATEGY_FINGERPRINT[:12],
        strategy_run.market_data_fingerprint[:12],
        decision.trend_positive,
        decision.residual_positive,
        decision.raw_soxl_weight,
        decision.overlay_state.overlay_active,
        decision.overlay_state.soxl_weight,
        decision.transition_reason,
        decision.lifecycle_stage,
        decision.lifecycle_reason,
        decision.estimated_investor_age,
        decision.failure_reason or "NONE",
        plan.rebalance_due,
        plan.reason,
        plan.one_way_turnover,
        plan.individual_orders,
        strategy_run.current_weights,
        decision.target_weights,
        plan.execution_weights,
    )


# ---------------------------------------------------------------------------
# Append-only chained production signal ledger
# ---------------------------------------------------------------------------
def _load_shadow_ledger() -> list[dict[str, object]]:
    if not SHADOW_LEDGER_FILE.exists():
        return []
    try:
        lines = SHADOW_LEDGER_FILE.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError("Could not read the shadow ledger") from exc
    records: list[dict[str, object]] = []
    previous_hash = ""
    previous_date: date | None = None
    current_observation_fields = {
        item.name for item in fields(ShadowObservation)
    }
    legacy_observation_fields = current_observation_fields - {
        "lifecycle_stage",
        "forward_experiment_fingerprint",
        "forward_targets",
    }
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise RuntimeError(
                f"Shadow ledger contains a blank line at {line_number}"
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Shadow ledger line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(record, dict) or set(record) != {
            "schema_version",
            "strategy_fingerprint",
            "previous_hash",
            "record_hash",
            "observation",
        }:
            raise RuntimeError("Shadow ledger record schema is invalid")
        schema_version = record["schema_version"]
        if schema_version not in {
            LEGACY_SHADOW_LEDGER_SCHEMA_VERSION,
            SHADOW_LEDGER_SCHEMA_VERSION,
        }:
            raise RuntimeError("Shadow ledger schema version is invalid")
        if not _is_sha256(record["strategy_fingerprint"]):
            raise RuntimeError("Shadow ledger strategy fingerprint is invalid")
        if record["previous_hash"] != previous_hash:
            raise RuntimeError("Shadow ledger chain linkage is invalid")
        observation = record["observation"]
        expected_fields = (
            legacy_observation_fields
            if schema_version == LEGACY_SHADOW_LEDGER_SCHEMA_VERSION
            else current_observation_fields
        )
        if (
            not isinstance(observation, dict)
            or set(observation) != expected_fields
        ):
            raise RuntimeError("Shadow ledger observation schema is invalid")
        signal_date = _parse_iso_date(
            str(observation["signal_date"]),
            "shadow_signal_date",
        )
        if signal_date is None or (
            previous_date is not None and signal_date <= previous_date
        ):
            raise RuntimeError("Shadow ledger dates are not strictly increasing")
        if not _is_sha256(observation["data_fingerprint"]):
            raise RuntimeError("Shadow ledger data fingerprint is invalid")
        if not isinstance(
            observation["trend_positive"],
            bool,
        ) or not isinstance(observation["residual_positive"], bool):
            raise RuntimeError("Shadow ledger signals must be boolean")
        if not isinstance(
            observation["overlay_active"],
            bool,
        ) or not isinstance(observation["structural_change"], bool):
            raise RuntimeError("Shadow ledger state flags must be boolean")
        if not _valid_soxl_weight(observation["raw_soxl_weight"]) or not (
            _valid_soxl_weight(observation["soxl_weight"])
        ):
            raise RuntimeError("Shadow ledger SOXL weights are invalid")
        if schema_version == SHADOW_LEDGER_SCHEMA_VERSION:
            if observation["lifecycle_stage"] not in LIFECYCLE_STAGES:
                raise RuntimeError("Shadow ledger lifecycle stage is invalid")
            experiment_fingerprint = observation[
                "forward_experiment_fingerprint"
            ]
            if experiment_fingerprint not in {
                LEGACY_FORWARD_EXPERIMENT_FINGERPRINT,
                FORWARD_EXPERIMENT_FINGERPRINT,
            }:
                raise RuntimeError(
                    "Shadow ledger forward experiment fingerprint is invalid"
                )
            expected_targets = forward_shadow_targets(
                float(observation["soxl_weight"]),
                str(observation["lifecycle_stage"]),
                str(experiment_fingerprint),
            )
            if observation["forward_targets"] != expected_targets:
                raise RuntimeError("Shadow ledger forward targets are invalid")
        hash_payload = {
            "schema_version": schema_version,
            "strategy_fingerprint": record["strategy_fingerprint"],
            "previous_hash": record["previous_hash"],
            "observation": observation,
        }
        expected_hash = canonical_sha256(hash_payload)
        if record["record_hash"] != expected_hash:
            raise RuntimeError("Shadow ledger record hash is invalid")
        records.append(record)
        previous_hash = expected_hash
        previous_date = signal_date
    return records


def _shadow_ledger_summary_from_records(
    records: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": SHADOW_LEDGER_SCHEMA_VERSION,
        "sessions": len(records),
        "structural_decisions": sum(
            bool(record["observation"]["structural_change"])
            for record in records
        ),
        "first_signal_date": (
            records[0]["observation"]["signal_date"] if records else ""
        ),
        "last_signal_date": (
            records[-1]["observation"]["signal_date"] if records else ""
        ),
        "chain_hash": records[-1]["record_hash"] if records else "",
    }


def shadow_ledger_summary() -> dict[str, object]:
    return _shadow_ledger_summary_from_records(_load_shadow_ledger())


def _validate_shadow_ledger_anchor(
    state: PortfolioState,
    records: list[dict[str, object]],
) -> int:
    """Prove that the persisted state anchor is present in this chain."""

    anchored = state.shadow_ledger_sessions
    if len(records) < anchored:
        raise RuntimeError(
            "Shadow ledger is missing or truncated relative to portfolio state"
        )
    if anchored:
        record = records[anchored - 1]
        if (
            record["record_hash"] != state.shadow_ledger_chain_hash
            or record["observation"]["signal_date"]
            != state.shadow_ledger_last_signal_date
        ):
            raise RuntimeError(
                "Shadow ledger does not match the persisted state anchor"
            )
    return anchored


def _expected_shadow_observation_for_record(
    expected: dict[str, object],
    record: dict[str, object],
) -> dict[str, object]:
    """Render a deterministic replay in the record's historical protocol."""
    normalized = copy.deepcopy(expected)
    schema_version = record["schema_version"]
    if schema_version == LEGACY_SHADOW_LEDGER_SCHEMA_VERSION:
        for field_name in (
            "lifecycle_stage",
            "forward_experiment_fingerprint",
            "forward_targets",
        ):
            normalized.pop(field_name)
        return normalized
    if schema_version != SHADOW_LEDGER_SCHEMA_VERSION:
        raise RuntimeError("Shadow ledger schema version is invalid")
    recorded_observation = record["observation"]
    if not isinstance(recorded_observation, dict):
        raise RuntimeError("Shadow ledger observation schema is invalid")
    experiment_fingerprint = str(
        recorded_observation["forward_experiment_fingerprint"]
    )
    normalized["forward_experiment_fingerprint"] = experiment_fingerprint
    normalized["forward_targets"] = forward_shadow_targets(
        float(normalized["soxl_weight"]),
        str(normalized["lifecycle_stage"]),
        experiment_fingerprint,
    )
    return normalized


def append_shadow_ledger(strategy_run: StrategyRun) -> dict[str, object]:
    """Atomically append distinct causal observations to the hash chain."""
    records = _load_shadow_ledger()
    anchored = _validate_shadow_ledger_anchor(strategy_run.state, records)
    observations = tuple(strategy_run.decision.shadow_observations)
    expected_by_date = {
        observation.signal_date: asdict(observation)
        for observation in observations
    }
    if len(expected_by_date) != len(observations):
        raise RuntimeError("Shadow observations contain duplicate dates")

    # A crash can leave an atomically written chain ahead of the older state
    # artifact. Accept only the exact deterministic replay of those records.
    for record in records[anchored:]:
        observation = record["observation"]
        signal_date = str(observation["signal_date"])
        expected = expected_by_date.get(signal_date)
        if (
            record["strategy_fingerprint"] != STRATEGY_FINGERPRINT
            or expected is None
            or _expected_shadow_observation_for_record(expected, record)
            != observation
        ):
            raise RuntimeError(
                "Shadow ledger contains unanchored observations that do not "
                "match deterministic state replay"
            )

    last_date = (
        pd.Timestamp(records[-1]["observation"]["signal_date"]).date()
        if records
        else None
    )
    previous_hash = str(records[-1]["record_hash"]) if records else ""
    appended = False
    for observation in strategy_run.decision.shadow_observations:
        observation_date = pd.Timestamp(observation.signal_date).date()
        if last_date is not None and observation_date <= last_date:
            continue
        observation_payload = asdict(observation)
        hash_payload = {
            "schema_version": SHADOW_LEDGER_SCHEMA_VERSION,
            "strategy_fingerprint": STRATEGY_FINGERPRINT,
            "previous_hash": previous_hash,
            "observation": observation_payload,
        }
        record_hash = canonical_sha256(hash_payload)
        records.append({**hash_payload, "record_hash": record_hash})
        previous_hash = record_hash
        last_date = observation_date
        appended = True
    if appended:
        rendered = "\n".join(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for record in records
        ) + "\n"
        temporary = SHADOW_LEDGER_FILE.with_suffix(
            f"{SHADOW_LEDGER_FILE.suffix}.tmp"
        )
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, SHADOW_LEDGER_FILE)
    summary = _shadow_ledger_summary_from_records(records)
    strategy_run.state.shadow_ledger_sessions = int(summary["sessions"])
    strategy_run.state.shadow_ledger_last_signal_date = str(
        summary["last_signal_date"]
    )
    strategy_run.state.shadow_ledger_chain_hash = str(summary["chain_hash"])
    return summary


# ---------------------------------------------------------------------------
# Structured decision audit
# ---------------------------------------------------------------------------
def _ridge_audit(fit: core.RidgeFit) -> dict[str, object]:
    return {
        "feature_names": list(fit.feature_names),
        "information_cutoff": fit.training_end.date().isoformat(),
        "last_label_origin": fit.last_label_origin.date().isoformat(),
        "sample_count": fit.sample_count,
        "feature_means": list(fit.feature_means),
        "feature_scales": list(fit.feature_scales),
        "intercept": fit.intercept,
        "coefficients": list(fit.coefficients),
        "current_features": list(fit.current_features),
        "prediction": fit.prediction,
        "smearing_factor": fit.smearing_factor,
    }


def _variance_audit(
    forecast: core.VarianceForecast,
) -> dict[str, object]:
    return {
        "model_volatility": forecast.model_volatility,
        "trailing_volatility_21": forecast.trailing_volatility_21,
        "trailing_volatility_63": forecast.trailing_volatility_63,
        "sizing_volatility": forecast.sizing_volatility,
        "ridge_fit": _ridge_audit(forecast.ridge_fit),
    }


def build_decision_audit(
    strategy_run: StrategyRun,
    notification: NotificationDecision,
    delivery_status: str,
) -> dict[str, object]:
    if delivery_status not in {"NOT_REQUIRED", "STAGED", "DELIVERED"}:
        raise ValueError("Unsupported audit delivery status")
    if notification.should_send and delivery_status == "NOT_REQUIRED":
        raise ValueError("A notification cannot be NOT_REQUIRED")
    if not notification.should_send and delivery_status != "NOT_REQUIRED":
        raise ValueError("A HOLD audit must be NOT_REQUIRED")
    decision = strategy_run.decision
    plan = strategy_run.rebalance_plan
    execution = strategy_run.execution_diagnostics
    residual = decision.residual_signal
    volatility = decision.portfolio_volatility
    core_payload: dict[str, object] = {
        "signal_date": strategy_run.signal_date.date().isoformat(),
        "strategy": {
            "revision": STRATEGY_REVISION,
            "fingerprint": STRATEGY_FINGERPRINT,
            "manifest": strategy_manifest(),
            "experimental_live": EXPERIMENTAL_LIVE,
            "research_promotion_complete": False,
        },
        "operations": operational_manifest(),
        "market_data": {
            "provider": "yfinance",
            "auto_adjust": True,
            "fingerprint": strategy_run.market_data_fingerprint,
            "latest_prices": {
                ticker: float(strategy_run.price_data[ticker].iloc[-1])
                for ticker in ALL_TICKERS
            },
        },
        "indicators": {
            "qqq_close": decision.qqq_close,
            "qqq_sma_200": decision.qqq_sma_200,
            "trend_positive": decision.trend_positive,
            "trend_margin": (
                decision.qqq_close / decision.qqq_sma_200 - 1.0
            ),
            "residual_positive": decision.residual_positive,
            "residual": (
                None
                if residual is None
                else {
                    "intercept": residual.intercept,
                    "beta": residual.beta,
                    "momentum": residual.residual_momentum,
                    "sigma": residual.residual_sigma,
                    "z": residual.residual_z,
                    "estimation_start": (
                        residual.estimation_start.date().isoformat()
                    ),
                    "estimation_end": (
                        residual.estimation_end.date().isoformat()
                    ),
                    "scoring_start": residual.scoring_start.date().isoformat(),
                    "scoring_end": residual.scoring_end.date().isoformat(),
                }
            ),
            "portfolio_volatility": (
                None
                if volatility is None
                else {
                    "qld_proxy": _variance_audit(volatility.qld),
                    "soxl": _variance_audit(volatility.soxl),
                    "correlation_21": volatility.correlation_21,
                    "correlation_63": volatility.correlation_63,
                    "sizing_correlation": volatility.sizing_correlation,
                    "raw_soxl_weight": volatility.raw_soxl_weight,
                    "raw_portfolio_volatility": (
                        volatility.raw_portfolio_volatility
                    ),
                }
            ),
            "failure_reason": decision.failure_reason,
        },
        "decision": {
            "transition_reason": decision.transition_reason,
            "lifecycle": {
                "stage": decision.lifecycle_stage,
                "reason": decision.lifecycle_reason,
                "stage_advanced": decision.lifecycle_stage_advanced,
                "estimated_investor_age": decision.estimated_investor_age,
                "value_stage": decision.lifecycle_value_stage,
                "age_stage": decision.lifecycle_age_stage,
                "exposure_ceiling": LIFECYCLE_EXPOSURE_CEILINGS[
                    decision.lifecycle_stage
                ],
            },
            "processed_signal_dates": list(
                decision.processed_signal_dates
            ),
            "transition_path": list(decision.transition_path),
            "alpha_reviewed": decision.alpha_reviewed,
            "structural_change": decision.structural_change,
            "overlay_state": asdict(decision.overlay_state),
            "strategic_weights": _with_cash_target(
                decision.target_weights
            ),
            "current_weights": dict(strategy_run.current_weights),
            "execution_weights": dict(plan.execution_weights),
            "rebalance_due": plan.rebalance_due,
            "full_transition": plan.full_transition,
            "reason": plan.reason,
            "individual_drift_triggered": (
                plan.individual_drift_triggered
            ),
            "aggregate_equity_drift_triggered": (
                plan.aggregate_equity_drift_triggered
            ),
            "one_way_turnover": plan.one_way_turnover,
            "gross_security_trade_fraction": (
                execution.gross_security_trade_fraction
            ),
            "current_daily_exposure": execution.current_daily_exposure,
            "strategic_daily_exposure": (
                execution.strategic_daily_exposure
            ),
            "destination_daily_exposure": (
                execution.destination_daily_exposure
            ),
            "cost_sensitivity_fraction": {
                str(bps): (
                    execution.gross_security_trade_fraction * bps / 10_000.0
                )
                for bps in TRANSACTION_COST_SCENARIOS_BPS
            },
        },
        "notification": {
            "kind": notification.kind,
            "reason": notification.reason,
            "previous_recommendation_date": (
                notification.previous_recommendation_date
            ),
            "supersedes_recommendation_date": (
                notification.supersedes_recommendation_date
            ),
        },
    }
    decision_hash = canonical_sha256(core_payload)
    if delivery_status == "DELIVERED":
        last_confirmed: dict[str, str] | None = {
            "decision_hash": decision_hash,
            "signal_date": strategy_run.signal_date.date().isoformat(),
            "notification_kind": notification.kind,
        }
    elif strategy_run.state.last_delivered_decision_hash:
        last_confirmed = {
            "decision_hash": strategy_run.state.last_delivered_decision_hash,
            "signal_date": strategy_run.state.last_delivered_signal_date,
            "notification_kind": (
                strategy_run.state.last_delivered_notification_kind
            ),
        }
    else:
        last_confirmed = None
    state_hash = (
        hashlib.sha256(STATE_FILE.read_bytes()).hexdigest()
        if STATE_FILE.exists()
        else ""
    )
    shadow_hash = (
        hashlib.sha256(SHADOW_LEDGER_FILE.read_bytes()).hexdigest()
        if SHADOW_LEDGER_FILE.exists()
        else ""
    )
    return {
        "audit_schema_version": DECISION_AUDIT_SCHEMA_VERSION,
        "decision_hash": decision_hash,
        **core_payload,
        "delivery": {
            "status": delivery_status,
            "last_confirmed": last_confirmed,
        },
        "runtime": {
            "created_at": datetime.now(NEW_YORK).isoformat(),
            "github_sha": os.environ.get("GITHUB_SHA", ""),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "implementation_sha256": calculate_implementation_fingerprint(),
            "state_sha256": state_hash,
            "shadow_ledger_sha256": shadow_hash,
            "shadow_ledger": shadow_ledger_summary(),
        },
    }


def write_decision_audit(
    strategy_run: StrategyRun,
    notification: NotificationDecision,
    delivery_status: str,
) -> dict[str, object]:
    payload = build_decision_audit(
        strategy_run,
        notification,
        delivery_status,
    )
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    temporary = DECISION_AUDIT_FILE.with_suffix(
        f"{DECISION_AUDIT_FILE.suffix}.tmp"
    )
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, DECISION_AUDIT_FILE)
    return payload


# ---------------------------------------------------------------------------
# Human-readable report and email
# ---------------------------------------------------------------------------
def build_dashboard(strategy_run: StrategyRun) -> str:
    decision = strategy_run.decision
    plan = strategy_run.rebalance_plan
    execution = strategy_run.execution_diagnostics
    residual_z = (
        "INVALID"
        if decision.residual_signal is None
        else f"{decision.residual_signal.residual_z:+.2f}"
    )
    sizing_volatility = (
        "INVALID"
        if decision.portfolio_volatility is None
        else (
            f"QLD proxy {decision.portfolio_volatility.qld.sizing_volatility:.1%}, "
            f"SOXL {decision.portfolio_volatility.soxl.sizing_volatility:.1%}"
        )
    )
    lines = [
        "=" * 112,
        "ROTH IRA - LIFECYCLE ALPHA ALLOCATION",
        "=" * 112,
        (
            f"Signal close: {strategy_run.signal_date.date()} | "
            f"Portfolio value: ${strategy_run.portfolio_value:,.2f}"
        ),
        (
            "Governance: EXPERIMENTAL LIVE - "
            "research promotion_complete=false"
        ),
        (
            f"QQQ trend: {'POSITIVE' if decision.trend_positive else 'NEGATIVE'} "
            f"({decision.qqq_close / decision.qqq_sma_200 - 1:+.2%} vs SMA200) | "
            f"SMH residual z: {residual_z}"
        ),
        (
            f"Volatility sizing: {sizing_volatility} | "
            f"raw SOXL {decision.raw_soxl_weight:.0%} | "
            f"state SOXL {decision.overlay_state.soxl_weight:.0%}"
        ),
        (
            f"Lifecycle: {decision.lifecycle_stage} "
            f"({decision.lifecycle_reason}) | estimated age "
            f"{decision.estimated_investor_age:.1f} | value/age gates "
            f"{decision.lifecycle_value_stage}/{decision.lifecycle_age_stage}"
        ),
        (
            f"Transition: {decision.transition_reason} | "
            f"Rebalance: {'YES' if plan.rebalance_due else 'NO'} | "
            f"Reason: {plan.reason} | One-way turnover: {plan.one_way_turnover:.1%}"
        ),
        (
            f"Advertised daily exposure: current "
            f"{execution.current_daily_exposure:.2f}x, strategic "
            f"{execution.strategic_daily_exposure:.2f}x, destination "
            f"{execution.destination_daily_exposure:.2f}x"
        ),
        (
            "Cost sensitivity: "
            + " / ".join(
                f"{bps}bp ${execution.estimated_costs[bps]:,.2f}"
                for bps in TRANSACTION_COST_SCENARIOS_BPS
            )
        ),
        (
            f"Strategy {STRATEGY_REVISION} ({STRATEGY_FINGERPRINT[:12]}) | "
            f"Data {strategy_run.market_data_fingerprint[:12]}"
        ),
        "-" * 112,
        (
            f"{'Ticker':<8}{'Price':>12}{'Current':>14}{'Target':>11}"
            f"{'Est. units':>15}{'Delta':>14}{'Est. trade':>16}{'Action':>13}"
        ),
        "-" * 112,
    ]
    if len(decision.processed_signal_dates) > 1:
        lines.insert(
            4,
            (
                f"Catch-up replay: {len(decision.processed_signal_dates)} "
                "completed sessions | "
                f"last path {' -> '.join(decision.transition_path[-10:])}"
            ),
        )
    for _, row in strategy_run.execution_table.iterrows():
        lines.append(
            f"{row['Ticker']:<8}"
            f"${row['Price']:>11,.2f}"
            f"{row['CurrentUnits']:>14,.4f}"
            f"{row['TargetPct']:>10.1%}"
            f"{row['EstimatedUnits']:>15,.4f}"
            f"{row['DeltaUnits']:>14,.4f}"
            f"${row['DeltaValue']:>15,.2f}"
            f"{row['Action']:>13}"
        )
    lines.extend(
        [
            "-" * 112,
            (
                "Quantities use the signal close. Recalculate orders from "
                "next-session executable prices, then confirm complete holdings."
            ),
            (
                f"Drift trigger/non-overlay destination: {REBALANCE_BAND:.0%}/"
                f"{REBALANCE_DESTINATION:.1%}; the semiconductor sleeve "
                "returns to its exact latent 0/15/25/35% source tier; "
                f"cap {core.MAX_SOXL_WEIGHT:.0%}; "
                "no margin or options."
            ),
            "=" * 112,
        ]
    )
    return "\n".join(lines)


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
        raise ValueError("A NONE notification has no subject")
    subject_date = (
        notification.previous_recommendation_date
        if notification.kind in {"RETRY", "UPDATE_RETRY"}
        else strategy_run.signal_date.date().isoformat()
    )
    prefix = "ROTH IRA Experimental" if EXPERIMENTAL_LIVE else "ROTH IRA"
    return f"{prefix} {label} - {subject_date}"


def notification_text_body(
    strategy_run: StrategyRun,
    notification: NotificationDecision,
    dashboard: str,
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


def build_email_html(
    strategy_run: StrategyRun,
    notification: NotificationDecision,
) -> str:
    plan = strategy_run.rebalance_plan
    rows = "".join(
        (
            "<tr>"
            f"<td>{row['Ticker']}</td><td>${row['Price']:,.2f}</td>"
            f"<td>{row['CurrentUnits']:,.4f}</td>"
            f"<td>{row['TargetPct']:.1%}</td>"
            f"<td>{row['EstimatedUnits']:,.4f}</td>"
            f"<td>{row['DeltaUnits']:,.4f}</td>"
            f"<td>{row['Action']}</td></tr>"
        )
        for _, row in strategy_run.execution_table.iterrows()
    )
    status = {
        "ACTION": "ACTION REQUIRED",
        "UPDATE": "UPDATED ACTION REQUIRED",
        "RETRY": "ACTION DELIVERY RETRY",
        "UPDATE_RETRY": "UPDATED ACTION DELIVERY RETRY",
        "CANCELLATION": "PREVIOUS ACTION CANCELLED",
    }[notification.kind]
    return f"""<!doctype html><html><body style="font-family:Arial,sans-serif">
<h2>ROTH IRA Lifecycle Alpha Allocation (Experimental)</h2>
<p><strong>Research promotion_complete=false.</strong></p>
<h3>{status}: {plan.reason}</h3>
<p>Signal close {strategy_run.signal_date.date()} · Portfolio
${strategy_run.portfolio_value:,.2f} · Lifecycle stage
{strategy_run.decision.lifecycle_stage} · Latent SOXL tier
{strategy_run.decision.overlay_state.soxl_weight:.0%}</p>
<table style="border-collapse:collapse" cellpadding="7">
<tr><th>Ticker</th><th>Price</th><th>Current</th><th>Target</th>
<th>Est. units</th><th>Delta</th><th>Action</th></tr>{rows}</table>
<p>Signal-close estimates only. Recalculate at executable prices next session,
then confirm complete post-trade holdings and CASH.</p></body></html>"""


def send_email(subject: str, text_body: str, html_body: str) -> None:
    address = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("RECEIVER_EMAIL")
    if not all((address, password, recipient)):
        raise RuntimeError(
            "GMAIL_ADDRESS, GMAIL_APP_PASSWORD, and RECEIVER_EMAIL are "
            "required only when a portfolio notification is due"
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


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
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
            raise ValueError(
                f"Value must be nonnegative and finite: {entry!r}"
            )
        if name == CASH_ASSET:
            if cash is not None:
                raise ValueError("CASH was supplied more than once")
            cash = value
        elif name not in TRADED_TICKERS or name in holdings:
            raise ValueError(f"Invalid or duplicate holding: {entry!r}")
        else:
            holdings[name] = value
    if cash is None:
        raise ValueError(
            "Complete holdings must include CASH, even when CASH=0"
        )
    return holdings, cash


def parse_signal_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "--executed-signal-date must be a valid date"
        ) from exc


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROTH IRA lifecycle alpha allocation engine"
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
        help="Synchronize complete broker holdings when no action is pending",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate a report without state, email, audit, or log persistence",
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
        help="Complete broker holdings, including CASH",
    )
    parser.add_argument(
        "--executed-signal-date",
        metavar="YYYY-MM-DD",
        help="Pending signal date being confirmed",
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
        executed_shares, executed_cash = parse_executed_shares(
            args.executed_shares
        )
        executed_signal_date = parse_signal_date(args.executed_signal_date)
    except ValueError as exc:
        parser.error(str(exc))

    if args.confirm_execution:
        if args.test or args.roth_amount is not None:
            parser.error(
                "--confirm-execution cannot be combined with --test or "
                "--roth-amount"
            )
        if executed_shares is None or executed_signal_date is None:
            parser.error(
                "--confirm-execution requires --executed-shares and "
                "--executed-signal-date"
            )
        confirm_execution(
            executed_shares,
            executed_cash,
            executed_signal_date,
        )
        print(
            f"Confirmed execution for signal {executed_signal_date}: "
            f"{len(executed_shares)} holdings, cash ${executed_cash:,.2f}"
        )
        return

    if args.sync_holdings:
        if args.test or args.roth_amount is not None or executed_signal_date:
            parser.error(
                "--sync-holdings cannot be combined with --test, "
                "--roth-amount, or --executed-signal-date"
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
    dashboard = build_dashboard(strategy_run)
    print(dashboard)
    notification = decide_notification(strategy_run)
    if args.test:
        return

    log_decision(strategy_run)
    logger.info(
        "notification kind=%s reason=%s",
        notification.kind,
        notification.reason,
    )
    # The exact outbox and actual holdings are checkpointed before SMTP.
    prepare_notification_delivery(strategy_run, notification)
    append_shadow_ledger(strategy_run)
    persist_signal_run(strategy_run)
    audit = write_decision_audit(
        strategy_run,
        notification,
        "STAGED" if notification.should_send else "NOT_REQUIRED",
    )
    if notification.should_send:
        send_email(
            notification_subject(strategy_run, notification),
            notification_text_body(
                strategy_run,
                notification,
                dashboard,
            ),
            build_email_html(strategy_run, notification),
        )
        # Delivery proof is persisted before the final audit write so an audit
        # filesystem error can never cause a duplicate email.
        persist_notification_delivery(
            strategy_run,
            notification,
            delivered_decision_hash=str(audit["decision_hash"]),
        )
        write_decision_audit(
            strategy_run,
            notification,
            "DELIVERED",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("ROTH IRA engine failed")
        sys.exit(1)

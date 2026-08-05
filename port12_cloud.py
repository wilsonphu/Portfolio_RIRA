#!/usr/bin/env python3
"""
ROTH IRA - Barbell Momentum Allocation Engine
==============================================
Production daily allocation engine for a Roth IRA.

Strategy Architecture
---------------------
  Regime Filter (2-of-3 consensus on MARKET_INDEX):
    - 200-day SMA, 50-day Donchian Midband, 50-day VWMA
    -> BULL if at least 2 agree, otherwise BEAR.

  Three-Tier Volatility Scaling (Bull regime):
    - Low Vol    (<15%):  45% Leader / 15% Follower / 25% SMH / 15% QLD
    - Moderate   (15-22%): 25% Leader / 10% Follower / 45% SMH / 20% QLD
    - High Vol   (>22%):  85% SMH / 15% GLD (de-leveraged)

  Leader/Follower Dynamic:
    - SOXL vs TECL 15-day momentum; winner holds the Leader slot.
    - Switches only on >5% momentum divergence (hysteresis).

  Bear Regime:
    - 80% SPMO (defensive equity momentum) / 20% GLD (hedge).

  Risk Controls:
    - 5% rebalance band vs last target weights
    - 45% cap on any single leveraged sector position
    - Monthly (~21 day) sector review

Environment variables:
  ROTH_IRA_AMOUNT      Current Roth IRA balance (required on a first run)
  GMAIL_ADDRESS        Sender Gmail address  (optional, for email reports)
  GMAIL_APP_PASSWORD   Gmail app password    (optional)
  RECEIVER_EMAIL       Report recipient      (optional)

CLI usage:
  python3 port12_cloud.py                 # run, print dashboard, save state, send email
  python3 port12_cloud.py --test          # run, print dashboard, do not save state or send email
  python3 port12_cloud.py --roth-amount 5000.00
"""

import argparse
import json
import logging
import os
import smtplib
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 1. MARKET UNIVERSE
# ==========================================
MARKET_INDEX = "QQQ"              # regime + volatility signal asset (never traded)

LEVERAGED_SEMICONDUCTOR = "SOXL"  # 3x semiconductors (leader candidate)
LEVERAGED_TECH = "TECL"           # 3x technology (leader candidate)
SEMICONDUCTOR_ETF = "SMH"         # 1x semiconductors (stabilizer)
LEVERAGED_INDEX = "QLD"           # 2x broad Nasdaq-100 (core)

DEFENSIVE_EQUITY = "SPMO"         # S&P 500 momentum (bear-regime equity)
HEDGE_ASSET = "GLD"               # gold (hedge)

LEADER_CANDIDATES = [LEVERAGED_SEMICONDUCTOR, LEVERAGED_TECH]
LEVERAGED_SECTOR_ETFS = {LEVERAGED_SEMICONDUCTOR, LEVERAGED_TECH}

ALL_TICKERS = [
    MARKET_INDEX,
    LEVERAGED_SEMICONDUCTOR,
    LEVERAGED_TECH,
    SEMICONDUCTOR_ETF,
    LEVERAGED_INDEX,
    DEFENSIVE_EQUITY,
    HEDGE_ASSET,
]
TRADED_TICKERS = frozenset(ALL_TICKERS) - {MARKET_INDEX}

# ==========================================
# 2. CONSTANTS
# ==========================================
# Market data
SMA_WINDOW = 200
DONCHIAN_WINDOW = 50
VWMA_WINDOW = 50
VOLATILITY_WINDOW = 20
MOMENTUM_WINDOW = 15
HISTORY_DAYS = 750

# Volatility thresholds
LOW_VOL_THRESHOLD = 0.15
MODERATE_VOL_THRESHOLD = 0.22

# Risk controls
REBALANCE_BAND = 0.05
MAX_LEVERAGED_POSITION = 0.45
LEADER_SWITCH_THRESHOLD = 0.05
SECTOR_REBALANCE_DAYS = 21

# Allocation templates ("Leader"/"Follower" slots resolve at runtime)
LOW_VOL_ALLOCATION = {"Leader": 0.45, "Follower": 0.15, SEMICONDUCTOR_ETF: 0.25, LEVERAGED_INDEX: 0.15}
MODERATE_VOL_ALLOCATION = {"Leader": 0.25, "Follower": 0.10, SEMICONDUCTOR_ETF: 0.45, LEVERAGED_INDEX: 0.20}
HIGH_VOL_ALLOCATION = {SEMICONDUCTOR_ETF: 0.85, HEDGE_ASSET: 0.15}
BEAR_ALLOCATION = {DEFENSIVE_EQUITY: 0.80, HEDGE_ASSET: 0.20}

# Account
NEW_YORK = ZoneInfo("America/New_York")
MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_BUFFER_TIME = time(16, 15)
configured_roth_amount = os.environ.get("ROTH_IRA_AMOUNT")
try:
    ROTH_IRA_AMOUNT = float(configured_roth_amount) if configured_roth_amount is not None else None
except ValueError as exc:
    raise RuntimeError("ROTH_IRA_AMOUNT must be numeric") from exc
if ROTH_IRA_AMOUNT is not None and (
        not np.isfinite(ROTH_IRA_AMOUNT) or ROTH_IRA_AMOUNT <= 0):
    raise RuntimeError("ROTH_IRA_AMOUNT must be positive and finite")
APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / "roth_ira_state.json"
LOG_FILE = APP_DIR / "roth_ira.log"
STATE_VERSION = 1

# ==========================================
# 3. DATA CLASSES
# ==========================================
@dataclass
class StrategyResult:
    """Output of the allocation engine: everything a report or state needs."""
    target_weights: dict[str, float]
    regime: str               # "BULL" | "BEAR"
    leader: str               # "SOXL" | "TECL"
    volatility_tier: str      # "LOW" | "MODERATE" | "HIGH" | "N/A"
    annualized_volatility: float


@dataclass
class PortfolioState:
    """Persisted account state between runs."""
    state_version: int = STATE_VERSION
    shares: dict[str, float] = field(default_factory=dict)
    target_weights: dict[str, float] = field(default_factory=dict)
    portfolio_value: float = 0.0
    leader: str | None = None
    volatility_tier: str = "N/A"
    regime: str = "UNKNOWN"
    last_sector_rebalance: str = ""
    last_updated: str = ""


@dataclass(frozen=True)
class StrategyRun:
    """All point-in-time values produced during a single signal run."""
    price_data: pd.DataFrame
    latest_indicators: pd.Series
    result: StrategyResult
    state: PortfolioState
    portfolio_value: float
    target_portfolio: pd.DataFrame
    signal_date: pd.Timestamp
    rebalance_due: bool
    sector_review_due: bool

# ==========================================
# 4. LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("port12_cloud")

# ==========================================
# 5. STATE PERSISTENCE
# ==========================================
def load_state() -> PortfolioState:
    if not STATE_FILE.exists():
        return PortfolioState()
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {STATE_FILE}; refusing to infer holdings") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Portfolio state must be a JSON object")
    required_fields = {
        "state_version",
        "shares",
        "target_weights",
        "portfolio_value",
        "leader",
        "volatility_tier",
        "regime",
        "last_sector_rebalance",
        "last_updated",
    }
    missing_fields = required_fields - set(payload)
    if missing_fields:
        raise RuntimeError(f"Portfolio state is missing required fields: {sorted(missing_fields)}")
    if payload.get("state_version") != STATE_VERSION:
        raise RuntimeError(
            f"Unsupported portfolio state version: {payload.get('state_version')!r}; "
            "migrate it explicitly"
        )
    if not isinstance(payload.get("shares"), dict):
        raise RuntimeError("Portfolio state is missing a valid shares mapping")
    if not isinstance(payload.get("target_weights"), dict):
        raise RuntimeError("Portfolio state is missing a valid target_weights mapping")
    try:
        state = PortfolioState(**payload)
    except TypeError as exc:
        raise RuntimeError(f"Unsupported fields in {STATE_FILE}; migrate it explicitly") from exc

    invalid_tickers = (
        (set(state.shares) | set(state.target_weights)) - TRADED_TICKERS
    )
    if invalid_tickers:
        raise RuntimeError(
            "Unsupported portfolio state; migrate it explicitly "
            f"(invalid_tickers={sorted(invalid_tickers)})"
        )
    if any(isinstance(shares, bool) or not isinstance(shares, (int, float))
           or not np.isfinite(shares) or shares < 0
           for shares in state.shares.values()):
        raise RuntimeError("Portfolio state contains invalid share counts")
    if any(isinstance(weight, bool) or not isinstance(weight, (int, float))
           or not np.isfinite(weight) or weight < 0
           for weight in state.target_weights.values()):
        raise RuntimeError("Portfolio state contains invalid target weights")
    if (isinstance(state.portfolio_value, bool) or not isinstance(state.portfolio_value, (int, float))
            or not np.isfinite(state.portfolio_value) or state.portfolio_value < 0):
        raise RuntimeError("Portfolio state contains an invalid portfolio value")
    if state.leader is not None and state.leader not in LEADER_CANDIDATES:
        raise RuntimeError("Portfolio state contains an invalid leader")
    if state.regime not in {"UNKNOWN", "BULL", "BEAR"}:
        raise RuntimeError("Portfolio state contains an invalid regime")
    if state.volatility_tier not in {"N/A", "LOW", "MODERATE", "HIGH"}:
        raise RuntimeError("Portfolio state contains an invalid volatility tier")
    if not isinstance(state.last_updated, str) or not isinstance(state.last_sector_rebalance, str):
        raise RuntimeError("Portfolio state contains invalid timestamps")
    if state.last_sector_rebalance:
        try:
            pd.Timestamp(state.last_sector_rebalance)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Portfolio state contains an invalid sector-review date") from exc
    return state


def save_state(state: PortfolioState) -> None:
    """Atomically save validated state or fail the run."""
    state.last_updated = datetime.now(NEW_YORK).isoformat()
    payload = json.dumps(asdict(state), indent=4, allow_nan=False)
    temporary_file = STATE_FILE.with_suffix(".json.tmp")
    temporary_file.write_text(payload, encoding="utf-8")
    os.replace(temporary_file, STATE_FILE)
    logger.info(
        "State saved: regime=%s, leader=%s, tier=%s, value=$%.2f",
        state.regime, state.leader, state.volatility_tier, state.portfolio_value,
    )

# ==========================================
# 6. DATA ACQUISITION (real daily data only)
# ==========================================
def download_market_data(tickers: list[str], days: int = HISTORY_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    now_new_york = datetime.now(NEW_YORK)
    today_new_york = now_new_york.date()
    if today_new_york.weekday() < 5 and MARKET_OPEN_TIME <= now_new_york.time() < MARKET_CLOSE_BUFFER_TIME:
        raise RuntimeError(
            "Daily signal is not complete during market hours; run before 09:30 or after 16:15 America/New_York"
        )
    start = (today_new_york - timedelta(days=days)).isoformat()
    # yfinance treats end as exclusive.  After the close include today's completed bar;
    # before the open use the prior completed session.
    end_date = today_new_york + timedelta(days=1) if now_new_york.time() >= MARKET_CLOSE_BUFFER_TIME else today_new_york
    end = end_date.isoformat()
    logger.info(f"Downloading {tickers} from {start} to {end}")

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}")

    try:
        price_data = data["Close"].copy() if isinstance(data.columns, pd.MultiIndex) else pd.DataFrame(data["Close"])
        volume_data = data["Volume"].copy() if isinstance(data.columns, pd.MultiIndex) else pd.DataFrame(data["Volume"])
    except KeyError as exc:
        raise RuntimeError("yfinance response is missing Close or Volume data") from exc

    missing_price_columns = set(tickers) - set(price_data.columns)
    missing_volume_columns = set(tickers) - set(volume_data.columns)
    if missing_price_columns or missing_volume_columns:
        raise RuntimeError(
            "Incomplete market-data universe: "
            f"prices={sorted(missing_price_columns)}, volumes={sorted(missing_volume_columns)}"
        )
    if not price_data.index.equals(volume_data.index):
        raise RuntimeError("Price and volume dates differ")

    price_data = price_data.reindex(columns=tickers)
    volume_data = volume_data.reindex(columns=tickers)
    complete_rows = price_data.notna().all(axis=1) & volume_data.notna().all(axis=1)
    if not complete_rows.any():
        raise RuntimeError("No complete market-data rows were returned")
    if not complete_rows.iloc[-1]:
        missing = [
            ticker for ticker in tickers
            if pd.isna(price_data.iloc[-1][ticker]) or pd.isna(volume_data.iloc[-1][ticker])
        ]
        raise RuntimeError(f"Latest market-data row is partial: {missing}")
    price_data = price_data.loc[complete_rows]
    volume_data = volume_data.loc[complete_rows]
    try:
        prices_are_valid = np.isfinite(price_data.to_numpy(dtype=float)).all() and (price_data > 0).all().all()
        volumes_are_valid = (
            np.isfinite(volume_data.to_numpy(dtype=float)).all()
            and (volume_data >= 0).all().all()
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Market data contains non-numeric values") from exc
    if not prices_are_valid:
        raise RuntimeError("Market data contains invalid prices")
    if not volumes_are_valid or float(volume_data[MARKET_INDEX].iloc[-1]) <= 0:
        raise RuntimeError("Market data contains invalid volume")

    if len(price_data) < SMA_WINDOW + 5:
        raise RuntimeError(f"Insufficient history: {len(price_data)} rows (need > {SMA_WINDOW + 5})")
    return price_data, volume_data

# ==========================================
# 7. INDICATORS (point-in-time, no lookahead)
# ==========================================
def calculate_indicators(price_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.DataFrame:
    indicators = pd.DataFrame(index=price_data.index)
    index_close = price_data[MARKET_INDEX]

    indicators["sma_200"] = index_close.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
    channel_high = index_close.rolling(DONCHIAN_WINDOW, min_periods=DONCHIAN_WINDOW).max()
    channel_low = index_close.rolling(DONCHIAN_WINDOW, min_periods=DONCHIAN_WINDOW).min()
    indicators["donchian_mid"] = (channel_high + channel_low) / 2.0

    price_x_volume = index_close * volume_data[MARKET_INDEX]
    indicators["vwma_50"] = (
        price_x_volume.rolling(VWMA_WINDOW, min_periods=VWMA_WINDOW).sum()
        / volume_data[MARKET_INDEX].rolling(VWMA_WINDOW, min_periods=VWMA_WINDOW).sum()
    )

    indicators["sma_signal"] = (index_close >= indicators["sma_200"]).astype(float)
    indicators["donchian_signal"] = (index_close >= indicators["donchian_mid"]).astype(float)
    indicators["vwma_signal"] = (index_close >= indicators["vwma_50"]).astype(float)
    indicators["bullish_consensus"] = (
        (indicators["sma_signal"] + indicators["donchian_signal"] + indicators["vwma_signal"]) >= 2
    ).astype(int)

    indicators["annualized_volatility"] = (
        index_close.pct_change().rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std() * np.sqrt(252)
    )

    indicators["soxl_momentum"] = price_data[LEVERAGED_SEMICONDUCTOR].pct_change(MOMENTUM_WINDOW)
    indicators["tecl_momentum"] = price_data[LEVERAGED_TECH].pct_change(MOMENTUM_WINDOW)
    return indicators

# ==========================================
# 8. STRATEGY CORE
# ==========================================
def classify_volatility(annualized_volatility: float) -> str:
    """Map annualized volatility to a regime tier."""
    if not np.isfinite(annualized_volatility):
        raise ValueError("Volatility is unavailable")
    if annualized_volatility < LOW_VOL_THRESHOLD:
        return "LOW"
    if annualized_volatility <= MODERATE_VOL_THRESHOLD:
        return "MODERATE"
    return "HIGH"


def select_leader(soxl_momentum: float, tecl_momentum: float, existing_leader: str | None) -> str:
    """Pick the momentum leader with hysteresis to prevent whipsaws."""
    if not np.isfinite(soxl_momentum) or not np.isfinite(tecl_momentum):
        raise ValueError("Leader momentum is unavailable")
    if existing_leader == LEVERAGED_SEMICONDUCTOR:
        if (tecl_momentum - soxl_momentum) > LEADER_SWITCH_THRESHOLD:
            return LEVERAGED_TECH
        return LEVERAGED_SEMICONDUCTOR
    if existing_leader == LEVERAGED_TECH:
        if (soxl_momentum - tecl_momentum) > LEADER_SWITCH_THRESHOLD:
            return LEVERAGED_SEMICONDUCTOR
        return LEVERAGED_TECH
    return LEVERAGED_SEMICONDUCTOR if soxl_momentum >= tecl_momentum else LEVERAGED_TECH


def select_allocation_template(volatility_tier: str) -> dict[str, float]:
    if volatility_tier == "LOW":
        return LOW_VOL_ALLOCATION.copy()
    if volatility_tier == "MODERATE":
        return MODERATE_VOL_ALLOCATION.copy()
    return HIGH_VOL_ALLOCATION.copy()


def apply_allocation_template(template: dict[str, float], leader: str) -> dict[str, float]:
    """Resolve Leader/Follower slots and enforce the leveraged position cap."""
    follower = LEVERAGED_TECH if leader == LEVERAGED_SEMICONDUCTOR else LEVERAGED_SEMICONDUCTOR
    target_weights: dict[str, float] = {}

    for slot, proportion in template.items():
        if slot == "Leader":
            ticker = leader
        elif slot == "Follower":
            ticker = follower
        else:
            ticker = slot

        # Cap any single leveraged sector position; spill excess into the stabilizer
        if ticker in LEVERAGED_SECTOR_ETFS and proportion > MAX_LEVERAGED_POSITION:
            excess = proportion - MAX_LEVERAGED_POSITION
            proportion = MAX_LEVERAGED_POSITION
            target_weights[SEMICONDUCTOR_ETF] = target_weights.get(SEMICONDUCTOR_ETF, 0.0) + excess

        target_weights[ticker] = target_weights.get(ticker, 0.0) + proportion

    return target_weights


def validate_configuration() -> None:
    """Fail fast if a future configuration edit makes allocations unsafe."""
    if len(ALL_TICKERS) != len(set(ALL_TICKERS)):
        raise RuntimeError("Ticker universe contains conflicting entries")
    allowed_slots = (set(ALL_TICKERS) - {MARKET_INDEX}) | {"Leader", "Follower"}
    for name, template in {
        "LOW_VOL_ALLOCATION": LOW_VOL_ALLOCATION,
        "MODERATE_VOL_ALLOCATION": MODERATE_VOL_ALLOCATION,
        "HIGH_VOL_ALLOCATION": HIGH_VOL_ALLOCATION,
        "BEAR_ALLOCATION": BEAR_ALLOCATION,
    }.items():
        if not np.isclose(sum(template.values()), 1.0, atol=1e-12):
            raise RuntimeError(f"{name} must sum to 1.0")
        if set(template) - allowed_slots or MARKET_INDEX in template:
            raise RuntimeError(f"{name} contains an invalid traded ticker")


def determine_target_allocation(latest_indicators: pd.Series, existing_leader: str | None,
                                allow_leader_review: bool) -> StrategyResult:
    """Heart of the strategy: regime, leader, volatility tier, and target weights."""
    is_bull = int(latest_indicators["bullish_consensus"]) == 1
    annualized_volatility = float(latest_indicators["annualized_volatility"])
    leader = existing_leader or LEVERAGED_SEMICONDUCTOR
    if allow_leader_review:
        leader = select_leader(
            float(latest_indicators["soxl_momentum"]),
            float(latest_indicators["tecl_momentum"]),
            existing_leader,
        )

    if not is_bull:
        return StrategyResult(
            target_weights=dict(BEAR_ALLOCATION),
            regime="BEAR",
            leader=leader,
            volatility_tier="N/A",
            annualized_volatility=annualized_volatility,
        )

    volatility_tier = classify_volatility(latest_indicators["annualized_volatility"])
    template = select_allocation_template(volatility_tier)
    target_weights = apply_allocation_template(template, leader)

    total = sum(target_weights.values())
    if not np.isclose(total, 1.0, atol=1e-12):
        raise RuntimeError(f"Target allocation must sum to 1.0 before normalization; got {total}")
    if any(target_weights.get(ticker, 0.0) > MAX_LEVERAGED_POSITION
           for ticker in LEVERAGED_SECTOR_ETFS):
        raise RuntimeError("Target allocation exceeds the leveraged-position cap")

    return StrategyResult(
        target_weights=target_weights,
        regime="BULL",
        leader=leader,
        volatility_tier=volatility_tier,
        annualized_volatility=annualized_volatility,
    )

# ==========================================
# 9. PORTFOLIO MATH
# ==========================================
def validate_latest_indicators(latest_indicators: pd.Series) -> None:
    """Prevent incomplete data from becoming a plausible allocation signal."""
    required_indicators = [
        "sma_200",
        "donchian_mid",
        "vwma_50",
        "annualized_volatility",
        "soxl_momentum",
        "tecl_momentum",
    ]
    invalid = []
    for name in required_indicators:
        try:
            valid = np.isfinite(float(latest_indicators[name]))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            invalid.append(name)
    if invalid:
        raise RuntimeError(f"Invalid latest indicators: {invalid}")


def validate_holdings_against_prices(state: PortfolioState, price_data: pd.DataFrame) -> None:
    """Ensure every recorded holding has a real, current price."""
    unknown_holdings = set(state.shares) - set(price_data.columns)
    if unknown_holdings:
        raise RuntimeError(
            "State contains holdings with no current price data: "
            f"{sorted(unknown_holdings)}"
        )
    if MARKET_INDEX in state.shares:
        raise RuntimeError(f"{MARKET_INDEX} is a signal asset and cannot be a holding")


def calculate_target_portfolio(price_data: pd.DataFrame, target_weights: dict[str, float],
                               portfolio_value: float) -> pd.DataFrame:
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise RuntimeError(f"Invalid portfolio value: {portfolio_value!r}")
    rows = []
    for ticker, proportion in sorted(target_weights.items(), key=lambda item: -item[1]):
        if proportion < 0.001:
            continue
        price = float(price_data[ticker].iloc[-1])
        if not np.isfinite(price) or price <= 0:
            raise RuntimeError(f"Invalid latest price for {ticker}: {price!r}")
        value = proportion * portfolio_value
        rows.append({
            "Ticker": ticker,
            "Price": price,
            "TargetPct": proportion,
            "TargetValue": value,
            "TargetShares": round(value / price, 4),
        })
    return pd.DataFrame(rows)


def existing_portfolio_value(state: PortfolioState, price_data: pd.DataFrame) -> float:
    validate_holdings_against_prices(state, price_data)
    total = 0.0
    for ticker, share_count in state.shares.items():
        price = float(price_data[ticker].iloc[-1])
        if not np.isfinite(price) or price <= 0:
            raise RuntimeError(f"Invalid latest price for held {ticker}: {price!r}")
        total += share_count * price
    if not np.isfinite(total):
        raise RuntimeError(f"Invalid marked-to-market portfolio value: {total!r}")
    return total


def existing_weights(state: PortfolioState, price_data: pd.DataFrame) -> dict[str, float]:
    total = existing_portfolio_value(state, price_data)
    if total <= 0:
        return {}
    return {
        ticker: share_count * float(price_data[ticker].iloc[-1]) / total
        for ticker, share_count in state.shares.items()
        if share_count > 0
    }


def should_rebalance(existing: dict[str, float], target: dict[str, float],
                     band: float = REBALANCE_BAND) -> bool:
    if not existing:
        return True
    tickers = set(existing) | set(target)
    return any(abs(target.get(t, 0.0) - existing.get(t, 0.0)) > band for t in tickers)


def sector_review_due(last_sector_rebalance: str, signal_date: pd.Timestamp,
                      trading_dates: pd.DatetimeIndex) -> bool:
    if not last_sector_rebalance:
        return True
    try:
        last_review = pd.Timestamp(last_sector_rebalance)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Portfolio state has an invalid sector-review date") from exc
    if last_review.tzinfo is not None:
        last_review = last_review.tz_localize(None)
    completed_sessions = trading_dates[
        (trading_dates > last_review.normalize()) & (trading_dates <= signal_date)
    ]
    return len(completed_sessions) >= SECTOR_REBALANCE_DAYS

# ==========================================
# 10. DASHBOARD (composed from sections)
# ==========================================
def build_header(date_str: str, result: StrategyResult, latest_indicators: pd.Series,
                 portfolio_value: float, rebalance_due: bool, sector_due: bool) -> str:
    border = "=" * 92
    legs = (f"SMA:{int(latest_indicators['sma_signal'])} "
            f"Donchian:{int(latest_indicators['donchian_signal'])} "
            f"VWMA:{int(latest_indicators['vwma_signal'])}")
    follower = LEVERAGED_TECH if result.leader == LEVERAGED_SEMICONDUCTOR else LEVERAGED_SEMICONDUCTOR
    return "\n".join([
        border,
        "  ROTH IRA - BARBELL MOMENTUM ENGINE",
        border,
        f"  Signal Date: {date_str}   Portfolio Value: ${portfolio_value:,.2f}",
        f"  Regime: {result.regime}   Consensus: {legs}",
        f"  {MARKET_INDEX} Vol: {result.annualized_volatility*100:.1f}%   Vol Tier: {result.volatility_tier}",
        f"  Momentum Leader: {result.leader}   Follower: {follower}",
        f"  Rebalance Due: {'YES' if rebalance_due else 'NO'}   Sector Review: {'YES' if sector_due else 'NO'}",
    ])


def build_allocation_table(target_portfolio: pd.DataFrame) -> str:
    divider = "-" * 92
    lines = [
        divider,
        "  TARGET ALLOCATION",
        divider,
        f"  {'Ticker':<8}{'Price':>10}{'Target %':>11}{'Target $':>14}{'Shares':>14}",
        divider,
    ]
    for _, row in target_portfolio.iterrows():
        lines.append(
            f"  {row['Ticker']:<8}${row['Price']:>9.2f}{row['TargetPct']*100:>10.1f}%"
            f"${row['TargetValue']:>13,.2f}{row['TargetShares']:>14.4f}"
        )
    return "\n".join(lines)


def build_rules_section(result: StrategyResult) -> str:
    divider = "-" * 92
    if result.regime == "BULL":
        structure = {
            "LOW": "  Low Vol: 45% Leader / 15% Follower / 25% SMH / 15% QLD",
            "MODERATE": "  Moderate Vol: 25% Leader / 10% Follower / 45% SMH / 20% QLD",
            "HIGH": "  High Vol: 85% SMH / 15% GLD (de-leveraged)",
        }[result.volatility_tier]
    else:
        structure = "  Bear Regime: 80% SPMO / 20% GLD"
    return "\n".join([
        divider,
        "  RULES & RISK CONTROLS",
        divider,
        structure,
        divider,
        f"  - Max single leveraged position : {MAX_LEVERAGED_POSITION:.0%}",
        f"  - Rebalance band                : {REBALANCE_BAND:.0%} (trade only on >5% drift)",
        f"  - Leader switch threshold       : {LEADER_SWITCH_THRESHOLD:.0%} momentum divergence",
        f"  - Sector review cadence         : ~{SECTOR_REBALANCE_DAYS} trading sessions",
        "=" * 92,
    ])


def build_dashboard(date_str: str, result: StrategyResult, latest_indicators: pd.Series,
                    portfolio_value: float, target_portfolio: pd.DataFrame,
                    rebalance_due: bool, sector_due: bool) -> str:
    return "\n".join([
        build_header(date_str, result, latest_indicators, portfolio_value, rebalance_due, sector_due),
        build_allocation_table(target_portfolio),
        build_rules_section(result),
    ])

# ==========================================
# 11. EMAIL (composed from sections)
# ==========================================
def build_email_header(date_str: str, result: StrategyResult,
                       portfolio_value: float, rebalance_due: bool) -> str:
    color = "#27ae60" if rebalance_due else "#7f8c8d"
    status = "ACTION REQUIRED: Execute Rebalance" if rebalance_due else "Hold Current Allocation"
    return f"""
        <div style="background:#1a252f;color:#fff;padding:24px;text-align:center;">
          <h2 style="margin:0;">ROTH IRA - BARBELL ENGINE</h2>
          <p style="margin:6px 0 0;color:#bdc3c7;">Signal date: {date_str}</p>
        </div>
        <div style="padding:20px;background:#f1f4f8;border-bottom:1px solid #e9ecef;text-align:center;">
          <div style="font-size:15px;font-weight:bold;color:{color};">{status}</div>
          <div style="font-size:12px;color:#7f8c8d;margin-top:4px;">
            Regime: {result.regime} | Vol Tier: {result.volatility_tier} | Leader: {result.leader}
            | Portfolio: ${portfolio_value:,.2f}</div>
        </div>"""


def build_email_table(target_portfolio: pd.DataFrame) -> str:
    rows_html = ""
    for _, row in target_portfolio.iterrows():
        rows_html += f"""
        <tr style="border-bottom:1px solid #e9ecef;">
          <td style="padding:10px;font-weight:bold;">{row['Ticker']}</td>
          <td style="padding:10px;">${row['Price']:,.2f}</td>
          <td style="padding:10px;font-weight:bold;color:#0056b3;">{row['TargetPct']*100:.1f}%</td>
          <td style="padding:10px;font-weight:bold;">${row['TargetValue']:,.2f}</td>
          <td style="padding:10px;font-weight:bold;color:#27ae60;">{row['TargetShares']:,.4f}</td>
        </tr>"""
    return f"""
          <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead><tr style="background:#f8f9fa;border-bottom:2px solid #e9ecef;">
              <th style="padding:8px;text-align:left;">Ticker</th>
              <th style="padding:8px;text-align:left;">Price</th>
              <th style="padding:8px;text-align:left;">%</th>
              <th style="padding:8px;text-align:left;">Value</th>
              <th style="padding:8px;text-align:left;">Shares</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>"""


def build_email_html(date_str: str, result: StrategyResult, portfolio_value: float,
                     target_portfolio: pd.DataFrame, rebalance_due: bool) -> str:
    return f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"></head>
    <body style="font-family:-apple-system,sans-serif;background:#f8f9fa;padding:20px;color:#333;">
      <div style="max-width:700px;background:#fff;margin:0 auto;border-radius:8px;
                  box-shadow:0 4px 12px rgba(0,0,0,.08);overflow:hidden;">
        {build_email_header(date_str, result, portfolio_value, rebalance_due)}
        <div style="padding:20px;">
        {build_email_table(target_portfolio)}
        </div>
      </div>
    </body></html>"""


def send_email(subject: str, text_body: str, html_body: str) -> None:
    address = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("RECEIVER_EMAIL")
    if not all([address, password, recipient]):
        logger.info("Email env vars not set; skipping.")
        return
    try:
        message = EmailMessage()
        message["Subject"], message["From"], message["To"] = subject, address, recipient
        message.set_content(text_body)
        message.add_alternative(html_body, subtype="html")
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(address, password)
            server.send_message(message)
        logger.info("Email sent.")
    except Exception:
        logger.exception("Email delivery failed")
        raise

# ==========================================
# 12. ORCHESTRATION
# ==========================================
def resolve_portfolio_value(roth_amount: float | None, state: PortfolioState,
                            price_data: pd.DataFrame) -> float:
    """Use an explicit amount, actual holdings, or a configured first-run amount."""
    if roth_amount is not None:
        portfolio_value = roth_amount
    elif state.shares:
        portfolio_value = existing_portfolio_value(state, price_data)
    elif ROTH_IRA_AMOUNT is not None:
        portfolio_value = ROTH_IRA_AMOUNT
    else:
        raise RuntimeError(
            "No portfolio value is available; supply --roth-amount, ROTH_IRA_AMOUNT, "
            "or confirmed holdings"
        )
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise RuntimeError(f"Invalid portfolio value: {portfolio_value!r}")
    return float(portfolio_value)


def run_strategy(roth_amount: float | None) -> StrategyRun:
    """Download data, compute the point-in-time signal, and decide whether to trade."""
    price_data, volume_data = download_market_data(ALL_TICKERS)
    indicators = calculate_indicators(price_data, volume_data)
    latest_indicators = indicators.iloc[-1]
    validate_latest_indicators(latest_indicators)
    state = load_state()
    signal_date = price_data.index[-1]
    sector_due = sector_review_due(state.last_sector_rebalance, signal_date, price_data.index)
    result = determine_target_allocation(latest_indicators, state.leader, sector_due)
    portfolio_value = resolve_portfolio_value(roth_amount, state, price_data)
    target_portfolio = calculate_target_portfolio(price_data, result.target_weights, portfolio_value)
    rebalance_due = should_rebalance(existing_weights(state, price_data), result.target_weights)
    return StrategyRun(
        price_data=price_data,
        latest_indicators=latest_indicators,
        result=result,
        state=state,
        portfolio_value=portfolio_value,
        target_portfolio=target_portfolio,
        signal_date=signal_date,
        rebalance_due=rebalance_due,
        sector_review_due=sector_due,
    )


def persist_state(state: PortfolioState, result: StrategyResult, portfolio_value: float,
                  signal_date: pd.Timestamp, sector_due: bool,
                  executed_shares: dict[str, float] | None,
                  price_data: pd.DataFrame) -> None:
    """Record reviews and confirmed holdings; recommendations never imply execution."""
    if sector_due:
        state.last_sector_rebalance = signal_date.date().isoformat()
        state.leader = result.leader
    if executed_shares is not None:
        state.shares = executed_shares
        executed_value = existing_portfolio_value(state, price_data)
        if executed_value <= 0:
            raise RuntimeError("Confirmed holdings have no positive marked-to-market value")
        state.target_weights = existing_weights(state, price_data)
        state.portfolio_value = round(executed_value, 2)
    else:
        state.portfolio_value = round(portfolio_value, 2)
    state.volatility_tier = result.volatility_tier
    state.regime = result.regime
    save_state(state)


def log_decision(strategy_run: StrategyRun, execution_confirmed: bool) -> None:
    """Write enough context to reconstruct a recommendation later."""
    current_weights = existing_weights(strategy_run.state, strategy_run.price_data)
    tickers = set(current_weights) | set(strategy_run.result.target_weights)
    drifts = {
        ticker: strategy_run.result.target_weights.get(ticker, 0.0)
        - current_weights.get(ticker, 0.0)
        for ticker in sorted(tickers)
    }
    logger.info(
        "decision signal_date=%s regime=%s tier=%s leader=%s volatility=%.6f "
        "soxl_momentum=%.6f tecl_momentum=%.6f sector_due=%s rebalance_due=%s "
        "execution_confirmed=%s current=%s target=%s drift=%s",
        strategy_run.signal_date.date(),
        strategy_run.result.regime,
        strategy_run.result.volatility_tier,
        strategy_run.result.leader,
        strategy_run.result.annualized_volatility,
        strategy_run.latest_indicators["soxl_momentum"],
        strategy_run.latest_indicators["tecl_momentum"],
        strategy_run.sector_review_due,
        strategy_run.rebalance_due,
        execution_confirmed,
        current_weights,
        strategy_run.result.target_weights,
        drifts,
    )


def parse_executed_shares(entries: list[str] | None) -> dict[str, float] | None:
    """Parse explicit post-trade holdings supplied as TICKER=SHARES values."""
    if entries is None:
        return None
    holdings: dict[str, float] = {}
    for entry in entries:
        ticker, separator, share_text = entry.partition("=")
        ticker = ticker.upper()
        if separator != "=" or ticker not in TRADED_TICKERS:
            raise ValueError(f"Invalid executed holding: {entry!r}")
        try:
            shares = float(share_text)
        except ValueError as exc:
            raise ValueError(f"Invalid share count for {ticker}: {share_text!r}") from exc
        if not np.isfinite(shares) or shares < 0 or ticker in holdings:
            raise ValueError(f"Invalid executed holding: {entry!r}")
        holdings[ticker] = shares
    return holdings


def main() -> None:
    validate_configuration()
    parser = argparse.ArgumentParser(description="ROTH IRA - Barbell Momentum Allocation Engine")
    parser.add_argument("--test", action="store_true", help="Print dashboard without saving state or sending email")
    parser.add_argument("--roth-amount", type=float, default=None, help="Override Roth IRA balance")
    parser.add_argument(
        "--executed-shares", nargs="+", metavar="TICKER=SHARES",
        help="Record confirmed post-trade holdings; e.g. SOXL=1.25 SMH=3.0",
    )
    args = parser.parse_args()
    if args.roth_amount is not None and (
            not np.isfinite(args.roth_amount) or args.roth_amount <= 0):
        parser.error("--roth-amount must be positive and finite")
    executed_shares = parse_executed_shares(args.executed_shares)

    strategy_run = run_strategy(args.roth_amount)
    log_decision(strategy_run, execution_confirmed=executed_shares is not None)

    signal_date_str = strategy_run.signal_date.strftime("%Y-%m-%d")
    dashboard = build_dashboard(
        signal_date_str,
        strategy_run.result,
        strategy_run.latest_indicators,
        strategy_run.portfolio_value,
        strategy_run.target_portfolio,
        strategy_run.rebalance_due,
        strategy_run.sector_review_due,
    )
    print(dashboard)

    if not args.test:
        send_email(
            f"ROTH IRA Report - {signal_date_str}",
            dashboard,
            build_email_html(
                signal_date_str,
                strategy_run.result,
                strategy_run.portfolio_value,
                strategy_run.target_portfolio,
                strategy_run.rebalance_due,
            ),
        )
        persist_state(
            strategy_run.state,
            strategy_run.result,
            strategy_run.portfolio_value,
            strategy_run.signal_date,
            strategy_run.sector_review_due,
            executed_shares,
            strategy_run.price_data,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("port12_cloud.py failed")
        sys.exit(1)

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
  ROTH_IRA_AMOUNT      Current Roth IRA balance (default 1025.97)
  GMAIL_ADDRESS        Sender Gmail address  (optional, for email reports)
  GMAIL_APP_PASSWORD   Gmail app password    (optional)
  RECEIVER_EMAIL       Report recipient      (optional)

CLI usage:
  python3 roth_ira.py                 # run, print dashboard, save state, send email
  python3 roth_ira.py --test          # run, print dashboard, save state, NO email
  python3 roth_ira.py --roth-amount 5000.00
"""

import argparse
import json
import logging
import os
import smtplib
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

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
ROTH_IRA_AMOUNT = float(os.environ.get("ROTH_IRA_AMOUNT", 1025.97))
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

# ==========================================
# 4. LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("roth_ira")

# ==========================================
# 5. STATE PERSISTENCE
# ==========================================
def load_state() -> PortfolioState:
    if not STATE_FILE.exists():
        return PortfolioState()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as state_file:
            payload = json.load(state_file)
    except Exception as exc:
        raise RuntimeError(f"Could not read {STATE_FILE}; refusing to infer holdings") from exc
    if payload.get("state_version") != STATE_VERSION:
        raise RuntimeError(
            f"Unsupported portfolio state version: {payload.get('state_version')!r}; "
            "migrate it explicitly"
        )
    try:
        state = PortfolioState(**payload)
    except TypeError as exc:
        raise RuntimeError(f"Unsupported fields in {STATE_FILE}; migrate it explicitly") from exc

    unknown_tickers = (set(state.shares) | set(state.target_weights)) - set(ALL_TICKERS)
    if unknown_tickers:
        raise RuntimeError(
            "Unsupported portfolio state; migrate it explicitly "
            f"(unknown_tickers={sorted(unknown_tickers)})"
        )
    if any(not isinstance(shares, (int, float)) or shares < 0 for shares in state.shares.values()):
        raise RuntimeError("Portfolio state contains invalid share counts")
    return state


def save_state(state: PortfolioState) -> None:
    try:
        state.last_updated = datetime.now().isoformat()
        with open(STATE_FILE, "w", encoding="utf-8") as state_file:
            json.dump(asdict(state), state_file, indent=4)
        logger.info(
            f"State saved: regime={state.regime}, leader={state.leader}, "
            f"tier={state.volatility_tier}, value=${state.portfolio_value:,.2f}"
        )
    except Exception as e:
        logger.error(f"Error saving state: {e}")

# ==========================================
# 6. DATA ACQUISITION (real daily data only)
# ==========================================
def download_market_data(tickers: list[str], days: int = HISTORY_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
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
    if price_data.isna().any().any() or volume_data.isna().any().any():
        raise RuntimeError("Incomplete market data; refusing to forward-fill prices or volumes")
    if not price_data.index.equals(volume_data.index):
        raise RuntimeError("Price and volume dates differ")

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
    if pd.isna(annualized_volatility):
        return "MODERATE"
    if annualized_volatility < LOW_VOL_THRESHOLD:
        return "LOW"
    if annualized_volatility < MODERATE_VOL_THRESHOLD:
        return "MODERATE"
    return "HIGH"


def select_leader(soxl_momentum: float, tecl_momentum: float, existing_leader: str | None) -> str:
    """Pick the momentum leader with hysteresis to prevent whipsaws."""
    if pd.isna(soxl_momentum) or pd.isna(tecl_momentum):
        return existing_leader or LEVERAGED_SEMICONDUCTOR
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
        if not np.isclose(sum(template.values()), 1.0):
            raise RuntimeError(f"{name} must sum to 1.0")
        if set(template) - allowed_slots or MARKET_INDEX in template:
            raise RuntimeError(f"{name} contains an invalid traded ticker")


def determine_target_allocation(latest_indicators: pd.Series, existing_leader: str | None) -> StrategyResult:
    """Heart of the strategy: regime, leader, volatility tier, and target weights."""
    required_indicators = [
        "bullish_consensus", "annualized_volatility", "soxl_momentum", "tecl_momentum",
    ]
    if latest_indicators[required_indicators].isna().any():
        raise RuntimeError("Latest indicator set is incomplete; no recommendation generated")
    is_bull = int(latest_indicators["bullish_consensus"]) == 1
    annualized_volatility = (
        float(latest_indicators["annualized_volatility"])
        if not pd.isna(latest_indicators["annualized_volatility"])
        else MODERATE_VOL_THRESHOLD
    )

    if not is_bull:
        return StrategyResult(
            target_weights=dict(BEAR_ALLOCATION),
            regime="BEAR",
            leader=existing_leader or LEVERAGED_SEMICONDUCTOR,
            volatility_tier="N/A",
            annualized_volatility=annualized_volatility,
        )

    volatility_tier = classify_volatility(latest_indicators["annualized_volatility"])
    leader = select_leader(
        float(latest_indicators.get("soxl_momentum", np.nan)),
        float(latest_indicators.get("tecl_momentum", np.nan)),
        existing_leader,
    )
    template = select_allocation_template(volatility_tier)
    target_weights = apply_allocation_template(template, leader)

    total = sum(target_weights.values())
    if total > 0:
        target_weights = {ticker: weight / total for ticker, weight in target_weights.items()}
    if not np.isclose(sum(target_weights.values()), 1.0):
        raise RuntimeError("Target allocation does not sum to 1.0")
    if any(target_weights.get(ticker, 0.0) > MAX_LEVERAGED_POSITION
           for ticker in LEVERAGED_SECTOR_ETFS):
        raise RuntimeError("Normalization exceeded the leveraged-position cap")

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
def calculate_target_portfolio(price_data: pd.DataFrame, target_weights: dict[str, float],
                               portfolio_value: float) -> pd.DataFrame:
    rows = []
    for ticker, proportion in sorted(target_weights.items(), key=lambda item: -item[1]):
        if proportion < 0.001:
            continue
        price = float(price_data[ticker].iloc[-1])
        value = proportion * portfolio_value
        rows.append({
            "Ticker": ticker,
            "Price": price,
            "TargetPct": proportion,
            "TargetValue": value,
            "TargetShares": round(value / price, 4) if price > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def existing_portfolio_value(state: PortfolioState, price_data: pd.DataFrame) -> float:
    total = 0.0
    for ticker, share_count in state.shares.items():
        if ticker in price_data.columns:
            total += share_count * float(price_data[ticker].iloc[-1])
    return total


def existing_weights(state: PortfolioState, price_data: pd.DataFrame) -> dict[str, float]:
    total = existing_portfolio_value(state, price_data)
    if total <= 0:
        return {}
    return {
        ticker: share_count * float(price_data[ticker].iloc[-1]) / total
        for ticker, share_count in state.shares.items()
        if ticker in price_data.columns and share_count > 0
    }


def should_rebalance(existing: dict[str, float], target: dict[str, float],
                     band: float = REBALANCE_BAND) -> bool:
    if not existing:
        return True
    tickers = set(existing) | set(target)
    return any(abs(target.get(t, 0.0) - existing.get(t, 0.0)) > band for t in tickers)


def sector_review_due(last_sector_rebalance: str, trading_dates: pd.DatetimeIndex) -> bool:
    if not last_sector_rebalance:
        return True
    try:
        last_review = pd.Timestamp(last_sector_rebalance)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Portfolio state has an invalid sector-review date") from exc
    if last_review.tzinfo is not None:
        last_review = last_review.tz_localize(None)
    return int((trading_dates > last_review).sum()) >= SECTOR_REBALANCE_DAYS

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
        f"  Date: {date_str}   Portfolio Value: ${portfolio_value:,.2f}",
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
        f"  - Sector review cadence         : ~{SECTOR_REBALANCE_DAYS} days",
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
          <p style="margin:6px 0 0;color:#bdc3c7;">{date_str}</p>
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
    except Exception as e:
        logger.error(f"Email error: {e}")

# ==========================================
# 12. ORCHESTRATION
# ==========================================
def run_strategy(roth_amount: float | None):
    """Download data, compute indicators, determine allocation, decide trades."""
    price_data, volume_data = download_market_data(ALL_TICKERS)
    indicators = calculate_indicators(price_data, volume_data)
    latest_indicators = indicators.iloc[-1]

    if pd.isna(latest_indicators["sma_200"]):
        raise RuntimeError("Latest bar has no 200-day SMA - insufficient history.")

    state = load_state()
    result = determine_target_allocation(latest_indicators, state.leader)

    portfolio_value = roth_amount or state.portfolio_value or ROTH_IRA_AMOUNT
    target_portfolio = calculate_target_portfolio(price_data, result.target_weights, portfolio_value)

    trade_date = price_data.index[-1]
    rebalance_due = should_rebalance(existing_weights(state, price_data), result.target_weights)
    sector_due = sector_review_due(state.last_sector_rebalance, price_data.index)

    return price_data, latest_indicators, result, state, portfolio_value, target_portfolio, \
        trade_date, rebalance_due, sector_due


def persist_state(state: PortfolioState, result: StrategyResult, portfolio_value: float,
                  trade_date: datetime, executed_shares: dict[str, float] | None) -> None:
    """Record only confirmed holdings; recommendations never imply execution."""
    if executed_shares is not None:
        state.shares = executed_shares
        state.target_weights = result.target_weights
        state.last_sector_rebalance = trade_date.isoformat()
    state.portfolio_value = round(portfolio_value, 2)
    state.leader = result.leader
    state.volatility_tier = result.volatility_tier
    state.regime = result.regime
    save_state(state)


def parse_executed_shares(entries: list[str] | None) -> dict[str, float] | None:
    """Parse explicit post-trade holdings supplied as TICKER=SHARES values."""
    if entries is None:
        return None
    holdings: dict[str, float] = {}
    for entry in entries:
        ticker, separator, share_text = entry.partition("=")
        ticker = ticker.upper()
        if separator != "=" or ticker not in set(ALL_TICKERS) - {MARKET_INDEX}:
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
    parser.add_argument("--test", action="store_true", help="Print dashboard and save state, skip email")
    parser.add_argument("--roth-amount", type=float, default=None, help="Override Roth IRA balance")
    parser.add_argument(
        "--executed-shares", nargs="+", metavar="TICKER=SHARES",
        help="Record confirmed post-trade holdings; e.g. SOXL=1.25 SMH=3.0",
    )
    args = parser.parse_args()
    if args.roth_amount is not None and args.roth_amount <= 0:
        parser.error("--roth-amount must be positive")
    executed_shares = parse_executed_shares(args.executed_shares)

    (price_data, latest_indicators, result, state, portfolio_value,
     target_portfolio, trade_date, rebalance_due, sector_due) = run_strategy(args.roth_amount)

    date_str = trade_date.strftime("%Y-%m-%d")
    dashboard = build_dashboard(date_str, result, latest_indicators, portfolio_value,
                                target_portfolio, rebalance_due, sector_due)
    print(dashboard)

    persist_state(state, result, portfolio_value, trade_date, executed_shares)

    if not args.test:
        send_email(
            f"ROTH IRA Report - {date_str}",
            dashboard,
            build_email_html(date_str, result, portfolio_value, target_portfolio, rebalance_due),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("roth_ira.py failed")
        sys.exit(1)

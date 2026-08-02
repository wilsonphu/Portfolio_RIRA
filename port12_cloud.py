import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ====================================================================
# 1. CLOUD CONFIGURATION & SECRETS
# ====================================================================

SENDER_EMAIL = os.environ.get("GMAIL_ADDRESS")
SENDER_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

# Portfolio Allocations
BULL_ALLOCATION: Dict[str, float] = {"TQQQ": 30.0, "SMH": 60.0}
BEAR_ALLOCATION: Dict[str, float] = {"SPMO": 100.0}

CRASH_ALERT_THRESHOLD = -5.0

# ====================================================================
# 2. EMAIL PROTOCOL
# ====================================================================


def send_institutional_alert(subject: str, body: str) -> None:
    """Dispatches automated summary reports via Gmail SMTP SSL."""
    if not SENDER_EMAIL or not SENDER_APP_PASSWORD or not RECEIVER_EMAIL:
        print("\n[✘] Error: Email credentials not found in environment variables.")
        return

    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = f"Port12 Quant System <{SENDER_EMAIL}>"
        msg["To"] = RECEIVER_EMAIL

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            server.send_message(msg)

        print("\n[✔] Automated Summary Report successfully dispatched via Email.")
    except Exception as e:
        print(f"\n[✘] Failed to dispatch Email. System Error: {e}")


# ====================================================================
# 3. QUANTITATIVE HELPERS
# ====================================================================


def calculate_regimes(
    close: pd.Series, ema200: pd.Series
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Calculates raw trend signals (with ±2% hysteresis bands) and applies
    the 5-day confirmation rule for regime switches.
    """
    upper_band = ema200 * 1.04
    lower_band = ema200 * 0.96

    n = len(close)
    raw_signals = np.ones(n, dtype=int)
    current_raw = 1

    # 1. Generate hysteresis signals
    for i in range(200, n):
        price = close.iloc[i]
        if price > upper_band.iloc[i]:
            current_raw = 1
        elif price < lower_band.iloc[i]:
            current_raw = 0
        raw_signals[i] = current_raw

    # 2. Apply 5-day confirmation rule
    confirmed_regimes = np.ones(n, dtype=int)
    current_regime = 1
    days_in_new_state = 0

    for i in range(1, n):
        prev_raw = raw_signals[i - 1]
        curr_raw = raw_signals[i]

        if curr_raw != current_regime:
            if curr_raw == prev_raw:
                days_in_new_state += 1
            else:
                days_in_new_state = 1

            if days_in_new_state >= 5:
                current_regime = curr_raw
                days_in_new_state = 0
        else:
            days_in_new_state = 0

        confirmed_regimes[i] = current_regime

    return raw_signals, confirmed_regimes, days_in_new_state


# ====================================================================
# 4. STATELESS QUANTITATIVE ENGINE (5-DAY RULE)
# ====================================================================


def run_portfolio() -> None:
    print("[*] Initializing Port12 Cloud Engine (5-Day Rule)...")

    data = yf.download("QQQ", period="5y", auto_adjust=True, progress=False)
    if data.empty:
        print("[✘] Error: Failed to fetch market data from Yahoo Finance.")
        return

    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=0).ffill().dropna()
    else:
        close = data["Close"].ffill().dropna()

    # If yfinance returns a DataFrame with 1 column, convert to Series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    ema200 = close.ewm(span=200, adjust=False).mean()

    raw_signals, confirmed_regimes, days_in_new_state = calculate_regimes(close, ema200)

    yesterday_regime = confirmed_regimes[-2]
    current_regime = confirmed_regimes[-1]

    latest_date = close.index[-1].strftime("%Y-%m-%d")
    latest_price = float(close.iloc[-1])
    yesterday_price = float(close.iloc[-2])
    latest_ema = float(ema200.iloc[-1])

    daily_pct_change = ((latest_price / yesterday_price) - 1.0) * 100.0
    distance_to_ema = ((latest_price / latest_ema) - 1.0) * 100.0

    bull_assets = ", ".join(BULL_ALLOCATION.keys())
    bear_assets = ", ".join(BEAR_ALLOCATION.keys())

    if current_regime == 1:
        regime_title = "BULL MARKET (Risk-On)"
        leverage_ratio = "1.8x Momentum"
        target_dict = BULL_ALLOCATION
        bull_alloc_str = ", ".join(
            [f"{val}% {key}" for key, val in BULL_ALLOCATION.items()]
        )
        instructions = (
            f" 1. LIQUIDATE all defensive assets ({bear_assets}) to 0%.\n"
            f" 2. ALLOCATE precisely: {bull_alloc_str}."
        )
    else:
        regime_title = "BEAR MARKET (Risk-Off)"
        leverage_ratio = "Defensive / Uncorrelated"
        target_dict = BEAR_ALLOCATION
        bear_alloc_str = ", ".join(
            [f"{val}% {key}" for key, val in BEAR_ALLOCATION.items()]
        )
        instructions = (
            f" 1. LIQUIDATE all Nasdaq leverage ({bull_assets}) to 0%.\n"
            f" 2. ALLOCATE precisely: {bear_alloc_str}."
        )

    is_regime_flip = current_regime != yesterday_regime
    is_new_month = close.index[-1].month != close.index[-2].month
    is_crash_event = daily_pct_change <= CRASH_ALERT_THRESHOLD

    event_flags: List[str] = []
    if is_regime_flip:
        event_flags.append("[!] MACRO TREND SHIFT DETECTED")
    if is_new_month:
        event_flags.append("[!] MONTHLY REBALANCE REQUIRED")
    if is_crash_event:
        event_flags.append("[!] HIGH VOLATILITY EVENT LOGGED")

    if days_in_new_state > 0:
        event_flags.append(
            f"[*] WHIPSAW FILTER: Raw trend broke {days_in_new_state} day(s) ago. Awaiting 5-day confirmation."
        )

    report_lines = [
        "=========================================================",
        "PORTFOLIO 12: CLOUD SYSTEM ALLOCATION REPORT",
        "=========================================================",
        f"Date Generated: {latest_date}",
        f"Run Time (UTC): {datetime.utcnow().strftime('%H:%M:%S')}",
        "",
        "EXECUTIVE SUMMARY",
        "---------------------------------------------------------",
    ]

    if event_flags:
        for flag in event_flags:
            report_lines.append(f" {flag}")
    else:
        report_lines.append(
            " [✔] Normal Operations. No immediate action required."
        )

    report_lines.extend(
        [
            "",
            "MARKET DATA (Nasdaq-100 / QQQ)",
            "---------------------------------------------------------",
            f"Closing Price:    ${latest_price:.2f} ({daily_pct_change:+.2f}%)",
            f"200-Day EMA:      ${latest_ema:.2f}",
            f"Distance to Trend: {distance_to_ema:+.2f}%",
            "",
            "PORTFOLIO POSTURE",
            "---------------------------------------------------------",
            f"Current Regime:   {regime_title}",
            f"Exposure Profile: {leverage_ratio}",
            "",
            "TARGET ALLOCATIONS",
        ]
    )

    for asset, weight in target_dict.items():
        report_lines.append(f" - {asset:<6} : {weight:>5.1f}%")

    if event_flags:
        report_lines.extend(
            [
                "",
                "=========================================================",
                " TRADE EXECUTION INSTRUCTIONS",
                "=========================================================",
                " Execute at next market open:",
                instructions,
                "=========================================================",
            ]
        )
    else:
        report_lines.extend(
            [
                "",
                "=========================================================",
                " INSTRUCTIONS: Maintain current allocations.",
                "=========================================================",
            ]
        )

    report = "\n".join(report_lines)
    print(report)

    if is_regime_flip or is_new_month or is_crash_event:
        subject_line = (
            "PORTFOLIO 12: Action Required"
            if (is_regime_flip or is_new_month)
            else "PORTFOLIO 12: Volatility Alert"
        )
        send_institutional_alert(subject_line, report)


if __name__ == "__main__":
    run_portfolio()

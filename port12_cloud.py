import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

# ==================================================================== #
# 1. CLOUD CONFIGURATION & SECRETS                                     #
# ==================================================================== #
SENDER_EMAIL = os.environ.get("GMAIL_ADDRESS")
SENDER_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

BULL_ALLOCATION: Dict[str, float] = {"TECL": 20.0, "SOXL": 20.0, "SMH": 60.0}
BEAR_ALLOCATION: Dict[str, float] = {"SPMO": 100.0}
CRASH_ALERT_THRESHOLD = -5.0

# ==================================================================== #
# 2. EMAIL PROTOCOL                                                   #
# ==================================================================== #
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

# ==================================================================== #
# 3. VECTORIZED QUANTITATIVE ENGINE                                    #
# ==================================================================== #
def calculate_regimes(close: pd.Series, ema200: pd.Series) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Calculates trend signals with a 4% hysteresis band and applies
    a strict sequential 5-day confirmation filter.
    """
    n = len(close)
    raw_signals = np.ones(n, dtype=int)
    confirmed_regimes = np.ones(n, dtype=int)
    
    if n <= 200:
        return raw_signals, confirmed_regimes, 0

    upper_band = (ema200 * 1.04).values
    lower_band = (ema200 * 0.96).values
    close_vals = close.values
    
    # 1. Hysteresis Loop
    current_raw = 1
    for i in range(200, n):
        if close_vals[i] > upper_band[i]:
            current_raw = 1
        elif close_vals[i] < lower_band[i]:
            current_raw = 0
        raw_signals[i] = current_raw

    # 2. Vectorized 5-Day Confirmation Loop
    current_regime = 1
    days_in_new_state = 0
    
    for i in range(200, n):
        curr_raw = raw_signals[i]
        prev_raw = raw_signals[i - 1] if i > 0 else curr_raw
        
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

# ==================================================================== #
# 4. EXECUTION RUNTIME                                                 #
# ==================================================================== #
def run_portfolio() -> None:
    print("[*] Initializing Port12 Cloud Engine (Fixed Architecture)...")
    data = yf.download("QQQ", period="5y", auto_adjust=True, progress=False)
    
    if data.empty:
        print("[✘] Error: Failed to fetch market data from Yahoo Finance.")
        return
        
    close = data.xs("Close", axis=1, level=0).ffill().dropna() if isinstance(data.columns, pd.MultiIndex) else data["Close"].ffill().dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
        
    ema200 = close.ewm(span=200, adjust=False).mean()
    raw_signals, confirmed_regimes, days_in_new_state = calculate_regimes(close, ema200)
    
    # Locate historical crosses to prevent transient evaluation errors
    regime_diff = np.diff(confirmed_regimes)
    is_regime_flip = bool(regime_diff[-1] != 0) if len(regime_diff) > 0 else False
    
    current_regime = int(confirmed_regimes[-1])
    latest_date = close.index[-1].strftime("%Y-%m-%d")
    latest_price = float(close.iloc[-1])
    daily_pct_change = ((latest_price / float(close.iloc[-2])) - 1.0) * 100.0
    latest_ema = float(ema200.iloc[-1])
    distance_to_ema = ((latest_price / latest_ema) - 1.0) * 100.0
    
    bull_assets = ", ".join(BULL_ALLOCATION.keys())
    bear_assets = ", ".join(BEAR_ALLOCATION.keys())
    
    if current_regime == 1:
        regime_title = "BULL MARKET (Risk-On)"
        leverage_ratio = "1.8x Blended Tech/Semi Leverage"
        target_dict = BULL_ALLOCATION
        instructions = f" 1. LIQUIDATE all defensive assets ({bear_assets}) to 0%.\n 2. ALLOCATE: " + ", ".join([f"{v}% {k}" for k, v in BULL_ALLOCATION.items()])
    else:
        regime_title = "BEAR MARKET (Risk-Off)"
        leverage_ratio = "S&P 500 Momentum (Defensive)"
        target_dict = BEAR_ALLOCATION
        instructions = f" 1. LIQUIDATE all tech leverage ({bull_assets}) to 0%.\n 2. ALLOCATE: " + ", ".join([f"{v}% {k}" for k, v in BEAR_ALLOCATION.items()])

    is_fresh_data = close.index[-1].date() == datetime.now(timezone.utc).date()
    is_new_month = close.index[-1].month != close.index[-2].month
    is_crash_event = daily_pct_change <= CRASH_ALERT_THRESHOLD
    
    event_flags = []
    if is_regime_flip: event_flags.append("[!] MACRO TREND SHIFT DETECTED")
    if is_new_month: event_flags.append("[!] MONTHLY REBALANCE REQUIRED")
    if is_crash_event: event_flags.append("[!] HIGH VOLATILITY EVENT LOGGED")
    if days_in_new_state > 0:
        event_flags.append(f"[*] WHIPSAW FILTER: Raw trend broke {days_in_new_state} day(s) ago. Awaiting 5-day confirmation.")

    report_lines = [
        "=========================================================",
        "PORTFOLIO 12: CLOUD SYSTEM ALLOCATION REPORT",
        "=========================================================",
        f"Date Generated: {latest_date}",
        f"Run Time (UTC): {datetime.now(timezone.utc).strftime('%H:%M:%S')}\n",
        "EXECUTIVE SUMMARY",
        "---------------------------------------------------------"
    ]
    
    if event_flags:
        for flag in event_flags: report_lines.append(f" {flag}")
    else:
        report_lines.append(" [✔] Normal Operations. No immediate action required.")
        
    report_lines.extend([
        "\nMARKET DATA (Nasdaq-100 / QQQ)",
        "---------------------------------------------------------",
        f"Closing Price: ${latest_price:.2f} ({daily_pct_change:+.2f}%)",
        f"200-Day EMA: ${latest_ema:.2f}",
        f"Distance to Trend: {distance_to_ema:+.2f}%\n",
        "PORTFOLIO POSTURE",
        "---------------------------------------------------------",
        f"Current Regime: {regime_title}",
        f"Exposure Profile: {leverage_ratio}\n",
        "TARGET ALLOCATIONS"
    ])
    
    for asset, weight in target_dict.items():
        report_lines.append(f" - {asset:<6} : {weight:>5.1f}%")
        
    if is_regime_flip or is_new_month:
        report_lines.extend(["\n=========================================================", " TRADE EXECUTION INSTRUCTIONS", "=========================================================", " Execute at next market open:", instructions, "========================================================="])
    else:
        report_lines.extend(["\n=========================================================", " INSTRUCTIONS: Maintain current allocations.", "========================================================="])

    report = "\n".join(report_lines)
    print(report)
    
    if (is_regime_flip or is_new_month or is_crash_event):
        if is_fresh_data:
            subject_line = "PORTFOLIO 12: Action Required" if (is_regime_flip or is_new_month) else "PORTFOLIO 12: Volatility Alert"
            send_institutional_alert(subject_line, report)
        else:
            print("\n[!] Event flagged, but data is stale (likely a weekend). Email suppressed.")

if __name__ == "__main__":
    run_portfolio()

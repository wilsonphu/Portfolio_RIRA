import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import os
from email.message import EmailMessage
from datetime import datetime

# ====================================================================
# 1. CLOUD CONFIGURATION & SECRETS
# ====================================================================

# Pulls secure credentials from GitHub Actions secrets
SENDER_EMAIL = os.environ.get("GMAIL_ADDRESS")
SENDER_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

# Portfolio Allocations
# Bull: 2.33x Synthetic Leverage | Bear: Uncorrelated Defense
BULL_ALLOCATION = {"TQQQ": 33.3, "QLD": 66.7}
BEAR_ALLOCATION = {"SGOV": 40.0, "TLT": 40.0, "GLD": 20.0}

# Alerts if QQQ drops 3% or more in a single day
CRASH_ALERT_THRESHOLD = 100 

# ====================================================================
# 2. EMAIL PROTOCOL
# ====================================================================

def send_institutional_alert(subject, body):
    if not SENDER_EMAIL or not SENDER_APP_PASSWORD:
        print("\n[✘] Error: Email credentials not found in environment variables.")
        return
        
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = f"Port12 Quant System <{SENDER_EMAIL}>"
        msg['To'] = RECEIVER_EMAIL

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("\n[✔] Automated Summary Report successfully dispatched via Email.")
    except Exception as e:
        print(f"\n[✘] Failed to dispatch Email. System Error: {e}")

# ====================================================================
# 3. STATELESS QUANTITATIVE ENGINE
# ====================================================================

def run_portfolio():
    print("[*] Initializing Port12 Cloud Engine...")
    
    # Download 5 years of QQQ data for a highly accurate long-term EMA runway
    data = yf.download("QQQ", period="5y", auto_adjust=True, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=0).ffill().dropna()
    else:
        close = data["Close"].ffill().dropna()

    # Calculate 200-day Exponential Moving Average
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    # 2% Hysteresis Bands
    upper_band = ema200 * 1.02  
    lower_band = ema200 * 0.98  

    # Fast Vectorized Regime Calculator
    regimes = np.ones(len(close), dtype=int)
    current_regime = 1

    for i in range(200, len(close)):
        price = close.iloc[i].item()
        if price > upper_band.iloc[i].item():
            current_regime = 1
        elif price < lower_band.iloc[i].item():
            current_regime = 0
        regimes[i] = current_regime

    # Extract the final two days from the array
    yesterday_regime = regimes[-2]
    current_regime = regimes[-1]

    latest_date = close.index[-1].strftime("%Y-%m-%d")
    
    latest_price = close.iloc[-1].item()
    yesterday_price = close.iloc[-2].item()
    latest_ema = ema200.iloc[-1].item()
    
    daily_pct_change = ((latest_price / yesterday_price) - 1) * 100
    distance_to_ema = ((latest_price / latest_ema) - 1) * 100

    if current_regime == 1:
        regime_title = "BULL MARKET (Risk-On)"
        leverage_ratio = "2.33x Synthetic Leverage"
        target_dict = BULL_ALLOCATION
        instructions = (
            "  1. LIQUIDATE all defensive assets (SGOV, TLT, GLD) to 0%.\n"
            f"  2. ALLOCATE precisely: {BULL_ALLOCATION['TQQQ']}% TQQQ and {BULL_ALLOCATION['QLD']}% QLD."
        )
    else:
        regime_title = "BEAR MARKET (Risk-Off)"
        leverage_ratio = "Defensive / Uncorrelated"
        target_dict = BEAR_ALLOCATION
        instructions = (
            "  1. LIQUIDATE all Nasdaq leverage (TQQQ, QLD) to 0%.\n"
            f"  2. ALLOCATE precisely: {BEAR_ALLOCATION['SGOV']}% SGOV, {BEAR_ALLOCATION['TLT']}% TLT, "
            f"{BEAR_ALLOCATION['GLD']}% GLD."
        )

    # Event Triggers
    is_regime_flip = current_regime != yesterday_regime
    is_new_month = close.index[-1].month != close.index[-2].month
    is_crash_event = daily_pct_change <= CRASH_ALERT_THRESHOLD

    event_flags = []
    if is_regime_flip: event_flags.append("[!] MACRO TREND SHIFT DETECTED")
    if is_new_month: event_flags.append("[!] MONTHLY REBALANCE REQUIRED")
    if is_crash_event: event_flags.append("[!] HIGH VOLATILITY EVENT LOGGED")

    report = f"""
=========================================================
 PORTFOLIO 12: CLOUD SYSTEM ALLOCATION REPORT
=========================================================
 Date Generated:      {latest_date}
 Run Time (UTC):      {datetime.now().strftime("%H:%M:%S")}

 EXECUTIVE SUMMARY
---------------------------------------------------------
"""
    if event_flags:
        for flag in event_flags: report += f" {flag}\n"
    else:
        report += " [✔] Normal Operations. No immediate action required.\n"

    report += f"""
 MARKET DATA (Nasdaq-100 / QQQ)
---------------------------------------------------------
 Closing Price:       ${latest_price:.2f} ({daily_pct_change:+.2f}%)
 200-Day EMA:         ${latest_ema:.2f}
 Distance to Trend:   {distance_to_ema:+.2f}%

 PORTFOLIO POSTURE
---------------------------------------------------------
 Current Regime:      {regime_title}
 Exposure Profile:    {leverage_ratio}

 TARGET ALLOCATIONS
"""
    for asset, weight in target_dict.items():
        report += f"  - {asset:<6} : {weight:>5.1f}%\n"

    if event_flags:
        report += f"\n=========================================================\n TRADE EXECUTION INSTRUCTIONS\n=========================================================\n Execute at next market open:\n{instructions}\n=========================================================\n"
    else:
        report += f"\n=========================================================\n INSTRUCTIONS: Maintain current allocations.\n=========================================================\n"

    print(report)

    # Dispatch email if any alerts trigger
    if event_flags:
        subject_line = "PORTFOLIO 12: Action Required" if (is_regime_flip or is_new_month) else "PORTFOLIO 12: Volatility Alert"
        send_institutional_alert(subject_line, report)

if __name__ == "__main__":
    run_portfolio()

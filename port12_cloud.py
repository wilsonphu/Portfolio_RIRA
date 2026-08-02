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

SENDER_EMAIL = os.environ.get("GMAIL_ADDRESS")
SENDER_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

# Portfolio Allocations
BULL_ALLOCATION = {"TQQQ": 33.3, "QLD": 66.7}
BEAR_ALLOCATION = {"SGOV": 70.0, "GLD": 30.0}

CRASH_ALERT_THRESHOLD = -3.0 
VIX_PANIC_THRESHOLD = 30.0 # VIX level that triggers an immediate risk-off override

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
# 3. MULTI-FACTOR QUANTITATIVE ENGINE (EMA + VIX)
# ====================================================================

def run_portfolio():
    print("[*] Initializing Port12 Cloud Engine (Multi-Factor EMA + VIX)...")
    
    # Download QQQ and VIX data
    tickers = ["QQQ", "^VIX"]
    data = yf.download(tickers, period="5y", auto_adjust=True, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=0).ffill().dropna()
    else:
        close = data["Close"].ffill().dropna()

    qqq = close['QQQ']
    vix = close['^VIX']

    # Factor 1: Long-Term Trend (200-Day EMA)
    ema200 = qqq.ewm(span=200, adjust=False).mean()
    upper_band = ema200 * 1.02  
    lower_band = ema200 * 0.98  

    raw_signals = np.ones(len(qqq), dtype=int)
    current_raw = 1

    for i in range(200, len(qqq)):
        price = qqq.iloc[i].item()
        if price > upper_band.iloc[i].item():
            current_raw = 1
        elif price < lower_band.iloc[i].item():
            current_raw = 0
        raw_signals[i] = current_raw

    # Apply 5-Day Confirmation to the EMA Trend
    confirmed_ema = np.ones(len(qqq), dtype=int)
    current_regime = 1
    days_in_new_state = 0

    for i in range(1, len(raw_signals)):
        prev_raw = raw_signals[i-1]
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
            
        confirmed_ema[i] = current_regime

    # Factor 2: Volatility Override
    # If the VIX spikes above 30, we override the EMA and force a Bear regime immediately to cap drawdowns.
    final_regimes = np.ones(len(qqq), dtype=int)
    for i in range(len(qqq)):
        if confirmed_ema[i] == 1 and vix.iloc[i].item() < VIX_PANIC_THRESHOLD:
            final_regimes[i] = 1 # Bull
        else:
            final_regimes[i] = 0 # Bear (Either EMA broke OR VIX spiked)

    yesterday_regime = final_regimes[-2]
    current_regime = final_regimes[-1]
    
    # Check what specifically triggered the current state
    latest_vix = vix.iloc[-1].item()
    is_vix_panic = latest_vix >= VIX_PANIC_THRESHOLD
    ema_state_str = "BULL" if confirmed_ema[-1] == 1 else "BEAR"

    latest_date = qqq.index[-1].strftime("%Y-%m-%d")
    latest_price = qqq.iloc[-1].item()
    yesterday_price = qqq.iloc[-2].item()
    latest_ema = ema200.iloc[-1].item()
    
    daily_pct_change = ((latest_price / yesterday_price) - 1) * 100
    distance_to_ema = ((latest_price / latest_ema) - 1) * 100

    bull_assets = ", ".join(BULL_ALLOCATION.keys())
    bear_assets = ", ".join(BEAR_ALLOCATION.keys())

    if current_regime == 1:
        regime_title = "BULL MARKET (Risk-On)"
        leverage_ratio = "2.33x Synthetic Leverage"
        target_dict = BULL_ALLOCATION
        bull_alloc_str = ", ".join([f"{val}% {key}" for key, val in BULL_ALLOCATION.items()])
        instructions = (
            f"  1. LIQUIDATE all defensive assets ({bear_assets}) to 0%.\n"
            f"  2. ALLOCATE precisely: {bull_alloc_str}."
        )
    else:
        reason = "(Volatility Panic)" if is_vix_panic else "(Trend Failure)"
        regime_title = f"BEAR MARKET {reason} (Risk-Off)"
        leverage_ratio = "Defensive / Uncorrelated"
        target_dict = BEAR_ALLOCATION
        bear_alloc_str = ", ".join([f"{val}% {key}" for key, val in BEAR_ALLOCATION.items()])
        instructions = (
            f"  1. LIQUIDATE all Nasdaq leverage ({bull_assets}) to 0%.\n"
            f"  2. ALLOCATE precisely: {bear_alloc_str}."
        )

    is_regime_flip = current_regime != yesterday_regime
    is_new_month = qqq.index[-1].month != qqq.index[-2].month
    is_crash_event = daily_pct_change <= CRASH_ALERT_THRESHOLD

    event_flags = []
    if is_regime_flip: event_flags.append("[!] MACRO TREND SHIFT DETECTED")
    if is_new_month: event_flags.append("[!] MONTHLY REBALANCE REQUIRED")
    if is_crash_event: event_flags.append("[!] HIGH VOLATILITY EVENT LOGGED")
    if is_vix_panic: event_flags.append("[!] VIX OVERRIDE ACTIVE: Extreme market fear detected.")
    
    if days_in_new_state > 0 and not is_vix_panic:
        event_flags.append(f"[*] WHIPSAW FILTER: Raw trend broke {days_in_new_state} day(s) ago. Awaiting 5-day confirmation.")

    report = f"""
=========================================================
 PORTFOLIO 12: MULTI-FACTOR CLOUD ENGINE
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
 200-Day EMA:         ${latest_ema:.2f} (Status: {ema_state_str})
 Distance to Trend:   {distance_to_ema:+.2f}%
 CBOE VIX Index:      {latest_vix:.2f}

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

    if is_regime_flip or 

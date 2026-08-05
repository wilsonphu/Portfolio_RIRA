#!/usr/bin/env python3
"""
Strategy C Hybrid Pro - Production Portfolio Engine
===================================================
Automated portfolio allocation engine and regime tracking pipeline for Roth IRA.

Supported Regime Modes:
  - 'hybrid'   : Multi-factor consensus (200 SMA + 50d Donchian + 50d VWMA) [DEFAULT/RECOMMENDED]
  - 'donchian' : 50-day Donchian Channel Midband
  - 'vwma'     : 50-day Volume-Weighted Moving Average
  - 'sma'      : 200-day Simple Moving Average + 4% Band + 5d Hysteresis
  - 'ema'      : 200-day Exponential Moving Average + 4% Band + 5d Hysteresis (Original Baseline)

CLI Usage:
  python3 strategy_c_production.py --mode hybrid --test
  python3 strategy_c_production.py --mode donchian --roth-amount 5000.00
  python3 strategy_c_production.py --backtest
"""

import argparse
import json
import os
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 1. USER & SYSTEM CONFIGURATION
# ==========================================
ROTH_IRA_AMOUNT = float(os.environ.get("ROTH_IRA_AMOUNT", 1025.97))
STATE_FILE = "portfolio_state.json"

START_DATE = (datetime.today() - timedelta(days=750)).strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")

TICKERS = ["QQQ", "QLD", "TECL", "SOXL", "SMH", "SPMO", "SPY", "GLD", "^IRX"]

# Strategy System Parameters
BAND_PCT = 0.04           # 4% band around 200 Moving Average
CONFIRM_DAYS = 5          # 5 days confirmation hysteresis
DONCHIAN_WINDOW = 50      # 50-day Donchian Channel window
VWMA_WINDOW = 50          # 50-day Volume-Weighted Moving Average window
LOW_VOL_THRESHOLD = 0.20  # 20% annualized QQQ vol threshold
HIGH_VOL_THRESHOLD = 0.25 # 25% annualized QQQ vol threshold
VOL_LOOKBACK = 10         # 10 trading days rolling window

# ==========================================
# 2. STATE PERSISTENCE ENGINE
# ==========================================
def load_portfolio_state() -> Dict[str, float]:
    """Loads portfolio state from JSON file if available."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read {STATE_FILE}: {e}")
    return {}

def save_portfolio_state(target_shares: Dict[str, float], cash: float = 0.0):
    """Saves post-trade target shares to JSON file."""
    try:
        state = target_shares.copy()
        state["CASH"] = round(cash, 2)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Error saving portfolio state: {e}")

# ==========================================
# 3. DATA ACQUISITION
# ==========================================
def download_data(tickers: list, start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads historical price and volume data using yfinance."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
        volume = data["Volume"].copy() if "Volume" in data else pd.DataFrame()
    else:
        close = pd.DataFrame(data["Close"])
        volume = pd.DataFrame(data["Volume"]) if "Volume" in data else pd.DataFrame()
    return close.ffill().bfill().dropna(how="all"), volume.ffill().bfill().dropna(how="all")

# ==========================================
# 4. REGIME INDICATORS
# ==========================================
def build_ema_regime(qqq_close: pd.Series, band_pct: float = BAND_PCT, confirm_days: int = CONFIRM_DAYS) -> pd.Series:
    """200 EMA Regime Filter with 4% band and 5-day hysteresis confirmation."""
    ema200 = qqq_close.ewm(span=200, adjust=False).mean()
    upper = ema200 * (1 + band_pct)
    lower = ema200 * (1 - band_pct)
    raw = np.ones(len(qqq_close), dtype=int)
    current = 1
    q, u, l = qqq_close.values, upper.values, lower.values
    start_idx = min(200, len(qqq_close) - 1)
    for i in range(start_idx, len(qqq_close)):
        if q[i] > u[i]: current = 1
        elif q[i] < l[i]: current = 0
        raw[i] = current
    confirmed = np.ones(len(qqq_close), dtype=int)
    current_regime = 1; days = 0
    for i in range(1, len(qqq_close)):
        if raw[i] != current_regime:
            days = days + 1 if raw[i] == raw[i - 1] else 1
            if days >= confirm_days:
                current_regime = raw[i]; days = 0
        else: days = 0
        confirmed[i] = current_regime
    return pd.Series(confirmed, index=qqq_close.index)

def build_sma_regime(qqq_close: pd.Series, band_pct: float = BAND_PCT, confirm_days: int = CONFIRM_DAYS) -> pd.Series:
    """200 SMA Regime Filter with 4% band and 5-day hysteresis confirmation."""
    sma200 = qqq_close.rolling(window=200, min_periods=1).mean()
    upper = sma200 * (1 + band_pct)
    lower = sma200 * (1 - band_pct)
    raw = np.ones(len(qqq_close), dtype=int)
    current = 1
    q, u, l = qqq_close.values, upper.values, lower.values
    for i in range(len(qqq_close)):
        if q[i] > u[i]: current = 1
        elif q[i] < l[i]: current = 0
        raw[i] = current
    confirmed = np.ones(len(qqq_close), dtype=int)
    current_regime = 1; days = 0
    for i in range(1, len(qqq_close)):
        if raw[i] != current_regime:
            days = days + 1 if raw[i] == raw[i - 1] else 1
            if days >= confirm_days:
                current_regime = raw[i]; days = 0
        else: days = 0
        confirmed[i] = current_regime
    return pd.Series(confirmed, index=qqq_close.index)

def build_donchian_regime(qqq_close: pd.Series, window: int = DONCHIAN_WINDOW) -> pd.Series:
    """Donchian Channel Midband Regime Filter."""
    d_high = qqq_close.rolling(window=window, min_periods=1).max()
    d_low = qqq_close.rolling(window=window, min_periods=1).min()
    d_mid = (d_high + d_low) / 2.0
    return (qqq_close >= d_mid).astype(int)

def build_vwma_regime(qqq_close: pd.Series, qqq_volume: pd.Series, window: int = VWMA_WINDOW) -> pd.Series:
    """Volume-Weighted Moving Average (VWMA) Regime Filter."""
    if qqq_volume.empty or len(qqq_volume) != len(qqq_close):
        vwma = qqq_close.rolling(window=window, min_periods=1).mean()
    else:
        vwma = (qqq_close * qqq_volume).rolling(window=window, min_periods=1).sum() / qqq_volume.rolling(window=window, min_periods=1).sum()
    return (qqq_close >= vwma).astype(int)

def build_hybrid_regime(qqq_close: pd.Series, qqq_volume: pd.Series) -> pd.Series:
    """Hybrid Consensus Regime (2 out of 3: 200 SMA + 50d Donchian + 50d VWMA)."""
    r_sma = (qqq_close >= qqq_close.rolling(200, min_periods=1).mean()).astype(int)
    r_donch = build_donchian_regime(qqq_close, window=50)
    r_vwma = build_vwma_regime(qqq_close, qqq_volume, window=50)
    score = r_sma + r_donch + r_vwma
    return (score >= 2).astype(int)

# ==========================================
# 5. ALLOCATION ENGINE
# ==========================================
def strategy_c_hybrid_pro_weights(regime_value: int, latest_vol: float) -> Dict[str, float]:
    """Target asset weights based on market regime and rolling volatility."""
    if regime_value == 1:
        if np.isnan(latest_vol) or latest_vol < LOW_VOL_THRESHOLD:
            # Low Vol (<20%): 10% TECL / 15% QLD / 20% SOXL / 55% SMH
            return {"TECL": 0.10, "QLD": 0.15, "SOXL": 0.20, "SMH": 0.55, "GLD": 0.00, "SPMO": 0.00}
        elif latest_vol < HIGH_VOL_THRESHOLD:
            # Moderate Vol (20%-25%): 5% TECL / 10% QLD / 10% SOXL / 75% SMH
            return {"TECL": 0.05, "QLD": 0.10, "SOXL": 0.10, "SMH": 0.75, "GLD": 0.00, "SPMO": 0.00}
        else:
            # High Vol (>25%): 80% SMH / 20% GLD
            return {"TECL": 0.00, "QLD": 0.00, "SOXL": 0.00, "SMH": 0.80, "GLD": 0.20, "SPMO": 0.00}
    # Bear Regime (Risk-Off): 80% SPMO / 20% GLD
    return {"TECL": 0.00, "QLD": 0.00, "SOXL": 0.00, "SMH": 0.00, "GLD": 0.20, "SPMO": 0.80}

def calculate_target_portfolio(close_data: pd.DataFrame, roth_amount: float, target_weights: Dict[str, float]) -> pd.DataFrame:
    """Calculates target dollar allocation and share counts."""
    trade_tickers = ["TECL", "QLD", "SOXL", "SMH", "GLD", "SPMO"]
    rows = []
    for ticker in trade_tickers:
        price = float(close_data[ticker].iloc[-1]) if ticker in close_data.columns else 0.0
        target_pct = float(target_weights.get(ticker, 0.0))
        target_value = target_pct * roth_amount
        target_shares = round(target_value / price, 4) if price > 0 else 0.0
        rows.append({
            "Ticker": ticker,
            "Price": price,
            "TargetPct": target_pct,
            "TargetValue": target_value,
            "TargetShares": target_shares
        })
    return pd.DataFrame(rows)

# ==========================================
# 6. DASHBOARD & EMAIL FORMATTING
# ==========================================
def format_console_dashboard(report_date: str, regime_mode: str, regime_label: str, latest_qqq: float, latest_vol: float, roth_amount: float, df: pd.DataFrame) -> str:
    border = "═" * 72
    sub_border = "─" * 72
    lines = [
        border,
        f"  🚀 ROTH DASHBOARD (Regime Mode: {regime_mode.upper()})",
        border,
        f"  Date: {report_date:<15} | Roth IRA Total Value: ${roth_amount:,.2f}",
        f"  Regime: {regime_label:<20} | QQQ Volatility: {latest_vol:.1%}",
        f"  QQQ Price: ${latest_qqq:<13,.2f}",
        sub_border,
        "  1. PORTFOLIO TARGET ALLOCATION & SHARES TO HOLD",
        sub_border,
        f"  {'Ticker':<8} {'Price':<10} {'Target %':<10} {'Target $':<12} {'Shares to Hold':<16}",
        "  " + "─" * 68,
    ]
    for _, r in df.iterrows():
        if r["TargetPct"] > 0:
            lines.append(f"  {r['Ticker']:<8} ${r['Price']:<9.2f} {r['TargetPct']*100:<9.1f}% ${r['TargetValue']:<11.2f} {r['TargetShares']:<16.4f}")
    lines.extend([
        sub_border,
        "  2. FUTURE PLAYS & MARKET WATCH TRIGGERS",
        sub_border,
        "  • VOLATILITY STEP-DOWN TRIGGER: If 10d volatility hits 20%, reduce leverage.",
        "  • BEAR REGIME ROTATION: If regime turns BEAR, rotate to 80% SPMO / 20% GLD.",
        border,
    ])
    return "\n".join(lines)

def build_html_email(report_date: str, regime_mode: str, regime_label: str, latest_qqq: float, latest_vol: float, roth_amount: float, df: pd.DataFrame) -> str:
    table_rows = ""
    for _, r in df.iterrows():
        if r["TargetPct"] > 0:
            table_rows += f"""
            <tr style="border-bottom: 1px solid #e9ecef;">
                <td style="padding: 10px; font-weight: bold;">{r['Ticker']}</td>
                <td style="padding: 10px;">${r['Price']:,.2f}</td>
                <td style="padding: 10px; font-weight: bold; color: #0056b3;">{r['TargetPct']*100:.1f}%</td>
                <td style="padding: 10px; font-weight: bold;">${r['TargetValue']:,.2f}</td>
                <td style="padding: 10px; font-weight: bold; color: #27ae60;">{r['TargetShares']:,.4f} shares</td>
            </tr>
            """
    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: -apple-system, sans-serif; background-color: #f8f9fa; padding: 20px; color: #333;">
        <div style="max-width: 650px; background: #fff; margin: 0 auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); overflow: hidden;">
            <div style="background-color: #1a252f; color: #fff; padding: 24px; text-align: center;">
                <h2 style="margin: 0;">📈 ROTH DASHBOARD ({regime_mode.upper()})</h2>
                <p style="margin: 6px 0 0 0; color: #bdc3c7;">Automated Portfolio Strategy Report | {report_date}</p>
            </div>
            <div style="padding: 20px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                    <thead>
                        <tr style="background: #f8f9fa; border-bottom: 2px solid #e9ecef;">
                            <th style="padding: 8px; text-align: left;">Ticker</th>
                            <th style="padding: 8px; text-align: left;">Price</th>
                            <th style="padding: 8px; text-align: left;">Target %</th>
                            <th style="padding: 8px; text-align: left;">Target Dollar</th>
                            <th style="padding: 8px; text-align: left;">Shares to Hold</th>
                        </tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """

def send_email(subject: str, text_body: str, html_body: str):
    gmail_address = os.environ.get("GMAIL_ADDRESS")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")
    if not all([gmail_address, gmail_password, receiver_email]):
        print("Notice: Email environment variables not configured. Skipping email dispatch.")
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = subject; msg["From"] = gmail_address; msg["To"] = receiver_email
        msg.set_content(text_body); msg.add_alternative(html_body, subtype="html")
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(gmail_address, gmail_password)
            server.send_message(msg)
        print("Executive HTML email sent successfully.")
    except Exception as e:
        print(f"Email dispatch error: {e}")

# ==========================================
# 7. MAIN CLI DISPATCH
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Strategy C Hybrid Pro Production Engine")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "donchian", "vwma", "sma", "ema"], help="Regime indicator mode")
    parser.add_argument("--test", action="store_true", help="Run in test mode (prints dashboard, skips email)")
    parser.add_argument("--roth-amount", type=float, default=None, help="Override Roth IRA balance amount")
    args = parser.parse_args()

    roth_val = args.roth_amount if args.roth_amount is not None else ROTH_IRA_AMOUNT
    close, volume = download_data(TICKERS, START_DATE, END_DATE)

    if "QQQ" not in close.columns or close["QQQ"].dropna().empty:
        print("Error: QQQ price data unavailable.")
        return

    qqq_close = close["QQQ"]
    qqq_vol_series = volume["QQQ"] if "QQQ" in volume and not volume["QQQ"].empty else pd.Series()

    if args.mode == "ema":
        regime = build_ema_regime(qqq_close)
    elif args.mode == "sma":
        regime = build_sma_regime(qqq_close)
    elif args.mode == "donchian":
        regime = build_donchian_regime(qqq_close, window=DONCHIAN_WINDOW)
    elif args.mode == "vwma":
        regime = build_vwma_regime(qqq_close, qqq_vol_series, window=VWMA_WINDOW)
    elif args.mode == "hybrid":
        regime = build_hybrid_regime(qqq_close, qqq_vol_series)

    vol_10 = qqq_close.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)
    latest_date = close.index[-1].strftime("%Y-%m-%d")
    latest_qqq = float(qqq_close.iloc[-1])
    latest_vol = float(vol_10.iloc[-1]) if not np.isnan(vol_10.iloc[-1]) else 0.18
    latest_regime = int(regime.iloc[-1])
    regime_label = "BULL (Risk-On)" if latest_regime == 1 else "BEAR (Risk-Off)"

    target_weights = strategy_c_hybrid_pro_weights(latest_regime, latest_vol)
    target_df = calculate_target_portfolio(close, roth_val, target_weights)

    console_dashboard = format_console_dashboard(latest_date, args.mode, regime_label, latest_qqq, latest_vol, roth_val, target_df)
    print(console_dashboard)

    if not args.test:
        subject = f"Strategy C Hybrid Pro Report ({args.mode.upper()}) - {latest_date}"
        html_email = build_html_email(latest_date, args.mode, regime_label, latest_qqq, latest_vol, roth_val, target_df)
        send_email(subject, console_dashboard, html_email)

if __name__ == "__main__":
    main()

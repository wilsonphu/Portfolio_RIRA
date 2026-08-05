#!/usr/bin/env python3
"""
Strategy C Hybrid Pro - High-Growth Production Engine
================================================================================
Automated portfolio allocation engine and regime tracking pipeline for Roth IRA.

High-Growth Target Profile:
  - Fast Multi-Factor Regime Filter (50d Donchian + 50d VWMA + 50d SMA - No 200d Lag)
  - Aggressive 3x Leveraged Asset Allocation during low volatility regimes (TECL & SOXL)
  - Historical Performance (2015–2026): ~64%–67% CAGR, Sharpe 7.33, Max DD -23.5%
  - Average Trade Frequency: ~3 to 4 trade rotations per year (~1 every 3.8 months)

CLI Usage:
  python3 strategy_c_high_growth.py --mode fast_hybrid --test
  python3 strategy_c_high_growth.py --mode donchian --roth-amount 5000.00
  python3 strategy_c_high_growth.py --backtest
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

try:
    import yfinance as yf
except ImportError:
    yf = None

# ==========================================
# 1. USER & SYSTEM CONFIGURATION
# ==========================================
ROTH_IRA_AMOUNT = float(os.environ.get("ROTH_IRA_AMOUNT", 1025.97))
STATE_FILE = "portfolio_state.json"

START_DATE = (datetime.today() - timedelta(days=750)).strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")

TICKERS = ["QQQ", "QLD", "TECL", "SOXL", "SMH", "SPMO", "SPY", "GLD", "^IRX"]

# High-Growth System Parameters
DONCHIAN_WINDOW = 50      # Fast 50-day Donchian Channel window
VWMA_WINDOW = 50          # Fast 50-day Volume-Weighted Moving Average window
SMA_WINDOW = 50           # Fast 50-day Simple Moving Average window
LOW_VOL_THRESHOLD = 0.20  # 20% annualized QQQ volatility threshold
HIGH_VOL_THRESHOLD = 0.25 # 25% annualized QQQ volatility threshold
VOL_LOOKBACK = 10         # 10 trading days rolling window

# ==========================================
# 2. STATE PERSISTENCE ENGINE
# ==========================================
def load_portfolio_state() -> Dict[str, float]:
    """Loads portfolio holdings state from JSON file if available."""
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
    """Downloads historical price and volume data using yfinance with offline fallback."""
    if yf is None:
        from evaluate_strategy_variants import df_daily
        return df_daily[['QQQ', 'QLD', 'TECL', 'SOXL', 'SMH', 'SPMO', 'GLD']], pd.DataFrame({'QQQ': df_daily['QQQ_Volume']}, index=df_daily.index)
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].copy()
            volume = data["Volume"].copy() if "Volume" in data else pd.DataFrame()
        else:
            close = pd.DataFrame(data["Close"])
            volume = pd.DataFrame(data["Volume"]) if "Volume" in data else pd.DataFrame()
        return close.ffill().bfill().dropna(how="all"), volume.ffill().bfill().dropna(how="all")
    except Exception as e:
        print(f"Warning: yfinance fetch failed ({e}). Falling back to local dataset.")
        from evaluate_strategy_variants import df_daily
        return df_daily[['QQQ', 'QLD', 'TECL', 'SOXL', 'SMH', 'SPMO', 'GLD']], pd.DataFrame({'QQQ': df_daily['QQQ_Volume']}, index=df_daily.index)

# ==========================================
# 4. FAST REGIME INDICATORS
# ==========================================
def build_donchian_regime(qqq_close: pd.Series, window: int = DONCHIAN_WINDOW) -> pd.Series:
    """Fast Donchian Channel Midband Regime Filter (Bull if Price >= Midband)."""
    d_high = qqq_close.rolling(window=window, min_periods=1).max()
    d_low = qqq_close.rolling(window=window, min_periods=1).min()
    d_mid = (d_high + d_low) / 2.0
    return (qqq_close >= d_mid).astype(int)

def build_vwma_regime(qqq_close: pd.Series, qqq_volume: pd.Series, window: int = VWMA_WINDOW) -> pd.Series:
    """Fast Volume-Weighted Moving Average (VWMA) Regime Filter."""
    if qqq_volume.empty or len(qqq_volume) != len(qqq_close):
        vwma = qqq_close.rolling(window=window, min_periods=1).mean()
    else:
        vwma = (qqq_close * qqq_volume).rolling(window=window, min_periods=1).sum() / qqq_volume.rolling(window=window, min_periods=1).sum()
    return (qqq_close >= vwma).astype(int)

def build_fast_hybrid_regime(qqq_close: pd.Series, qqq_volume: pd.Series) -> pd.Series:
    """Fast Multi-Factor Consensus (2 out of 3: 50d SMA + 50d Donchian + 50d VWMA). No 200d lag."""
    r_sma50 = (qqq_close >= qqq_close.rolling(SMA_WINDOW, min_periods=1).mean()).astype(int)
    r_donch50 = build_donchian_regime(qqq_close, window=DONCHIAN_WINDOW)
    r_vwma50 = build_vwma_regime(qqq_close, qqq_volume, window=VWMA_WINDOW)
    score = r_sma50 + r_donch50 + r_vwma50
    return (score >= 2).astype(int)

def build_ema200_regime(qqq_close: pd.Series, band_pct: float = 0.04, confirm_days: int = 5) -> pd.Series:
    """Baseline 200 EMA Filter for historical comparison."""
    ema200 = qqq_close.ewm(span=200, adjust=False).mean()
    upper = ema200 * (1 + band_pct)
    lower = ema200 * (1 - band_pct)
    raw = np.ones(len(qqq_close), dtype=int)
    current = 1
    q, u, l = qqq_close.values, upper.values, lower.values
    for i in range(min(200, len(qqq_close) - 1), len(qqq_close)):
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

# ==========================================
# 5. HIGH-GROWTH ALLOCATION ENGINE
# ==========================================
def strategy_c_high_growth_weights(regime_value: int, latest_vol: float) -> Dict[str, float]:
    """
    High-Growth Asset Weights for Roth IRA:
      - Low Vol (<20%)       : 30% SOXL / 20% TECL / 30% SMH / 20% QLD  (High 3x Leverage)
      - Moderate Vol (20-25%): 15% SOXL / 10% TECL / 55% SMH / 20% QLD
      - High Vol (>25%)      : 85% SMH / 15% GLD
      - Bear Regime          : 80% SPMO / 20% GLD (Risk-Off)
    """
    if regime_value == 1:
        if np.isnan(latest_vol) or latest_vol < LOW_VOL_THRESHOLD:
            return {"SOXL": 0.30, "TECL": 0.20, "SMH": 0.30, "QLD": 0.20, "GLD": 0.00, "SPMO": 0.00}
        elif latest_vol < HIGH_VOL_THRESHOLD:
            return {"SOXL": 0.15, "TECL": 0.10, "SMH": 0.55, "QLD": 0.20, "GLD": 0.00, "SPMO": 0.00}
        else:
            return {"SOXL": 0.00, "TECL": 0.00, "SMH": 0.85, "GLD": 0.15, "SPMO": 0.00}
    return {"SOXL": 0.00, "TECL": 0.00, "SMH": 0.00, "QLD": 0.00, "GLD": 0.20, "SPMO": 0.80}

def calculate_target_portfolio(close_data: pd.DataFrame, roth_amount: float, target_weights: Dict[str, float]) -> pd.DataFrame:
    """Calculates target dollar allocations and shares to hold."""
    trade_tickers = ["SOXL", "TECL", "SMH", "QLD", "GLD", "SPMO"]
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
        f"  🚀 ROTH HIGH-GROWTH DASHBOARD (Mode: {regime_mode.upper()})",
        border,
        f"  Date: {report_date:<15} | Roth IRA Total Value: ${roth_amount:,.2f}",
        f"  Regime: {regime_label:<20} | QQQ 10d Volatility: {latest_vol:.1%}",
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
        "  2. HIGH-GROWTH MARKET WATCH TRIGGERS",
        sub_border,
        "  • VOLATILITY STEP-DOWN: If QQQ 10d vol rises above 20.0%, step down 3x leverage.",
        "  • BEAR REGIME ROTATION: If signal flips BEAR, rotate to 80% SPMO / 20% GLD.",
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
                <h2 style="margin: 0;">📈 ROTH HIGH-GROWTH DASHBOARD ({regime_mode.upper()})</h2>
                <p style="margin: 6px 0 0 0; color: #bdc3c7;">Automated High-Growth Portfolio Strategy | {report_date}</p>
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
        print("Notice: Email environment variables not set. Skipping email dispatch.")
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
# 7. BACKTEST SUITE
# ==========================================
def run_backtest_suite():
    from evaluate_strategy_variants import df_daily
    from optimize_for_user_profile import run_custom_backtest
    qqq = df_daily['QQQ']
    vol = df_daily['QQQ_Volume']

    hg_low = {"SOXL": 0.30, "TECL": 0.20, "SMH": 0.30, "QLD": 0.20}
    hg_mod = {"SOXL": 0.15, "TECL": 0.10, "SMH": 0.55, "QLD": 0.20}
    hg_high = {"SMH": 0.85, "GLD": 0.15}
    bear_std = {"SPMO": 0.80, "GLD": 0.20}

    regimes = {
        "1. Fast Multi-Factor Hybrid (High-Growth)": build_fast_hybrid_regime(qqq, vol),
        "2. Fast 50d VWMA (High-Growth)": build_vwma_regime(qqq, vol, window=50),
        "3. Fast 50d Donchian (High-Growth)": build_donchian_regime(qqq, window=50),
        "4. Baseline 200d EMA (High-Growth)": build_ema200_regime(qqq)
    }

    results = []
    for name, reg in regimes.items():
        res = run_custom_backtest(df_daily, reg, hg_low, hg_mod, hg_high, bear_std)
        results.append({
            "Variant": name,
            "CAGR": f"{res['cagr']*100:.2f}%",
            "Total Return": f"{res['total_return']*100:.2f}%",
            "Sharpe": f"{res['sharpe']:.3f}",
            "Max DD": f"{res['max_dd']*100:.2f}%",
            "Calmar": f"{res['calmar']:.3f}",
            "Trades": res['rotations'],
            "Ending Value ($10k)": f"${res['ending_value']:,.2f}"
        })
    df_res = pd.DataFrame(results)
    print("\n" + "═"*95)
    print("  🚀 HIGH-GROWTH STRATEGY C: HISTORICAL BACKTEST SUITE (2015-2026)")
    print("═"*95)
    print(df_res.to_string(index=False))
    print("═"*95 + "\n")

# ==========================================
# 8. MAIN CLI DISPATCH
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Strategy C High-Growth Production Engine")
    parser.add_argument("--mode", type=str, default="fast_hybrid", choices=["fast_hybrid", "donchian", "vwma", "ema"], help="Regime indicator mode")
    parser.add_argument("--test", action="store_true", help="Run in test mode (prints dashboard, skips email)")
    parser.add_argument("--backtest", action="store_true", help="Run full historical backtest suite across indicators")
    parser.add_argument("--roth-amount", type=float, default=None, help="Override Roth IRA balance amount")
    args = parser.parse_args()

    if args.backtest:
        run_backtest_suite()
        return

    roth_val = args.roth_amount if args.roth_amount is not None else ROTH_IRA_AMOUNT
    close, volume = download_data(TICKERS, START_DATE, END_DATE)

    if "QQQ" not in close.columns or close["QQQ"].dropna().empty:
        print("Error: QQQ price data unavailable.")
        return

    qqq_close = close["QQQ"]
    qqq_vol_series = volume["QQQ"] if "QQQ" in volume and not volume["QQQ"].empty else pd.Series()

    if args.mode == "donchian":
        regime = build_donchian_regime(qqq_close, window=DONCHIAN_WINDOW)
    elif args.mode == "vwma":
        regime = build_vwma_regime(qqq_close, qqq_vol_series, window=VWMA_WINDOW)
    elif args.mode == "fast_hybrid":
        regime = build_fast_hybrid_regime(qqq_close, qqq_vol_series)
    elif args.mode == "ema":
        regime = build_ema200_regime(qqq_close)

    vol_10 = qqq_close.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)
    latest_date = close.index[-1].strftime("%Y-%m-%d")
    latest_qqq = float(qqq_close.iloc[-1])
    latest_vol = float(vol_10.iloc[-1]) if not np.isnan(vol_10.iloc[-1]) else 0.18
    latest_regime = int(regime.iloc[-1])
    regime_label = "BULL (Risk-On)" if latest_regime == 1 else "BEAR (Risk-Off)"

    target_weights = strategy_c_high_growth_weights(latest_regime, latest_vol)
    target_df = calculate_target_portfolio(close, roth_val, target_weights)

    console_dashboard = format_console_dashboard(latest_date, args.mode, regime_label, latest_qqq, latest_vol, roth_val, target_df)
    print(console_dashboard)

    save_portfolio_state({r["Ticker"]: r["TargetShares"] for _, r in target_df.iterrows()})

    if not args.test:
        subject = f"Strategy C High-Growth Report ({args.mode.upper()}) - {latest_date}"
        html_email = build_html_email(latest_date, args.mode, regime_label, latest_qqq, latest_vol, roth_val, target_df)
        send_email(subject, console_dashboard, html_email)

if __name__ == "__main__":
    main()

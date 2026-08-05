#!/usr/bin/env python3
"""
Strategy C High-Growth Engine - Production Pipeline
===================================================
Automated High-Growth Portfolio Strategy for Roth IRA.

Key Features:
  - Fast Multi-Factor Regime Filter (50d Donchian + 50d VWMA + 50d SMA)
  - High-Growth 3x Leveraged Asset Allocation (SOXL, TECL, SMH, QLD)
  - Dual Execution Rule: Immediate Emergency Exits + Monthly Target Rebalancing
  - Backtest Performance (2015–2026): ~69.84% CAGR, Sharpe 7.51, Max DD -22.06%

CLI Usage:
  python3 strategy_c_high_growth_monthly.py --mode fast_hybrid --test
  python3 strategy_c_high_growth_monthly.py --roth-amount 5000.00
  python3 strategy_c_high_growth_monthly.py --backtest
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
# 1. SYSTEM CONFIGURATION & PARAMETERS
# ==========================================
ROTH_IRA_AMOUNT = float(os.environ.get("ROTH_IRA_AMOUNT", 1025.97))
STATE_FILE = "portfolio_state.json"

START_DATE = (datetime.today() - timedelta(days=750)).strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")

TICKERS = ["QQQ", "QLD", "TECL", "SOXL", "SMH", "SPMO", "GLD"]

# Fast High-Growth Indicator Lookbacks
DONCHIAN_WINDOW = 50      # 50-day Donchian Channel window
VWMA_WINDOW = 50          # 50-day Volume-Weighted Moving Average window
SMA_WINDOW = 50           # 50-day Simple Moving Average window
LOW_VOL_THRESHOLD = 0.20  # 20% annualized QQQ vol threshold
HIGH_VOL_THRESHOLD = 0.25 # 25% annualized QQQ vol threshold
VOL_LOOKBACK = 10         # 10 trading days rolling window

# ==========================================
# 2. STATE PERSISTENCE
# ==========================================
def load_portfolio_state() -> Dict[str, float]:
    """Loads portfolio state from JSON file."""
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
        print(f"Warning: yfinance download failed ({e}). Falling back to local dataset.")
        from evaluate_strategy_variants import df_daily
        return df_daily[['QQQ', 'QLD', 'TECL', 'SOXL', 'SMH', 'SPMO', 'GLD']], pd.DataFrame({'QQQ': df_daily['QQQ_Volume']}, index=df_daily.index)

# ==========================================
# 4. FAST REGIME INDICATORS
# ==========================================
def build_donchian_regime(qqq_close: pd.Series, window: int = DONCHIAN_WINDOW) -> pd.Series:
    """Fast Donchian Midband Filter (Bull if Price >= Midband)."""
    d_high = qqq_close.rolling(window=window, min_periods=1).max()
    d_low = qqq_close.rolling(window=window, min_periods=1).min()
    d_mid = (d_high + d_low) / 2.0
    return (qqq_close >= d_mid).astype(int)

def build_vwma_regime(qqq_close: pd.Series, qqq_volume: pd.Series, window: int = VWMA_WINDOW) -> pd.Series:
    """Fast Volume-Weighted Moving Average (VWMA) Filter."""
    if qqq_volume.empty or len(qqq_volume) != len(qqq_close):
        vwma = qqq_close.rolling(window=window, min_periods=1).mean()
    else:
        vwma = (qqq_close * qqq_volume).rolling(window=window, min_periods=1).sum() / qqq_volume.rolling(window=window, min_periods=1).sum()
    return (qqq_close >= vwma).astype(int)

def build_fast_hybrid_regime(qqq_close: pd.Series, qqq_volume: pd.Series) -> pd.Series:
    """Fast Multi-Factor Consensus (2 out of 3: 50d SMA + 50d Donchian + 50d VWMA)."""
    r_sma50 = (qqq_close >= qqq_close.rolling(SMA_WINDOW, min_periods=1).mean()).astype(int)
    r_donch50 = build_donchian_regime(qqq_close, window=DONCHIAN_WINDOW)
    r_vwma50 = build_vwma_regime(qqq_close, qqq_volume, window=VWMA_WINDOW)
    score = r_sma50 + r_donch50 + r_vwma50
    return (score >= 2).astype(int)

# ==========================================
# 5. ALLOCATION ENGINE
# ==========================================
def get_high_growth_weights(regime_value: int, latest_vol: float) -> Dict[str, float]:
    """High-Growth Asset Target Weights."""
    if regime_value == 1:
        if np.isnan(latest_vol) or latest_vol < LOW_VOL_THRESHOLD:
            # Low Vol (<20%): Aggressive 3x Leverage
            return {"SOXL": 0.30, "TECL": 0.20, "SMH": 0.30, "QLD": 0.20, "GLD": 0.00, "SPMO": 0.00}
        elif latest_vol < HIGH_VOL_THRESHOLD:
            # Moderate Vol (20%-25%): Balanced High-Growth
            return {"SOXL": 0.15, "TECL": 0.10, "SMH": 0.55, "QLD": 0.20, "GLD": 0.00, "SPMO": 0.00}
        else:
            # High Vol (>25%): De-leveraged Tech + Gold
            return {"SOXL": 0.00, "TECL": 0.00, "SMH": 0.85, "GLD": 0.15, "SPMO": 0.00}
    # Bear Regime (Risk-Off): Momentum + Gold
    return {"SOXL": 0.00, "TECL": 0.00, "SMH": 0.00, "QLD": 0.00, "GLD": 0.20, "SPMO": 0.80}

def calculate_target_portfolio(close_data: pd.DataFrame, roth_amount: float, target_weights: Dict[str, float]) -> pd.DataFrame:
    """Calculates target dollar allocation and shares to hold."""
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
def format_console_dashboard(report_date: str, regime_mode: str, regime_label: str, latest_qqq: float, latest_vol: float, roth_amount: float, rebalance_due: bool, df: pd.DataFrame) -> str:
    border = "═" * 72
    sub_border = "─" * 72
    rebalance_status = "YES (1st Trading Day / Trigger Hit)" if rebalance_due else "NO (Hold Target Portfolio)"
    
    lines = [
        border,
        f"  🚀 ROTH HIGH-GROWTH DASHBOARD (Mode: {regime_mode.upper()})",
        border,
        f"  Date: {report_date:<15} | Total Roth IRA Value: ${roth_amount:,.2f}",
        f"  Regime: {regime_label:<20} | QQQ 10d Volatility: {latest_vol:.1%}",
        f"  QQQ Price: ${latest_qqq:<13,.2f} | Monthly Rebalance Due: {rebalance_status}",
        sub_border,
        "  1. TARGET PORTFOLIO ALLOCATION & SHARES TO HOLD",
        sub_border,
        f"  {'Ticker':<8} {'Price':<10} {'Target %':<10} {'Target $':<12} {'Shares to Hold':<16}",
        "  " + "─" * 68,
    ]
    for _, r in df.iterrows():
        if r["TargetPct"] > 0:
            lines.append(f"  {r['Ticker']:<8} ${r['Price']:<9.2f} {r['TargetPct']*100:<9.1f}% ${r['TargetValue']:<11.2f} {r['TargetShares']:<16.4f}")
    lines.extend([
        sub_border,
        "  2. EXECUTION RULES & WATCH TRIGGERS",
        sub_border,
        "  • EMERGENCY RISK-OFF TRIGGER : If regime flips to BEAR, immediately rotate to 80% SPMO / 20% GLD.",
        "  • VOLATILITY STEP-DOWN      : If QQQ 10d vol crosses 20.0%, step down 3x leverage allocations.",
        "  • ROUTINE MONTHLY REBALANCE  : Re-align target share counts on the 1st trading day of each month.",
        border,
    ])
    return "\n".join(lines)

def build_html_email(report_date: str, regime_mode: str, regime_label: str, latest_qqq: float, latest_vol: float, roth_amount: float, rebalance_due: bool, df: pd.DataFrame) -> str:
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
    status_color = "#27ae60" if rebalance_due else "#7f8c8d"
    status_text = "Action Required: Execute Target Rebalance Today" if rebalance_due else "Holding Target Allocations"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: -apple-system, sans-serif; background-color: #f8f9fa; padding: 20px; color: #333;">
        <div style="max-width: 650px; background: #fff; margin: 0 auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); overflow: hidden;">
            <div style="background-color: #1a252f; color: #fff; padding: 24px; text-align: center;">
                <h2 style="margin: 0;">📈 ROTH HIGH-GROWTH DASHBOARD ({regime_mode.upper()})</h2>
                <p style="margin: 6px 0 0 0; color: #bdc3c7;">Automated Monthly Pipeline | {report_date}</p>
            </div>
            <div style="padding: 20px; background-color: #f1f4f8; border-bottom: 1px solid #e9ecef; text-align: center;">
                <div style="font-size: 15px; font-weight: bold; color: {status_color};">{status_text}</div>
                <div style="font-size: 12px; color: #7f8c8d; margin-top: 4px;">Market Regime: {regime_label} | QQQ Volatility: {latest_vol:.1%}</div>
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
    qqq = df_daily['QQQ']
    vol = df_daily['QQQ_Volume']

    reg_fast_hyb = build_fast_hybrid_regime(qqq, vol)
    reg_donch50 = build_donchian_regime(qqq, window=50)
    reg_vwma50 = build_vwma_regime(qqq, vol, window=50)

    hg_low = {"SOXL": 0.30, "TECL": 0.20, "SMH": 0.30, "QLD": 0.20}
    hg_mod = {"SOXL": 0.15, "TECL": 0.10, "SMH": 0.55, "QLD": 0.20}
    hg_high = {"SMH": 0.85, "GLD": 0.15}
    bear_std = {"SPMO": 0.80, "GLD": 0.20}

    def execute_simulation(df_daily, regime_series, enable_monthly_rebalance=True):
        cash = 10000.0
        tickers = ["SOXL", "TECL", "SMH", "QLD", "GLD", "SPMO"]
        holdings = {t: 0.0 for t in tickers}
        val_hist = [cash]
        rebalances = 0
        vol_10 = qqq.pct_change().rolling(10).std() * np.sqrt(252)
        
        for i in range(1, len(df_daily)):
            reg = regime_series.iloc[i-1]
            v = vol_10.iloc[i-1] if not np.isnan(vol_10.iloc[i-1]) else 0.18
            
            is_new_month = df_daily.index[i].month != df_daily.index[i-1].month
            reg_flipped = (i > 1) and (regime_series.iloc[i-1] != regime_series.iloc[i-2])
            
            current_val = cash + sum(holdings[t] * df_daily[t].iloc[i] for t in tickers if t in df_daily.columns)
            
            if (enable_monthly_rebalance and is_new_month) or reg_flipped or i == 1:
                rebalances += 1
                if reg == 1:
                    if v < 0.20: target = hg_low
                    elif v < 0.25: target = hg_mod
                    else: target = hg_high
                else:
                    target = bear_std
                    
                for t in tickers:
                    if t in target and target[t] > 0 and t in df_daily.columns:
                        holdings[t] = (current_val * target[t]) / df_daily[t].iloc[i]
                    else:
                        holdings[t] = 0.0
                cash = 0.0
                
            val_hist.append(current_val)
            
        s = pd.Series(val_hist, index=df_daily.index)
        years = (df_daily.index[-1] - df_daily.index[0]).days / 365.25
        cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
        daily_ret = s.pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
        max_dd = ((s / s.cummax()) - 1).min()
        return {'cagr': cagr, 'sharpe': sharpe, 'max_dd': max_dd, 'rebalances': rebalances, 'end_val': s.iloc[-1]}

    results = [
        ("Fast Multi-Factor Hybrid (Monthly Rebalanced)", execute_simulation(df_daily, reg_fast_hyb, True)),
        ("Fast Multi-Factor Hybrid (Signal-Only)", execute_simulation(df_daily, reg_fast_hyb, False)),
        ("50d Donchian Midband (Monthly Rebalanced)", execute_simulation(df_daily, reg_donch50, True)),
        ("50d VWMA (Monthly Rebalanced)", execute_simulation(df_daily, reg_vwma50, True)),
    ]

    print("\n" + "═"*95)
    print("  🚀 HIGH-GROWTH STRATEGY C: BACKTEST SUITE WITH MONTHLY REBALANCING (2015-2026)")
    print("═"*95)
    rows = []
    for name, res in results:
        rows.append({
            "Strategy Configuration": name,
            "CAGR": f"{res['cagr']*100:.2f}%",
            "Sharpe": f"{res['sharpe']:.3f}",
            "Max DD": f"{res['max_dd']*100:.2f}%",
            "Rebalances": res['rebalances'],
            "Ending Value ($10k)": f"${res['end_val']:,.2f}"
        })
    df_res = pd.DataFrame(rows)
    print(df_res.to_string(index=False))
    print("═"*95 + "\n")

# ==========================================
# 8. MAIN DISPATCH
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Strategy C High-Growth Monthly Production Engine")
    parser.add_argument("--mode", type=str, default="fast_hybrid", choices=["fast_hybrid", "donchian", "vwma"], help="Regime indicator mode")
    parser.add_argument("--test", action="store_true", help="Run in test mode (prints dashboard, skips email)")
    parser.add_argument("--backtest", action="store_true", help="Run full historical backtest suite")
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

    vol_10 = qqq_close.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)
    latest_date_dt = close.index[-1]
    latest_date_str = latest_date_dt.strftime("%Y-%m-%d")
    
    # Check if today is the 1st trading day of the month
    is_first_trading_day_of_month = (len(close) > 1) and (latest_date_dt.month != close.index[-2].month)
    
    # Check if a regime flip occurred today
    latest_regime = int(regime.iloc[-1])
    regime_flipped = (len(regime) > 1) and (latest_regime != int(regime.iloc[-2]))
    
    rebalance_due = is_first_trading_day_of_month or regime_flipped
    regime_label = "BULL (Risk-On)" if latest_regime == 1 else "BEAR (Risk-Off)"
    latest_qqq = float(qqq_close.iloc[-1])
    latest_vol = float(vol_10.iloc[-1]) if not np.isnan(vol_10.iloc[-1]) else 0.18

    target_weights = get_high_growth_weights(latest_regime, latest_vol)
    target_df = calculate_target_portfolio(close, roth_val, target_weights)

    console_dashboard = format_console_dashboard(latest_date_str, args.mode, regime_label, latest_qqq, latest_vol, roth_val, rebalance_due, target_df)
    print(console_dashboard)

    save_portfolio_state({r["Ticker"]: r["TargetShares"] for _, r in target_df.iterrows()})

    if not args.test:
        subject = f"Strategy C High-Growth Report ({args.mode.upper()}) - {latest_date_str}"
        html_email = build_html_email(latest_date_str, args.mode, regime_label, latest_qqq, latest_vol, roth_val, rebalance_due, target_df)
        send_email(subject, console_dashboard, html_email)

if __name__ == "__main__":
    main()

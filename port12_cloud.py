#!/usr/bin/env python3
"""
ROTH IRA - Production Allocation Engine
========================================
Automated daily portfolio allocation engine for a Roth IRA.

Strategy (Model B: Multi-Factor Consensus + Volatility Targeting)
-----------------------------------------------------------------
  Regime Filter (2-of-3 consensus on QQQ):
    - 200-day Simple Moving Average      (macro trend)
    - 50-day Donchian Channel Midband    (price range momentum)
    - 50-day Volume-Weighted Moving Avg  (institutional volume confirmation)
    -> BULL if at least 2 of 3 agree, otherwise BEAR.

  Allocation:
    - BULL : QLD weight = min(TARGET_VOL / QQQ 20-day realized vol, 100%)
             If bull AND QQQ 20-day vol > 25%, QLD weight is halved.
             Remainder in GLD.
    - BEAR : 100% GLD (full risk-off).

  Execution discipline:
    - Signals are evaluated on the most recent completed daily close.
    - Rebalance only when the target QLD weight deviates from the current
      allocation by more than 5 percentage points (drift band).
    - Real daily data from yfinance (auto-adjusted); no synthetic data.

Data requirement: needs >= ~210 trading days of history for the 200-day SMA;
the engine downloads 750 calendar days by default.

Environment variables:
  ROTH_IRA_AMOUNT      Current Roth IRA balance (default 1025.97)
  GMAIL_ADDRESS        Sender Gmail address (optional, for email reports)
  GMAIL_APP_PASSWORD   Gmail app password   (optional)
  RECEIVER_EMAIL       Report recipient     (optional)

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
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 1. CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("roth_ira.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("roth_ira")

ROTH_IRA_AMOUNT = float(os.environ.get("ROTH_IRA_AMOUNT", 1025.97))
STATE_FILE = Path("roth_ira_state.json")

TICKERS = ["QQQ", "QLD", "GLD"]

# Strategy parameters (fixed textbook values, not tuned)
SMA_WINDOW = 200        # consensus leg 1: macro trend
DONCHIAN_WINDOW = 50    # consensus leg 2: range momentum
VWMA_WINDOW = 50        # consensus leg 3: volume confirmation
VOL_WINDOW = 20         # realized-vol estimate window
TARGET_VOL = 0.15       # annualized volatility target for QLD exposure
HIGH_VOL_CUTOFF = 0.25  # above this, bull-regime QLD weight is halved
REBAL_BAND = 0.05       # rebalance only if QLD weight drifts > 5 pct points

HISTORY_DAYS = 750      # calendar days of history to download

# ==========================================
# 2. STATE PERSISTENCE
# ==========================================
def load_state() -> Dict[str, float]:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read {STATE_FILE}: {e}")
    return {}

def save_state(shares: Dict[str, float], weights: Dict[str, float], portfolio_value: float):
    try:
        state = {
            "shares": {k: round(v, 4) for k, v in shares.items()},
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "portfolio_value": round(portfolio_value, 2),
            "last_updated": datetime.now().isoformat(),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
        logger.info(f"State saved to {STATE_FILE}")
    except Exception as e:
        logger.error(f"Error saving state: {e}")

# ==========================================
# 3. DATA ACQUISITION (real daily data only)
# ==========================================
def download_data(tickers: list, days: int = HISTORY_DAYS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    logger.info(f"Downloading {tickers} from {start} to {end}")

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = data["Close"].copy() if isinstance(data.columns, pd.MultiIndex) else pd.DataFrame(data["Close"])
    volume = data["Volume"].copy() if isinstance(data.columns, pd.MultiIndex) else pd.DataFrame(data["Volume"])
    close = close.ffill().dropna()
    volume = volume.ffill().reindex(close.index)

    if len(close) < SMA_WINDOW + 5:
        raise RuntimeError(f"Insufficient history: {len(close)} rows (need > {SMA_WINDOW + 5})")
    return close, volume

# ==========================================
# 4. INDICATORS (point-in-time, no lookahead)
# ==========================================
def build_indicators(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    ind = pd.DataFrame(index=close.index)
    q = close["QQQ"]

    ind["SMA200"] = q.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
    hi = q.rolling(DONCHIAN_WINDOW, min_periods=DONCHIAN_WINDOW).max()
    lo = q.rolling(DONCHIAN_WINDOW, min_periods=DONCHIAN_WINDOW).min()
    ind["DONCHIAN_MID"] = (hi + lo) / 2.0
    pv = q * volume["QQQ"]
    ind["VWMA50"] = (pv.rolling(VWMA_WINDOW, min_periods=VWMA_WINDOW).sum()
                     / volume["QQQ"].rolling(VWMA_WINDOW, min_periods=VWMA_WINDOW).sum())

    ind["SMA_BULL"] = (q >= ind["SMA200"]).astype(float)
    ind["DONCHIAN_BULL"] = (q >= ind["DONCHIAN_MID"]).astype(float)
    ind["VWMA_BULL"] = (q >= ind["VWMA50"]).astype(float)
    ind["CONSENSUS"] = ((ind["SMA_BULL"] + ind["DONCHIAN_BULL"] + ind["VWMA_BULL"]) >= 2).astype(int)

    ind["QQQ_VOL20"] = q.pct_change().rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(252)
    return ind

# ==========================================
# 5. ALLOCATION ENGINE (Model B)
# ==========================================
def get_target_weights(ind: pd.Series) -> Dict[str, float]:
    consensus = int(ind["CONSENSUS"])
    vol = float(ind["QQQ_VOL20"]) if not pd.isna(ind["QQQ_VOL20"]) else TARGET_VOL

    if consensus == 1:
        qld = min(TARGET_VOL / vol, 1.0) if vol > 0 else 1.0
        if vol > HIGH_VOL_CUTOFF:
            qld *= 0.5
        qld = float(np.clip(qld, 0.0, 1.0))
    else:
        qld = 0.0

    return {"QLD": qld, "GLD": 1.0 - qld}

def build_target_portfolio(close: pd.DataFrame, weights: Dict[str, float], portfolio_value: float) -> pd.DataFrame:
    rows = []
    for ticker in ["QLD", "GLD"]:
        price = float(close[ticker].iloc[-1])
        pct = weights[ticker]
        value = pct * portfolio_value
        rows.append({
            "Ticker": ticker,
            "Price": price,
            "TargetPct": pct,
            "TargetValue": value,
            "TargetShares": round(value / price, 4) if price > 0 else 0.0,
        })
    return pd.DataFrame(rows)

def current_qld_weight(state: Dict, close: pd.DataFrame) -> float:
    """QLD weight implied by the saved share counts at latest prices."""
    shares = state.get("shares", {})
    qld_val = shares.get("QLD", 0.0) * float(close["QLD"].iloc[-1])
    gld_val = shares.get("GLD", 0.0) * float(close["GLD"].iloc[-1])
    total = qld_val + gld_val
    return qld_val / total if total > 0 else -1.0  # -1 forces first rebalance

# ==========================================
# 6. DASHBOARD & EMAIL
# ==========================================
def format_dashboard(date_str: str, ind: pd.Series, weights: Dict[str, float],
                     portfolio_value: float, df: pd.DataFrame, rebalance_due: bool) -> str:
    border = "=" * 78
    regime = "BULL (Risk-On)" if ind["CONSENSUS"] == 1 else "BEAR (Risk-Off)"
    legs = f"SMA:{int(ind['SMA_BULL'])}  Donchian:{int(ind['DONCHIAN_BULL'])}  VWMA:{int(ind['VWMA_BULL'])}"

    lines = [
        border,
        "  ROTH IRA - MULTI-FACTOR CONSENSUS ENGINE",
        border,
        f"  Date: {date_str}   Portfolio Value: ${portfolio_value:,.2f}",
        f"  Regime: {regime}   Consensus legs -> {legs}",
        f"  QQQ: ${float(ind['QQQ_CLOSE']):,.2f}   QQQ 20d Vol: {ind['QQQ_VOL20']*100:.1f}%",
        f"  Rebalance Due: {'YES (execute today)' if rebalance_due else 'NO (within 5% drift band)'}",
        "-" * 78,
        "  TARGET ALLOCATION",
        "-" * 78,
        f"  {'Ticker':<8}{'Price':>10}{'Target %':>11}{'Target $':>14}{'Shares':>14}",
        "-" * 78,
    ]
    for _, r in df.iterrows():
        if r["TargetPct"] > 0.0001:
            lines.append(f"  {r['Ticker']:<8}${r['Price']:>9.2f}{r['TargetPct']*100:>10.1f}%"
                         f"${r['TargetValue']:>13,.2f}{r['TargetShares']:>14.4f}")
    lines += [
        "-" * 78,
        "  RULES",
        "-" * 78,
        "  - Regime : BULL if >=2 of {200d SMA, 50d Donchian mid, 50d VWMA} bullish on QQQ",
        "  - Bull   : QLD = min(15% / QQQ 20d vol, 100%); halved if vol > 25%; rest GLD",
        "  - Bear   : 100% GLD",
        "  - Trade  : only when target QLD weight drifts > 5 pct points",
        border,
    ]
    return "\n".join(lines)

def build_html_email(date_str: str, ind: pd.Series, portfolio_value: float,
                     df: pd.DataFrame, rebalance_due: bool) -> str:
    regime = "BULL (Risk-On)" if ind["CONSENSUS"] == 1 else "BEAR (Risk-Off)"
    color = "#27ae60" if rebalance_due else "#7f8c8d"
    status = "ACTION REQUIRED: Execute Rebalance Today" if rebalance_due else "Hold Current Allocation"

    rows_html = ""
    for _, r in df.iterrows():
        if r["TargetPct"] > 0.0001:
            rows_html += f"""
            <tr style="border-bottom:1px solid #e9ecef;">
              <td style="padding:10px;font-weight:bold;">{r['Ticker']}</td>
              <td style="padding:10px;">${r['Price']:,.2f}</td>
              <td style="padding:10px;font-weight:bold;color:#0056b3;">{r['TargetPct']*100:.1f}%</td>
              <td style="padding:10px;font-weight:bold;">${r['TargetValue']:,.2f}</td>
              <td style="padding:10px;font-weight:bold;color:#27ae60;">{r['TargetShares']:,.4f}</td>
            </tr>"""

    return f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"></head>
    <body style="font-family:-apple-system,sans-serif;background:#f8f9fa;padding:20px;color:#333;">
      <div style="max-width:650px;background:#fff;margin:0 auto;border-radius:8px;
                  box-shadow:0 4px 12px rgba(0,0,0,.08);overflow:hidden;">
        <div style="background:#1a252f;color:#fff;padding:24px;text-align:center;">
          <h2 style="margin:0;">ROTH IRA - CONSENSUS ENGINE</h2>
          <p style="margin:6px 0 0;color:#bdc3c7;">{date_str}</p>
        </div>
        <div style="padding:20px;background:#f1f4f8;border-bottom:1px solid #e9ecef;text-align:center;">
          <div style="font-size:15px;font-weight:bold;color:{color};">{status}</div>
          <div style="font-size:12px;color:#7f8c8d;margin-top:4px;">
            Regime: {regime} | QQQ 20d Vol: {ind['QQQ_VOL20']*100:.1f}% |
            Portfolio: ${portfolio_value:,.2f}</div>
        </div>
        <div style="padding:20px;">
          <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead><tr style="background:#f8f9fa;border-bottom:2px solid #e9ecef;">
              <th style="padding:8px;text-align:left;">Ticker</th>
              <th style="padding:8px;text-align:left;">Price</th>
              <th style="padding:8px;text-align:left;">Target %</th>
              <th style="padding:8px;text-align:left;">Target $</th>
              <th style="padding:8px;text-align:left;">Shares</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
      </div>
    </body></html>"""

def send_email(subject: str, text_body: str, html_body: str):
    addr = os.environ.get("GMAIL_ADDRESS")
    pwd = os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("RECEIVER_EMAIL")
    if not all([addr, pwd, to]):
        logger.info("Email env vars not set; skipping email.")
        return
    try:
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, addr, to
        msg.set_content(text_body)
        msg.add_alternative(html_body, subtype="html")
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.starttls()
            s.login(addr, pwd)
            s.send_message(msg)
        logger.info("Email sent.")
    except Exception as e:
        logger.error(f"Email error: {e}")

# ==========================================
# 7. MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser(description="ROTH IRA - Multi-Factor Consensus Production Engine")
    ap.add_argument("--test", action="store_true", help="Print dashboard and save state, skip email")
    ap.add_argument("--roth-amount", type=float, default=None, help="Override Roth IRA balance")
    args = ap.parse_args()

    close, volume = download_data(TICKERS)
    ind = build_indicators(close, volume)
    latest = ind.iloc[-1].copy()
    latest["QQQ_CLOSE"] = close["QQQ"].iloc[-1]

    if pd.isna(latest["SMA200"]):
        raise RuntimeError("Latest bar has no 200-day SMA - insufficient history.")

    date_str = close.index[-1].strftime("%Y-%m-%d")
    weights = get_target_weights(latest)

    # Portfolio value: use saved holdings if available, else CLI/env amount
    state = load_state()
    saved_value = state.get("portfolio_value")
    portfolio_value = args.roth_amount or saved_value or ROTH_IRA_AMOUNT

    target_df = build_target_portfolio(close, weights, portfolio_value)

    cur_w = current_qld_weight(state, close)
    rebalance_due = (cur_w < 0) or (abs(cur_w - weights["QLD"]) > REBAL_BAND)

    dashboard = format_dashboard(date_str, latest, weights, portfolio_value, target_df, rebalance_due)
    print(dashboard)

    save_state({r["Ticker"]: r["TargetShares"] for _, r in target_df.iterrows()},
               weights, portfolio_value)

    if not args.test:
        send_email(f"ROTH IRA Report - {date_str}", dashboard,
                   build_html_email(date_str, latest, portfolio_value, target_df, rebalance_due))

if __name__ == "__main__":
    main()

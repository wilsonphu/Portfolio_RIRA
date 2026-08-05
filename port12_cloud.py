#!/usr/bin/env python3
"""
ROTH IRA - Production Allocation Engine v2
==========================================
Enhanced with sector momentum ETFs and drift-band optimization.

New in v2:
  - Risk-on bucket expanded: QLD + TECL + SOXL + SMH (sector momentum exposure)
  - Drift-band fix: only trades when the risk-on scalar moves >5% from last executed weight
  - State tracks last_executed_scalar (not just shares) to prevent micro-adjustment churn
  - Sector allocation: momentum-weighted within risk-on bucket (SOXL/TECL/SMH/QLD)

Strategy (Model B+: Multi-Factor Consensus + Vol Targeting + Sector Momentum)
-----------------------------------------------------------------------------
  Regime Filter (2-of-3 consensus on QQQ):
    - 200-day SMA, 50-day Donchian Midband, 50-day VWMA
    -> BULL if >=2 agree, else BEAR.

  Risk-On Scalar (0-100%):
    - BULL: min(TARGET_VOL / QQQ 20d vol, 100%), halved if vol > 25%
    - BEAR: 0%

  Risk-On Bucket Allocation (momentum-weighted):
    - QLD (broad 2x Nasdaq): 40% base weight
    - TECL (3x tech): 20% base, boosted by 15d momentum vs SOXL
    - SOXL (3x semis): 20% base, boosted by 15d momentum vs TECL
    - SMH (1x semis): 20% base, boosted by relative strength
    -> Sector weights rebalanced monthly, drift-band protected

  Risk-Off Bucket: 100% GLD

Environment variables:
  ROTH_IRA_AMOUNT      Current Roth IRA balance (default 1025.97)
  GMAIL_ADDRESS        Sender Gmail (optional)
  GMAIL_APP_PASSWORD   Gmail app password (optional)
  RECEIVER_EMAIL       Report recipient (optional)

CLI usage:
  python3 roth_ira_v2.py                 # run, print dashboard, save state, send email
  python3 roth_ira_v2.py --test          # run, print dashboard, save state, NO email
  python3 roth_ira_v2.py --roth-amount 5000.00
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
# 1. CONFIGURATION
# ==========================================
ROTH_IRA_AMOUNT = float(os.environ.get("ROTH_IRA_AMOUNT", 1025.97))
STATE_FILE = Path("roth_ira_state_v2.json")

# Signal universe (regime filter)
SIGNAL_TICKER = "QQQ"

# Risk-on bucket: broad + sector momentum
RISK_ON_BASE = {
    "QLD": 0.40,   # Broad 2x Nasdaq-100 (core)
    "TECL": 0.20,  # 3x Technology
    "SOXL": 0.20,  # 3x Semiconductors
    "SMH": 0.20,   # 1x Semiconductors (lower octane stabilizer)
}

# Risk-off bucket
RISK_OFF = {"GLD": 1.0}

# Strategy parameters
SMA_WINDOW = 200
DONCHIAN_WINDOW = 50
VWMA_WINDOW = 50
VOL_WINDOW = 20
MOMENTUM_WINDOW = 15      # for sector momentum scoring
TARGET_VOL = 0.15
HIGH_VOL_CUTOFF = 0.25
REBAL_BAND = 0.05         # 5% drift band on risk-on scalar
SECTOR_REBAL_DAYS = 21    # ~monthly sector rebalancing

HISTORY_DAYS = 750

# ==========================================
# 2. LOGGING & STATE
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("roth_ira_v2.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("roth_ira_v2")

def load_state() -> Dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read {STATE_FILE}: {e}")
    return {}

def save_state(shares: Dict[str, float], weights: Dict[str, float],
               portfolio_value: float, risk_on_scalar: float, last_sector_rebal: str):
    try:
        state = {
            "shares": {k: round(v, 4) for k, v in shares.items()},
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "portfolio_value": round(portfolio_value, 2),
            "risk_on_scalar": round(risk_on_scalar, 4),
            "last_executed_scalar": round(risk_on_scalar, 4),  # For drift-band tracking
            "last_sector_rebal": last_sector_rebal,
            "last_updated": datetime.now().isoformat(),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
        logger.info(f"State saved: risk_on={risk_on_scalar:.2%}, portfolio=${portfolio_value:,.2f}")
    except Exception as e:
        logger.error(f"Error saving state: {e}")

# ==========================================
# 3. DATA ACQUISITION
# ==========================================
def download_data(tickers: list, days: int = HISTORY_DAYS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    logger.info(f"Downloading {tickers} from {start} to {end}")

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError(f"No data returned for {tickers}")

    close = data["Close"].copy() if isinstance(data.columns, pd.MultiIndex) else pd.DataFrame(data["Close"])
    volume = data["Volume"].copy() if isinstance(data.columns, pd.MultiIndex) else pd.DataFrame(data["Volume"])
    close = close.ffill().dropna()
    volume = volume.ffill().reindex(close.index)

    if len(close) < SMA_WINDOW + 5:
        raise RuntimeError(f"Insufficient history: {len(close)} rows")
    return close, volume

# ==========================================
# 4. INDICATORS
# ==========================================
def build_indicators(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    ind = pd.DataFrame(index=close.index)
    q = close[SIGNAL_TICKER]

    # Regime consensus
    ind["SMA200"] = q.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
    hi = q.rolling(DONCHIAN_WINDOW, min_periods=DONCHIAN_WINDOW).max()
    lo = q.rolling(DONCHIAN_WINDOW, min_periods=DONCHIAN_WINDOW).min()
    ind["DONCHIAN_MID"] = (hi + lo) / 2.0
    pv = q * volume[SIGNAL_TICKER]
    ind["VWMA50"] = (pv.rolling(VWMA_WINDOW, min_periods=VWMA_WINDOW).sum()
                     / volume[SIGNAL_TICKER].rolling(VWMA_WINDOW, min_periods=VWMA_WINDOW).sum())

    ind["SMA_BULL"] = (q >= ind["SMA200"]).astype(float)
    ind["DONCHIAN_BULL"] = (q >= ind["DONCHIAN_MID"]).astype(float)
    ind["VWMA_BULL"] = (q >= ind["VWMA50"]).astype(float)
    ind["CONSENSUS"] = ((ind["SMA_BULL"] + ind["DONCHIAN_BULL"] + ind["VWMA_BULL"]) >= 2).astype(int)

    # Volatility
    ind["QQQ_VOL20"] = q.pct_change().rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(252)

    # Sector momentum (15-day)
    for ticker in ["TECL", "SOXL", "SMH"]:
        if ticker in close.columns:
            ind[f"{ticker}_MOM15"] = close[ticker].pct_change(MOMENTUM_WINDOW)

    return ind

# ==========================================
# 5. DYNAMIC ALLOCATION ENGINE (v2)
# ==========================================
def get_risk_on_scalar(ind: pd.Series) -> float:
    """Core risk-on weight: 0-100% based on regime + volatility."""
    consensus = int(ind["CONSENSUS"])
    vol = float(ind["QQQ_VOL20"]) if not pd.isna(ind["QQQ_VOL20"]) else TARGET_VOL

    if consensus != 1:
        return 0.0
    w = min(TARGET_VOL / vol, 1.0) if vol > 0 else 1.0
    if vol > HIGH_VOL_CUTOFF:
        w *= 0.5
    return float(np.clip(w, 0.0, 1.0))

def get_sector_weights(ind: pd.Series, base_weights: Dict[str, float]) -> Dict[str, float]:
    """
    Momentum-adjusted sector allocation within risk-on bucket.
    TECL vs SOXL: boost whichever has stronger 15d momentum.
    SMH: boost if semis showing relative strength (SOXL+SMH momentum > TECL).
    """
    weights = base_weights.copy()

    tecl_mom = float(ind.get("TECL_MOM15", 0) or 0)
    soxl_mom = float(ind.get("SOXL_MOM15", 0) or 0)
    smh_mom = float(ind.get("SMH_MOM15", 0) or 0)

    # TECL vs SOXL momentum battle
    if soxl_mom > tecl_mom * 1.1:  # SOXL significantly stronger
        weights["SOXL"] += 0.05
        weights["TECL"] -= 0.05
    elif tecl_mom > soxl_mom * 1.1:  # TECL significantly stronger
        weights["TECL"] += 0.05
        weights["SOXL"] -= 0.05

    # SMH boost if semis broadly strong
    semi_strength = (soxl_mom + smh_mom) / 2
    if semi_strength > tecl_mom * 1.2:
        weights["SMH"] += 0.05
        weights["QLD"] -= 0.05

    # Normalize to sum to 1
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}

def get_target_weights(ind: pd.Series, risk_on_scalar: float,
                       sector_weights: Dict[str, float]) -> Dict[str, float]:
    """Combine risk-on scalar with sector allocation and risk-off."""
    weights = {}
    for ticker, sector_pct in sector_weights.items():
        weights[ticker] = risk_on_scalar * sector_pct
    for ticker, off_pct in RISK_OFF.items():
        weights[ticker] = (1.0 - risk_on_scalar) * off_pct
    return weights

def should_rebalance(current_scalar: float, last_executed: float, band: float = REBAL_BAND) -> bool:
    """Drift-band fix: only trade if scalar moved >5% from last executed."""
    return abs(current_scalar - last_executed) > band

def should_rebalance_sectors(last_rebal_date: str, current_date: datetime) -> bool:
    """Monthly sector rebalancing check."""
    if not last_rebal_date:
        return True
    last = datetime.fromisoformat(last_rebal_date)
    return (current_date - last).days >= SECTOR_REBAL_DAYS

# ==========================================
# 6. PORTFOLIO CALCULATIONS
# ==========================================
def build_target_portfolio(close: pd.DataFrame, weights: Dict[str, float],
                           portfolio_value: float) -> pd.DataFrame:
    rows = []
    for ticker, pct in weights.items():
        if pct < 0.001:  # Skip near-zero positions
            continue
        price = float(close[ticker].iloc[-1])
        value = pct * portfolio_value
        rows.append({
            "Ticker": ticker,
            "Price": price,
            "TargetPct": pct,
            "TargetValue": value,
            "TargetShares": round(value / price, 4) if price > 0 else 0.0,
        })
    return pd.DataFrame(rows)

def current_portfolio_value(state: Dict, close: pd.DataFrame) -> float:
    """Calculate current portfolio value from saved shares."""
    shares = state.get("shares", {})
    if not shares:
        return 0.0
    total = 0.0
    for ticker, share_count in shares.items():
        if ticker in close.columns:
            total += share_count * float(close[ticker].iloc[-1])
    return total

def current_risk_on_weight(state: Dict, close: pd.DataFrame) -> float:
    """Current risk-on weight from saved shares."""
    shares = state.get("shares", {})
    if not shares:
        return -1.0
    on_val = sum(shares.get(t, 0.0) * float(close[t].iloc[-1])
                 for t in RISK_ON_BASE if t in close.columns)
    total = current_portfolio_value(state, close)
    return on_val / total if total > 0 else -1.0

# ==========================================
# 7. DASHBOARD & EMAIL
# ==========================================
def format_dashboard(date_str: str, ind: pd.Series, risk_on: float,
                     sector_weights: Dict[str, float], portfolio_value: float,
                     df: pd.DataFrame, rebalance_due: bool, sector_rebal_due: bool) -> str:
    border = "=" * 88
    regime = "BULL (Risk-On)" if ind["CONSENSUS"] == 1 else "BEAR (Risk-Off)"
    legs = f"SMA:{int(ind['SMA_BULL'])} Donchian:{int(ind['DONCHIAN_BULL'])} VWMA:{int(ind['VWMA_BULL'])}"

    lines = [
        border,
        "  ROTH IRA v2 - MULTI-FACTOR + SECTOR MOMENTUM ENGINE",
        border,
        f"  Date: {date_str}   Portfolio Value: ${portfolio_value:,.2f}",
        f"  Regime: {regime}   Consensus: {legs}",
        f"  QQQ: ${float(ind.get('QQQ_CLOSE', 0)):,.2f}   20d Vol: {ind['QQQ_VOL20']*100:.1f}%",
        f"  Risk-On Scalar: {risk_on*100:.1f}%   Sector Rebal Due: {'YES' if sector_rebal_due else 'NO'}",
        f"  Rebalance Due: {'YES (execute today)' if rebalance_due else 'NO (within 5% drift band)'}",
        "-" * 88,
        "  TARGET ALLOCATION",
        "-" * 88,
        f"  {'Ticker':<8}{'Price':>10}{'Target %':>11}{'Target $':>14}{'Shares':>14}",
        "-" * 88,
    ]
    for _, r in df.iterrows():
        lines.append(f"  {r['Ticker']:<8}${r['Price']:>9.2f}{r['TargetPct']*100:>10.1f}%"
                     f"${r['TargetValue']:>13,.2f}{r['TargetShares']:>14.4f}")
    lines += [
        "-" * 88,
        "  SECTOR WEIGHTS (within risk-on)",
        "-" * 88,
    ]
    for ticker, w in sorted(sector_weights.items(), key=lambda x: -x[1]):
        lines.append(f"  {ticker:<8}: {w*100:.1f}%")
    lines += [
        "-" * 88,
        "  RULES",
        "-" * 88,
        "  - Regime: BULL if >=2 of {200d SMA, 50d Donchian, 50d VWMA} bullish on QQQ",
        f"  - Risk-On: min({TARGET_VOL:.0%}/vol, 100%), halved if vol>{HIGH_VOL_CUTOFF:.0%}",
        "  - Sectors: 40% QLD / 20% TECL / 20% SOXL / 20% SMH (momentum-adjusted)",
        "  - Trade: only when risk-on scalar drifts >5% from last executed",
        "  - Sector rebal: ~monthly (21 days)",
        border,
    ]
    return "\n".join(lines)

def build_html_email(date_str: str, ind: pd.Series, risk_on: float,
                     sector_weights: Dict[str, float], portfolio_value: float,
                     df: pd.DataFrame, rebalance_due: bool) -> str:
    regime = "BULL (Risk-On)" if ind["CONSENSUS"] == 1 else "BEAR (Risk-Off)"
    color = "#27ae60" if rebalance_due else "#7f8c8d"
    status = "ACTION REQUIRED: Execute Rebalance" if rebalance_due else "Hold Current Allocation"

    rows_html = ""
    for _, r in df.iterrows():
        rows_html += f"""
        <tr style="border-bottom:1px solid #e9ecef;">
          <td style="padding:10px;font-weight:bold;">{r['Ticker']}</td>
          <td style="padding:10px;">${r['Price']:,.2f}</td>
          <td style="padding:10px;font-weight:bold;color:#0056b3;">{r['TargetPct']*100:.1f}%</td>
          <td style="padding:10px;font-weight:bold;">${r['TargetValue']:,.2f}</td>
          <td style="padding:10px;font-weight:bold;color:#27ae60;">{r['TargetShares']:,.4f}</td>
        </tr>"""

    sector_html = "".join(f"<li>{t}: {w*100:.1f}%</li>" for t, w in sorted(sector_weights.items(), key=lambda x: -x[1]))

    return f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"></head>
    <body style="font-family:-apple-system,sans-serif;background:#f8f9fa;padding:20px;color:#333;">
      <div style="max-width:700px;background:#fff;margin:0 auto;border-radius:8px;
                  box-shadow:0 4px 12px rgba(0,0,0,.08);overflow:hidden;">
        <div style="background:#1a252f;color:#fff;padding:24px;text-align:center;">
          <h2 style="margin:0;">ROTH IRA v2 - SECTOR MOMENTUM ENGINE</h2>
          <p style="margin:6px 0 0;color:#bdc3c7;">{date_str}</p>
        </div>
        <div style="padding:20px;background:#f1f4f8;border-bottom:1px solid #e9ecef;text-align:center;">
          <div style="font-size:15px;font-weight:bold;color:{color};">{status}</div>
          <div style="font-size:12px;color:#7f8c8d;margin-top:4px;">
            Regime: {regime} | Risk-On: {risk_on*100:.1f}% | Portfolio: ${portfolio_value:,.2f}</div>
        </div>
        <div style="padding:20px;display:flex;">
          <div style="flex:1;">
            <h4>Target Allocation</h4>
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
              <thead><tr style="background:#f8f9fa;border-bottom:2px solid #e9ecef;">
                <th style="padding:8px;text-align:left;">Ticker</th>
                <th style="padding:8px;text-align:left;">Price</th>
                <th style="padding:8px;text-align:left;">%</th>
                <th style="padding:8px;text-align:left;">Value</th>
                <th style="padding:8px;text-align:left;">Shares</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
          </div>
          <div style="flex:0 0 180px;padding-left:20px;">
            <h4>Sector Weights</h4>
            <ul style="font-size:13px;color:#555;">{sector_html}</ul>
          </div>
        </div>
      </div>
    </body></html>"""

def send_email(subject: str, text_body: str, html_body: str):
    addr = os.environ.get("GMAIL_ADDRESS")
    pwd = os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("RECEIVER_EMAIL")
    if not all([addr, pwd, to]):
        logger.info("Email env vars not set; skipping.")
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
# 8. MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser(description="ROTH IRA v2 - Sector Momentum Production Engine")
    ap.add_argument("--test", action="store_true", help="Print dashboard, save state, skip email")
    ap.add_argument("--roth-amount", type=float, default=None, help="Override Roth IRA balance")
    args = ap.parse_args()

    # Download all tickers
    all_tickers = [SIGNAL_TICKER] + list(RISK_ON_BASE.keys()) + list(RISK_OFF.keys())
    close, volume = download_data(all_tickers)

    # Build indicators
    ind = build_indicators(close, volume)
    latest = ind.iloc[-1].copy()
    latest["QQQ_CLOSE"] = close[SIGNAL_TICKER].iloc[-1]

    if pd.isna(latest["SMA200"]):
        raise RuntimeError("Insufficient history for 200-day SMA")

    date_str = close.index[-1].strftime("%Y-%m-%d")
    current_date = close.index[-1]

    # Load state
    state = load_state()
    last_executed_scalar = state.get("last_executed_scalar", -1.0)
    last_sector_rebal = state.get("last_sector_rebal", "")

    # Calculate risk-on scalar
    risk_on = get_risk_on_scalar(latest)

    # Drift-band check (v2 fix: compare to last EXECUTED scalar, not just shares)
    rebalance_due = should_rebalance(risk_on, last_executed_scalar, REBAL_BAND)

    # Sector rebalancing check
    sector_rebal_due = should_rebalance_sectors(last_sector_rebal, current_date)

    # Get sector weights (momentum-adjusted)
    sector_weights = get_sector_weights(latest, RISK_ON_BASE)

    # Full target weights
    weights = get_target_weights(latest, risk_on, sector_weights)

    # Portfolio value: saved -> CLI -> env
    saved_value = state.get("portfolio_value", 0)
    portfolio_value = args.roth_amount or saved_value or ROTH_IRA_AMOUNT

    # Build target portfolio
    target_df = build_target_portfolio(close, weights, portfolio_value)

    # Dashboard
    dashboard = format_dashboard(date_str, latest, risk_on, sector_weights,
                                  portfolio_value, target_df, rebalance_due, sector_rebal_due)
    print(dashboard)

    # Save state (v2: track last_executed_scalar for drift-band)
    if rebalance_due or sector_rebal_due:
        save_state(
            {r["Ticker"]: r["TargetShares"] for _, r in target_df.iterrows()},
            weights,
            portfolio_value,
            risk_on,
            current_date.isoformat() if sector_rebal_due else last_sector_rebal
        )
    else:
        # No trade: preserve previous state, update only portfolio_value
        save_state(
            state.get("shares", {}),
            state.get("weights", weights),
            portfolio_value,
            last_executed_scalar,  # Preserve, don't update
            last_sector_rebal
        )

    if not args.test:
        send_email(f"ROTH IRA v2 Report - {date_str}", dashboard,
                   build_html_email(date_str, latest, risk_on, sector_weights,
                                     portfolio_value, target_df, rebalance_due))

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("roth_ira_v2.py failed")
        sys.exit(1)

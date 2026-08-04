import os
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
CURRENT_HOLDINGS = {
    "TECL": 5.9043,
    "SOXL": 0.0,
    "SMH": 0.0,
    "GLD": 0.0,
    "SPMO": 0.0,
    "CASH": 1.28,
}

# Email only if a position is off target by at least this much.
DRIFT_THRESHOLD = 0.02     # 2 percentage points
MIN_NOTIONAL_TRADE = 25.00   # Ignore tiny trades under $25

# ==========================================
# 2. SYSTEM PARAMETERS
# ==========================================
START_DATE = (datetime.today() - timedelta(days=550)).strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")

TICKERS = ["QQQ", "TECL", "SOXL", "SMH", "SPMO", "SPY", "GLD", "^IRX"]

BAND_PCT = 0.04
CONFIRM_DAYS = 5
LOW_VOL_THRESHOLD = 0.20
HIGH_VOL_THRESHOLD = 0.25
VOL_LOOKBACK = 10

# ==========================================
# 3. DATA + SIGNALS
# ==========================================
def download_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Downloads historical data and returns forward-filled closing prices."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    
    # yfinance returns a MultiIndex if multiple tickers are passed
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = pd.DataFrame(data["Close"])
        
    return close.ffill()

def build_regime_filter(qqq_close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray]:
    """Calculates EMA regimes and applies confirmation smoothing."""
    ema200 = qqq_close.ewm(span=200, adjust=False).mean()
    upper = ema200 * (1 + BAND_PCT)
    lower = ema200 * (1 - BAND_PCT)

    raw = np.ones(len(qqq_close), dtype=int)
    current = 1

    q = qqq_close.values
    u = upper.values
    l = lower.values

    # Generate raw signals
    for i in range(200, len(qqq_close)):
        if q[i] > u[i]:
            current = 1
        elif q[i] < l[i]:
            current = 0
        raw[i] = current

    # Apply confirmation smoothing
    confirmed = np.ones(len(qqq_close), dtype=int)
    current_regime = 1
    days = 0

    for i in range(1, len(qqq_close)):
        if raw[i] != current_regime:
            days = days + 1 if raw[i] == raw[i - 1] else 1
            if days >= CONFIRM_DAYS:
                current_regime = raw[i]
                days = 0
        else:
            days = 0
        confirmed[i] = current_regime

    return ema200, upper, lower, confirmed

def barbell_weights(v10: float) -> Dict[str, float]:
    """Returns target weights based on volatility."""
    if np.isnan(v10) or v10 < LOW_VOL_THRESHOLD:
        return {"TECL": 0.20, "SOXL": 0.20, "SMH": 0.60, "GLD": 0.00, "SPMO": 0.00}
    elif v10 < HIGH_VOL_THRESHOLD:
        return {"TECL": 0.10, "SOXL": 0.10, "SMH": 0.80, "GLD": 0.00, "SPMO": 0.00}
    
    return {"TECL": 0.00, "SOXL": 0.00, "SMH": 0.80, "GLD": 0.20, "SPMO": 0.00}

def current_target_weights(regime_value: int, latest_vol: float) -> Dict[str, float]:
    """Determines final target allocation based on market regime and volatility."""
    if regime_value == 1:
        return barbell_weights(latest_vol)
    return {"TECL": 0.0, "SOXL": 0.0, "SMH": 0.0, "GLD": 0.0, "SPMO": 1.0}

# ==========================================
# 4. REBALANCE ENGINE
# ==========================================
def build_rebalance_table(close_data: pd.DataFrame, target_weights: Dict[str, float]) -> Tuple[pd.DataFrame, float, bool]:
    """Calculates necessary trades based on current drift from target weights."""
    latest_prices = {t: float(close_data[t].iloc[-1]) for t in CURRENT_HOLDINGS if t != "CASH"}
    portfolio_value = float(CURRENT_HOLDINGS.get("CASH", 0.0))

    # Calculate total portfolio value
    for ticker, shares in CURRENT_HOLDINGS.items():
        if ticker != "CASH":
            portfolio_value += float(shares) * latest_prices.get(ticker, 0.0)

    rows = []
    needs_rebalance = False
    trade_tickers = ["TECL", "SOXL", "SMH", "GLD", "SPMO"]

    for ticker in trade_tickers:
        current_shares = float(CURRENT_HOLDINGS.get(ticker, 0.0))
        price = latest_prices.get(ticker, 0.0)
        current_value = current_shares * price
        current_pct = current_value / portfolio_value if portfolio_value > 0 else 0.0

        target_pct = float(target_weights.get(ticker, 0.0))
        target_value = target_pct * portfolio_value
        target_shares = round(target_value / price, 4) if price > 0 else 0.0
        
        share_diff = round(target_shares - current_shares, 4)
        trade_value = abs(share_diff) * price
        pct_drift = abs(target_pct - current_pct)

        actionable = (pct_drift >= DRIFT_THRESHOLD) and (trade_value >= MIN_NOTIONAL_TRADE)
        if actionable:
            needs_rebalance = True
            action = f"BUY {share_diff}" if share_diff > 0 else f"SELL {abs(share_diff)}"
        else:
            action = "HOLD"

        rows.append({
            "Ticker": ticker,
            "Price": price,
            "CurrentShares": current_shares,
            "CurrentPct": current_pct,
            "TargetPct": target_pct,
            "TargetShares": target_shares,
            "ShareDiff": share_diff,
            "TradeValue": trade_value,
            "PctDrift": pct_drift,
            "Action": action,
        })

    df = pd.DataFrame(rows)
    return df, portfolio_value, needs_rebalance

# ==========================================
# 5. EMAIL
# ==========================================
def build_email_body(report_date: str, regime_label: str, latest_vol: float, 
                     latest_qqq: float, rebalance_df: pd.DataFrame, portfolio_value: float) -> str:
    """Formats the rebalance alert email."""
    lines = [
        "Barbell rebalance alert",
        "",
        f"Date: {report_date}",
        f"Regime: {regime_label}",
        f"QQQ Price: ${latest_qqq:,.2f}",
        f"QQQ 10-Day Volatility: {latest_vol:.2%}",
        f"Portfolio Value: ${portfolio_value:,.2f}",
        "",
        "Trades required:"
    ]

    actionable = rebalance_df[rebalance_df["Action"] != "HOLD"]
    for _, row in actionable.iterrows():
        lines.append(
            f"- {row['Ticker']}: {row['Action']} shares | "
            f"Current {row['CurrentPct']:.1%} -> Target {row['TargetPct']:.1%} | "
            f"Approx ${row['TradeValue']:.2f}"
        )

    lines.extend(["", "Full allocation snapshot:"])
    for _, row in rebalance_df.iterrows():
        lines.append(
            f"- {row['Ticker']}: current {row['CurrentPct']:.1%}, "
            f"target {row['TargetPct']:.1%}, action {row['Action']}"
        )

    return "\n".join(lines)

def send_email(subject, body):
    gmail_address = os.environ.get("GMAIL_ADDRESS")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    print(f"DEBUG: GMAIL_ADDRESS set? {bool(gmail_address)}")
    print(f"DEBUG: GMAIL_APP_PASSWORD set? {bool(gmail_password)}")
    print(f"DEBUG: RECEIVER_EMAIL = {receiver_email}")

    if not all([gmail_address, gmail_password, receiver_email]):
        raise ValueError("Missing one or more required email environment variables")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = receiver_email
    msg.set_content(body)

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls()
        server.login(gmail_address, gmail_password)
        server.send_message(msg)

    print("DEBUG: email sent successfully")

# ==========================================
# 6. MAIN
# ==========================================
def main():
    close = download_data(TICKERS, START_DATE, END_DATE)
    qqq = close["QQQ"]
    _, _, _, regime = build_regime_filter(qqq)
    
    # Calculate 10-day historical volatility annualized
    vol_10 = qqq.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)

    latest_date = close.index[-1].strftime("%Y-%m-%d")
    latest_qqq = float(qqq.iloc[-1])
    latest_vol = float(vol_10.iloc[-1])
    latest_regime = int(regime[-1])
    regime_label = "BULL (Risk-On)" if latest_regime == 1 else "BEAR (Risk-Off)"

    target_weights = current_target_weights(latest_regime, latest_vol)
    rebalance_df, portfolio_value, needs_rebalance = build_rebalance_table(close, target_weights)

    # Format the dataframe for cleaner terminal output
    print(f"Date: {latest_date}")
    print(f"Regime: {regime_label}")
    print(f"Needs rebalance: {needs_rebalance}\n")
    
    display_df = rebalance_df[['Ticker', 'CurrentPct', 'TargetPct', 'Action']].copy()
    display_df['CurrentPct'] = display_df['CurrentPct'].apply(lambda x: f"{x:.1%}")
    display_df['TargetPct'] = display_df['TargetPct'].apply(lambda x: f"{x:.1%}")
    print(display_df.to_string(index=False))
    print("-" * 50)

    if not needs_rebalance:
        print("No rebalance needed. No email sent.")
        return

    subject = f"Barbell Rebalance Alert - {latest_date}"
    body = build_email_body(latest_date, regime_label, latest_vol, latest_qqq, rebalance_df, portfolio_value)
    
    send_email(subject, body)
    print("Rebalance logic complete. (Email sent if configured properly).")

if __name__ == "__main__":
    main()

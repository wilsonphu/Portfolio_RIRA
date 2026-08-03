import os
import smtplib
from email.message import EmailMessage
from datetime import datetime

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
    "CASH": 0.00,
}

# Email only if a position is off target by at least this much.
DRIFT_THRESHOLD = 0.02       # 2 percentage points
MIN_NOTIONAL_TRADE = 25.00   # Ignore tiny trades under $25

# ==========================================
# 2. SYSTEM PARAMETERS
# ==========================================
START_DATE = "2014-01-01"
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
def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=0).copy()
    else:
        close = data["Close"].copy()
    return close.ffill()

def build_regime_filter(qqq_close):
    ema200 = qqq_close.ewm(span=200, adjust=False).mean()
    upper = ema200 * (1 + BAND_PCT)
    lower = ema200 * (1 - BAND_PCT)

    raw = np.ones(len(qqq_close), dtype=int)
    current = 1

    q = qqq_close.values
    u = upper.values
    l = lower.values

    for i in range(200, len(qqq_close)):
        if q[i] > u[i]:
            current = 1
        elif q[i] < l[i]:
            current = 0
        raw[i] = current

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

def barbell_weights(v10):
    if np.isnan(v10) or v10 < LOW_VOL_THRESHOLD:
        return {"TECL": 0.20, "SOXL": 0.20, "SMH": 0.60, "GLD": 0.00, "SPMO": 0.00}
    elif v10 < HIGH_VOL_THRESHOLD:
        return {"TECL": 0.10, "SOXL": 0.10, "SMH": 0.80, "GLD": 0.00, "SPMO": 0.00}
    else:
        return {"TECL": 0.00, "SOXL": 0.00, "SMH": 0.80, "GLD": 0.20, "SPMO": 0.00}

def current_target_weights(regime_value, latest_vol):
    if regime_value == 1:
        return barbell_weights(latest_vol)
    return {"TECL": 0.0, "SOXL": 0.0, "SMH": 0.0, "GLD": 0.0, "SPMO": 1.0}

# ==========================================
# 4. REBALANCE ENGINE
# ==========================================
def build_rebalance_table(close_data, target_weights):
    latest_prices = {t: float(close_data[t].iloc[-1]) for t in CURRENT_HOLDINGS if t != "CASH"}
    portfolio_value = float(CURRENT_HOLDINGS.get("CASH", 0.0))

    for ticker, shares in CURRENT_HOLDINGS.items():
        if ticker != "CASH":
            portfolio_value += float(shares) * latest_prices[ticker]

    rows = []
    needs_rebalance = False

    for ticker in ["TECL", "SOXL", "SMH", "GLD", "SPMO"]:
        current_shares = float(CURRENT_HOLDINGS.get(ticker, 0.0))
        price = latest_prices[ticker]
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

        if actionable and share_diff > 0:
            action = f"BUY {share_diff}"
        elif actionable and share_diff < 0:
            action = f"SELL {abs(share_diff)}"
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
def build_email_body(report_date, regime_label, latest_vol, latest_qqq, rebalance_df, portfolio_value):
    lines = []
    lines.append("Barbell rebalance alert")
    lines.append("")
    lines.append(f"Date: {report_date}")
    lines.append(f"Regime: {regime_label}")
    lines.append(f"QQQ Price: ${latest_qqq:,.2f}")
    lines.append(f"QQQ 10-Day Volatility: {latest_vol:.2%}")
    lines.append(f"Portfolio Value: ${portfolio_value:,.2f}")
    lines.append("")
    lines.append("Trades required:")

    actionable = rebalance_df[rebalance_df["Action"] != "HOLD"].copy()
    for _, row in actionable.iterrows():
        lines.append(
            f"- {row['Ticker']}: {row['Action']} shares | "
            f"Current {row['CurrentPct']:.1%} -> Target {row['TargetPct']:.1%} | "
            f"Approx ${row['TradeValue']:.2f}"
        )

    lines.append("")
    lines.append("Full allocation snapshot:")
    for _, row in rebalance_df.iterrows():
        lines.append(
            f"- {row['Ticker']}: current {row['CurrentPct']:.1%}, "
            f"target {row['TargetPct']:.1%}, action {row['Action']}"
        )

    return "\n".join(lines)

def send_email(subject, body):
    smtp_host = os.environ["SMTP_HOST"]
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ["SMTP_USER"]
    smtp_password = os.environ["SMTP_PASSWORD"]
    from_email = os.environ["FROM_EMAIL"]
    to_email = os.environ["TO_EMAIL"]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)

# ==========================================
# 6. MAIN
# ==========================================
def main():
    close = download_data(TICKERS, START_DATE, END_DATE)
    qqq = close["QQQ"]
    _, _, _, regime = build_regime_filter(qqq)
    vol_10 = qqq.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)

    latest_date = close.index[-1].strftime("%Y-%m-%d")
    latest_qqq = float(qqq.iloc[-1])
    latest_vol = float(vol_10.iloc[-1])
    latest_regime = int(regime[-1])
    regime_label = "BULL (Risk-On)" if latest_regime == 1 else "BEAR (Risk-Off)"

    target_weights = current_target_weights(latest_regime, latest_vol)
    rebalance_df, portfolio_value, needs_rebalance = build_rebalance_table(close, target_weights)

    print(f"Date: {latest_date}")
    print(f"Regime: {regime_label}")
    print(f"Needs rebalance: {needs_rebalance}")
    print(rebalance_df[['Ticker', 'CurrentPct', 'TargetPct', 'Action']].to_string(index=False))

    if not needs_rebalance:
        print("No rebalance needed. No email sent.")
        return

    subject = f"Barbell Rebalance Alert - {latest_date}"
    body = build_email_body(latest_date, regime_label, latest_vol, latest_qqq, rebalance_df, portfolio_value)
    send_email(subject, body)
    print("Rebalance email sent.")

if __name__ == "__main__":
    main()

import os
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
# Includes TECL so if you currently hold TECL, the script automatically generates the SELL transition trade.
CURRENT_HOLDINGS = {
    "QLD": 0.0,
    "TECL": 5.9043,
    "SOXL": 0.0,
    "SMH": 0.0,
    "GLD": 0.0,
    "SPMO": 0.0,
    "CASH": 1.28,
}

# Rebalance Triggers
DRIFT_THRESHOLD = 0.02  # 2 percentage points drift required to trade
MIN_NOTIONAL_TRADE = 25.00  # Ignore tiny trades under $25

# ==========================================
# 2. SYSTEM PARAMETERS (STRATEGY C HIGH-GROWTH)
# ==========================================
# Extended lookback window to 750 days for 200 EMA warmup stability
START_DATE = (datetime.today() - timedelta(days=750)).strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")

TICKERS = ["QQQ", "QLD", "TECL", "SOXL", "SMH", "SPMO", "SPY", "GLD", "^IRX"]

BAND_PCT = 0.04  # 4% band around 200 EMA
CONFIRM_DAYS = 5  # 5 days confirmation hysteresis
LOW_VOL_THRESHOLD = 0.20  # 20% annualized QQQ vol
HIGH_VOL_THRESHOLD = 0.25  # 25% annualized QQQ vol
VOL_LOOKBACK = 10  # 10 trading days rolling window


# ==========================================
# 3. DATA + SIGNALS
# ==========================================
def download_data(tickers: list, start: str, end: str) -> pd.DataFrame:
  """Downloads historical data and returns forward-filled closing prices."""
  data = yf.download(
      tickers, start=start, end=end, auto_adjust=True, progress=False
  )

  if isinstance(data.columns, pd.MultiIndex):
    close = data["Close"].copy()
  else:
    close = pd.DataFrame(data["Close"])

  return close.ffill().bfill().dropna(how="all")


def build_regime_filter(
    qqq_close: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray]:
  """Calculates 200 EMA regime filter with confirmation smoothing."""
  ema200 = qqq_close.ewm(span=200, adjust=False).mean()
  upper = ema200 * (1 + BAND_PCT)
  lower = ema200 * (1 - BAND_PCT)

  raw = np.ones(len(qqq_close), dtype=int)
  current = 1

  q = qqq_close.values
  u = upper.values
  l = lower.values

  start_idx = min(200, len(qqq_close) - 1)
  for i in range(start_idx, len(qqq_close)):
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


def strategy_c_high_growth_weights(
    regime_value: int, latest_vol: float
) -> Dict[str, float]:
  """Target weights for Strategy C High-Growth Variant (20% TECL / 30% QLD / 15% SOXL / 35% SMH)."""
  if regime_value == 1:
    # Bull Regime Allocation (Risk-On)
    if np.isnan(latest_vol) or latest_vol < LOW_VOL_THRESHOLD:
      # Low Volatility (<20%): 20% TECL / 30% QLD / 15% SOXL / 35% SMH
      return {
          "TECL": 0.20,
          "QLD": 0.30,
          "SOXL": 0.15,
          "SMH": 0.35,
          "GLD": 0.00,
          "SPMO": 0.00,
      }
    elif latest_vol < HIGH_VOL_THRESHOLD:
      # Moderate Volatility (20%-25%): 10% TECL / 15% QLD / 10% SOXL / 65% SMH
      return {
          "TECL": 0.10,
          "QLD": 0.15,
          "SOXL": 0.10,
          "SMH": 0.65,
          "GLD": 0.00,
          "SPMO": 0.00,
      }
    else:
      # High Volatility (>25%): 80% SMH / 20% GLD
      return {
          "TECL": 0.00,
          "QLD": 0.00,
          "SOXL": 0.00,
          "SMH": 0.80,
          "GLD": 0.20,
          "SPMO": 0.00,
      }

  # Bear Regime Allocation (Risk-Off): 80% SPMO / 20% GLD Hedge
  return {
      "TECL": 0.00,
      "QLD": 0.00,
      "SOXL": 0.00,
      "SMH": 0.00,
      "GLD": 0.20,
      "SPMO": 0.80,
  }


# ==========================================
# 4. REBALANCE ENGINE
# ==========================================
def build_rebalance_table(
    close_data: pd.DataFrame, target_weights: Dict[str, float]
) -> Tuple[pd.DataFrame, float, bool]:
  """Calculates necessary trades based on current drift from target weights."""
  latest_prices = {}
  for t in CURRENT_HOLDINGS:
    if t != "CASH":
      if t in close_data.columns:
        latest_prices[t] = float(close_data[t].iloc[-1])
      else:
        latest_prices[t] = 0.0

  portfolio_value = float(CURRENT_HOLDINGS.get("CASH", 0.0))
  for ticker, shares in CURRENT_HOLDINGS.items():
    if ticker != "CASH":
      portfolio_value += float(shares) * latest_prices.get(ticker, 0.0)

  rows = []
  needs_rebalance = False
  trade_tickers = ["TECL", "QLD", "SOXL", "SMH", "GLD", "SPMO"]

  for ticker in trade_tickers:
    current_shares = float(CURRENT_HOLDINGS.get(ticker, 0.0))
    price = latest_prices.get(ticker, 0.0)
    current_value = current_shares * price
    current_pct = (
        current_value / portfolio_value if portfolio_value > 0 else 0.0
    )

    target_pct = float(target_weights.get(ticker, 0.0))
    target_value = target_pct * portfolio_value
    target_shares = round(target_value / price, 4) if price > 0 else 0.0

    share_diff = round(target_shares - current_shares, 4)
    trade_value = abs(share_diff) * price
    pct_drift = abs(target_pct - current_pct)

    actionable = (pct_drift >= DRIFT_THRESHOLD) and (
        trade_value >= MIN_NOTIONAL_TRADE
    )
    if actionable:
      needs_rebalance = True
      action = (
          f"BUY {share_diff}" if share_diff > 0 else f"SELL {abs(share_diff)}"
      )
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
# 5. EMAIL ALERT ENGINE
# ==========================================
def build_email_body(
    report_date: str,
    regime_label: str,
    latest_vol: float,
    latest_qqq: float,
    rebalance_df: pd.DataFrame,
    portfolio_value: float,
) -> str:
  """Formats the rebalance alert email."""
  lines = [
      "ROTH IRA Rebalance Alert",
      "",
      f"Date: {report_date}",
      f"Regime: {regime_label}",
      f"QQQ Price: ${latest_qqq:,.2f}",
      f"QQQ 10-Day Volatility: {latest_vol:.2%}",
      f"Portfolio Value: ${portfolio_value:,.2f}",
      "",
      "Actionable Trades Required:",
  ]

  actionable = rebalance_df[rebalance_df["Action"] != "HOLD"]
  for _, row in actionable.iterrows():
    lines.append(
        f"- {row['Ticker']}: {row['Action']} | "
        f"Current {row['CurrentPct']:.1%} -> Target {row['TargetPct']:.1%} | "
        f"Approx ${row['TradeValue']:.2f}"
    )

  lines.extend(["", "Full Allocation Snapshot:"])
  for _, row in rebalance_df.iterrows():
    lines.append(
        f"- {row['Ticker']}: Current {row['CurrentPct']:.1%}, "
        f"Target {row['TargetPct']:.1%}, Action: {row['Action']}"
    )

  return "\n".join(lines)


def send_email(subject, body):
  gmail_address = os.environ.get("GMAIL_ADDRESS")
  gmail_password = os.environ.get("GMAIL_APP_PASSWORD")
  receiver_email = os.environ.get("RECEIVER_EMAIL")

  if not all([gmail_address, gmail_password, receiver_email]):
    print(
        "Warning: Missing email environment variables. Skipping email"
        " dispatch."
    )
    return

  try:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = receiver_email
    msg.set_content(body)

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
      server.starttls()
      server.login(gmail_address, gmail_password)
      server.send_message(msg)

    print("DEBUG: Email sent successfully.")
  except Exception as e:
    print(f"Error sending email: {e}")


# ==========================================
# 6. MAIN
# ==========================================
def main():
  close = download_data(TICKERS, START_DATE, END_DATE)

  if "QQQ" not in close.columns or close["QQQ"].dropna().empty:
    print("Error: Could not retrieve QQQ price data.")
    return

  qqq = close["QQQ"]
  _, _, _, regime = build_regime_filter(qqq)

  vol_10 = qqq.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)

  latest_date = close.index[-1].strftime("%Y-%m-%d")
  latest_qqq = float(qqq.iloc[-1])
  latest_vol = float(vol_10.iloc[-1]) if not np.isnan(vol_10.iloc[-1]) else 0.18
  latest_regime = int(regime[-1])
  regime_label = "BULL (Risk-On)" if latest_regime == 1 else "BEAR (Risk-Off)"

  target_weights = strategy_c_high_growth_weights(latest_regime, latest_vol)
  rebalance_df, portfolio_value, needs_rebalance = build_rebalance_table(
      close, target_weights
  )

  print(f"Date: {latest_date}")
  print(f"Regime: {regime_label}")
  print(f"Needs rebalance: {needs_rebalance}\n")

  display_df = rebalance_df[
      ["Ticker", "CurrentPct", "TargetPct", "Action"]
  ].copy()
  display_df["CurrentPct"] = display_df["CurrentPct"].apply(
      lambda x: f"{x:.1%}"
  )
  display_df["TargetPct"] = display_df["TargetPct"].apply(lambda x: f"{x:.1%}")
  print(display_df.to_string(index=False))
  print("-" * 50)

  if needs_rebalance:
    subject = f"Strategy C High-Growth Rebalance Alert - {latest_date}"
    body = build_email_body(
        latest_date,
        regime_label,
        latest_vol,
        latest_qqq,
        rebalance_df,
        portfolio_value,
    )
    send_email(subject, body)
  else:
    print("No rebalance needed. Thresholds not met.")


if __name__ == "__main__":
  main()

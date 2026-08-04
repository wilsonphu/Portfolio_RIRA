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
CURRENT_HOLDINGS = {
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
# 2. SYSTEM PARAMETERS (STRATEGY F PRO)
# ==========================================
START_DATE = (datetime.today() - timedelta(days=550)).strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")

TICKERS = ["QQQ", "TECL", "SOXL", "SMH", "SPMO", "SPY", "GLD", "^IRX"]

BAND_PCT = 0.04
CONFIRM_DAYS = 5

# Fine-Tuned Strategy F Pro Controls
MOM_LOOKBACK_DAYS = 15  # 15 trading days (~3 weeks) for leader detection
LOW_VOL_THRESHOLD = 0.22  # 22% annualized QQQ vol
HIGH_VOL_THRESHOLD = 0.26  # 26% annualized QQQ vol

# Low-Vol Allocation Targets
LEADER_WEIGHT_LOW = 0.45
LAGGARD_WEIGHT_LOW = 0.20
SMH_WEIGHT_LOW = 0.35


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

  return close.ffill()


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


def get_fine_tuned_target_weights(
    close_data: pd.DataFrame, regime_val: int
) -> Dict[str, float]:
  """Determines target weights via 15-day relative momentum ranking and volatility tiers."""
  if regime_val == 0:
    # Bear Regime: 85% SPMO, 15% GLD
    return {"TECL": 0.0, "SOXL": 0.0, "SMH": 0.0, "GLD": 0.15, "SPMO": 0.85}

  # Calculate 10-day annualized volatility of QQQ
  qqq_close = close_data["QQQ"]
  vol_10 = qqq_close.pct_change().rolling(10).std().iloc[-1] * np.sqrt(252)
  if np.isnan(vol_10):
    vol_10 = 0.18

  # Calculate 15-day relative momentum
  tecl_series = close_data["TECL"]
  soxl_series = close_data["SOXL"]

  if len(close_data) >= MOM_LOOKBACK_DAYS:
    tecl_mom = (
        tecl_series.iloc[-1] / tecl_series.iloc[-MOM_LOOKBACK_DAYS]
    ) - 1
    soxl_mom = (
        soxl_series.iloc[-1] / soxl_series.iloc[-MOM_LOOKBACK_DAYS]
    ) - 1
  else:
    tecl_mom, soxl_mom = 0.0, 0.0

  # Leader Selection
  leader, laggard = ("SOXL", "TECL") if soxl_mom >= tecl_mom else ("TECL", "SOXL")

  # Volatility-Tiered Allocation
  if vol_10 < LOW_VOL_THRESHOLD:
    # Low Vol (<22%): 45% Leader / 20% Laggard / 35% SMH
    return {
        leader: LEADER_WEIGHT_LOW,
        laggard: LAGGARD_WEIGHT_LOW,
        "SMH": SMH_WEIGHT_LOW,
        "GLD": 0.00,
        "SPMO": 0.00,
    }
  elif vol_10 < HIGH_VOL_THRESHOLD:
    # Mid Vol (22%-26%): 22.5% Leader / 10% Laggard / 67.5% SMH
    return {
        leader: 0.225,
        laggard: 0.10,
        "SMH": 0.675,
        "GLD": 0.00,
        "SPMO": 0.00,
    }
  else:
    # High Vol (>26%): 70% SMH / 15% GLD / 15% SPMO
    return {"TECL": 0.00, "SOXL": 0.00, "SMH": 0.70, "GLD": 0.15, "SPMO": 0.15}


# ==========================================
# 4. REBALANCE ENGINE
# ==========================================
def build_rebalance_table(
    close_data: pd.DataFrame, target_weights: Dict[str, float]
) -> Tuple[pd.DataFrame, float, bool]:
  """Calculates necessary trades based on current drift from target weights."""
  latest_prices = {
      t: float(close_data[t].iloc[-1])
      for t in CURRENT_HOLDINGS
      if t != "CASH"
  }
  portfolio_value = float(CURRENT_HOLDINGS.get("CASH", 0.0))

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
      f"Portfolio Value: ${portfolio_value:,.2f}",
      "",
      "Actionable Trades Required:",
  ]

  actionable = rebalance_df[rebalance_df["Action"] != "HOLD"]
  for _, row in actionable.iterrows():
    lines.append(
        f"- {row['Ticker']}: {row['Action']} shares | "
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


# ==========================================
# 6. MAIN
# ==========================================
def main():
  close = download_data(TICKERS, START_DATE, END_DATE)
  qqq = close["QQQ"]
  _, _, _, regime = build_regime_filter(qqq)

  latest_date = close.index[-1].strftime("%Y-%m-%d")
  latest_qqq = float(qqq.iloc[-1])
  latest_regime = int(regime[-1])
  regime_label = "BULL (Risk-On)" if latest_regime == 1 else "BEAR (Risk-Off)"

  target_weights = get_fine_tuned_target_weights(close, latest_regime)
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
    subject = f"Strategy F Pro Rebalance Alert - {latest_date}"
    body = build_email_body(
        latest_date,
        regime_label,
        latest_qqq,
        rebalance_df,
        portfolio_value,
    )
    send_email(subject, body)
  else:
    print("No rebalance needed. Thresholds not met.")


if __name__ == "__main__":
  main()

import argparse
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
# All the script needs is your total Roth IRA dollar balance.
# Can also be set via environment variable ROTH_IRA_AMOUNT
ROTH_IRA_AMOUNT = float(os.environ.get("ROTH_IRA_AMOUNT", 1025.97))

# ==========================================
# 2. SYSTEM PARAMETERS (STRATEGY C HYBRID PRO)
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


def strategy_c_hybrid_pro_weights(
    regime_value: int, latest_vol: float
) -> Dict[str, float]:
  """Target weights for Strategy C Hybrid Pro (10% TECL / 15% QLD / 20% SOXL / 55% SMH)."""
  if regime_value == 1:
    # Bull Regime Allocation (Risk-On)
    if np.isnan(latest_vol) or latest_vol < LOW_VOL_THRESHOLD:
      # Low Volatility (<20%): 10% TECL / 15% QLD / 20% SOXL / 55% SMH
      return {
          "TECL": 0.10,
          "QLD": 0.15,
          "SOXL": 0.20,
          "SMH": 0.55,
          "GLD": 0.00,
          "SPMO": 0.00,
      }
    elif latest_vol < HIGH_VOL_THRESHOLD:
      # Moderate Volatility (20%-25%): 5% TECL / 10% QLD / 10% SOXL / 75% SMH
      return {
          "TECL": 0.05,
          "QLD": 0.10,
          "SOXL": 0.10,
          "SMH": 0.75,
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
# 4. PORTFOLIO ALLOCATION ENGINE
# ==========================================
def calculate_target_portfolio(
    close_data: pd.DataFrame,
    roth_amount: float,
    target_weights: Dict[str, float],
) -> pd.DataFrame:
  """Calculates target dollar allocations and target shares directly from total Roth IRA balance."""
  latest_prices = {}
  trade_tickers = ["TECL", "QLD", "SOXL", "SMH", "GLD", "SPMO"]

  for t in trade_tickers:
    if t in close_data.columns:
      latest_prices[t] = float(close_data[t].iloc[-1])
    else:
      latest_prices[t] = 0.0

  rows = []
  for ticker in trade_tickers:
    price = latest_prices.get(ticker, 0.0)
    target_pct = float(target_weights.get(ticker, 0.0))
    target_value = target_pct * roth_amount
    target_shares = round(target_value / price, 4) if price > 0 else 0.0

    rows.append({
        "Ticker": ticker,
        "Price": price,
        "TargetPct": target_pct,
        "TargetValue": target_value,
        "TargetShares": target_shares,
    })

  return pd.DataFrame(rows)


# ==========================================
# 5. DASHBOARD & EMAIL FORMATTER
# ==========================================
def format_console_dashboard(
    report_date: str,
    regime_label: str,
    latest_qqq: float,
    latest_vol: float,
    lower_band_val: float,
    roth_amount: float,
    df: pd.DataFrame,
) -> str:
  """Formats clean, high-readability terminal logs."""
  border = "═" * 72
  sub_border = "─" * 72

  lines = [
      border,
      "  🚀 ROTH IRA STRATEGY C HYBRID PRO DASHBOARD",
      border,
      (
          f"  Date: {report_date:<15} | Roth IRA Total Value:"
          f" ${roth_amount:,.2f}"
      ),
      (
          f"  Regime: {regime_label:<20} | QQQ Volatility:"
          f" {latest_vol:.1%}"
      ),
      (
          f"  QQQ Price: ${latest_qqq:<13,.2f} | Bear Regime Pivot Price: <"
          f" ${lower_band_val:,.2f}"
      ),
      sub_border,
      "  1. PORTFOLIO TARGET ALLOCATION & SHARES TO HOLD",
      sub_border,
      (
          f"  {'Ticker':<8} {'Price':<10} {'Target %':<10} {'Target $':<12}"
          f" {'Shares to Hold':<16}"
      ),
      "  " + "─" * 68,
  ]

  for _, r in df.iterrows():
    if r["TargetPct"] > 0:
      lines.append(
          f"  {r['Ticker']:<8} ${r['Price']:<9.2f} {r['TargetPct']*100:<9.1f}%"
          f" ${r['TargetValue']:<11.2f} {r['TargetShares']:<16.4f}"
      )

  lines.extend([
      sub_border,
      "  2. FUTURE PLAYS & MARKET WATCH TRIGGERS",
      sub_border,
      "  • NEXT VOLATILITY STEP-DOWN TRIGGER:",
      (
          f"    If QQQ 10-day volatility rises from {latest_vol:.1%} to 20.0%,"
          " strategy steps down"
      ),
      "    leverage to: 5% TECL / 10% QLD / 10% SOXL / 75% SMH.",
      "",
      "  • BEAR REGIME ROTATION TRIGGER:",
      (
          f"    If QQQ drops below ${lower_band_val:,.2f} (200 EMA - 4%),"
          " strategy triggers risk-off"
      ),
      "    rotation to: 80% SPMO / 20% GLD.",
      border,
  ])

  return "\n".join(lines)


def build_html_email(
    report_date: str,
    regime_label: str,
    latest_qqq: float,
    latest_vol: float,
    lower_band_val: float,
    roth_amount: float,
    df: pd.DataFrame,
) -> str:
  """Formats rich executive HTML report for Gmail alerts."""
  table_rows_html = ""
  for _, r in df.iterrows():
    if r["TargetPct"] > 0:
      table_rows_html += f"""
            <tr style="border-bottom: 1px solid #f2f2f2;">
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
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8f9fa; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 650px; background: #ffffff; margin: 0 auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); overflow: hidden; }}
            .header {{ background-color: #1a252f; color: #ffffff; padding: 24px; text-align: center; }}
            .header h2 {{ margin: 0; font-size: 20px; letter-spacing: 0.5px; }}
            .header p {{ margin: 6px 0 0 0; color: #bdc3c7; font-size: 13px; }}
            .card {{ padding: 20px; border-bottom: 1px solid #e9ecef; }}
            .card-title {{ font-size: 14px; font-weight: bold; color: #2c3e50; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }}
            .metric-grid {{ display: flex; justify-content: space-between; background: #f1f4f8; padding: 12px 16px; border-radius: 6px; }}
            .metric {{ text-align: center; }}
            .metric-val {{ font-size: 16px; font-weight: bold; color: #2c3e50; }}
            .metric-lbl {{ font-size: 11px; color: #7f8c8d; text-transform: uppercase; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
            th {{ background: #f8f9fa; text-align: left; padding: 8px; color: #7f8c8d; font-size: 11px; text-transform: uppercase; border-bottom: 2px solid #e9ecef; }}
            .future-play {{ background-color: #fcf8e3; border-left: 4px solid #f0ad4e; padding: 12px 16px; border-radius: 4px; font-size: 13px; margin-top: 8px; }}
            .footer {{ background: #f8f9fa; text-align: center; padding: 14px; font-size: 11px; color: #95a5a6; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>📈 ROTH IRA STRATEGY C HYBRID PRO DASHBOARD</h2>
                <p>Automated Portfolio Strategy & Execution Report | {report_date}</p>
            </div>
            
            <div class="card">
                <div class="metric-grid">
                    <div class="metric"><div class="metric-val">${roth_amount:,.2f}</div><div class="metric-lbl">Total Roth IRA Balance</div></div>
                    <div class="metric"><div class="metric-val" style="color: #27ae60;">{regime_label}</div><div class="metric-lbl">Market Regime</div></div>
                    <div class="metric"><div class="metric-val">${latest_qqq:,.2f}</div><div class="metric-lbl">QQQ Price</div></div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">1. Target Portfolio Shares & Allocation</div>
                <table>
                    <thead>
                        <tr><th>Ticker</th><th>Price</th><th>Target %</th><th>Target Dollar</th><th>Shares to Hold</th></tr>
                    </thead>
                    <tbody>
                        {table_rows_html}
                    </tbody>
                </table>
            </div>

            <div class="card">
                <div class="card-title">2. Future Plays & Market Watch Triggers</div>
                <div class="future-play">
                    <strong>⚡ Volatility Step-Down Trigger:</strong><br>
                    If QQQ 10-day volatility rises from <strong>{latest_vol:.1%}</strong> to <strong>20.0%</strong>, strategy steps down leverage to: <em>5% TECL / 10% QLD / 10% SOXL / 75% SMH</em>.
                </div>
                <div class="future-play" style="background-color: #f2dede; border-left-color: #d9534f; margin-top: 10px;">
                    <strong>🛡️ Bear Regime Rotation Trigger:</strong><br>
                    If QQQ drops below <strong>${lower_band_val:,.2f}</strong> (200 EMA - 4%), strategy triggers risk-off rotation to: <em>80% SPMO / 20% GLD</em>.
                </div>
            </div>

            <div class="footer">
                Strategy C Hybrid Pro Engine | Automated GitHub Pipeline
            </div>
        </div>
    </body>
    </html>
    """


def send_email(subject, text_body, html_body):
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
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
      server.starttls()
      server.login(gmail_address, gmail_password)
      server.send_message(msg)

    print("DEBUG: Executive HTML Email sent successfully.")
  except Exception as e:
    print(f"Error sending email: {e}")


# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
  parser = argparse.ArgumentParser(
      description="Strategy C Hybrid Pro Portfolio Engine"
  )
  parser.add_argument(
      "--test",
      action="store_true",
      help="Run in test mode (prints dashboard to console, skips email)",
  )
  args = parser.parse_args()

  close = download_data(TICKERS, START_DATE, END_DATE)

  if "QQQ" not in close.columns or close["QQQ"].dropna().empty:
    print("Error: Could not retrieve QQQ price data.")
    return

  qqq = close["QQQ"]
  _, upper_band, lower_band, regime = build_regime_filter(qqq)

  vol_10 = qqq.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)

  latest_date = close.index[-1].strftime("%Y-%m-%d")
  latest_qqq = float(qqq.iloc[-1])
  latest_vol = float(vol_10.iloc[-1]) if not np.isnan(vol_10.iloc[-1]) else 0.18
  lower_band_val = float(lower_band.iloc[-1])
  latest_regime = int(regime[-1])
  regime_label = "BULL (Risk-On)" if latest_regime == 1 else "BEAR (Risk-Off)"

  target_weights = strategy_c_hybrid_pro_weights(latest_regime, latest_vol)
  target_df = calculate_target_portfolio(
      close, ROTH_IRA_AMOUNT, target_weights
  )

  # Generate Dashboard Output
  console_dashboard = format_console_dashboard(
      latest_date,
      regime_label,
      latest_qqq,
      latest_vol,
      lower_band_val,
      ROTH_IRA_AMOUNT,
      target_df,
  )
  print(console_dashboard)

  if args.test:
    print("\n[TEST MODE ACTIVE]: Email dispatch skipped.")
  else:
    subject = f"Strategy C Hybrid Pro Report - {latest_date}"
    html_email = build_html_email(
        latest_date,
        regime_label,
        latest_qqq,
        latest_vol,
        lower_band_val,
        ROTH_IRA_AMOUNT,
        target_df,
    )
    send_email(subject, console_dashboard, html_email)


if __name__ == "__main__":
  main()

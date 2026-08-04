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

DRIFT_THRESHOLD = 0.02      # 2 percentage points drift required to trade
MIN_NOTIONAL_TRADE = 25.00   # Ignore tiny trades under $25

# ==========================================
# 2. SYSTEM PARAMETERS (STRATEGY F DYNAMIC ALPHA)
# ==========================================
START_DATE = (datetime.today() - timedelta(days=550)).strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")

TICKERS = ["QQQ", "TECL", "SOXL", "SMH", "SPMO", "SPY", "GLD", "^IRX"]

BAND_PCT = 0.04
CONFIRM_DAYS = 5
VOL_LOOKBACK = 10
MOM_LOOKBACK_DAYS = 20      # 4-week lookback for relative momentum tilt

# ==========================================
# 3. DATA + SIGNALS
# ==========================================
def download_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Downloads historical data and returns forward-filled closing prices."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = pd.DataFrame(data["Close"])
        
    return close.ffill()

def build_regime_filter(qqq_close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray]:
    """Calculates EMA 200 regime and applies confirmation smoothing."""
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

def get_dynamic_target_weights(close_data: pd.DataFrame, regime_val: int) -> Dict[str, float]:
    """Determines target weights by evaluating relative 4-week momentum between TECL & SOXL."""
    if regime_val == 0:
        # Bear Regime: 85% SPMO, 15% GLD
        return {"TECL": 0.0, "SOXL": 0.0, "SMH": 0.0, "GLD": 0.15, "SPMO": 0.85}

    # Bull Regime: Calculate 20-day (~4 week) returns for TECL and SOXL
    tecl_series = close_data["TECL"]
    soxl_series = close_data["SOXL"]
    
    if len(close_data) >= MOM_LOOKBACK_DAYS:
        tecl_mom = (tecl_series.iloc[-1] / tecl_series.iloc[-MOM_LOOKBACK_DAYS]) - 1
        soxl_mom = (soxl_series.iloc[-1] / soxl_series.iloc[-MOM_LOOKBACK_DAYS]) - 1
    else:
        tecl_mom, soxl_mom = 0.0, 0.0

    # Dynamic Tilt to the Momentum Leader
    if soxl_mom > tecl_mom:
        # Semiconductor Lead: Overweight SOXL
        return {"TECL": 0.20, "SOXL": 0.40, "SMH": 0.40, "GLD": 0.00, "SPMO": 0.00}
    else:
        # Tech Broad Lead: Overweight TECL
        return {"TECL": 0.40, "SOXL": 0.20, "SMH": 0.40, "GLD": 0.00, "SPMO": 0.00}

# ==========================================
# 4. REBALANCE ENGINE
# ==========================================
def build_rebalance_table(close_data: pd.DataFrame, target_weights: Dict[str, float]) -> Tuple[pd.DataFrame, float, bool]:
    """Calculates necessary trades based on current drift from target weights."""
    latest_prices = {t: float(close_data[t].iloc[-1]) for t in CURRENT_HOLDINGS if t != "CASH"}
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
# 5. MAIN
# ==========================================
def main():
    close = download_data(TICKERS, START_DATE, END_DATE)
    qqq = close["QQQ"]
    _, _, _, regime = build_regime_filter(qqq)

    latest_date = close.index[-1].strftime("%Y-%m-%d")
    latest_qqq = float(qqq.iloc[-1])
    latest_regime = int(regime[-1])
    regime_label = "BULL (Risk-On)" if latest_regime == 1 else "BEAR (Risk-Off)"

    target_weights = get_dynamic_target_weights(close, latest_regime)
    rebalance_df, portfolio_value, needs_rebalance = build_rebalance_table(close, target_weights)

    print(f"Date: {latest_date}")
    print(f"Regime: {regime_label}")
    print(f"Needs rebalance: {needs_rebalance}\n")
    
    display_df = rebalance_df[['Ticker', 'CurrentPct', 'TargetPct', 'Action']].copy()
    display_df['CurrentPct'] = display_df['CurrentPct'].apply(lambda x: f"{x:.1%}")
    display_df['TargetPct'] = display_df['TargetPct'].apply(lambda x: f"{x:.1%}")
    print(display_df.to_string(index=False))
    print("-" * 50)

if __name__ == "__main__":
    main()

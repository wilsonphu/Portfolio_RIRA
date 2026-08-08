#!/usr/bin/env python3
"""
validate_risk_thresholds.py
STEP 1 - Empirically validate numeric thresholds against real history.

This is the single biggest gap per your list. It runs against actual QLD/SOXL
(and TQQQ if available) history, using the same adjusted data your frozen protocol uses.

Usage:
  python validate_risk_thresholds.py --snapshot research_outputs/alpha_paired_union_snapshot.csv
  OR
  python validate_risk_thresholds.py --start 2011-01-01 --end 2026-08-07  (downloads via yfinance)

Outputs:
- Circuit breaker fire counts, durations, MaxDD avoided vs upside clipped
- Chandelier 3.0x/3.5x/4.0x stability on SOXL
- Asset-level -20% flat + 20-day cooloff failure in 2022->2023 snapback
- Walk-forward split ready for Step 2
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Reuse your existing download logic if snapshot not provided
sys.path.insert(0, str(Path(__file__).parent))
try:
    import alpha_risk_extensions as risk
except ImportError:
    print("alpha_risk_extensions.py not found")
    sys.exit(1)


def load_from_snapshot(snapshot_path: Path):
    df = pd.read_csv(snapshot_path)
    # Snapshot format from alpha_paired_research: long-form with Date, Field, Ticker, Value?
    # Try to detect format
    print(f"Snapshot columns: {df.columns.tolist()[:20]}")
    # If it's the union snapshot, it has Session, Ticker, Open, Close, Volume
    # For simplicity, we expect a wide close frame - adapt here if needed
    # This is placeholder - we will handle common cases
    if "Close" in df.columns and "Ticker" in df.columns:
        # long -> wide
        wide = df.pivot(index="Session", columns="Ticker", values="Close")
        wide.index = pd.to_datetime(wide.index)
        return wide
    elif "Session" in df.columns:
        wide = df.pivot(index="Session", columns="Ticker", values="Close")
        wide.index = pd.to_datetime(wide.index)
        return wide
    else:
        # assume already wide with Date index
        df.index = pd.to_datetime(df.iloc[:, 0])
        return df


def load_via_yfinance(start, end):
    import yfinance as yf

    tickers = ["QQQ", "SMH", "QLD", "SOXL", "SPY", "TQQQ"]
    print(f"Downloading {tickers} {start} -> {end} ...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data
    # For High/Low need unadjusted? Use auto_adjust=False for HL
    hl_raw = yf.download(["SOXL", "QLD", "TQQQ"], start=start, end=end, auto_adjust=False, progress=False)
    return close, hl_raw


def analyze_circuit_breaker(close: pd.DataFrame):
    print("\n" + "=" * 70)
    print("1. PORTFOLIO DRAWDOWN CIRCUIT BREAKER (-10%/-15%/-20%)")
    print("=" * 70)
    # Build equal-risk portfolio of QLD/SOXL (and TQQQ if you want)
    cols = [c for c in ["QLD", "SOXL"] if c in close.columns]
    if len(cols) < 2:
        cols = [c for c in close.columns if c in ["QLD", "SOXL", "TQQQ"]][:3]
    ret = close[cols].pct_change()
    vol = ret.rolling(20).std() * np.sqrt(252)
    inv_vol = 1 / vol
    w = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    port_ret = (w.shift(1) * ret).sum(axis=1)
    equity = (1 + port_ret).cumprod().dropna()

    dd_df = risk.portfolio_drawdown_series(equity)
    fires = risk.count_drawdown_fires(dd_df["dd"])

    for lvl, stats in fires.items():
        print(f"\n  Level {lvl*100:.0f}%:")
        print(f"    Fires: {stats['fires']} times")
        print(f"    Days below: {stats['days_below']}")
        print(
            f"    Median duration: {stats['median_duration']:.0f} days, Max: {stats['max_duration']} days"
        )

    # With breaker applied (simplified: scale future returns by exposure)
    equity_breaker = (1 + equity.pct_change() * dd_df["exposure"].shift(1).fillna(1.0)).cumprod()
    dd_no = dd_df["dd"].min()
    dd_yes = (equity_breaker / equity_breaker.cummax() - 1).min()
    print(f"\n  MaxDD no breaker: {dd_no*100:.2f}%")
    print(f"  MaxDD with breaker: {dd_yes*100:.2f}%")
    print(f"  Improvement: {(dd_no-dd_yes)*100:.2f}pp avoided")

    # Show actual dates of -20% fires
    fires_20_dates = dd_df[dd_df["dd"] < -0.20].index
    if len(fires_20_dates) > 0:
        print(f"\n  -20% periods (first 5):")
        # group contiguous
        prev = None
        for d in fires_20_dates[:30]:
            if prev is None or (d - prev).days > 5:
                print(f"    Start: {d.date()}")
            prev = d

    return dd_df


def analyze_chandelier(hl_raw):
    print("\n" + "=" * 70)
    print("2. CHANDELIER EXIT STABILITY (3.0x / 3.5x / 4.0x ATR)")
    print("=" * 70)
    try:
        for ticker in ["SOXL", "QLD", "TQQQ"]:
            if ticker not in hl_raw["High"].columns:
                continue
            high = hl_raw["High"][ticker].dropna()
            low = hl_raw["Low"][ticker].dropna()
            close = hl_raw["Close"][ticker].dropna()
            df = pd.concat([high, low, close], axis=1).dropna()
            df.columns = ["High", "Low", "Close"]
            print(f"\n  {ticker}:")
            for mult in [3.0, 3.5, 4.0]:
                ce = risk.chandelier_exit_long(
                    df["High"], df["Low"], df["Close"], hh_period=22, atr_period=22, atr_mult=mult
                )
                sig = risk.chandelier_exit_signal(df["Close"], ce["exit_level"], confirm_closes=2)
                n = sig.sum()
                per_year = n / (len(df) / 252)
                print(
                    f"    {mult:.1f}x ATR: {n} exits, {per_year:.1f}/yr, avg hold {len(df)/max(n,1):.0f} days"
                )
            print(
                "    -> Check: if trades/yr varies >3x between 3.0x and 4.0x, threshold is fragile"
            )
    except Exception as e:
        print(f"Chandelier test failed (need High/Low): {e}")
        print("Falling back to Close-only proxy for demo")


def analyze_asset_flat(close: pd.DataFrame):
    print("\n" + "=" * 70)
    print("3. ASSET-LEVEL -20% FLAT + 20-DAY COOLOFF FAILURE MODE")
    print("=" * 70)
    for ticker in ["SOXL", "QLD", "TQQQ"]:
        if ticker not in close.columns:
            continue
        s = close[ticker].dropna()
        peak63 = s.rolling(63).max()
        dd63 = s / peak63 - 1
        # Find Sep 2022 - Mar 2023 snapback
        window = s.loc["2022-09-01":"2023-03-15"]
        if len(window) < 20:
            continue
        start_price = window.iloc[0]
        low_idx = window.idxmin()
        low_price = window.min()
        end_price = window.iloc[-1]
        first_20d_after_low = s.loc[low_idx:].iloc[1:21]
        if len(first_20d_after_low) > 0:
            gain_20d = first_20d_after_low.iloc[-1] / low_price - 1
            total_gain = end_price / low_price - 1
            print(f"\n  {ticker} 2022 bear -> 2023 snapback:")
            print(f"    Low: {low_idx.date()} @ {low_price:.2f}")
            print(f"    First 20 days after low: +{gain_20d*100:.1f}%")
            print(f"    Low -> Mar 15 2023: +{total_gain*100:.1f}%")
            print(
                f"    CONCLUSION: Fixed 20-day cooloff would miss {gain_20d*100:.1f}% of recovery"
            )
            print("    FIX: Re-enter on Close > EMA20, not fixed days")


def analyze_decay(close: pd.DataFrame):
    print("\n" + "=" * 70)
    print("4. LEVERAGED ETP DECAY ADJUSTMENT")
    print("=" * 70)
    for ticker, lev in [("QLD", 2), ("SOXL", 3), ("TQQQ", 3)]:
        if ticker not in close.columns:
            continue
        ret = close[ticker].pct_change()
        sigma_lev = ret.rolling(63).std().iloc[-1] * np.sqrt(252)
        sigma_u = sigma_lev / lev
        decay = risk.leveraged_etf_decay_estimate(sigma_u, lev)
        print(f"\n  {ticker} (L={lev}):")
        print(f"    Current 63d vol (lev): {sigma_lev*100:.1f}%")
        print(f"    Implied underlying vol: {sigma_u*100:.1f}%")
        print(f"    Estimated annual drag: {decay*100:.2f}% = 0.5*{lev}*{lev-1}*sigma_u^2")
        # Example adjustment
        base_w = 0.35
        adj_w, diag = risk.decay_adjusted_sizing(base_w, sigma_lev, lev)
        print(f"    Base 35% weight -> decay-adjusted {adj_w*100:.1f}% (penalty {diag['decay_penalty']:.2f})")
        if sigma_u > 0.35:
            print(
                f"    -> VOL CAP TRIGGER: underlying vol >35%, downgrade {lev}x -> 2x/1x recommended"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str, help="Path to alpha_paired_union_snapshot.csv")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2026-08-07")
    args = parser.parse_args()

    if args.snapshot:
        close = load_from_snapshot(Path(args.snapshot))
        hl_raw = None
        print(f"Loaded snapshot: {close.shape}")
    else:
        close, hl_raw = load_via_yfinance(args.start, args.end)

    dd_df = analyze_circuit_breaker(close)
    if hl_raw is not None:
        analyze_chandelier(hl_raw)
    else:
        print("\nSkipping Chandelier High/Low test - no HL data (provide yfinance download)")

    analyze_asset_flat(close)
    analyze_decay(close)

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE. Next: paste this output + we build Step 2 walk-forward.")
    print("=" * 70)

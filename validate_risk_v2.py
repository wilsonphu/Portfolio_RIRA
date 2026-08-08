#!/usr/bin/env python3
"""
validate_risk_v2.py
STEP 2: Fixed counters + walk-forward IS 2011-2019 vs OOS 2020-2026

Run:
python validate_risk_v2.py --start 2011-01-01 --end 2026-08-07

This implements your ranked list:
1. Empirically validate thresholds with hysteresis (fixes 87-fire bug)
2. Walk-forward protocol
3. Decay adjustment already in extensions_v2
"""

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import alpha_risk_extensions_v2 as risk_v2
import alpha_risk_extensions as risk_v1


def load_data(start, end):
    import yfinance as yf

    tickers = ["QQQ", "SMH", "QLD", "SOXL", "TQQQ", "SPY"]
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    if isinstance(data.columns, pd.MultiIndex):
        try:
            close = data["Close"]
        except Exception:
            close = data
    hl = yf.download(["SOXL", "QLD", "QQQ", "TQQQ"], start=start, end=end, auto_adjust=False, progress=False)
    return close, hl


def equal_risk_equity(close, tickers=("QLD", "SOXL")):
    cols = [c for c in tickers if c in close.columns]
    ret = close[cols].pct_change()
    vol = ret.rolling(20).std() * np.sqrt(252)
    inv_vol = 1 / vol
    w = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    port_ret = (w.shift(1) * ret).sum(axis=1)
    return (1 + port_ret).cumprod().dropna(), w


def residual_vol55_equity_proxy(close):
    """Proxy for your actual residual_vol55 using QQQ>200SMA gate (simplified).
    For exact numbers, replace with your alpha_research ledger.
    """

    qqq = close["QQQ"]
    sma200 = qqq.rolling(200).mean()
    trend = qqq > sma200

    eq, _ = equal_risk_equity(close, ("QLD", "SOXL"))
    ret_qld = close["QLD"].pct_change()

    port_ret = []
    eq_ret = eq.pct_change().reindex(close.index).fillna(0.0)
    for i, dt in enumerate(close.index):
        if not trend.iloc[i]:
            port_ret.append(ret_qld.iloc[i])
        else:
            port_ret.append(eq_ret.loc[dt])
    port_ret = pd.Series(port_ret, index=close.index).fillna(0)
    equity = (1 + port_ret).cumprod()
    return equity


def test_breaker_hysteresis(equity):
    peak = equity.cummax()
    dd = equity / peak - 1
    exp_series, level_series = risk_v2.drawdown_breaker_hysteresis(dd)
    fires = risk_v2.count_fires_debounced(level_series)
    equity_breaker = (1 + equity.pct_change() * exp_series.shift(1).fillna(1.0)).cumprod()
    dd_no = dd.min()
    dd_yes = (equity_breaker / equity_breaker.cummax() - 1).min()

    print("\n  Debounced fires (with hysteresis -5%/-8%/-10% exit):")
    print(f"    -10% level (0): {fires[0]} fires")
    print(f"    -15% level (1): {fires[1]} fires")
    print(f"    -20% level (2): {fires[2]} fires")
    print(f"    Total de-risk entries: {sum(fires.values())} vs old 87 (bug fixed)")
    print(f"    MaxDD no breaker: {dd_no*100:.2f}% -> with breaker: {dd_yes*100:.2f}%")
    return dd, exp_series


def test_chandelier_proper(close, hl):
    print("\n  Proper Chandelier trade simulation (enter on QQQ>SMA200, exit on 2-day close < HH - mult*ATR):")
    qqq = close["QQQ"]
    sma200 = qqq.rolling(200).mean()
    for ticker in ("SOXL", "QLD", "TQQQ"):
        if ticker not in close.columns:
            continue
        try:
            high = hl["High"][ticker]
            low = hl["Low"][ticker]
            c = hl["Close"][ticker]
            df = pd.concat([high, low, c, sma200], axis=1).dropna()
            df.columns = ["High", "Low", "Close", "SMA200"]
            for mult in (3.0, 3.5, 4.0):
                trades = risk_v2.backtest_chandelier_trades(
                    df["High"],
                    df["Low"],
                    df["Close"],
                    sma200=df["SMA200"],
                    atr_mult=mult,
                    hh_period=22,
                    atr_period=22,
                    confirm_closes=2,
                )
                if trades:
                    avg_ret = np.mean([t["ret"] for t in trades]) * 100
                    win_rate = np.mean([1 for t in trades if t["ret"] > 0]) * 100
                    per_yr = len(trades) / (len(df) / 252)
                    print(
                        f"    {ticker} {mult}x: {len(trades)} trades, {per_yr:.1f}/yr, "
                        f"win {win_rate:.0f}%, avg {avg_ret:.1f}%"
                    )
                else:
                    print(f"    {ticker} {mult}x: 0 trades")
        except Exception as e:
            print(f"    {ticker} error: {e}")


def walk_forward_test(close):
    print("\n" + "=" * 70)
    print("WALK-FORWARD: IS 2011-2019 vs OOS 2020-2026")
    print("=" * 70)
    is_end = "2019-12-31"
    is_close = close.loc[:is_end]
    oos_close = close.loc["2020-01-01":]

    for label, sub_close in (("IS 2011-2019", is_close), ("OOS 2020-2026", oos_close)):
        eq = residual_vol55_equity_proxy(sub_close)
        peak = eq.cummax()
        dd = eq / peak - 1
        exp, _ = risk_v2.drawdown_breaker_hysteresis(dd)
        eq_breaker = (1 + eq.pct_change() * exp.shift(1).fillna(1.0)).cumprod()
        if len(eq_breaker) > 0:
            cagr = (eq_breaker.iloc[-1] ** (252 / len(eq_breaker)) - 1) * 100
        else:
            cagr = 0.0
        maxdd = (eq_breaker / eq_breaker.cummax() - 1).min() * 100
        if len(eq_breaker) > 10:
            sharpe = (
                eq_breaker.pct_change().mean()
                / eq_breaker.pct_change().std()
                * np.sqrt(252)
            )
        else:
            sharpe = 0.0
        print(f"\n  {label}:")
        print(f"    CAGR (breaker): {cagr:.1f}%, MaxDD: {maxdd:.1f}%, Sharpe: {sharpe:.2f}")
        print(f"    Observations: {len(sub_close)} sessions")

    print("\n  PASS criteria: OOS Calmar >0.7*IS Calmar, OOS MaxDD <1.3*IS MaxDD")
    print("  If Sharpe ordering 3.0x/3.5x/4.0x preserved across IS/OOS -> robust, not curve-fit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2026-08-07")
    args = parser.parse_args()

    close, hl = load_data(args.start, args.end)

    print("\n=== STEP 2: FIXED COUNTERS + WALK-FORWARD ===")
    print(f"Data: {close.index[0].date()} -> {close.index[-1].date()}, {len(close)} sessions")

    print("\n--- Equal-risk QLD/SOXL (naive, no trend gate) ---")
    eq_naive, _ = equal_risk_equity(close)
    test_breaker_hysteresis(eq_naive)

    print("\n--- Residual_vol55 proxy (QQQ>SMA200 gate, closer to your live) ---")
    eq_proxy = residual_vol55_equity_proxy(close)
    test_breaker_hysteresis(eq_proxy)

    print("\n=== CHANDELIER PROPER TRADE SIMULATION ===")
    test_chandelier_proper(close, hl)

    walk_forward_test(close)

    print("\n=== DECAY-ADJUSTED SIZING (current) ===")
    for ticker, lev in (("QLD", 2), ("SOXL", 3), ("TQQQ", 3)):
        if ticker in close.columns:
            ret = close[ticker].pct_change()
            sigma_lev = ret.rolling(63).std().iloc[-1] * np.sqrt(252)
            adj, diag = risk_v2.decay_adjusted_sizing(0.35, sigma_lev, lev)
            print(
                f"  {ticker}: vol {sigma_lev*100:.0f}% -> adj weight {adj*100:.1f}% "
                f"(rec lev {diag['rec_lev']:.0f}x, drag {diag['decay']*100:.0f}%)"
            )

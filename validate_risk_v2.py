#!/usr/bin/env python3
"""
validate_risk_v3.py - Recalibrated thresholds for 55% vol target + fixed Chandelier

Run:
python validate_risk_v3.py

This is Step 2b - fixes the 41-fire bug and 0-trade bug from v2.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import alpha_risk_extensions_v3 as risk


def load_data(start="2011-01-01", end="2026-08-07"):
    import yfinance as yf
    tickers = ["QQQ","SMH","QLD","SOXL","TQQQ","SPY"]
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    if isinstance(data.columns, pd.MultiIndex):
        try:
            close = data['Close']
        except:
            close = data
    hl = yf.download(["SOXL","QLD","QQQ","TQQQ"], start=start, end=end, auto_adjust=False, progress=False)
    return close, hl


def equal_risk_equity(close, tickers=["QLD","SOXL"]):
    cols = [c for c in tickers if c in close.columns]
    ret = close[cols].pct_change()
    vol = ret.rolling(20).std() * np.sqrt(252)
    inv_vol = 1/vol
    w = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    port_ret = (w.shift(1) * ret).sum(axis=1)
    return (1+port_ret).cumprod().dropna()


def residual_proxy_equity(close):
    qqq = close["QQQ"]
    sma200 = qqq.rolling(200).mean()
    trend = qqq > sma200
    ret_qld = close["QLD"].pct_change()
    ret_soxl = close["SOXL"].pct_change()
    ret = close[["QLD","SOXL"]].pct_change()
    vol = ret.rolling(20).std() * np.sqrt(252)
    inv_vol = 1/vol
    w = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    port_ret = []
    for i in range(len(close)):
        if i==0:
            port_ret.append(0)
            continue
        if not trend.iloc[i]:
            port_ret.append(ret_qld.iloc[i])
        else:
            w_row = w.iloc[i-1] if i-1 < len(w) else pd.Series([0.5,0.5], index=["QLD","SOXL"])
            r = 0
            if "QLD" in w_row.index:
                r += w_row["QLD"]*ret_qld.iloc[i]
            if "SOXL" in w_row.index:
                r += w_row["SOXL"]*ret_soxl.iloc[i]
            port_ret.append(r)
    port_ret = pd.Series(port_ret, index=close.index).fillna(0)
    return (1+port_ret).cumprod()


def test_breaker_recalibrated(equity, label):
    peak = equity.cummax()
    dd = equity/peak -1
    exp_old, lvl_old = risk.drawdown_breaker_v3(dd, entry_levels=(-0.10,-0.15,-0.20), exit_levels=(-0.05,-0.08,-0.10), confirm_days=1, min_hold_days=1)
    fires_old = risk.count_fires_v3(lvl_old)
    exp_new, lvl_new = risk.drawdown_breaker_v3(dd, entry_levels=(-0.15,-0.25,-0.35), exit_levels=(-0.08,-0.15,-0.20), confirm_days=3, min_hold_days=10)
    fires_new = risk.count_fires_v3(lvl_new)
    eq_old = (1 + equity.pct_change()*exp_old.shift(1).fillna(1.0)).cumprod()
    eq_new = (1 + equity.pct_change()*exp_new.shift(1).fillna(1.0)).cumprod()
    print(f"\n{label}:")
    print(f"  No breaker MaxDD: {dd.min()*100:.1f}%")
    print(f"  OLD -10/-15/-20 (1d confirm): fires {sum(fires_old.values())} (15%:{fires_old[0]}, 25%:{fires_old[1]}, 35%:{fires_old[2]}), MaxDD {(eq_old/eq_old.cummax()-1).min()*100:.1f}%")
    print(f"  NEW -15/-25/-35 (3d confirm, 10d hold): fires {sum(fires_new.values())} (15%:{fires_new[0]}, 25%:{fires_new[1]}, 35%:{fires_new[2]}), MaxDD {(eq_new/eq_new.cummax()-1).min()*100:.1f}%")
    print(f"  -> NEW is calibrated for 55% vol target (monthly sigma 15.9%), -15% = 1 sigma, not noise")
    return exp_new


if __name__ == "__main__":
    close, hl = load_data()
    print(f"Data: {close.index[0].date()} -> {close.index[-1].date()}, {len(close)} sessions")
    eq_naive = equal_risk_equity(close)
    eq_proxy = residual_proxy_equity(close)
    print("\n=== RECALIBRATED BREAKER: OLD vs NEW ===")
    test_breaker_recalibrated(eq_naive, "Equal-risk QLD/SOXL naive")
    test_breaker_recalibrated(eq_proxy, "Residual_vol55 proxy (QQQ>SMA200 gate)")
    print("\n=== FIXED CHANDELIER (QQQ close vs QQQ SMA200, not SOXL vs QQQ) ===")
    qqq_close = close["QQQ"]
    qqq_sma200 = qqq_close.rolling(200).mean()
    for ticker in ["SOXL","QLD","TQQQ"]:
        if ticker not in close.columns:
            continue
        try:
            high = hl['High'][ticker]
            low = hl['Low'][ticker]
            c_close = hl['Close'][ticker]
            ema20 = c_close.ewm(span=20, adjust=False).mean()
            df = pd.concat([high, low, c_close, qqq_close, qqq_sma200, ema20], axis=1).dropna()
            df.columns = ['High','Low','Close','QQQ_Close','QQQ_SMA200','EMA20']
            print(f"\n  {ticker}:")
            for mult in [3.0, 3.5, 4.0]:
                trades = risk.backtest_chandelier_fixed(
                    df['High'], df['Low'], df['Close'],
                    qqq_close=df['QQQ_Close'], qqq_sma200=df['QQQ_SMA200'],
                    ticker_ema20=df['EMA20'],
                    atr_mult=mult
                )
                if trades:
                    avg_ret = np.mean([t['ret'] for t in trades])*100
                    win = np.mean([1 for t in trades if t['ret']>0])*100
                    per_yr = len(trades)/(len(df)/252)
                    print(f"    {mult}x: {len(trades)} trades, {per_yr:.1f}/yr, win {win:.0f}%, avg {avg_ret:+.1f}%")
                else:
                    print(f"    {mult}x: 0 trades (still bearish or EMA20 filter too tight)")
            print(f"    Stability check: trades/yr should vary <2x across 3.0/3.5/4.0")
        except Exception as e:
            import traceback
            print(f"    {ticker} error: {e}")
            traceback.print_exc()
    print("\n=== DECAY-ADJUSTED SIZING (current vol) ===")
    for ticker, lev in [("QLD",2), ("SOXL",3), ("TQQQ",3)]:
        if ticker in close.columns:
            ret = close[ticker].pct_change()
            sigma_lev = ret.rolling(63).std().iloc[-1]*np.sqrt(252)
            adj, diag = risk.decay_adjusted_weight(0.35, sigma_lev, lev)
            print(f"  {ticker}: lev vol {sigma_lev*100:.0f}% -> underlying {diag['sigma_u']*100:.0f}%, drag {diag['decay']*100:.0f}%, adj weight {adj*100:.1f}% (rec lev {diag['rec_lev']:.0f}x)")
    print("\n=== NEXT: PRODUCTION PATCH ===")
    print("After this, we add to alpha_core.py:")
    print("  def decay_adjusted_soxl_weight() + drawdown breaker exposure")
    print("  New candidate residual_vol55_decay_breaker in alpha_research.py")

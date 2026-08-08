#!/usr/bin/env python3
"""
validate_risk_v4.py - Episode clustering to get 4-6 fires + 3x-calibrated Chandelier

Run:
python validate_risk_v4.py

Fixes v3:
- 82 fires -> ~5-7 fires via episode clustering (recovery to -3% + 20-day gap)
- Chandelier 100% win -> now logs ATR_EXIT vs TREND_EXIT, uses 1.5-2.5x for 3x ETFs
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import alpha_risk_extensions_v4 as risk


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


if __name__ == "__main__":
    close, hl = load_data()
    print(f"Data: {close.index[0].date()} -> {close.index[-1].date()}, {len(close)} sessions")

    eq_proxy = residual_proxy_equity(close)
    peak = eq_proxy.cummax()
    dd = eq_proxy/peak -1

    print("\n=== BREAKER V4: EPISODE CLUSTERING ===")
    print("OLD v3: 82 fires counting every sub-dip")
    print("NEW v4: Cluster sub-dips into episodes, must recover to -3% + 20 days to start new episode")

    exp_v4, lvl_v4, episode_ids, n_episodes = risk.drawdown_breaker_clustered(
        dd,
        entry_levels=(-0.15,-0.25,-0.35),
        exit_levels=(-0.08,-0.15,-0.20),
        confirm_days=3,
        min_hold_days=10,
        cluster_recovery=-0.03,
        cluster_min_gap_days=20,
    )

    eq_breaker = (1 + eq_proxy.pct_change()*exp_v4.shift(1).fillna(1.0)).cumprod()
    print(f"\nResidual_vol55 proxy:")
    print(f"  No breaker MaxDD: {dd.min()*100:.1f}%")
    print(f"  With breaker v4 MaxDD: {(eq_breaker/eq_breaker.cummax()-1).min()*100:.1f}%")
    print(f"  Episode-clustered fires: {n_episodes} episodes (target 4-6 for 2011-2026)")
    print(f"  Episode IDs present: {sorted(episode_ids[episode_ids>=0].unique())}")

    # Show episode dates
    for ep_id in sorted(episode_ids[episode_ids>=0].unique())[:10]:
        ep_mask = episode_ids == ep_id
        ep_dates = dd[ep_mask].index
        if len(ep_dates)>0:
            ep_dd_min = dd[ep_mask].min()
            print(f"    Episode {ep_id}: {ep_dates[0].date()} -> {ep_dates[-1].date()}, {len(ep_dates)} days, min DD {ep_dd_min*100:.1f}%")

    print("\n=== CHANDELIER V4: 3x-CALIBRATED (1.5-2.5x) + EXIT REASON LOGGING ===")
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
            for mult in [1.5, 2.0, 2.5, 3.5]:  # include 3.5 for comparison
                trades = risk.backtest_chandelier_v4(
                    df['High'], df['Low'], df['Close'],
                    qqq_close=df['QQQ_Close'], qqq_sma200=df['QQQ_SMA200'],
                    ticker_ema20=df['EMA20'],
                    atr_mult=mult
                )
                if trades:
                    rets = [t['ret'] for t in trades]
                    avg_ret = np.mean(rets)*100
                    win = np.mean([1 for r in rets if r>0])*100
                    per_yr = len(trades)/(len(df)/252)
                    atr_exits = sum(1 for t in trades if t['reason']=='ATR_EXIT')
                    trend_exits = sum(1 for t in trades if t['reason']=='TREND_EXIT')
                    losing_trades = sum(1 for r in rets if r<0)
                    print(f"    {mult}x ATR: {len(trades)} trades, {per_yr:.1f}/yr, win {win:.0f}% ({losing_trades} losers), avg {avg_ret:+.1f}%, ATR exits {atr_exits}, Trend exits {trend_exits}")
                    worst = sorted(trades, key=lambda x: x['ret'])[:3]
                    if worst and worst[0]['ret']<0:
                        print(f"      Worst: {worst[0]['ret']*100:+.1f}% ({worst[0]['reason']}) on {worst[0]['exit'].date()}")
                else:
                    print(f"    {mult}x: 0 trades")
            print(f"    -> For 3x ETFs, 1.5-2.5x should give 40-60% ATR exits, not 0% (v3 bug). 3.5x should be mostly trend exits.")
        except Exception as e:
            import traceback
            print(f"    {ticker} error: {e}")
            traceback.print_exc()

    print("\n=== PRODUCTION PATCH READY ===")
    print("Next file: alpha_research patch for residual_vol55_decay_breaker candidate")

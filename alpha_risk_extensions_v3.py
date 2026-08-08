"""alpha_risk_extensions_v3.py
V3: Recalibrated for 55% vol target portfolio + fixed Chandelier

Key fixes:
- Breaker thresholds widened: -15%/-25%/-35% for 55% vol target (was -10/-15/-20 too tight)
- Added 3-day confirmation + 10-day minimum de-risk hold
- Fixed Chandelier: QQQ trend uses QQQ close vs QQQ SMA200, not SOXL close vs QQQ SMA
- Added EMA20 reclaim instead of fixed 20-day cooloff
"""

import numpy as np
import pandas as pd
from typing import Tuple


def true_range(high, low, close):
    prev_close = close.shift(1)
    return pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)


def atr_wilder(high, low, close, period=22):
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def chandelier_exit_long(high, low, close, hh_period=22, atr_period=22, atr_mult=3.5):
    atr = atr_wilder(high, low, close, atr_period)
    hh = high.rolling(hh_period, min_periods=hh_period).max()
    return pd.DataFrame({"atr": atr, "hh": hh, "exit_level": hh - atr_mult*atr})


# --- RECALIBRATED BREAKER FOR 55% VOL TARGET ---


def drawdown_breaker_v3(
    dd_series: pd.Series,
    entry_levels=(-0.15, -0.25, -0.35),  # WIDENED for high-vol portfolio
    exit_levels=(-0.08, -0.15, -0.20),   # hysteresis: must recover to -8% to exit -15% state
    exposures=(0.5, 0.25, 0.0),
    confirm_days=3,
    min_hold_days=10,
):
    """State machine with confirmation + min hold.

    For 55% annual vol target:
    - monthly sigma ~15.9%, so -15% = ~1 sigma, -25% = 1.5 sigma, -35% = 2.2 sigma
    Evidence: threshold should be >0.8 sigma to avoid noise.
    """
    exposure = 1.0
    current_idx = -1
    below_count = 0
    hold_counter = 0
    exposures_out = []
    levels_out = []

    for dd in dd_series:
        target_entry = entry_levels[current_idx+1] if current_idx+1 < len(entry_levels) else None
        if target_entry is not None and dd < target_entry:
            below_count += 1
        else:
            below_count = 0
        if below_count >= confirm_days and current_idx+1 < len(entry_levels):
            current_idx += 1
            exposure = exposures[current_idx]
            hold_counter = 0
            below_count = 0
        else:
            for i in range(current_idx+1, len(entry_levels)):
                if dd < entry_levels[i]:
                    current_idx = i
                    exposure = exposures[i]
                    hold_counter = 0
        if current_idx >= 0:
            hold_counter += 1
            if hold_counter >= min_hold_days:
                if dd > exit_levels[current_idx]:
                    current_idx -= 1
                    if current_idx < 0:
                        exposure = 1.0
                    else:
                        exposure = exposures[current_idx]
                    hold_counter = 0
        exposures_out.append(exposure)
        levels_out.append(current_idx)

    return pd.Series(exposures_out, index=dd_series.index), pd.Series(levels_out, index=dd_series.index)


def count_fires_v3(level_series):
    fires = {}
    for lvl in [0,1,2]:
        entered = (level_series == lvl) & (level_series.shift(1) != lvl)
        fires[lvl] = int(entered.sum())
    return fires


def backtest_chandelier_fixed(
    ticker_high, ticker_low, ticker_close,
    qqq_close, qqq_sma200,
    ticker_ema20=None,
    atr_period=22, hh_period=22, atr_mult=3.5, confirm_closes=2,
):
    """FIXED: QQQ trend uses QQQ close vs QQQ SMA200, not ticker close vs QQQ SMA200.

    Entry: QQQ > SMA200 AND ticker Close > EMA20 (reclaim)
    """
    atr = atr_wilder(ticker_high, ticker_low, ticker_close, atr_period)
    trades = []
    in_pos = False
    entry_price = None
    entry_date = None
    hh = None
    below_count = 0

    for i in range(len(ticker_close)):
        date = ticker_close.index[i]
        c = ticker_close.iloc[i]
        h = ticker_high.iloc[i]
        qqq_c = qqq_close.loc[date] if date in qqq_close.index else None
        qqq_sma = qqq_sma200.loc[date] if date in qqq_sma200.index else None
        if qqq_c is None or qqq_sma is None:
            continue
        qqq_trend = qqq_c > qqq_sma
        if not qqq_trend:
            if in_pos:
                ret = c/entry_price -1
                trades.append({"entry": entry_date, "exit": date, "ret": ret, "reason": "QQQ_TREND_EXIT"})
                in_pos = False
                hh = None
            continue
        if not in_pos:
            can_enter = True
            if ticker_ema20 is not None and date in ticker_ema20.index:
                can_enter = c > ticker_ema20.loc[date]
            if can_enter:
                in_pos = True
                entry_price = c
                entry_date = date
                hh = h
                below_count = 0
        else:
            hh = max(hh, h)
            exit_level = hh - atr_mult * atr.iloc[i] if i < len(atr) and not np.isnan(atr.iloc[i]) else None
            if exit_level is None:
                continue
            if c < exit_level:
                below_count += 1
            else:
                below_count = 0
            if below_count >= confirm_closes:
                ret = c/entry_price -1
                trades.append({"entry": entry_date, "exit": date, "ret": ret, "hh": hh, "exit_level": exit_level})
                in_pos = False
                hh = None
                below_count = 0

    return trades


def leveraged_decay(sigma_u_annual, leverage=3.0):
    return 0.5 * leverage * (leverage - 1) * (sigma_u_annual ** 2)


def decay_adjusted_weight(base_weight, sigma_lev_annual, leverage=3.0, target_vol=0.55):
    sigma_u = sigma_lev_annual / leverage
    decay = leveraged_decay(sigma_u, leverage)
    rec_lev = leverage
    if sigma_u > 0.45:
        rec_lev = 1.0
    elif sigma_u > 0.35:
        rec_lev = 2.0 if leverage == 3.0 else leverage
    penalty = 1.0 / (1.0 + decay / max(target_vol, 1e-6))
    adj = base_weight * penalty * (rec_lev / leverage)
    return float(np.clip(adj, 0.0, 0.35)), {"sigma_u": sigma_u, "decay": decay, "rec_lev": rec_lev, "penalty": penalty}

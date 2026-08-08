"""alpha_risk_extensions.py
Episode clustering + 3x-calibrated Chandelier + production-ready breaker

Fixes v3 issues:
- 82 fires -> ~5-7 fires via drawdown episode clustering
- 100% win rate -> proper ATR vs trend exit logging, recalibrated for 3x ETFs (1.5-2.5x not 3-4x)
"""

import numpy as np
import pandas as pd


def true_range(high, low, close):
    prev_close = close.shift(1)
    return pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)


def atr_wilder(high, low, close, period=22):
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def chandelier_level(high, low, close, hh_period=22, atr_period=22, atr_mult=2.0):
    """For 3x ETFs use 1.5-2.5x, for 1x/2x use 3.0-4.0x"""
    atr = atr_wilder(high, low, close, atr_period)
    hh = high.rolling(hh_period, min_periods=hh_period).max()
    return hh - atr_mult*atr, atr, hh


# --- EPISODE CLUSTERED BREAKER ---

def drawdown_breaker_clustered(
    dd_series: pd.Series,
    entry_levels=(-0.15, -0.25, -0.35),
    exit_levels=(-0.08, -0.15, -0.20),
    exposures=(0.5, 0.25, 0.0),
    confirm_days=3,
    min_hold_days=10,
    cluster_recovery=-0.03,  # must recover to -3% to close episode
    cluster_min_gap_days=20,  # must stay recovered 20 days to count new episode
):
    """Episode clustering so one fire per major bear market, not per wobble.

    Expect 2011, 2015, 2018, 2020, 2022 = about 5 episodes over 2011-2026.
    """
    exposure = 1.0
    current_idx = -1
    below_count = 0
    hold_counter = 0
    in_episode = False
    days_since_recovery = 999
    exposures_out = []
    levels_out = []
    episode_ids = []
    episode_id_counter = 0
    current_episode_id = -1

    for idx, dd in enumerate(dd_series):
        date = dd_series.index[idx]
        if dd > cluster_recovery:
            days_since_recovery += 1
        else:
            days_since_recovery = 0
        if in_episode and days_since_recovery >= cluster_min_gap_days:
            in_episode = False
            current_episode_id = -1

        target_entry = entry_levels[current_idx+1] if current_idx+1 < len(entry_levels) else None
        if target_entry is not None and dd < target_entry:
            below_count += 1
        else:
            entered_deeper = False
            for i in range(current_idx+1, len(entry_levels)):
                if dd < entry_levels[i]:
                    current_idx = i
                    exposure = exposures[i]
                    hold_counter = 0
                    entered_deeper = True
                    if not in_episode:
                        in_episode = True
                        episode_id_counter += 1
                        current_episode_id = episode_id_counter
                    break
            if not entered_deeper:
                below_count = 0

        if below_count >= confirm_days and current_idx+1 < len(entry_levels):
            current_idx += 1
            exposure = exposures[current_idx]
            hold_counter = 0
            below_count = 0
            if not in_episode:
                in_episode = True
                episode_id_counter += 1
                current_episode_id = episode_id_counter

        if current_idx >= 0:
            hold_counter += 1
            if hold_counter >= min_hold_days and dd > exit_levels[current_idx]:
                current_idx -= 1
                exposure = 1.0 if current_idx < 0 else exposures[current_idx]
                hold_counter = 0

        exposures_out.append(exposure)
        levels_out.append(current_idx)
        episode_ids.append(current_episode_id if in_episode else -1)

    return (
        pd.Series(exposures_out, index=dd_series.index),
        pd.Series(levels_out, index=dd_series.index),
        pd.Series(episode_ids, index=dd_series.index),
        episode_id_counter,
    )


def backtest_chandelier_v4(
    ticker_high, ticker_low, ticker_close,
    qqq_close, qqq_sma200,
    ticker_ema20=None,
    atr_period=22, hh_period=22,
    atr_mult=2.0,
    confirm_closes=2,
):
    """Proper exit reason logging, 3x-calibrated ATR multiples.

    Returns trades with reason: ATR_EXIT vs TREND_EXIT.
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
        if date not in qqq_close.index or date not in qqq_sma200.index:
            continue
        c = ticker_close.loc[date]
        h = ticker_high.loc[date]
        qqq_c = qqq_close.loc[date]
        qqq_sma = qqq_sma200.loc[date]
        qqq_trend = qqq_c > qqq_sma

        ema_ok = True
        if ticker_ema20 is not None and date in ticker_ema20.index:
            ema_ok = c > ticker_ema20.loc[date]

        if not qqq_trend:
            if in_pos:
                ret = c/entry_price -1
                trades.append({
                    "entry": entry_date, "exit": date, "ret": ret,
                    "reason": "TREND_EXIT", "entry_price": entry_price, "exit_price": c,
                    "hh": hh, "atr_mult": atr_mult,
                })
                in_pos = False
                hh = None
            continue

        if not in_pos:
            if ema_ok:
                in_pos = True
                entry_price = c
                entry_date = date
                hh = h
                below_count = 0
        else:
            hh = max(hh, h)
            atr_val = atr.loc[date] if date in atr.index else np.nan
            if np.isnan(atr_val):
                continue
            exit_level = hh - atr_mult * atr_val
            if c < exit_level:
                below_count += 1
            else:
                below_count = 0
            if below_count >= confirm_closes:
                ret = c/entry_price -1
                trades.append({
                    "entry": entry_date, "exit": date, "ret": ret,
                    "reason": "ATR_EXIT", "entry_price": entry_price, "exit_price": c,
                    "hh": hh, "exit_level": exit_level, "atr_mult": atr_mult,
                })
                in_pos = False
                hh = None
                below_count = 0

    return trades


def decay_adjusted_weight(base_weight, sigma_lev_annual, leverage=3.0, target_vol=0.55):
    sigma_u = sigma_lev_annual / leverage
    decay = 0.5 * leverage * (leverage - 1) * (sigma_u ** 2)
    rec_lev = leverage
    if sigma_u > 0.45:
        rec_lev = 1.0
    elif sigma_u > 0.35:
        rec_lev = 2.0 if leverage == 3.0 else leverage
    penalty = 1.0 / (1.0 + decay / max(target_vol, 1e-6))
    adj = base_weight * penalty * (rec_lev / leverage)
    return float(np.clip(adj, 0.0, 0.35)), {
        "sigma_u": sigma_u,
        "decay": decay,
        "rec_lev": rec_lev,
        "penalty": penalty,
    }

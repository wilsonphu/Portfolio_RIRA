"""alpha_risk_extensions_v2.py
V2: Fixed fire counting, proper trade simulation, hysteresis, walk-forward ready.

This replaces v1. All functions pure, adjusted closes.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 22) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def chandelier_exit_long(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    hh_period: int = 22,
    atr_period: int = 22,
    atr_mult: float = 3.5,
) -> pd.DataFrame:
    atr = atr_wilder(high, low, close, atr_period)
    hh = high.rolling(hh_period, min_periods=hh_period).max()
    return pd.DataFrame({"atr": atr, "highest_high": hh, "exit_level": hh - atr_mult * atr})


# ---- Debounced circuit breaker with hysteresis ----


def drawdown_breaker_hysteresis(
    dd_series: pd.Series,
    entry_levels=(-0.10, -0.15, -0.20),
    exit_levels=(-0.05, -0.08, -0.10),
    exposures=(0.5, 0.25, 0.0),
) -> Tuple[pd.Series, pd.Series]:
    """State machine to avoid 87-fire bug.

    Enters de-risk when DD < entry, exits only when DD > exit (hysteresis).
    Returns (exposure_series, level_index_series).
    """

    exposure = 1.0
    current_level_idx = -1  # -1 = no de-risk
    exposures_out: List[float] = []
    levels_out: List[int] = []

    for dd in dd_series:
        # deeper entry
        for i, entry in enumerate(entry_levels):
            if dd < entry and i > current_level_idx:
                current_level_idx = i
                exposure = exposures[i]

        # recovery / exit
        if current_level_idx >= 0:
            recovered = True
            for j in range(current_level_idx, -1, -1):
                if dd < exit_levels[j]:
                    recovered = False
                    break
            if recovered:
                if dd > exit_levels[0]:
                    current_level_idx = -1
                    exposure = 1.0
                else:
                    new_idx = -1
                    for j, entry in enumerate(entry_levels):
                        if dd < entry:
                            new_idx = j
                    current_level_idx = new_idx
                    exposure = exposures[new_idx] if new_idx >= 0 else 1.0

        exposures_out.append(exposure)
        levels_out.append(current_level_idx)

    return pd.Series(exposures_out, index=dd_series.index), pd.Series(levels_out, index=dd_series.index)


def count_fires_debounced(level_series: pd.Series) -> dict:
    """Count entries into de-risk states (transitions from -1 to 0,1,2)."""
    fires = {}
    for lvl in (0, 1, 2):
        entered = (level_series == lvl) & (level_series.shift(1) != lvl)
        fires[lvl] = int(entered.sum())
    return fires


# ---- Proper Chandelier trade simulation ----


def backtest_chandelier_trades(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    sma200: pd.Series | None = None,
    ema20: pd.Series | None = None,
    atr_period: int = 22,
    hh_period: int = 22,
    atr_mult: float = 3.5,
    confirm_closes: int = 2,
) -> list:
    """Real trade logic: enter on trend, track HH from entry, exit on 2-day close below.

    Returns list of trades with entry/exit dates and returns.
    """

    atr = atr_wilder(high, low, close, atr_period)
    trades: list = []
    in_pos = False
    entry_price = None
    entry_date = None
    hh = None
    below_count = 0

    for i in range(len(close)):
        date = close.index[i]
        c = close.iloc[i]
        h = high.iloc[i]

        # trend filter
        if sma200 is not None:
            if c < sma200.iloc[i]:
                if in_pos:
                    ret = c / entry_price - 1
                    trades.append({"entry": entry_date, "exit": date, "ret": ret, "reason": "TREND_EXIT"})
                    in_pos = False
                    hh = None
                continue

        if not in_pos:
            can_enter = True
            if ema20 is not None:
                can_enter = c > ema20.iloc[i]
            if can_enter:
                in_pos = True
                entry_price = c
                entry_date = date
                hh = h
                below_count = 0
        else:
            hh = max(hh, h)
            exit_level = hh - atr_mult * atr.iloc[i]
            if c < exit_level:
                below_count += 1
            else:
                below_count = 0

            if below_count >= confirm_closes:
                ret = c / entry_price - 1
                trades.append({"entry": entry_date, "exit": date, "ret": ret, "hh": hh, "exit_level": exit_level})
                in_pos = False
                hh = None
                below_count = 0

    return trades


def leveraged_etf_decay_estimate(sigma_u_annual: float, leverage: float = 3.0) -> float:
    return 0.5 * leverage * (leverage - 1) * (sigma_u_annual ** 2)


def decay_adjusted_sizing(
    base_weight: float,
    sigma_lev_annual: float,
    leverage: float = 3.0,
    target_vol: float = 0.55,
    cap: float = 0.35,
) -> Tuple[float, dict]:
    sigma_u = sigma_lev_annual / leverage
    decay = leveraged_etf_decay_estimate(sigma_u, leverage)
    rec_lev = leverage
    if sigma_u > 0.45:
        rec_lev = 1.0
    elif sigma_u > cap:
        rec_lev = 2.0 if leverage == 3.0 else leverage
    penalty = 1.0 / (1.0 + decay / max(target_vol, 1e-6))
    adj = base_weight * penalty * (rec_lev / leverage)
    return float(np.clip(adj, 0.0, 0.35)), {
        "sigma_u": sigma_u,
        "decay": decay,
        "rec_lev": rec_lev,
        "penalty": penalty,
    }

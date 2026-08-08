"""alpha_risk_extensions.py
Institutional risk extensions for QLD/SOXL core.

This module is ADDITIVE to alpha_core.py - it does not modify frozen candidates.
All functions are pure (no I/O) and use adjusted closes, matching ALPHA_RESEARCH_PROTOCOL.

Implements your 5 gap-closers:
1. Drawdown circuit breaker (-10%/-15%/-20%)
2. Chandelier Exit (3.0x/3.5x/4.0x ATR)
3. Leveraged ETP decay adjustment 0.5*(L^2-L)*sigma^2
4. Execution realism helpers
5. Regime failure mode detection
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# 1. True Range / ATR - use Wilder's, not simple mean
# ---------------------------------------------------------------------------

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range per Wilder. All inputs must be adjusted."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 22) -> pd.Series:
    """Wilder's ATR: EMA with alpha=1/period, not SMA."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# 2. Chandelier Exit (Longs)
# ---------------------------------------------------------------------------

def chandelier_exit_long(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    hh_period: int = 22,
    atr_period: int = 22,
    atr_mult: float = 3.5,
) -> pd.DataFrame:
    """Return DataFrame with atr, highest_high, exit_level.

    Signal: Close < exit_level. Uses adjusted High/Low/Close.

    Evidence: Retail heuristic (LeBeau) with institutional CTA wrapper -
    weak alpha alone, strong as volatility-scaled risk control.
    """
    atr = atr_wilder(high, low, close, atr_period)
    highest_high = high.rolling(hh_period, min_periods=hh_period).max()
    exit_level = highest_high - atr_mult * atr
    return pd.DataFrame({"atr": atr, "highest_high": highest_high, "exit_level": exit_level})


def chandelier_exit_signal(close: pd.Series, exit_level: pd.Series, confirm_closes: int = 2) -> pd.Series:
    """N-day close confirmation to avoid high-beta wick noise."""
    below = close < exit_level
    signal = below.rolling(confirm_closes, min_periods=confirm_closes).sum() == confirm_closes
    return signal.fillna(False)


# ---------------------------------------------------------------------------
# 3. Portfolio Drawdown Circuit Breaker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DrawdownState:
    dd: float  # current drawdown -0.10 = -10%
    peak: float
    equity: float
    exposure_factor: float  # 1.0, 0.5, 0.25, 0.0
    level_hit: str  # NONE, -10%, -15%, -20%


def drawdown_circuit_breaker_exposure(
    dd: float,
    thresholds: Tuple[float, float, float] = (-0.10, -0.15, -0.20),
    exposures: Tuple[float, float, float] = (0.5, 0.25, 0.0),
) -> Tuple[float, str]:
    """Pure function: maps DD to exposure.

    Evidence: STRONG institutional (Grossman-Zhou, CPPI). Not retail.
    """
    t10, t15, t20 = thresholds
    e10, e15, e20 = exposures
    if dd < t20:
        return e20, "-20%"
    if dd < t15:
        return e15, "-15%"
    if dd < t10:
        return e10, "-10%"
    return 1.0, "NONE"


def portfolio_drawdown_series(equity: pd.Series) -> pd.DataFrame:
    """From equity curve -> peak, dd, exposure."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    exposures = []
    levels = []
    for v in dd:
        exp, lvl = drawdown_circuit_breaker_exposure(float(v))
        exposures.append(exp)
        levels.append(lvl)
    return pd.DataFrame({"equity": equity, "peak": peak, "dd": dd, "exposure": exposures, "level": levels})


def count_drawdown_fires(dd_series: pd.Series, levels=(-0.10, -0.15, -0.20)) -> dict:
    """Count entries below each threshold (fires)."""
    fires = {}
    for lvl in levels:
        below = dd_series < lvl
        fire = (below & (~below.shift(1).fillna(False))).sum()
        days_below = below.sum()
        blocks = []
        cur = 0
        for b in below:
            if b:
                cur += 1
            else:
                if cur > 0:
                    blocks.append(cur)
                cur = 0
        if cur > 0:
            blocks.append(cur)
        fires[lvl] = {
            "fires": int(fire),
            "days_below": int(days_below),
            "median_duration": float(np.median(blocks)) if blocks else 0.0,
            "max_duration": int(np.max(blocks)) if blocks else 0,
            "blocks": blocks[:20],
        }
    return fires


# ---------------------------------------------------------------------------
# 4. Leveraged ETP Decay Adjustment
# ---------------------------------------------------------------------------


def leveraged_etf_decay_estimate(sigma_underlying_annual: float, leverage: float = 3.0) -> float:
    """Expected annual drag from daily reset: 0.5 * L*(L-1) * sigma_u^2.

    Evidence: STRONG academic (Avellaneda & Zhang 2010; Cheng & Madhavan 2009).
    """
    return 0.5 * leverage * (leverage - 1) * (sigma_underlying_annual ** 2)


def decay_adjusted_sizing(
    base_weight: float,
    sigma_leveraged_annual: float,
    leverage: float = 3.0,
    target_vol: float = 0.55,
    underlying_vol_cap: float = 0.35,
) -> Tuple[float, dict]:
    """Adjusts LETF weight for decay.

    1) If underlying vol > cap, downgrade leverage (3x -> 2x -> 1x).
    2) Discount weight by decay/target_vol ratio.

    Returns (adjusted_weight, diagnostics).
    Evidence: STRONG for concept, MODERATE for specific cap values.
    """
    sigma_u = sigma_leveraged_annual / leverage
    decay = leveraged_etf_decay_estimate(sigma_u, leverage)

    recommended_leverage = leverage
    if sigma_u > 0.45:
        recommended_leverage = 1.0
    elif sigma_u > underlying_vol_cap:
        recommended_leverage = 2.0 if leverage == 3.0 else leverage

    decay_penalty = 1.0 / (1.0 + decay / max(target_vol, 1e-6))
    adjusted = base_weight * decay_penalty
    if recommended_leverage < leverage:
        adjusted = adjusted * (recommended_leverage / leverage)

    diag = {
        "sigma_u": float(sigma_u),
        "sigma_leveraged": float(sigma_leveraged_annual),
        "decay_annual": float(decay),
        "recommended_leverage": float(recommended_leverage),
        "decay_penalty": float(decay_penalty),
        "base_weight": float(base_weight),
        "adjusted_weight": float(adjusted),
    }
    return float(np.clip(adjusted, 0.0, 0.35)), diag


# ---------------------------------------------------------------------------
# 5. Execution Realism & Regime Failure Mode
# ---------------------------------------------------------------------------


def execution_cost_bps(
    is_circuit_breaker_day: bool,
    vix_close: float | None = None,
    base_bps: float = 20.0,
) -> float:
    """Estimate one-way slippage + fees in bps.

    Normal: 20bps; VIX>30: 60bps; breaker day: 75-100bps.
    """
    if is_circuit_breaker_day:
        return 75.0 if vix_close is None or vix_close < 30 else 100.0
    if vix_close is not None and vix_close > 30:
        return 60.0
    if vix_close is not None and vix_close > 25:
        return 35.0
    return base_bps


def detect_chop_around_sma200(
    qqq_close: pd.Series,
    sma200: pd.Series,
    adx: pd.Series | None = None,
    window: int = 20,
    band_pct: float = 0.02,
) -> pd.Series:
    """Detect prolonged range-bound chop straddling SMA200.

    Returns True when price spends most of last `window` days inside a
    +/-band_pct band around SMA200; optionally requires ADX <20.
    """
    ratio = qqq_close / sma200
    inside_band = (ratio - 1.0).abs() < band_pct
    inside_count = inside_band.rolling(window).sum()
    chop = inside_count >= (window * 0.75)
    if adx is not None:
        chop = chop & (adx < 20)
    return chop.fillna(False)

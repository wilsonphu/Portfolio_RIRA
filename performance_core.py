"""Pure measurement functions. Nothing here may influence a trading decision.

Every function in this module is a diagnostic. The strategy's decision path
does not import, call, or branch on any result produced here, and the strategy
fingerprint does not cover it. That separation is deliberate: measurement that
can change the decision is no longer an independent measurement of it.

Three things are computed:

1. A paper track of the live rule against reference books, from price history.
2. Realised volatility and per-sleeve risk contribution, which the advertised
   daily multiplier does not capture.
3. Estimated financing drag on the levered sleeves, inferred from SGOV rather
   than hard-coded, so it moves with the rate regime.

IMPORTANT LIMITATION: the paper track is computed from price history assuming a
clean annually rebalanced book. It is NOT the dollar-weighted return of the
actual account, which depends on contribution timing and real fills. Treat it
as a comparison of strategy designs, not as a statement of realised profit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
TRAILING_WINDOWS = {"1y": 252, "3y": 756, "5y": 1260}
VOLATILITY_WINDOWS = {"21d": 21, "63d": 63, "252d": 252}
MIN_HISTORY = 30


@dataclass(frozen=True)
class PerformanceReport:
    paper_track: dict[str, dict[str, float]] = field(default_factory=dict)
    realized_volatility: dict[str, float] = field(default_factory=dict)
    sleeve_risk_contribution: dict[str, float] = field(default_factory=dict)
    estimated_short_rate: float = 0.0
    financing_drag: float = 0.0
    sample_sessions: int = 0


def _clean(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        raise ValueError(f"Missing price columns: {', '.join(missing)}")
    frame = prices.loc[:, tickers].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(how="any")


def daily_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    frame = _clean(prices, tickers)
    return frame.pct_change().dropna(how="any")


def annual_rebalanced_curve(
    returns: pd.DataFrame, weights: dict[str, float]
) -> pd.Series:
    """Grow a book, resetting to target on the first session of each year."""
    tickers = [t for t in weights if t in returns.columns]
    if not tickers:
        return pd.Series(dtype=float)
    target = np.array([weights[t] for t in tickers], dtype=float)
    total = target.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    target = target / total

    matrix = returns.loc[:, tickers].to_numpy(dtype=float)
    years = returns.index.year.to_numpy()
    sleeve = target.copy()
    values = np.empty(len(matrix))
    for day in range(len(matrix)):
        if day > 0 and years[day] != years[day - 1]:
            sleeve = target * sleeve.sum()
        sleeve = sleeve * (1.0 + matrix[day])
        values[day] = sleeve.sum()
    return pd.Series(values, index=returns.index)


def crisis_tactical_returns(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    index_ticker: str,
    confirmation_ticker: str,
    growth: str,
    hedge: str,
    sma_window: int,
    short_sma_window: int,
    entry_ratio: float,
    exit_ratio: float,
) -> pd.Series:
    """Replicate the live extreme-bear rule FOR MEASUREMENT ONLY.

    This reconstruction is never consulted by the allocator. It exists so the
    paper track reflects the rule actually in production rather than a
    permanently-held fund.
    """
    common = returns.index.union(prices.index)
    qqq = pd.to_numeric(prices[index_ticker], errors="coerce").reindex(common).sort_index()
    spy = pd.to_numeric(prices[confirmation_ticker], errors="coerce").reindex(common).sort_index()
    qqq_sma = qqq.rolling(sma_window, min_periods=sma_window).mean()
    spy_sma = spy.rolling(sma_window, min_periods=sma_window).mean()
    spy_short = spy.rolling(short_sma_window, min_periods=short_sma_window).mean()
    extreme = ((qqq <= entry_ratio * qqq_sma) & (spy < spy_sma)).reindex(returns.index).fillna(False)
    recovery = ((qqq >= exit_ratio * qqq_sma) & (spy > spy_short)).reindex(returns.index).fillna(False)
    active = False
    entry_streak = 0
    exit_streak = 0
    signal_state: list[bool] = []
    for is_extreme, is_recovery in zip(extreme, recovery):
        if active:
            exit_streak = exit_streak + 1 if is_recovery else 0
            if exit_streak >= 2:
                active = False
                exit_streak = 0
            entry_streak = 0
        else:
            entry_streak = entry_streak + 1 if is_extreme else 0
            if entry_streak >= 2:
                active = True
                entry_streak = 0
            exit_streak = 0
        signal_state.append(active)
    hold_hedge = pd.Series(signal_state, index=returns.index).shift(1, fill_value=False)
    hedged_return = 0.75 * returns[growth] + 0.25 * returns[hedge]
    return returns[growth].where(~hold_hedge, hedged_return)


def trailing_returns(curve: pd.Series) -> dict[str, float]:
    """Annualised return over each trailing window that history supports."""
    out: dict[str, float] = {}
    if len(curve) < MIN_HISTORY:
        return out
    for label, window in TRAILING_WINDOWS.items():
        if len(curve) < window:
            continue
        starting_value = 1.0 if len(curve) == window else float(curve.iloc[-(window + 1)])
        growth = float(curve.iloc[-1] / starting_value)
        if growth <= 0:
            continue
        out[label] = growth ** (TRADING_DAYS / window) - 1.0
    span = len(curve)
    growth = float(curve.iloc[-1])
    if growth > 0 and span > MIN_HISTORY:
        out["full"] = growth ** (TRADING_DAYS / span) - 1.0
    wealth = pd.concat([pd.Series([1.0]), curve.reset_index(drop=True)], ignore_index=True)
    out["max_drawdown"] = float((wealth / wealth.cummax() - 1.0).min())
    return out


def realized_volatility(portfolio_returns: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, window in VOLATILITY_WINDOWS.items():
        if len(portfolio_returns) < window:
            continue
        sample = portfolio_returns.iloc[-window:]
        out[label] = float(sample.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return out


def sleeve_risk_contribution(
    returns: pd.DataFrame, weights: dict[str, float], window: int = 252
) -> dict[str, float]:
    """Each sleeve's share of portfolio variance.

    This is the number the advertised daily multiplier cannot express: ZROZ
    carries a 1.0x multiplier and roughly equity-like volatility.
    """
    tickers = [t for t in weights if t in returns.columns and weights[t] > 0]
    if len(tickers) < 2 or len(returns) < window:
        return {}
    sample = returns.loc[:, tickers].iloc[-window:]
    w = np.array([weights[t] for t in tickers], dtype=float)
    total = w.sum()
    if total <= 0:
        return {}
    w = w / total
    cov = sample.cov().to_numpy(dtype=float) * TRADING_DAYS
    portfolio_var = float(w @ cov @ w)
    if not np.isfinite(portfolio_var) or portfolio_var <= 0:
        return {}
    contributions = w * (cov @ w) / portfolio_var
    return {t: float(c) for t, c in zip(tickers, contributions)}


def estimate_short_rate(prices: pd.DataFrame, cash_proxy: str, window: int = 63) -> float:
    """Infer the prevailing short rate from the cash proxy's total return."""
    if cash_proxy not in prices.columns:
        return 0.0
    series = pd.to_numeric(prices[cash_proxy], errors="coerce").dropna()
    if len(series) <= window:
        return 0.0
    growth = float(series.iloc[-1] / series.iloc[-(window + 1)])
    if growth <= 0:
        return 0.0
    rate = growth ** (TRADING_DAYS / window) - 1.0
    return float(np.clip(rate, 0.0, 0.25))


def financing_drag(
    weights: dict[str, float], multipliers: dict[str, float], short_rate: float
) -> float:
    """Annual cost of the borrowed notional inside levered wrappers."""
    borrowed = sum(
        weight * max(multipliers.get(ticker, 1.0) - 1.0, 0.0)
        for ticker, weight in weights.items()
        if weight > 0
    )
    return float(borrowed * short_rate)


def build_performance_report(
    prices: pd.DataFrame,
    *,
    reference_books: dict[str, dict[str, float]],
    live_book: dict[str, float],
    index_ticker: str,
    confirmation_ticker: str,
    growth: str,
    hedge: str,
    cash_proxy: str,
    sma_window: int,
    short_sma_window: int,
    entry_ratio: float,
    exit_ratio: float,
    multipliers: dict[str, float],
    current_weights: dict[str, float],
) -> PerformanceReport:
    required = {
        index_ticker, confirmation_ticker, growth, hedge,
        *live_book,
        *(t for book in reference_books.values() for t in book),
    }
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"Missing required performance data: {', '.join(missing)}")
    # The cash proxy estimates financing separately. Including it in the common
    # return panel would truncate every benchmark to the proxy's inception.
    needed = sorted(required)
    returns = daily_returns(prices, needed)
    if len(returns) < MIN_HISTORY:
        return PerformanceReport()

    paper: dict[str, dict[str, float]] = {}
    for label, book in reference_books.items():
        curve = annual_rebalanced_curve(returns, book)
        if len(curve):
            paper[label] = trailing_returns(curve)

    routed = returns.copy()
    routed["__TACTICAL__"] = crisis_tactical_returns(
        returns,
        prices,
        index_ticker,
        confirmation_ticker,
        growth,
        hedge,
        sma_window,
        short_sma_window,
        entry_ratio,
        exit_ratio,
    )
    live_mapped = {
        ("__TACTICAL__" if t == growth else t): w for t, w in live_book.items()
    }
    live_curve = annual_rebalanced_curve(routed, live_mapped)
    if len(live_curve):
        paper["live_rule"] = trailing_returns(live_curve)

    live_returns = live_curve.pct_change().dropna() if len(live_curve) else pd.Series(dtype=float)
    short_rate = estimate_short_rate(prices, cash_proxy)
    tradable = {
        ticker: weight
        for ticker, weight in current_weights.items()
        if ticker in returns.columns and weight > 0
    }
    # An explicitly all-cash account has no sleeve variance or financing drag.
    # Fall back to the model book only when a caller supplied no holdings at all.
    measured_book = tradable if current_weights else live_book
    return PerformanceReport(
        paper_track=paper,
        realized_volatility=realized_volatility(live_returns),
        sleeve_risk_contribution=sleeve_risk_contribution(returns, measured_book),
        estimated_short_rate=short_rate,
        financing_drag=financing_drag(measured_book, multipliers, short_rate),
        sample_sessions=int(len(returns)),
    )

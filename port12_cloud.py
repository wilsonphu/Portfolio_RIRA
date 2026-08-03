import numpy as np
import pandas as pd
import yfinance as yf

START_DATE = "2014-01-01"
END_DATE = "2026-08-01"
BACKTEST_START = "2016-01-01"

TICKERS = ["QQQ", "TECL", "SOXL", "SMH", "SPMO", "SPY", "GLD", "^IRX"]

BAND_PCT = 0.04
CONFIRM_DAYS = 5

LOW_VOL_THRESHOLD = 0.20
HIGH_VOL_THRESHOLD = 0.25
VOL_LOOKBACK = 10


def download_close_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=0).copy()
    else:
        close = data["Close"].copy()
    return close


def build_regime_filter(qqq_close):
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

    return ema200, raw, confirmed


def compute_returns(close):
    rets = {t: close[t].pct_change() for t in close.columns}

    if "SPMO" in close.columns and "SPY" in close.columns:
        rets["SPMO"] = rets["SPMO"].fillna(rets["SPY"])

    for k in rets:
        rets[k] = rets[k].fillna(0.0)

    rf_daily = ((close["^IRX"].ffill() / 100.0) / 252.0).fillna(0.0)
    return rets, rf_daily


def barbell_weights(v10):
    if np.isnan(v10) or v10 < LOW_VOL_THRESHOLD:
        return {"TECL": 0.20, "SOXL": 0.20, "SMH": 0.60, "GLD": 0.00}
    elif v10 < HIGH_VOL_THRESHOLD:
        return {"TECL": 0.10, "SOXL": 0.10, "SMH": 0.80, "GLD": 0.00}
    else:
        return {"TECL": 0.00, "SOXL": 0.00, "SMH": 0.80, "GLD": 0.20}


def performance_stats(daily_returns, rf_daily):
    daily_returns = np.asarray(daily_returns)
    rf_daily = np.asarray(rf_daily)

    equity = np.cumprod(1 + daily_returns)
    years = len(daily_returns) / 252.0

    cagr = equity[-1] ** (1 / years) - 1
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    excess = daily_returns - rf_daily
    sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252) if np.std(excess) > 0 else np.nan

    return {
        "CAGR": cagr,
        "MaxDrawdown": max_dd,
        "Sharpe": sharpe,
        "EquityCurve": equity,
    }


def run_backtest():
    close = download_close_data(TICKERS, START_DATE, END_DATE)
    close = close.ffill()

    qqq = close["QQQ"]
    _, _, regime = build_regime_filter(qqq)

    rets, rf_daily = compute_returns(close)

    vol_10 = qqq.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)

    start_idx = qqq.index.get_loc(qqq[qqq.index >= pd.to_datetime(BACKTEST_START)].index[0])

    strategy_returns = np.zeros(len(qqq))

    for i in range(start_idx, len(qqq)):
        if regime[i - 1] == 1:
            w = barbell_weights(vol_10.iloc[i - 1])
            strategy_returns[i] = (
                w["TECL"] * rets["TECL"].iloc[i]
                + w["SOXL"] * rets["SOXL"].iloc[i]
                + w["SMH"] * rets["SMH"].iloc[i]
                + w["GLD"] * rets["GLD"].iloc[i]
            )
        else:
            strategy_returns[i] = rets["SPMO"].iloc[i]

    bt_returns = strategy_returns[start_idx:]
    bt_rf = rf_daily.iloc[start_idx:].values
    bt_dates = qqq.index[start_idx:]

    stats = performance_stats(bt_returns, bt_rf)

    out = pd.DataFrame({
        "Date": bt_dates,
        "DailyReturn": bt_returns,
        "EquityCurve": stats["EquityCurve"],
        "Regime": regime[start_idx:],
        "Vol10": vol_10.iloc[start_idx:].values,
    })
    out.to_csv("barbell_backtest_results.csv", index=False)

    print("Original Barbell Backtest Complete")
    print(f"CAGR: {stats['CAGR']:.2%}")
    print(f"Max Drawdown: {stats['MaxDrawdown']:.2%}")
    print(f"Sharpe Ratio: {stats['Sharpe']:.3f}")

    latest_vol = vol_10.iloc[-1]
    latest_regime = "BULL" if regime[-1] == 1 else "BEAR"
    latest_weights = barbell_weights(latest_vol) if regime[-1] == 1 else {"SPMO": 1.00}

    print(f"Current Regime: {latest_regime}")
    print("Current Allocation:")
    for k, v in latest_weights.items():
        print(f"  {k}: {v:.0%}")


if __name__ == "__main__":
    run_backtest()

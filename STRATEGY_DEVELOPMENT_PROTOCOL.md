# Strategy development protocol

## Live decision boundary

Production is frozen as
`tqqq-upro40-dbmf20-zroz20-ugl20-sma200-annual-v2`:

```text
Bull-confirmed: 40% TQQQ / 20% DBMF / 20% ZROZ / 20% UGL
Trend failed:  40% UPRO / 20% DBMF / 20% ZROZ / 20% UGL
```

The QQQ close is bullish only when it is strictly above its trailing
200-session SMA. One failed completed close selects UPRO immediately. TQQQ
requires two distinct bullish completed closes. Signals observed after a close
are intended for the next executable session. No SOXL tier, residual-momentum
overlay, or alternative ETF can affect production.

The first completed NYSE signal in each calendar year forces an exact target
rebalance. QQQ's 50-session average and 252-session momentum are displayed as
supporting health checks only.

Between annual rebalances, a router change replaces the current TQQQ/UPRO
equity position without resetting the other sleeves. Ordinary weight drift does
not produce an email or a trade.

This is an index router, not a leverage-reduction rule: TQQQ and UPRO both
target 3x daily returns. The portfolio's base advertised exposure is 2.00x.
Age/value lifecycle ceilings scale all risky weights proportionally into SGOV
and ratchet in one direction only.

## Evidence behind promotion

The pre-promotion comparison used actual ETF histories from May 2019 through
September 2026 and charged 25 basis points per dollar bought or sold. Annual
rebalancing was retained for the fixed sleeves, while a switch traded only the
40% equity sleeve. Results from this short, unusually favorable sample were:

| Equity rule | CAGR | Max drawdown |
|---|---:|---:|
| Permanent UPRO | 21.11% | -34.00% |
| Permanent TQQQ | 29.22% | -40.33% |
| QQQ SMA200 TQQQ/UPRO router | 26.00% | -35.97% |
| 5% SMA band router | 27.48% | -41.09% |
| SMA/Donchian/VWMA 2-of-3, annual | 21.61% | -36.75% |
| SMA plus short-filter confirmation, annual | 20.72% | -36.68% |

The simple SMA rule was selected because it reduced the permanent-TQQQ
drawdown in this sample without adding a second fitted threshold. A proposed
Nasdaq-versus-S&P relative-strength condition was rejected. These figures are
not a forecast and do not establish statistical proof; DBMF's live history is
too short for a full-cycle inference.

The former SMH-on-QQQ OLS residual and HAR-style volatility forecast were
designed exclusively to admit and size SOXL. They were retired with that sleeve
rather than repurposed without evidence. Tests of the short-trend filter set
produced 76–86 equity switches versus 16 for SMA200 and materially reduced net
growth at the same 25-basis-point cost assumption.

## Rules for future changes

Every challenger must be isolated from the production decision path and frozen
before its final evaluation. Its protocol must state:

- hypothesis and causal rationale;
- exact universe, weights, signal timing, and rebalance rule;
- training, validation, and untouched holdout boundaries;
- purging/embargo where labels overlap;
- costs, turnover, distributions, and delisting assumptions;
- primary objective (net geometric growth) and drawdown constraint;
- sensitivity tests and explicit rejection criteria.

Promote only evidence that survives multiple market regimes, realistic costs,
parameter perturbations, and prospective observation. A higher in-sample CAGR
alone is not sufficient. Any production change requires a new revision, state
migration, fingerprint review, tests, and a written comparison against the
current live rule.

## Operational invariants

- Confirmed broker shares and cash are the source of truth.
- Missing, stale, partial, zero, infinite, or NaN market data fail closed.
- A duplicate close cannot mutate the signal state twice.
- Old holdings remain priceable until explicitly sold.
- State is atomic and private; credentials, balances, shares, and state files
  are never committed.
- HOLD is silent; notification retries cannot erase confirmed fills.
- Test mode cannot save, email, log, or write an audit.

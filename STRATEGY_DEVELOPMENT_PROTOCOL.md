# Strategy development protocol

## Live decision boundary

Production is frozen as
`tqqq40-btal10-extreme-bear-8-2-annual-v4`:

```text
Normal:       40% TQQQ / 20% DBMF / 20% ZROZ / 20% UGL
Extreme bear: 30% TQQQ / 10% BTAL / 20% DBMF / 20% ZROZ / 20% UGL
```

BTAL enters only after two distinct completed sessions where both conditions
are true:

1. QQQ closes at or below 92% of its trailing 200-session SMA.
2. SPY closes below its trailing 200-session SMA.

BTAL exits only after two distinct completed sessions where both conditions
are true:

1. QQQ closes at or above 98% of its trailing 200-session SMA.
2. SPY closes above its trailing 50-session SMA.

The asymmetric thresholds are an explicit deadband. Signals observed after a
completed close are intended for the next executable session. Duplicate dates
cannot advance either confirmation count, and missed sessions are replayed in
order. BTAL is capped at 10%; there are no intermediate hedge tiers.

The first completed NYSE signal in each calendar year forces an exact target
rebalance. Between annual rebalances, a crisis transition preserves the
current total TQQQ/BTAL sleeve and changes only its internal 100/0 or 75/25
split. Ordinary drift in DBMF, ZROZ, UGL, or cash does not produce a trade.

The normal portfolio's advertised gross daily exposure is 2.00x and the
extreme-bear target's is 1.80x. BTAL is dollar-neutral rather than guaranteed
beta-neutral, so advertised exposure is not a forecast of realized portfolio
beta. Age/value lifecycle ceilings continue to scale all risky weights
proportionally into SGOV and ratchet in one direction only.

## Evidence and limitations

The permanent 40% TQQQ / 20% DBMF / 20% ZROZ / 20% UGL allocation remains the
return engine. The BTAL rule is a narrowly capped crisis overlay selected to
reduce high-beta exposure only after severe, broad deterioration while
retaining 30% TQQQ for a rebound.

The 8% entry distance, 2% recovery distance, two-close confirmation, and 10%
cap are predeclared policy parameters. They are not statistically proven
optima. BTAL began trading in 2011 and changed from passive index tracking to
an active rules-based process in 2022, so its live history does not support a
full-cycle inference. UPRO is no longer a target; it remains accepted only as
a migration and liquidation-only holding so confirmed broker assets are never
silently discarded.

Measurement-only diagnostics reconstruct the same delayed two-close crisis
rule. They cannot influence or block a production decision.

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
alone is insufficient. Any production change requires a new revision, state
migration, fingerprint review, tests, and a written comparison with the live
rule.

## Operational invariants

- Confirmed broker shares and cash are the source of truth.
- Missing, stale, partial, zero, infinite, or NaN market data fail closed.
- A duplicate close cannot mutate signal state twice.
- Old holdings remain priceable until explicitly sold.
- State is atomic and private; credentials, balances, shares, and state files
  are never committed.
- HOLD is silent; notification retries cannot erase confirmed fills.
- Test mode cannot save, email, log, or write an audit.

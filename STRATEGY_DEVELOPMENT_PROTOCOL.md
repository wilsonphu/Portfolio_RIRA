# Production Strategy Protocol

## Frozen decision rule

Production is the static annual allocation:

```text
35% TQQQ
25% DBMF
20% UGL
15% ZROZ
 5% BTAL
```

The engine does not use market signals. It does not inspect moving averages,
volatility, momentum, correlations, macro regimes, or lifecycle thresholds.
It holds the target through the year and evaluates reallocation only on the
first completed NYSE session of a new calendar year.

## Execution and state invariants

- Confirmed broker shares and cash are the source of truth.
- A new annual target is delivered as a next-session manual order plan.
- Exact post-trade shares and CASH are confirmed afterward.
- Midyear drift does not create a trade or email.
- Missing, stale, partial, zero, infinite, or NaN prices fail closed.
- State is saved atomically and privately as a workflow artifact.
- A duplicate run cannot send a duplicate annual recommendation.
- Legacy UPRO shares remain priceable until explicitly sold during adoption of
  this static target; no holding is silently discarded.

## Contributions

An optional contribution plan releases the configured remaining annual budget
once at the annual allocation review and allocates it to target underweights.
No contribution decision depends on a market indicator.

## Evidence boundary

This is a policy simplification, not a claim of superior alpha. TQQQ and UGL
have daily leverage and can suffer path-dependent volatility drag; DBMF, ZROZ,
and BTAL can all behave differently from their intended diversifying roles in
stress. The fixed weights and annual schedule are deliberate user-selected
parameters and must be evaluated prospectively before being treated as proven.

Any future strategy change requires a new revision, state migration, tests,
and a written comparison against this frozen static rule.

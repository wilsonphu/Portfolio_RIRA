# Strategy development protocol

Revised on 2026-09-13. This document governs research around the Roth IRA
allocator. It is not a trade instruction and cannot change production state.

## Objective

The objective is maximum long-run net log growth subject to a predeclared
historical drawdown boundary of 70%. Report CAGR, Sharpe, Sortino, Calmar,
expected shortfall, recovery time, rolling returns, turnover, and costs, but do
not optimize CAGR in isolation.

Production remains the benchmark until a challenger clears every applicable
gate. Research cannot change holdings, send a recommendation, or change the
live strategy fingerprint.

## Frozen production control

The current sprint target is defined by one binary SOXL tier, `s ∈ {0%, 15%}`:

```text
TQQQ = 40% * (1 - s)
DBMF = 20% * (1 - s)
ZROZ = 20% * (1 - s)
GLD  = 20% * (1 - s)
SOXL = s
```

The 15% tier requires positive SMH-on-QQQ residual momentum from a separated
126-session regression and 63-session scoring window, with QQQ above its
200-session SMA. The QQQ trend rule is a defensive SOXL exit fuse, not an
independently proven return source. The causal HAR-style forecast and 55%
overlay budget decide whether the single tier fits. Trend failure exits
immediately; re-entry needs two distinct completed closes; a tier reduction is
immediate. There are no 25% or 35% SOXL production tiers.

The current revision intentionally does not implement SMA-based TQQQ-to-QLD or
TQQQ-to-QQQ de-leveraging. That proposal must be tested as a separate frozen
challenger before it can change live targets.

The one-way lifecycle ratchet remains a risk policy. Until product-level
deleveraging is promoted, a lifecycle ceiling scales the current target into
SGOV proportionally and never re-levers after a stage advance.

## Evidence and legacy variants

Historical QLD/UGL, TQQQ/UGL, SOXL multi-tier, SPY/SSO, and other allocations
remain useful research or migration controls. Their historical results do not
describe the current production target and do not establish future alpha.
The current revision was chosen to make the live portfolio easier to inspect:
four fixed foundation ETFs and one bounded tactical sleeve.

## Data and timing rules

- Use point-in-time completed daily observations only.
- Reject synthetic pre-inception history, interpolation, ticker substitution,
  stale or incomplete bars, invalid prices, and zero/NaN volatility.
- Indicators use close `t`; the earliest modeled action is session `t+1`.
  Same-date reruns cannot create additional evidence.
- Use blocked or walk-forward time-series validation with purging and embargo
  when labels overlap. Random k-fold validation is prohibited.
- Charge 10, 25, and 50 basis points per one-way turnover.
- Share quantities and cash must evolve through time; daily target replacement
  is an invalid free-rebalance assumption.

## Required benchmarks

Every complete study must report:

1. current production with its stateful alpha, lifecycle, and drift rules;
2. the same TQQQ/DBMF/ZROZ/GLD foundation without SOXL;
3. the permanent 15% SOXL overlay;
4. the prior QLD/UGL production control;
5. the proposed TQQQ deleveraging challenger with unchanged timing/costs; and
6. a zero-cost path only as attribution, never as the decision result.

## Promotion gates

A challenger must show positive net log-growth improvement at 10 and 25 bps,
nonnegative improvement at 50 bps, drawdown no worse than 70% and no more than
2.5 percentage points worse than production, improvement in at least three of
four fixed chronological slices including the final slice, and a moving-block
bootstrap probability of nonpositive improvement of at most 10%. It must also
survive a Deflated-Sharpe review using the honest trial count and show that no
single crisis, asset, or terminal subperiod explains over half the gain.

Historical passage is insufficient. Promotion additionally requires 252
prospective completed sessions under one frozen fingerprint, three prospective
structural SOXL decisions, positive net log-growth after modeled costs,
complete shadow/fill reconciliation, and a written negative-evidence review.

Changing a hypothesis, allocation, parameter neighborhood, benchmark, cost,
sample split, or gate creates a new dated protocol and a new trial.

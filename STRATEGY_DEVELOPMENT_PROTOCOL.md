# Strategy development protocol

Frozen on 2026-08-25. This document governs research around the Roth IRA
allocator. It is not a trade instruction and cannot change production state.

## Objective

The primary objective is maximum long-run net log growth subject to a
historical maximum-drawdown boundary of 70%. Report CAGR for readability, but
do not optimize it independently. Secondary diagnostics are Sharpe, Sortino,
Calmar, expected shortfall, recovery time, rolling returns, turnover, and cost.

Production remains the benchmark until a challenger clears every applicable
gate. Research cannot change holdings, send a recommendation, or change the
live strategy fingerprint.

## Frozen production control

At latent SOXL tier `s` in `{0%, 15%, 25%, 35%}`, the sprint target is:

```text
TQQQ = 65% * (1 - s)
UGL  = 35% * (1 - s)
SOXL = s
```

The semiconductor alpha gate is positive SMH-on-QQQ residual momentum from a
separated 126-session regression and 63-session scoring window, reviewed every
21 completed sessions. QQQ above its 200-session SMA is a defensive overlay
fuse. It is not treated as an independently proven alpha source.

The causal HAR-style forecast and its conservative trailing overrides admit the
highest SOXL tier fitting the 55% volatility budget. That budget ranks and
limits candidate tiers; it does not guarantee future realized volatility.
Trend failure exits immediately, re-entry needs two distinct closes,
volatility cuts are immediate, and increases wait five completed sessions.

The permanent lifecycle ratchet changes delivery leverage according to age and
inflation-adjusted account value. It is a risk policy, not an alpha claim.

## Evidence retained from the 2026-08-25 review

- The residual signal retained a positive next-21-session excess-return spread
  under a corrected moving-block bootstrap. It remains the primary alpha gate.
- The 200-session trend rule did not establish standalone return improvement.
  It remains only as an immediate defensive SOXL exit fuse.
- A 45% volatility budget reduced the measured drawdown by less than one
  percentage point while giving up roughly three annualized growth points in
  the reviewed full-path comparison. The aggressive 55% budget remains.
- Tiering reduced drawdown substantially versus a permanent 35% SOXL overlay,
  although permanent 35% SOXL had slightly higher raw growth. Tiering remains
  the selected growth/risk compromise.
- The lifecycle ratchet reduced drawdown and modestly improved recovery versus
  remaining permanently in the sprint allocation. It remains enabled.
- The tested TQQQ/QLD/SSO/UGL challenger did not improve both growth and
  drawdown versus production and is rejected, not shipped as a live or shadow
  allocation.

These results are exploratory because the same finite history has been examined
many times. They justify retaining the existing design, not claiming a durable
out-of-sample edge.

## Data and timing rules

- Use actual point-in-time daily observations only.
- Reject synthetic pre-inception history, interpolation, ticker substitution,
  stale or incomplete latest bars, invalid prices, and zero/NaN volatility.
- Indicators use completed close `t`; the earliest modeled action is the next
  session. Same-date reruns cannot create additional evidence.
- Use walk-forward or blocked time-series validation with purging and embargo
  when labels overlap. Random k-fold validation is prohibited.
- Charge 10, 25, and 50 basis points per one-way turnover.
- Include delistings and corporate actions when testing individual stocks.

## Required benchmarks

Every complete study must report:

1. production with its actual stateful alpha, lifecycle, and drift rules;
2. the same 65/35 TQQQ/UGL core without SOXL;
3. permanent sprint, isolating lifecycle effects;
4. QLD buy-and-hold as the simple leveraged-Nasdaq anchor;
5. the challenger with no favorable change to timing or cost assumptions; and
6. a zero-cost path only as attribution, never as the decision result.

Share quantities and cash must evolve through time. Replacing holdings with
target weights each day is an implicit free rebalance and invalidates a path.

## Statistical requirements

Test components before the full portfolio. Report fixed chronological slices,
expanding-origin deterioration, 21-session moving-block-bootstrap intervals,
and the honest number of strategies and parameters examined. Bootstrap samples
must advance one reproducible random-number generator rather than reinitialize
it inside the loop.

Only predeclared local-neighborhood checks are allowed for robustness. A search
over many alternatives must be reflected in multiple-testing correction; it
cannot be relabeled as one hypothesis after seeing the winner.

## Promotion gates

A historical growth challenger must satisfy all of the following at the frozen
parameters:

- positive net log-growth improvement at 10 and 25 bps and nonnegative
  improvement at 50 bps;
- maximum drawdown no worse than 70% and no more than 2.5 percentage points
  worse than production;
- positive improvement in at least three of four fixed chronological slices,
  including the final slice;
- moving-block-bootstrap probability of nonpositive improvement at most 10%;
- 95% Deflated-Sharpe confidence using consistent daily returns and the honest
  trial count; and
- no single crisis, asset, or terminal subperiod explaining over half the gain.

Historical passage is insufficient. Promotion additionally requires at least
252 prospectively recorded completed sessions under one frozen fingerprint,
three prospective structural SOXL decisions, positive net log-growth
difference after modeled costs, complete shadow/fill reconciliation, and a
written negative-evidence review.

## Revision rule

Changing a hypothesis, allocation, parameter neighborhood, benchmark, cost,
sample split, or gate creates a new dated protocol and a new trial. A
disappointing result is never a reason to edit acceptance criteria retroactively.

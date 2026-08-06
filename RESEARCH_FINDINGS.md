# Research findings — 2026-08-05

## Decision

Keep the live Original ROTH IRA Barbell unchanged. None of the five
prespecified, one-change-at-a-time candidates earned promotion.

This is an exploratory conclusion, not a claim that the Original will
outperform in the future. The historical span was already inspected, the old A
and B daily paths are unavailable, and no future shadow period has elapsed.

## Reproducible run

- Adjusted-data source: frozen yfinance snapshot downloaded after the
  2026-08-05 completed close.
- Full-data SHA-256:
  `60329bb1dd7774ff41b782f56b1d53e53c2000a37bebe6eaef6a3763e46061e4`
- Actual common inception: 2015-10-12.
- Final session: 2026-08-05.
- Execution: signal at completed close, fill at next adjusted open.
- Starting value: $10,000 with fractional shares and zero cash return.
- Primary cost: 10bp on gross security buys plus sells.
- Inference: 20,000 paired moving-block resamples with seed 20260805.

The market snapshot and detailed ledgers are local ignored artifacts. The
command and validation protocol are in `RESEARCH.md`.

## Primary 10bp results

| Strategy | CAGR | Max drawdown | Sharpe | Annual gross turnover |
|---|---:|---:|---:|---:|
| Original | 31.58% | -39.09% | 0.94 | 3,321.62% |
| C1 wider drift buffer | 31.60% | -39.09% | 0.94 | 3,320.11% |
| C2 two-close bull re-entry | 31.23% | -39.06% | 0.94 | 2,697.02% |
| C3 volatility hysteresis | 29.82% | -36.26% | 0.93 | 3,167.36% |
| C4 63-session leader momentum | 31.98% | -38.12% | 0.95 | 3,299.12% |
| C5 equal SOXL/TECL | 31.71% | -38.29% | 0.96 | 3,243.52% |
| 50/50 SPMO/SMH | 27.50% | -32.94% | 1.09 | 7.58% |
| QLD buy-and-hold | 32.53% | -63.68% | 0.86 | 0.00% |
| SPY buy-and-hold | 15.14% | -33.72% | 0.88 | 0.00% |

QLD had 0.95 percentage point more CAGR than Original over the longer actual
common history, but its drawdown was 24.60 percentage points deeper. The
Original therefore did not dominate QLD on terminal growth in this expanded
sample; it substantially improved the drawdown path.

## Candidate evidence

| Candidate | Descriptive CAGR difference | Positive chronological slices | 95% block-bootstrap excess-growth interval | BH-adjusted one-sided p |
|---|---:|---:|---:|---:|
| C1 wider drift buffer | +0.02pp | 1 of 4 | -0.13% to +0.18% | 0.6873 |
| C2 two-close bull re-entry | -0.35pp | 2 of 4 | -3.12% to +2.62% | 0.6873 |
| C3 volatility hysteresis | -1.76pp | 2 of 4 | -4.55% to +1.35% | 0.8124 |
| C4 63-session leader momentum | +0.40pp | 3 of 4 | -1.34% to +1.94% | 0.6873 |
| C5 equal SOXL/TECL | +0.13pp | 2 of 4 | -2.01% to +2.20% | 0.6873 |

The C4 point estimate was the best, but it missed the locked 1pp materiality
hurdle and its interval crossed zero widely. The candidate-matrix
selection-instability diagnostic was 91.43% (64 of 70 CSCV-style splits), so
the apparent winner was highly unstable across subperiod selections.

C2 reduced gross turnover by about 18.8%. It trailed Original at 10bp but led by
about 0.85pp descriptive CAGR at the 25bp stress. That makes it a useful
high-friction shadow candidate, not a production replacement.

## Why the earlier report favored Original

A sensitivity beginning 2020-12-10 closely reproduces the supplied
approximately-five-year-eight-month summary:

| Strategy | Supplied CAGR | Reproduced CAGR |
|---|---:|---:|
| Original | 33.73% | 33.69% |
| 50/50 SPMO/SMH | 29.67% | 29.71% |
| QLD | 24.75% | 24.82% |
| SPY | 15.76% | 15.78% |

The Original really did rank first in that shorter technology-heavy window.
The conclusion changes when the test begins at actual common inception.

The old turnover labels also mixed conventions:

- Supplied QLD/SPY turnover was 17.73%, even though buy-and-hold has zero
  ongoing turnover. That is the initial purchase annualized into the metric.
- Correct reproduced ongoing QLD/SPY turnover is 0%.
- Supplied Original turnover was 1,576.94%. Reproduced ongoing one-way turnover
  is 1,564.09%, while gross security turnover is 3,124.43%.

Thus the old report labeled approximately one-way turnover as total turnover
and included initial funding. Costs charged on buys plus sells should use gross
security notional, so the narrative estimate of roughly 1.58% annual drag was
not the correct gross-notional estimate.

## Turnover and liquidity diagnosis

Over the full common-inception run, Original executed:

| Filled reason | Events | Cumulative gross trade fraction |
|---|---:|---:|
| Regime transition | 150 | 272.52× |
| Volatility-tier transition | 104 | 77.03× |
| Leader transition | 27 | 8.36× |
| Drift band | 5 | 0.36× |

Wider drift bands cannot solve the turnover because drift caused only five
filled events. Regime and tier transitions dominate.

The historical liquidity screen also fails: the largest modeled Original trade
was 432.21% of trailing 20-session median dollar volume, driven by very thin
early SPMO trading. A constant 10bp implementation assumption is not credible
for those observations. This is an additional reason not to promote a
historical winner from this sample.

## Next safe step

Keep all live parameters fixed. Run the Original and C2/C4 in shadow mode on
future completed sessions, record broker-observable spread/fill data, and
revisit only after at least 252 sessions and three structural transitions.
Any new candidate family must be registered before results are viewed.
